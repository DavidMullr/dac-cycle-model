"""MOF screening spatial v1, a versioned extension of the retained V21 model.

The nominal model is a parallel-channel surrogate.  Each cell has its own solid
and gas state and its own pressure.  Feed, leak and withdrawal are split by the
declared channel shares; there is no transverse gas exchange.  The effective
bulk conductivity remains the only through-thickness heat path, so gas
conduction is not counted twice.

The source identity is new. Local gas/solid, heater/wall, pressure-control and
utilities are retained from the reviewed V21 implementation. Pump conditioning
uses the reviewed portable ideal 55 C knockout helper. The storage vector is
generalized from four species to one or two CO2 site states plus water and,
when supplied by the material, inert adsorbates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from math import isfinite
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

from ..airflow import airflow
from .. import pump_profile
from ..common import (CP_GAS, MW, R, SPECIES, ExpectedScenarioError,
                     gas_enthalpy_mol,
                     gas_internal_energy_mol, pump_profile_for,
                     water_saturation_pressure_Pa)
from ..coupled_control_v10 import (
    CAPACITY_PRESSURE_BASIS,
    pressure_control_observable,
    pressure_control_pressure,
)
from ..cycle_control_v8 import DesorptionControlRuntime, load_desorption_control
from ..feed_dryer import build_dryer
from ..geometry import Geometry
from ..systems import build_systems, Systems
from ..source_contract_v21 import (
    SourceContractV21Error,
    require_declared_thermal_conversion,
    source_execution_diagnostic,
)
from ..utilities import build_utilities
from ..vacuum import pump_export
from ..vacuum_qualification_v11 import MODES, rate_at_pressure
from .lumped_port.pump_conditioner import (
    evaluate_pump_conditioner,
    pump_conditioner_config,
)
from .spatial_jacobian import (
    BRANCH_POLICY_ENV,
    BRANCH_POLICY_V1,
    JACOBIAN_CACHE_POLICY_ENV,
    JACOBIAN_CACHE_POLICY_EXACT,
    ExactMaterialEvaluationCache,
    SpatialJacobianAudit,
    bounded_spatial_jacobian,
    bounded_spatial_jacobian_with_material_cache,
)
from .bdf_trial_recovery import make_recovering_bdf, merge_trackers

STEPS = ("adsorption", "evacuation", "desorption", "cooling", "repressurization")
PUMPED_PHASES = ("evacuation", "desorption", "cooling")
COLLECTING_PHASES = ("desorption", "cooling")
FIRST_LAW_RELATIVE_TOLERANCE = 1e-8
RUNTIME_FIRST_LAW_BACKSTOP = 1e-6
COOLING_TOLERANCE_K = 0.1
BED_OVERSHOOT_GUARD_K = 5.0
PUMP_FLOW_FLOOR_MOL_S = 1e-12
SPATIAL_SOLVER_POLICY_BDF = "recovering-bdf/v1"
SPATIAL_SOLVER_POLICY_RADAU = "radau-same-rhs-jac/v1"
SPATIAL_SOLVER_POLICY_ENV = "MOF_SPATIAL_SOLVER_POLICY"
SPATIAL_SOLVER_POLICY_SCHEMA = "mof-screening-spatial-solver-policy/1"


def _branch_jacobian_policy(c: dict) -> str | None:
    """Return the explicit source-bound branch policy or reject a typo."""

    runtime_control = c.get("_runtime_control") or {}
    policy = runtime_control.get(
        "branch_jacobian_policy", os.environ.get(BRANCH_POLICY_ENV)
    )
    if policy not in {None, BRANCH_POLICY_V1}:
        raise ValueError(
            "unknown branch-aware Jacobian policy: "
            f"{policy!r}; expected {BRANCH_POLICY_V1!r}"
        )
    return policy


def _jacobian_cache_policy(c: dict) -> str | None:
    """Return the explicit per-Jacobian material-cache policy."""

    runtime_control = c.get("_runtime_control") or {}
    policy = runtime_control.get(
        "jacobian_cache_policy", os.environ.get(JACOBIAN_CACHE_POLICY_ENV)
    )
    if policy not in {None, JACOBIAN_CACHE_POLICY_EXACT}:
        raise ValueError(
            "unknown Jacobian cache policy: "
            f"{policy!r}; expected {JACOBIAN_CACHE_POLICY_EXACT!r}"
        )
    return policy


def resolved_spatial_solver_policy(c: dict) -> dict:
    """Resolve the explicit spatial integration method without changing RHS."""

    runtime_control = c.get("_runtime_control") or {}
    configured = runtime_control.get(
        "spatial_solver_policy", os.environ.get(SPATIAL_SOLVER_POLICY_ENV)
    )
    if configured not in (
        None,
        SPATIAL_SOLVER_POLICY_BDF,
        SPATIAL_SOLVER_POLICY_RADAU,
    ):
        raise ValueError(
            "unknown spatial solver policy: "
            f"{configured!r}; expected one of "
            f"{SPATIAL_SOLVER_POLICY_BDF!r}, "
            f"{SPATIAL_SOLVER_POLICY_RADAU!r}"
        )
    policy = (
        SPATIAL_SOLVER_POLICY_BDF
        if configured is None
        else configured
    )
    radau = policy == SPATIAL_SOLVER_POLICY_RADAU
    return {
        "schema": SPATIAL_SOLVER_POLICY_SCHEMA,
        "policy": policy,
        "integration_method": "Radau" if radau else "recovering_BDF",
        "recovery_wrapper": not radau,
        "activation_source": (
            "default"
            if configured is None
            else "environment_or_runtime_control"
        ),
        "environment_variable": SPATIAL_SOLVER_POLICY_ENV,
        "validated": True,
    }


@dataclass(frozen=True)
class ReviewedConditionedStream:
    """Spatial view of the shared ideal pump-inlet conditioner result."""

    n_raw_mol_s: float
    n_vapour_mol_s: float
    n_condensate_mol_s: float
    vapour_composition: np.ndarray
    duty_W: float
    target_shortfall_K: float
    p_pump_Pa: float
    t_out_K: float
    details: dict


def pump_flow_basis(c: dict) -> str:
    value = (c.get("coupled_through_thickness") or {}).get("pump_flow_basis")
    if value not in MODES:
        raise ValueError(
            "coupled_through_thickness.pump_flow_basis is required and must be "
            f"one of {MODES}; got {value!r}."
        )
    return value


def _pump_rates_v21(c: dict, pressure_Pa: float, temperature_K: float,
                    *, profile=None) -> dict:
    """Total installed bank evaluated once.

    The default fixed-standard-reference interpretation treats the tabulated
    ordinate as volume at the pump profile's reference pressure/temperature.
    The actual-suction-volume alternative is retained only as an explicit
    scenario because the vendor curve does not settle that basis.
    """
    profile = pump_profile_for(c) if profile is None else profile
    if profile is None:
        raise ValueError("A reference pump profile is required for every V21 coupled case.")
    bank = pump_profile.bank_count(c)
    basis = pump_flow_basis(c)
    rate = rate_at_pressure(
        profile, pressure_Pa, temperature_K, bank_count=bank, mode=basis
    )
    return {
        "molar_flow_mol_s": rate["molar_flow_mol_s"],
        "electric_power_W": rate["electric_power_W"],
        "actual_inlet_volumetric_speed_m3_s": rate["actual_inlet_volumetric_speed_m3_s"],
        "plotted_curve_volumetric_flow_m3_s": rate["plotted_curve_volumetric_flow_m3_s"],
        "equivalent_reference_condition_volumetric_flow_m3_s":
            rate["equivalent_reference_condition_volumetric_flow_m3_s"],
        "pressure_normalised_reference_flow_speed_m3_s":
            rate["pressure_normalised_reference_flow_speed_m3_s"],
        "curve_clamped": rate["upper_domain_exceeded"],
        "media_temperature_overrun_K": rate["media_temperature_overrun_K"],
        "throttle": 1.0,
        "bank_count": bank,
        "flow_basis": basis,
        "pump_rate_model": basis,
        "profile_reference_pressure_Pa": rate["profile_reference_pressure_Pa"],
        "profile_reference_temperature_K": rate["profile_reference_temperature_K"],
    }


def _reviewed_conditioned_stream(
    c: dict,
    *,
    p_suction_Pa: float,
    bed_temperature_K: float,
    bed_mole_fractions: np.ndarray,
    opening: float,
    profile,
) -> tuple[ReviewedConditionedStream, dict]:
    """Evaluate the reviewed zero-drop pump conditioner for spatial cells."""
    conditioner = pump_conditioner_config(c)
    if conditioner is None:
        raise ValueError("reviewed pump conditioner is not configured")
    pump = _pump_rates_v21(
        c,
        p_suction_Pa,
        min(
            float(bed_temperature_K),
            conditioner.target_temperature_K,
        ),
        profile=profile,
    )
    details = evaluate_pump_conditioner(
        bed_mole_fractions=bed_mole_fractions,
        pressure_Pa=p_suction_Pa,
        bed_temperature_K=bed_temperature_K,
        pump_capacity_mol_s=float(pump["molar_flow_mol_s"]),
        alpha=opening,
        target_temperature_K=conditioner.target_temperature_K,
        total_cooling_power_W=conditioner.total_cooling_power_W,
        cooling_cop=conditioner.cooling_cop,
        pump_electric_power_W=float(pump["electric_power_W"]),
    )
    conditioned = np.asarray(
        details["conditioned_gas_mol_s"], dtype=float
    )
    conditioned_total = float(conditioned.sum())
    composition = np.divide(
        conditioned,
        conditioned_total,
        out=np.zeros(4, dtype=float),
        where=conditioned_total > 1e-30,
    )
    stream = ReviewedConditionedStream(
        n_raw_mol_s=float(details["bed_flow_mol_s"]),
        n_vapour_mol_s=conditioned_total,
        n_condensate_mol_s=float(details["liquid_total_flow_mol_s"]),
        vapour_composition=composition,
        duty_W=float(details["conditioner_cooling_W"]),
        target_shortfall_K=0.0,
        p_pump_Pa=float(p_suction_Pa),
        t_out_K=float(details["outlet_temperature_K"]),
        details=details,
    )
    return stream, pump


@dataclass(frozen=True)
class CoupledLayout:
    n_cells: int
    n_storage: int
    size: int
    extended: bool
    T_h: int = 0
    T_w: int = 1
    T_bed: slice = slice(2, 2)
    T_gas: slice = slice(2, 2)
    q: slice = slice(2, 2)
    n_gas: slice = slice(2, 2)
    inlet: slice = slice(2, 6)
    vent: slice = slice(6, 10)
    product: slice = slice(10, 14)
    E_heat: int = 14
    E_cool: int = 15
    E_fan: int = 16
    E_vac: int = 17
    E_cond: int = 18
    E_loss: int = 19
    E_netflow: int = 20
    E_electric_total: int = 21
    raw: slice | None = None
    condensate: int | None = None
    condensate_product: int | None = None
    e_source: int | None = None
    e_source_input: int | None = None
    e_source_aux: int | None = None
    e_cooler: int | None = None
    e_cooler_aux: int | None = None

    def as_dict(self) -> dict:
        return {
            "model": "local_gas_parallel_channel",
            "n_cells_half": self.n_cells,
            "n_storage": self.n_storage,
            "size": self.size,
            "extended": self.extended,
            "T_h": self.T_h, "T_w": self.T_w,
            "T_bed": [self.T_bed.start, self.T_bed.stop],
            "T_gas": [self.T_gas.start, self.T_gas.stop],
            "q": [self.q.start, self.q.stop],
            "n_gas": [self.n_gas.start, self.n_gas.stop],
            "inlet": [self.inlet.start, self.inlet.stop],
            "vent": [self.vent.start, self.vent.stop],
            "product": [self.product.start, self.product.stop],
            "E_heat": self.E_heat, "E_cool": self.E_cool,
            "E_fan": self.E_fan, "E_vac": self.E_vac,
            "E_cond": self.E_cond, "E_loss": self.E_loss,
            "E_netflow": self.E_netflow,
            "E_electric_total": self.E_electric_total,
            "raw": None if self.raw is None else [self.raw.start, self.raw.stop],
            "condensate": self.condensate,
            "condensate_product": self.condensate_product,
            "e_source": self.e_source,
            "e_source_input": self.e_source_input,
            "e_source_aux": self.e_source_aux,
            "e_cooler": self.e_cooler,
            "e_cooler_aux": self.e_cooler_aux,
        }


def state_layout(n_cells: int, *, extended: bool, n_storage: int = 4) -> CoupledLayout:
    if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells < 1:
        raise ValueError("n_cells_half must be a positive integer.")
    q0 = 2 + 2 * n_cells
    n0 = q0 + n_storage * n_cells
    inlet0 = n0 + 4 * n_cells
    vent0 = inlet0 + 4
    product0 = vent0 + 4
    e0 = product0 + 4
    base = e0 + 8
    layout = CoupledLayout(
        n_cells=n_cells, size=base, extended=extended,
        n_storage=n_storage,
        T_bed=slice(2, 2 + n_cells), T_gas=slice(2 + n_cells, 2 + 2 * n_cells),
        q=slice(q0, n0), n_gas=slice(n0, inlet0),
        inlet=slice(inlet0, vent0), vent=slice(vent0, product0),
        product=slice(product0, e0),
        E_heat=e0, E_cool=e0 + 1, E_fan=e0 + 2, E_vac=e0 + 3,
        E_cond=e0 + 4, E_loss=e0 + 5, E_netflow=e0 + 6,
        E_electric_total=e0 + 7,
    )
    if not extended:
        return layout
    raw0 = base
    return replace(
        layout, size=base + 11, extended=True,
        raw=slice(raw0, raw0 + 4), condensate=raw0 + 4,
        condensate_product=raw0 + 5, e_source=raw0 + 6,
        e_source_input=raw0 + 7, e_source_aux=raw0 + 8,
        e_cooler=raw0 + 9, e_cooler_aux=raw0 + 10,
    )


@dataclass(frozen=True)
class CoupledModel:
    layout: CoupledLayout
    n_cells: int
    n_storage: int
    dx_m: float
    half_face_area_m2: float
    half_bed_volume_m3: float
    half_gas_volume_m3: float
    cell_volume_m3: np.ndarray
    cell_gas_volume_m3: np.ndarray
    cell_mass_kg: np.ndarray
    cell_flow_share: np.ndarray
    solid_capacity_J_K: np.ndarray
    wall_capacity_J_K: float
    heater_capacity_J_K: float
    heater_wall_conductance_W_K: float
    wall_bed_resistance_K_W: float
    cell_conductance_W_K: float
    gas_solid_area_m2: np.ndarray
    gas_solid_h_W_m2_K: float
    wall_cooling_capacity_W: float | None
    symmetry_factor: int = 2

    def as_dict(self) -> dict:
        return {
            "model": "local_gas_parallel_channel",
            "n_cells_half": self.n_cells,
            "n_storage": self.n_storage,
            "dx_m": self.dx_m,
            "half_face_area_m2": self.half_face_area_m2,
            "half_bed_volume_m3": self.half_bed_volume_m3,
            "half_gas_volume_m3": self.half_gas_volume_m3,
            "cell_mass_kg": self.cell_mass_kg.tolist(),
            "cell_flow_share": self.cell_flow_share.tolist(),
            "wall_capacity_J_K": self.wall_capacity_J_K,
            "heater_capacity_J_K": self.heater_capacity_J_K,
            "wall_bed_resistance_K_W": self.wall_bed_resistance_K_W,
            "cell_conductance_W_K": self.cell_conductance_W_K,
            "gas_solid_h_W_m2_K": self.gas_solid_h_W_m2_K,
            "gas_solid_area_m2": self.gas_solid_area_m2.tolist(),
            "wall_cooling_capacity_W": self.wall_cooling_capacity_W,
            "symmetry_factor": self.symmetry_factor,
            "transverse_gas_exchange": "zero",
        }


def _required_number(section: dict, key: str, name: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"{name} is required and must be finite.")
    return float(value)


def build_model(c: dict, material, g: Geometry, *, systems: Systems | None = None,
                utilities=None) -> CoupledModel:
    systems = build_systems(c) if systems is None else systems
    utilities = build_utilities(c) if utilities is None else utilities
    section = c.get("coupled_through_thickness")
    if not isinstance(section, dict):
        raise ValueError("coupled_through_thickness section is required for the V21 backend.")
    n = section.get("n_cells_half")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("coupled_through_thickness.n_cells_half must be a positive integer.")
    hgs = _required_number(section, "gas_solid_heat_transfer_W_m2_K",
                           "coupled_through_thickness.gas_solid_heat_transfer_W_m2_K")
    if hgs < 0.0:
        raise ValueError("gas-solid heat-transfer coefficient must be nonnegative.")
    area = section.get("gas_solid_specific_area_m2_m3")
    if area is None:
        eps = float(c["bed"]["bed_void_fraction"])
        dp = float(c["bed"]["particle_diameter_m"])
        area = 6.0 * (1.0 - eps) / dp
    area = float(area)
    if area <= 0.0 or not isfinite(area):
        raise ValueError("gas-solid specific area must be positive and finite.")

    L = float(g.heat_path_m)
    A_h = 0.5 * float(g.heated_area_m2)
    V_bh = 0.5 * float(g.bed_volume_m3)
    V_gh = 0.5 * float(g.gas_volume_m3)
    dx = L / n
    V_bj = np.full(n, A_h * dx)
    V_gj = V_gh * (V_bj / V_bj.sum())
    m_j = np.full(n, 0.5 * float(c["bed"]["dry_sorbent_mass_kg"]) / n)
    share = V_bj / V_bj.sum()
    th = c["thermal"]
    source = systems.source
    wall_capacity = 0.5 * (float(g.metal_mass_kg) * float(c["geometry"]["metal_cp_J_kg_K"])
                           + (0.0 if source is None else source.jacket_heat_capacity_J_K))
    heater_capacity = 0.0 if source is not None else 0.5 * float(th["heater_heat_capacity_J_K"])
    k = float(th["effective_bed_conductivity_W_m_K"])
    h_contact = float(th["wall_contact_h_W_m2_K"])
    if n == 1:
        r_wb = L / (k * A_h) + 1.0 / (h_contact * A_h)
    else:
        r_wb = 1.0 / (h_contact * A_h) + dx / (2.0 * k * A_h)
    wall_cap = None if utilities is None else utilities.wall_cooling_capacity_W()
    if wall_cap is not None:
        wall_cap = 0.5 * float(wall_cap)
    return CoupledModel(
        layout=state_layout(
            n,
            extended=systems.layout.extended,
            n_storage=material.n_storage,
        ),
        n_cells=n, n_storage=material.n_storage, dx_m=dx, half_face_area_m2=A_h,
        half_bed_volume_m3=V_bh, half_gas_volume_m3=V_gh,
        cell_volume_m3=V_bj, cell_gas_volume_m3=V_gj, cell_mass_kg=m_j,
        cell_flow_share=share,
        solid_capacity_J_K=m_j * float(material.data["solid_cp_J_kg_K"]),
        wall_capacity_J_K=wall_capacity,
        heater_capacity_J_K=heater_capacity,
        heater_wall_conductance_W_K=0.5 * float(th["heater_wall_conductance_W_K"]),
        wall_bed_resistance_K_W=r_wb,
        cell_conductance_W_K=k * A_h / dx,
        gas_solid_area_m2=area * V_bj,
        gas_solid_h_W_m2_K=hgs,
        wall_cooling_capacity_W=wall_cap,
    )


def state_pressures(state: np.ndarray, model: CoupledModel) -> np.ndarray:
    lay = model.layout
    n = state[lay.n_gas].reshape(model.n_cells, 4)
    T = state[lay.T_gas]
    return np.maximum(n, 0.0).sum(axis=1) * R * T / model.cell_gas_volume_m3


def check_inventory(state: np.ndarray, model: CoupledModel, *, where: str = "state") -> None:
    """Reject a nonphysical negative local inventory instead of clipping it."""
    lay = model.layout
    for name, block in (("T_bed", state[lay.T_bed]), ("T_gas", state[lay.T_gas]),
                        ("q", state[lay.q]), ("n_gas", state[lay.n_gas])):
        if not np.all(np.isfinite(block)):
            raise ExpectedScenarioError(f"Non-finite {name} at {where}.")
    if float(np.min(state[lay.n_gas])) < -1e-7:
        raise ExpectedScenarioError(
            f"Negative local gas inventory at {where}: minimum {float(np.min(state[lay.n_gas])):g} mol.")
    if float(np.min(state[lay.q])) < -1e-6:
        raise ExpectedScenarioError(
            f"Negative local loading at {where}: minimum {float(np.min(state[lay.q])):g} mol/kg.")


def check_inventory_rows(states: np.ndarray, model: CoupledModel, *, where: str) -> None:
    for index, row in enumerate(np.atleast_2d(states)):
        check_inventory(row, model, where=f"{where}[{index}]")


def reset_current_cycle_state_meters(state: np.ndarray, layout) -> None:
    """Clear cumulative and species meters without using prior-cycle totals."""
    state[layout.inlet.start:] = 0.0


def species_relative_closure(residual: np.ndarray, throughput: np.ndarray,
                             *, floor: float = 1e-9) -> np.ndarray:
    """Per-species relative closure using that species' own throughput."""
    residual = np.asarray(residual, dtype=float)
    throughput = np.asarray(throughput, dtype=float)
    return np.abs(residual) / np.maximum(np.abs(throughput), np.full(4, floor))


def initial_state(c: dict, material, g: Geometry, model: CoupledModel) -> np.ndarray:
    lay = model.layout
    Ta = c["ambient"]["temperature_C"] + 273.15
    y = _ambient_feed_composition(c)
    state = np.zeros(lay.size)
    state[lay.T_h] = Ta
    state[lay.T_w] = Ta
    state[lay.T_bed] = Ta
    state[lay.T_gas] = Ta
    n_cell = c["ambient"]["pressure_Pa"] * model.cell_gas_volume_m3 / (R * Ta)
    state[lay.n_gas] = (n_cell[:, None] * y[None, :]).ravel()
    if c.get("initial_state", {}).get("basis") == "equilibrated":
        partials = y * float(c["ambient"]["pressure_Pa"])
        loading = material.equilibrium(Ta, partials)
        state[lay.q] = np.tile(loading, model.n_cells)
    return state


def _temperature_limit(c: dict) -> float:
    return float(c["material_maximum_temperature_C"]) if "material_maximum_temperature_C" in c else float("nan")


def energy_inventory(c: dict, material, g: Geometry, state: np.ndarray,
                     model: CoupledModel, *, systems: Systems | None = None) -> dict:
    systems = build_systems(c) if systems is None else systems
    lay = model.layout
    ref = float(material.reference_temperature_K)
    m = material.data
    q = state[lay.q].reshape(model.n_cells, model.n_storage)
    T = state[lay.T_bed]
    solid = float(np.sum(model.cell_mass_kg * material.solid_cp_J_kg_K * (T - ref)))
    storage_potential = [
        material.storage_energy_terms(float(value), q[index])
        for index, value in enumerate(T)
    ]
    adsorbed = float(np.sum([
        model.cell_mass_kg[index] * value[0]
        for index, value in enumerate(storage_potential)
    ]))
    n = state[lay.n_gas].reshape(model.n_cells, 4)
    Tg = state[lay.T_gas]
    gas = float(np.sum(n * gas_internal_energy_mol(Tg[:, None], ref)))
    heater = 0.0 if systems.source is not None else model.heater_capacity_J_K * (state[lay.T_h] - ref)
    metal = model.wall_capacity_J_K * (state[lay.T_w] - ref)
    return {"total_J": float(heater + metal + solid + adsorbed + gas),
            "heater_J": float(heater), "metal_J": float(metal),
            "solid_J": solid, "adsorbed_J": adsorbed, "gas_J": gas,
            "reference_temperature_K": ref}


def phase_actuator_request(
    c: dict,
    *,
    phase: str,
    bed_temperature_K: float,
    ambient_temperature_K: float,
    fixed_bed_balance_W: float,
) -> tuple[float, float, float]:
    """Return shared-policy heater/cooler requests and target in full watts."""
    policy = c["operating_policy"]["phases"][phase]
    request = policy["request"]
    regeneration_K = (
        float(c["cycle"]["regeneration_temperature_C"]) + 273.15
    )
    heater_limit = float(c["thermal"]["heater_power_W"])
    cooling_limit = float(
        c["utilities"]["wall_cooling"]["max_cooling_capacity_W"]
    )
    heat_W = 0.0
    cool_W = 0.0
    if request == "setpoint":
        gain = float(policy["gain_W_K"])
        target_K = ambient_temperature_K
        signed = gain * (target_K - bed_temperature_K)
        heat_W = min(max(signed, 0.0), heater_limit)
        cool_W = min(max(-signed, 0.0), cooling_limit)
    elif request == "positive_setpoint":
        gain = float(policy["gain_W_K"])
        target_K = regeneration_K
        heat_W = min(
            heater_limit,
            gain * max(target_K - bed_temperature_K, 0.0),
        )
    elif request == "positive_sensible_cooling":
        target_K = ambient_temperature_K
        gain = float(policy["gain_W_K"])
        cool_W = min(
            cooling_limit,
            gain * max(bed_temperature_K - target_K, 0.0),
        )
    elif request == "zero_net_bed_energy_imbalance":
        target_K = ambient_temperature_K
        heat_W = min(max(-fixed_bed_balance_W, 0.0), heater_limit)
        cool_W = min(max(fixed_bed_balance_W, 0.0), cooling_limit)
    else:
        raise ValueError(f"unsupported phase actuator request: {request}")
    return float(heat_W), float(cool_W), float(target_K)


def source_input_per_delivered_W(source: Systems.source | None) -> float:
    """Finite fixed-carrier conversion factor, evaluated once per factory use."""
    if source is None:
        return 0.0
    factor = source.thermal_input_W(1.0)
    if factor is None or not isfinite(float(factor)) or float(factor) < 0.0:
        raise SourceContractV21Error(
            "External source thermal conversion is unavailable or nonfinite."
        )
    return float(factor)


def _weather_feed_composition(row) -> np.ndarray:
    return _feed_composition(
        water_mole_fraction=float(row.water_mole_fraction),
        co2_ppm_dry=float(row.co2_ppm_dry),
    )


def _feed_composition(
    *, water_mole_fraction: float, co2_ppm_dry: float
) -> np.ndarray:
    water = float(water_mole_fraction)
    if not isfinite(water) or not 0.0 <= water < 1.0:
        raise ValueError("canonical water mole fraction must be in [0, 1)")
    co2 = float(co2_ppm_dry) * 1e-6
    dry = np.array(
        [co2, 0.0, (1.0 - co2) * 0.7905, (1.0 - co2) * 0.2095]
    )
    return dry * (1.0 - water) + np.array([0.0, water, 0.0, 0.0])


def _ambient_feed_composition(c: dict) -> np.ndarray:
    ambient = c["ambient"]
    if "water_mole_fraction" not in ambient:
        raise ValueError(
            "ambient water_mole_fraction is required; do not derive inlet "
            "forcing with the source material's Antoine reference law"
        )
    return _feed_composition(
        water_mole_fraction=float(ambient["water_mole_fraction"]),
        co2_ppm_dry=float(ambient["co2_ppm_dry"]),
    )


def cycle_rhs(c: dict, material, g: Geometry, phase: str, *, systems: Systems | None = None,
              dryer=None, utilities=None, model: CoupledModel | None = None):
    systems = build_systems(c) if systems is None else systems
    require_declared_thermal_conversion(c, systems.source)
    utilities = build_utilities(c) if utilities is None else utilities
    dryer = build_dryer(c) if dryer is None else dryer
    model = build_model(c, material, g, systems=systems, utilities=utilities) if model is None else model
    lay = model.layout
    m = material.data
    Ta = c["ambient"]["temperature_C"] + 273.15
    Patm = float(c["ambient"]["pressure_Pa"])
    ref = float(material.reference_temperature_K)
    y_feed = _ambient_feed_composition(c)
    y_leak = _ambient_feed_composition(c)
    flow = airflow(c, g)
    feed_total = flow["configured_basis_mol_s"]
    if dryer is not None:
        from ..cycle import conditioned_feed_mol_s
        feed_total = conditioned_feed_mol_s(dryer, feed_total)
    # One symmetric half of the full train. Apply the 0.5 factor exactly once
    # to every inlet/leak/repressurization stream before channel sharing.
    feed_half = 0.5 * float(feed_total)
    leak_half = (0.5 * float(c["pump"]["air_leak_Pa_m3_s"]) / (R * Ta)
                 if phase in PUMPED_PHASES else 0.0)
    repress_half = 0.5 * float(c["cycle"]["repressurization_flow_mol_s"])
    pump_on = phase in PUMPED_PHASES
    control_observable = pressure_control_observable(c)
    profile = pump_profile_for(c)
    source = systems.source
    source_half = None if source is None else replace(
        source,
        source_to_wall_UA_W_K=0.5 * source.source_to_wall_UA_W_K,
        power_limit_W=0.5 * source.power_limit_W,
        jacket_heat_capacity_J_K=0.5 * source.jacket_heat_capacity_J_K,
        auxiliary_electric_W=0.5 * source.auxiliary_electric_W,
    )
    source_input_factor = (
        source_input_per_delivered_W(source_half)
        if source_half is not None and phase == "desorption"
        else 0.0
    )
    conditioner_spec = systems.conditioner if pump_on else None
    reviewed_conditioner = (
        pump_conditioner_config(c) if pump_on else None
    )
    hgs = model.gas_solid_h_W_m2_K
    share = model.cell_flow_share
    pump_off = {"molar_flow_mol_s": 0.0, "electric_power_W": 0.0,
                "effective_speed_m3_s": 0.0, "unthrottled_speed_m3_s": 0.0,
                "equivalent_speed_m3_s": 0.0, "curve_clamped": False,
                "media_temperature_overrun_K": 0.0, "throttle": 0.0}
    weather_cache = {}
    branch_policy = _branch_jacobian_policy(c)
    branch_aware = branch_policy == BRANCH_POLICY_V1
    if branch_aware and not callable(
        getattr(material, "rates_with_directional_delta", None)
    ):
        raise ValueError(
            "branch-aware Jacobian requires directional-rate diagnostics"
        )
    cache_policy = _jacobian_cache_policy(c)
    material_cache = None
    if cache_policy == JACOBIAN_CACHE_POLICY_EXACT:
        if not callable(
            getattr(material, "rates_with_directional_delta", None)
        ) or not callable(
            getattr(material, "storage_energy_terms", None)
        ):
            raise ValueError(
                "exact Jacobian cache requires material rate and energy "
                "evaluation methods"
            )
        material_cache = ExactMaterialEvaluationCache(material)
        material = material_cache
    runtime_control = c.get("_runtime_control")
    if isinstance(runtime_control, dict):
        runtime_control["resolved_jacobian_cache_policy"] = cache_policy

    def evaluate(_t: float, y: np.ndarray) -> tuple[np.ndarray, dict]:
        runtime_control = c.get("_runtime_control") or {}
        cancel_check = runtime_control.get("cancel_check")
        if cancel_check is not None:
            cancel_check()
        deadline = runtime_control.get("deadline_monotonic")
        if deadline is not None and time.monotonic() > float(deadline):
            raise ExpectedScenarioError(
                "Integration exceeded its declared wall-time cap."
            )
        temperature_limit = (
            runtime_control["temperature_guard_limit_K"]
            if "temperature_guard_limit_K" in runtime_control
            else float(material.data["maximum_bed_temperature_C"]) + 273.15
        )
        if (
            temperature_limit is not None
            and float(np.max(y[model.layout.T_bed]))
            > float(temperature_limit) + 1e-9
        ):
            raise ExpectedScenarioError(
                "Scenario temperature guard reached; revise the policy or case."
            )
        progress_callback = runtime_control.get("progress_callback")
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "integration_step",
                    "cycle": int(runtime_control.get("cycle_number", 1)),
                    "phase": phase,
                    "elapsed_s": float(
                        runtime_control.get("cycle_start_s", 0.0)
                    )
                    + float(runtime_control.get("phase_offset_s", 0.0))
                    + float(_t),
                    "phase_elapsed_s": float(_t),
                    "phase_duration_s": float(
                        runtime_control.get("phase_duration_s", 0.0)
                    ),
                    "weather_index": int(
                        float(
                            runtime_control.get("cycle_start_s", 0.0)
                        )
                        // 3600.0
                    ),
                }
            )
        provider = runtime_control.get("weather_provider")
        current_weather = None
        if provider is not None:
            current_weather = provider(
                float(runtime_control.get("cycle_start_s", 0.0))
                + float(runtime_control.get("phase_offset_s", 0.0))
                + float(_t)
            )
            Ta = float(current_weather.temperature_C) + 273.15
            Patm = float(current_weather.pressure_Pa)
            y_feed = _weather_feed_composition(current_weather)
            y_leak = y_feed.copy()
            cache_key = current_weather.interval_end_utc
            if cache_key not in weather_cache:
                dynamic_config = dict(c)
                dynamic_config["ambient"] = {
                    "temperature_C": current_weather.temperature_C,
                    "pressure_Pa": current_weather.pressure_Pa,
                    "relative_humidity": current_weather.relative_humidity,
                    "co2_ppm_dry": current_weather.co2_ppm_dry,
                }
                weather_cache[cache_key] = airflow(dynamic_config, g)
            flow = weather_cache[cache_key]
            feed_total = float(flow["configured_basis_mol_s"])
            feed_half = 0.5 * feed_total
            leak_half = (
                0.5 * float(c["pump"]["air_leak_Pa_m3_s"]) / (R * Ta)
                if phase in PUMPED_PHASES
                else 0.0
            )
        else:
            Ta = float(c["ambient"]["temperature_C"]) + 273.15
            Patm = float(c["ambient"]["pressure_Pa"])
            y_feed = _ambient_feed_composition(c)
            y_leak = _ambient_feed_composition(c)
            flow = airflow(c, g)
            feed_total = float(flow["configured_basis_mol_s"])
            feed_half = 0.5 * feed_total
            leak_half = (
                0.5 * float(c["pump"]["air_leak_Pa_m3_s"]) / (R * Ta)
                if phase in PUMPED_PHASES
                else 0.0
            )
        Tbed = y[lay.T_bed]
        Tg = y[lay.T_gas]
        q = y[lay.q].reshape(model.n_cells, model.n_storage)
        ng = np.maximum(y[lay.n_gas].reshape(model.n_cells, 4), 0.0)
        ntot = ng.sum(axis=1)
        ygas = ng / np.maximum(ntot[:, None], 1e-30)
        p = ntot * R * Tg / model.cell_gas_volume_m3
        storage_rates = []
        species_rates = []
        directional_rate_deltas = []
        storage_cp_rows = []
        storage_gradient_rows = []
        species_energy_for_storage_rows = []
        for j in range(model.n_cells):
            partials_Pa = ygas[j] * p[j]
            try:
                if branch_aware:
                    dq, dn_species, branch_deltas = (
                        material.rates_with_directional_delta(
                            float(Tbed[j]), partials_Pa, q[j]
                        )
                    )
                    directional_rate_deltas.append(
                        tuple(
                            (j, index, delta)
                            for index, delta in branch_deltas
                        )
                    )
                else:
                    dq, dn_species = material.rates(
                        float(Tbed[j]), partials_Pa, q[j]
                    )
            except Exception as error:
                diagnostics = dict(getattr(error, "diagnostics", None) or {})
                diagnostics.setdefault("cell", int(j))
                diagnostics.setdefault(
                    "query_state_vector",
                    np.asarray(y, dtype=float).tolist(),
                )
                try:
                    error.diagnostics = diagnostics
                except Exception:
                    pass
                recorder = getattr(
                    material, "record_failure_context", None
                )
                if callable(recorder):
                    recorder(
                        {
                            "schema": (
                                "mof-screening-material-domain-failure/1"
                            ),
                            "phase": phase,
                            "local_time_s": float(_t),
                            "cell": int(j),
                            "cell_count": int(model.n_cells),
                            "temperature_K": float(Tbed[j]),
                            "partial_pressures_Pa": np.asarray(
                                partials_Pa, dtype=float
                            ).tolist(),
                            "raw_storage_mol_kg": np.asarray(
                                q[j], dtype=float
                            ).tolist(),
                            "rejected_state_vector": np.asarray(
                                y, dtype=float
                            ).tolist(),
                            "rejection_diagnostics": getattr(
                                error, "diagnostics", {}
                            ),
                            "accepted_checkpoint_not_updated": True,
                        }
                    )
                raise
            storage_rates.append(dq)
            species_rates.append(model.cell_mass_kg[j] * dn_species)
            _e_ads, cp_ads, h_ads = material.storage_energy_terms(
                float(Tbed[j]), q[j]
            )
            storage_cp_rows.append(float(cp_ads))
            storage_gradient_rows.append(h_ads)
            species_energy_for_storage_rows.append(
                gas_internal_energy_mol(float(Tg[j]), ref)[
                    material.species_for_storage
                ]
            )
        dq = np.vstack(storage_rates)
        if branch_aware:
            evaluate.last_directional_rate_deltas = tuple(
                item
                for cell_values in directional_rate_deltas
                for item in cell_values
            )
        S = np.vstack(species_rates)
        cp_ads = np.asarray(storage_cp_rows, dtype=float)
        h_ads = np.vstack(storage_gradient_rows)
        ua_species = np.vstack(species_energy_for_storage_rows)
        cp_mix = ygas @ CP_GAS
        cv_mix = ygas @ (CP_GAS - R)
        ug = gas_internal_energy_mol(Tg[:, None], ref)
        phi = hgs * model.gas_solid_area_m2 * (Tg - Tbed)

        # The valve command uses the selected observable.  Pump capacity and
        # the conditioner are always evaluated at the minimum actual suction
        # and the resulting conditioned pump inlet.
        p_min = float(np.min(p))
        p_suction = p_min if pump_on else Patm
        p_control = (pressure_control_pressure(p, control_observable)
                     if pump_on else Patm)
        opening = None
        stream = None
        pump_eval = None
        p_pump_inlet = None
        t_pump_inlet = None
        w_total_full = 0.0
        reviewed_conditioner_details = None
        y_mix = (ygas * share[:, None]).sum(axis=0)
        # Mix the outlet streams by species enthalpy, not by arithmetic
        # temperature.  Equal channel shares are the declared withdrawal rule.
        h_mix_mol = float(np.sum(share[:, None] * ygas
                                 * gas_enthalpy_mol(Tg[:, None], ref)))
        T_mix = float(ref + (h_mix_mol - R * ref) / max(float(y_mix @ CP_GAS), 1e-30))
        if pump_on:
            if reviewed_conditioner is not None:
                opening = float(np.clip((p_control - c["cycle"]["regeneration_pressure_Pa"])
                                        / c["pump"]["throttle_band_Pa"], 0.0, 1.0))
                stream, pump_eval = _reviewed_conditioned_stream(
                    c,
                    p_suction_Pa=p_suction,
                    bed_temperature_K=T_mix,
                    bed_mole_fractions=y_mix,
                    opening=opening,
                    profile=profile,
                )
                reviewed_conditioner_details = stream.details
                w_total_full = stream.n_raw_mol_s
                p_pump_inlet = float(stream.p_pump_Pa)
                t_pump_inlet = float(stream.t_out_K)
            else:
                pump = _pump_rates_v21(c, p_suction, T_mix, profile=profile)
                w_total_full = pump["molar_flow_mol_s"]
                p_pump_inlet = p_suction
                t_pump_inlet = T_mix
        w_total_half = 0.5 * w_total_full
        W = share * w_total_half

        # Adsorption has a cell-wise pressure-following vent.  Pumped phases use
        # the declared total withdrawal shares.
        if phase == "adsorption":
            W = np.zeros(model.n_cells)
            for j in range(model.n_cells):
                F_j = share[j] * feed_half
                L_j = share[j] * leak_half
                X = F_j + L_j - float(S[j].sum())
                corr = (p[j] - Patm) * model.cell_gas_volume_m3[j] / (R * Tg[j])
                B = ((F_j * float(y_feed @ CP_GAS) + L_j * float(y_leak @ CP_GAS)) * (Ta - Tg[j])
                     + R * Tg[j] * (F_j + L_j) - phi[j])
                cand = ((cv_mix[j] * Tg[j] * (X + corr) + B)
                        / ((cv_mix[j] + R) * Tg[j]))
                W[j] = max(cand, 0.0)
        elif phase == "repressurization":
            W = np.zeros(model.n_cells)

        F = np.zeros(model.n_cells)
        L = np.zeros(model.n_cells)
        if phase == "adsorption":
            F = share * feed_half
            L = share * leak_half
        elif phase == "repressurization":
            for j in range(model.n_cells):
                F[j] = share[j] * repress_half * float(np.clip(
                    (Patm - p[j]) / (0.01 * Patm), 0.0, 1.0))
        elif pump_on:
            L = share * leak_half

        # Gas equations and boundary terms.
        dng = F[:, None] * y_feed[None, :] + L[:, None] * y_leak[None, :] \
            - W[:, None] * ygas - S
        dTg = ((F * float(y_feed @ CP_GAS) + L * float(y_leak @ CP_GAS)) * (Ta - Tg)
               + R * Tg * (F + L - W) - phi) / (ntot * cv_mix)

        # Solid network, wall and heater.
        G = model.cell_conductance_W_K
        qcond = np.zeros(model.n_cells)
        for j in range(model.n_cells):
            if j > 0:
                qcond[j] += G * (Tbed[j - 1] - Tbed[j])
            if j < model.n_cells - 1:
                qcond[j] += G * (Tbed[j + 1] - Tbed[j])
        Tw = float(y[lay.T_w])
        q_wb = (Tw - Tbed[-1]) / model.wall_bed_resistance_K_W
        qcond[-1] += q_wb
        C_s = model.solid_capacity_J_K + model.cell_mass_kg * cp_ads
        phase_heat = np.sum(
            model.cell_mass_kg[:, None] * dq * (ua_species - h_ads),
            axis=1,
        )
        dT = (qcond + phi + phase_heat) / C_s

        T_control = float(np.min(Tbed))
        fixed_bed_balance_W = float(
            np.sum(qcond + phi + phase_heat)
        )
        heat_request_full, cool_request_full, target_K = (
            phase_actuator_request(
                c,
                phase=phase,
                bed_temperature_K=T_control,
                ambient_temperature_K=Ta,
                fixed_bed_balance_W=fixed_bed_balance_W,
            )
        )
        command = min(
            0.5 * heat_request_full,
            0.5 * float(c["thermal"]["heater_power_W"]),
        )
        ceiling = target_K + float(
            c["thermal"]["heater_temperature_margin_K"]
        )
        heating = heat_request_full > 0.0
        if source_half is not None:
            delivered, limit = source_half.delivered_W(Tw, T_control, command, heating=heating)
            q_hw = 0.0
            dTh = 0.0
            q_heater = 0.0
            q_source = delivered
        elif heating:
            headroom = float(np.clip((ceiling - float(y[lay.T_h]))
                                     / c["thermal"]["heater_ceiling_band_K"], 0.0, 1.0))
            q_heater = command * headroom
            q_hw = model.heater_wall_conductance_W_K * (float(y[lay.T_h]) - Tw)
            dTh = (q_heater - q_hw) / model.heater_capacity_J_K
            q_source = 0.0
            limit = ("heater power ceiling" if command >= 0.5 * c["thermal"]["heater_power_W"] - 1e-9
                     else "bed-temperature feedback")
            if headroom < 1.0:
                limit = "heater temperature ceiling"
        else:
            q_hw = model.heater_wall_conductance_W_K * (float(y[lay.T_h]) - Tw)
            q_heater = 0.0
            dTh = -q_hw / model.heater_capacity_J_K if model.heater_capacity_J_K > 0 else 0.0
            q_source = 0.0
            limit = "heater disabled by shared phase policy"
        q_loss = c["thermal"]["heat_loss_U_W_m2_K"] * model.half_face_area_m2 * (Tw - Ta)
        available_bed_cooling_full = (
            float(reviewed_conditioner_details["available_bed_cooling_W"])
            if reviewed_conditioner_details is not None
            else float(
                c["utilities"]["wall_cooling"]["max_cooling_capacity_W"]
            )
        )
        requested_wall_cool_half = min(
            0.5 * cool_request_full,
            0.5 * available_bed_cooling_full,
        )
        finite_transfer_cool_half = (
            c["thermal"]["cooling_h_W_m2_K"]
            * model.half_face_area_m2
            * max(Tw - target_K, 0.0)
        )
        q_cool = min(
            requested_wall_cool_half,
            finite_transfer_cool_half,
        )
        cooling_request_unmet_half = max(
            requested_wall_cool_half - q_cool, 0.0
        )
        if model.wall_cooling_capacity_W is not None and q_cool > model.wall_cooling_capacity_W:
            excess = q_cool - model.wall_cooling_capacity_W
            dTw = (q_hw + q_source - q_wb - q_loss - model.wall_cooling_capacity_W) / model.wall_capacity_J_K
            q_cool = model.wall_cooling_capacity_W
        else:
            dTw = (q_hw + q_source - q_wb - q_loss - q_cool) / model.wall_capacity_J_K
            excess = cooling_request_unmet_half
        prepump_duty_full_W = 0.0 if stream is None else float(stream.duty_W)
        combined_cooling_full_W = 2.0 * q_cool + prepump_duty_full_W
        if combined_cooling_full_W > 2000.0 + 1e-8:
            raise ExpectedScenarioError(
                "Shared cooling budget exceeded: "
                f"{combined_cooling_full_W:g} W > 2000 W."
            )

        derivative = np.zeros(lay.size)
        derivative[lay.T_h] = dTh
        derivative[lay.T_w] = dTw
        derivative[lay.T_bed] = dT
        derivative[lay.T_gas] = dTg
        derivative[lay.q] = dq.ravel()
        derivative[lay.n_gas] = dng.ravel()
        purge_s = float(c["cycle"].get("purge_s", 0.0))
        product_route = str(
            (c.get("collection") or {}).get(
                "product_route", "all_desorption_and_cooling"
            )
        )
        inlet = F[:, None] * y_feed[None, :] + L[:, None] * y_leak[None, :]
        derivative[lay.inlet] = inlet.sum(axis=0)
        if phase == "adsorption":
            derivative[lay.vent] = (W[:, None] * ygas).sum(axis=0)
            derivative[lay.product] = 0.0
        elif phase == "evacuation":
            vflow = 0.5 * (stream.n_vapour_mol_s if stream is not None else w_total_full)
            derivative[lay.vent] = vflow * (stream.vapour_composition if stream is not None else y_mix)
            derivative[lay.product] = 0.0
        elif phase == "desorption":
            pflow = 0.5 * (stream.n_vapour_mol_s if stream is not None else w_total_full)
            composition = stream.vapour_composition if stream is not None else y_mix
            if _t >= purge_s:
                derivative[lay.product] = pflow * composition
                derivative[lay.vent] = 0.0
            else:
                derivative[lay.vent] = pflow * composition
                derivative[lay.product] = 0.0
        elif phase == "cooling" and "cooling" in product_route:
            pflow = 0.5 * (stream.n_vapour_mol_s if stream is not None else w_total_full)
            derivative[lay.product] = pflow * (stream.vapour_composition if stream is not None else y_mix)
            derivative[lay.vent] = 0.0
        elif phase == "cooling":
            pflow = 0.5 * (
                stream.n_vapour_mol_s if stream is not None else w_total_full
            )
            derivative[lay.vent] = pflow * (
                stream.vapour_composition if stream is not None else y_mix
            )
            derivative[lay.product] = 0.0
        else:
            derivative[lay.vent] = 0.0
            derivative[lay.product] = 0.0
        derivative[lay.E_heat] = q_heater
        derivative[lay.E_cool] = q_cool
        derivative[lay.E_fan] = 0.5 * flow["fan_power_W"] if phase == "adsorption" else 0.0
        derivative[lay.E_vac] = 0.0
        # The total installed bank is evaluated once at the actual pump inlet.
        if pump_on:
            if stream is not None:
                pump_eval = _pump_rates_v21(c, stream.p_pump_Pa, stream.t_out_K, profile=profile)
            else:
                pump_eval = _pump_rates_v21(c, p_suction, T_mix, profile=profile)
            derivative[lay.E_vac] = 0.5 * pump_eval["electric_power_W"]
        derivative[lay.E_cond] = 0.0
        derivative[lay.E_loss] = q_loss
        h_in_feed = gas_enthalpy_mol(Ta, ref)
        h_in_leak = gas_enthalpy_mol(Ta, ref)
        net = (F * float(y_feed @ h_in_feed) + L * float(y_leak @ h_in_leak)
               - W * np.sum(ygas * gas_enthalpy_mol(Tg[:, None], ref), axis=1))
        derivative[lay.E_netflow] = float(np.sum(net))
        utility_electric_full = None
        if utilities is not None:
            utility_now = utilities.convert(
                wall_cooling_duty_W=2.0 * q_cool,
                prepump_cooling_duty_W=0.0 if stream is None else float(stream.duty_W),
            )
            utility_electric_full = utility_now["total_electric_W"]
        electric_boundary_complete = utility_electric_full is not None
        electric_boundary_half = (
            q_heater + derivative[lay.E_fan] + derivative[lay.E_vac]
            + derivative[lay.E_cond]
            + (source_half.auxiliary_electric_W
               if source_half is not None and heating else 0.0)
            + (0.5 * conditioner_spec.auxiliary_electric_W
               if conditioner_spec is not None else 0.0)
            + 0.5 * utility_electric_full
        ) if electric_boundary_complete else 0.0
        derivative[lay.E_electric_total] = electric_boundary_half
        if lay.extended:
            derivative[lay.raw] = (W[:, None] * ygas).sum(axis=0)
            cond = 0.0 if stream is None else 0.5 * stream.n_condensate_mol_s
            derivative[lay.condensate] = cond
            derivative[lay.condensate_product] = cond if phase in COLLECTING_PHASES else 0.0
            derivative[lay.e_source] = q_source
            if source_half is None:
                derivative[lay.e_source_input] = 0.0
            else:
                derivative[lay.e_source_input] = (
                    q_source * source_input_factor
                )
            derivative[lay.e_source_aux] = (source_half.auxiliary_electric_W
                                            if source_half is not None and heating else 0.0)
            derivative[lay.e_cooler] = 0.0 if stream is None else 0.5 * stream.duty_W
            derivative[lay.e_cooler_aux] = (
                0.5 * conditioner_spec.auxiliary_electric_W
                if conditioner_spec is not None else 0.0
            )

        inv = energy_inventory(c, material, g, y, model, systems=systems)
        # Analytic state-energy derivative for the instantaneous audit.
        storage_h = np.vstack([
            material.storage_energy_terms(float(Tbed[j]), q[j])[2]
            for j in range(model.n_cells)
        ])
        dU = (model.heater_capacity_J_K * dTh + model.wall_capacity_J_K * dTw
              + float(np.sum(C_s * dT))
              + float(np.sum(model.cell_mass_kg[:, None] * dq * storage_h))
              + float(np.sum(ntot * cv_mix * dTg))
              + float(np.sum(np.sum(dng * ug, axis=1))))
        boundary = (q_heater + q_source - q_cool - q_loss + derivative[lay.E_netflow])
        residual = dU - boundary
        mw_mix = ygas @ MW
        rho_gas = p * mw_mix / (R * Tg)
        local_q_full = 2.0 * W * R * Tg / np.maximum(p, 1e-20)
        local_area_full = share * float(g.flow_area_m2)
        local_u = local_q_full / np.maximum(local_area_full, 1e-20)
        local_u_i = local_u / float(c["bed"]["bed_void_fraction"])
        mu = float(c["airflow"]["air_viscosity_Pa_s"])
        dp = float(c["bed"]["particle_diameter_m"])
        obs = {
            "pressure_Pa": float(np.max(p)),
            "minimum_pressure_Pa": float(np.min(p)),
            "pump_suction_pressure_Pa": p_suction,
            "pressure_control_observable": control_observable,
            "pressure_control_pressure_Pa": p_control,
            "pressure_control_opening": opening,
            "pump_inlet_pressure_Pa": p_pump_inlet,
            "pump_inlet_temperature_K": t_pump_inlet,
            "pump_bank_count": (None if pump_eval is None else pump_eval.get("bank_count")),
            "pump_bank_evaluated_once": True,
            "pump_capacity_at_inlet_mol_s": (None if pump_eval is None
                                              else float(pump_eval["molar_flow_mol_s"])),
            "pump_rate_model": pump_flow_basis(c),
            "pump_profile_reference_pressure_Pa": (
                None if pump_eval is None else pump_eval.get("profile_reference_pressure_Pa")
            ),
            "pump_profile_reference_temperature_K": (
                None if pump_eval is None else pump_eval.get("profile_reference_temperature_K")
            ),
            "pump_actual_inlet_volumetric_speed_m3_s": (
                0.0 if pump_eval is None else pump_eval.get("actual_inlet_volumetric_speed_m3_s")
            ),
            "pump_plotted_curve_volumetric_flow_m3_s": (
                0.0 if pump_eval is None else pump_eval.get("plotted_curve_volumetric_flow_m3_s")
            ),
            "pump_equivalent_reference_condition_volumetric_flow_m3_s": (
                0.0 if pump_eval is None
                else pump_eval.get("equivalent_reference_condition_volumetric_flow_m3_s")
            ),
            "pressure_control_capacity_basis": CAPACITY_PRESSURE_BASIS,
            "pump_manifold_temperature_K": T_mix,
            "pump_manifold_composition": y_mix.tolist(),
            "channel_pressure_spread_Pa": float(np.max(p) - np.min(p)),
            "local_gas_temperature_K": Tg.tolist(),
            "local_pressure_Pa": p.tolist(),
            "local_withdrawal_mol_s": (2.0 * W).tolist(),
            "local_superficial_velocity_m_s": local_u.tolist(),
            "local_interstitial_velocity_m_s": local_u_i.tolist(),
            "local_reynolds_superficial": (rho_gas * local_u * dp / mu).tolist(),
            "local_reynolds_interstitial": (rho_gas * local_u_i * dp / mu).tolist(),
            "pump_flow_basis": pump_flow_basis(c),
            "withdrawal_total_mol_s": float(2.0 * np.sum(W)),
            "withdrawal_shares_nonnegative": bool(np.all(W >= 0.0)),
            "withdrawal_allocation_feasible": bool(np.all(np.isfinite(W)) and np.all(W >= 0.0)),
            "local_inventory_min_mol": float(np.min(ntot)),
            "local_inventory_min_mol_kg": float(np.min(q)),
            "pump_curve_clamped": bool(pump_eval["curve_clamped"]) if pump_on else False,
            "pump_media_temperature_overrun_K": (float(pump_eval["media_temperature_overrun_K"])
                                                  if pump_on else 0.0),
            "pump_pressure_domain_ok": bool(not pump_eval["curve_clamped"]) if pump_on else True,
            "electric_boundary_W": float(2.0 * electric_boundary_half),
            "electric_boundary_complete": bool(electric_boundary_complete),
            "heater_W": q_heater,
            "heater_request_full_W": heat_request_full,
            "cooler_request_full_W": cool_request_full,
            "actuator_target_K": target_K,
            "actuator_policy": c["operating_policy"]["phases"][phase],
            "source_delivered_W": q_source,
            "source_auxiliary_W": (source_half.auxiliary_electric_W
                                    if source_half is not None and heating else 0.0),
            "source_input_per_delivered": source_input_factor,
            "cooling_W": q_cool,
            "prepump_cooling_full_W": prepump_duty_full_W,
            "combined_cooling_full_W": combined_cooling_full_W,
            "combined_cooling_capacity_W": 2000.0,
            "cooler_duty_W": 0.0 if stream is None else float(stream.duty_W),
            "cooler_shortfall_K": 0.0 if stream is None else float(stream.target_shortfall_K),
            "cooler_auxiliary_W": (
                0.0 if conditioner_spec is None
                else float(conditioner_spec.auxiliary_electric_W)
            ),
            "conditioner_policy": (
                None
                if reviewed_conditioner_details is None
                else "pump_inlet_cooler_knockout_v1"
            ),
            "conditioner_pressure_drop_Pa": (
                None
                if stream is None
                else float(p_suction - stream.p_pump_Pa)
            ),
            "conditioner_bed_flow_mol_s": (
                None
                if reviewed_conditioner_details is None
                else float(reviewed_conditioner_details["bed_flow_mol_s"])
            ),
            "conditioner_pump_flow_mol_s": (
                None
                if reviewed_conditioner_details is None
                else float(
                    reviewed_conditioner_details["pump_flow_mol_s"]
                )
            ),
            "conditioner_liquid_flow_mol_s": (
                None
                if reviewed_conditioner_details is None
                else float(reviewed_conditioner_details["liquid_flow_mol_s"])
            ),
            "conditioner_available_bed_cooling_W": (
                None
                if reviewed_conditioner_details is None
                else float(
                    reviewed_conditioner_details[
                        "available_bed_cooling_W"
                    ]
                )
            ),
            "conditioner_capacity_limiting": (
                None
                if reviewed_conditioner_details is None
                else reviewed_conditioner_details["withdrawal_reason"]
            ),
            "cooling_capacity_shortfall_W": excess,
            "ambient_loss_W": q_loss,
            "fan_W": derivative[lay.E_fan],
            "vacuum_W": derivative[lay.E_vac],
            "first_law_residual_W": float(residual),
            "first_law_scale_W": float(abs(boundary) + 1.0),
            "product_CO2_mol_s": float(derivative[lay.product][0]),
            "vent_total_mol_s": float(derivative[lay.vent].sum()),
            "bed_mass_balance_mol_s": float(np.sum(dng + S, axis=0).sum()),
            "source_limit": limit,
        }
        return derivative, obs

    evaluate._jacobian_material_cache = material_cache
    evaluate._jacobian_cache_policy = cache_policy
    return evaluate


def _atol(model: CoupledModel) -> np.ndarray:
    n = model.n_cells
    atol = np.r_[
        np.full(2, 1e-5),
        np.full(n, 1e-5),
        np.full(n, 1e-5),
        np.full(model.n_storage * n, 1e-9),
        np.full(4 * n, 1e-10),
        np.full(12, 1e-9),
        np.full(8, 1e-3),
    ]
    if model.layout.extended:
        atol = np.r_[atol, np.full(4, 1e-9), np.full(1, 1e-9),
                     np.full(1, 1e-9), np.full(5, 1e-3)]
    return atol


def _events(c: dict, material, g: Geometry, model: CoupledModel, phase: str):
    lay = model.layout
    runtime_control = c.get("_runtime_control") or {}
    limit = (
        runtime_control["temperature_guard_limit_K"]
        if "temperature_guard_limit_K" in runtime_control
        else float(material.data["maximum_bed_temperature_C"]) + 273.15
    )
    provider = (c.get("_runtime_control") or {}).get("weather_provider")

    def local_weather(base_t: float):
        if provider is None:
            return None
        control = c.get("_runtime_control") or {}
        return provider(
            float(control.get("cycle_start_s", 0.0))
            + float(control.get("phase_offset_s", 0.0))
            + float(base_t)
        )
    ev = []
    if phase == "cooling":
        def cool(t, y):
            row = local_weather(t)
            ambient_C = (
                row.temperature_C if row is not None
                else c["ambient"]["temperature_C"]
            )
            end = ambient_C + 273.15 + c["cycle"]["cooling_approach_K"]
            return max(float(y[lay.T_w]), float(np.max(y[lay.T_bed]))) - end
        cool.terminal = True
        cool.direction = -1
        cool.event_kind = "cooling_completion"
        ev.append(cool)
    if phase == "repressurization":
        def full(t, y):
            row = local_weather(t)
            ambient_pressure = (
                row.pressure_Pa if row is not None
                else c["ambient"]["pressure_Pa"]
            )
            return (
                float(np.min(state_pressures(y, model)))
                - 0.995 * ambient_pressure
            )
        full.terminal = True
        full.direction = 1
        full.event_kind = "repressurization_completion"
        ev.append(full)
    if limit is not None:
        def over(_t, y):
            return float(np.max(y[lay.T_bed])) - float(limit)
        over.terminal = True
        over.direction = 1
        over.event_kind = "temperature_guard"
        ev.append(over)
    return ev


def _event_kind_fired(events, solution, event_kind: str) -> bool:
    times = solution.t_events or ()
    return any(
        getattr(event, "event_kind", None) == event_kind and len(fired) > 0
        for event, fired in zip(events, times)
    )


def _raise_for_temperature_guard(events, solution) -> None:
    if _event_kind_fired(events, solution, "temperature_guard"):
        raise ExpectedScenarioError(
            "Scenario bed-temperature limit reached; revise the case."
        )


def one_cycle(c: dict, material, g: Geometry, initial: np.ndarray, *, systems=None,
              dryer=None, utilities=None, model: CoupledModel | None = None,
              max_total_duration_s: float | None = None):
    systems = build_systems(c) if systems is None else systems
    require_declared_thermal_conversion(c, systems.source)
    utilities = build_utilities(c) if utilities is None else utilities
    dryer = build_dryer(c) if dryer is None else dryer
    if dryer is not None:
        raise ExpectedScenarioError(
            "Explicit feed-dryer operation is not qualified in the V21 preparation.")
    if c["conditioning"]["electric_J_per_kg_water_removed"] not in (None, 0.0):
        raise ExpectedScenarioError(
            "Nonzero legacy feed-conditioning energy is not qualified in the V21 preparation.")
    model = build_model(c, material, g, systems=systems, utilities=utilities) if model is None else model
    lay = model.layout
    jacobian_audit = SpatialJacobianAudit(
        physical_columns=int(lay.inlet.start),
        ledger_columns=int(lay.size - lay.inlet.start),
    )
    state = initial.copy()
    reset_current_cycle_state_meters(state, lay)
    check_inventory(state, model, where="cycle start")
    durations = (c["cycle"]["adsorption_s"], c["cycle"]["evacuation_s"],
                 c["cycle"]["desorption_s"], c["cycle"]["cooling_max_s"],
                 c["cycle"]["repressurization_max_s"])
    t_parts, y_parts, p_parts, windows, completion = [], [], [], [], {}
    audit_t_parts, audit_y_parts, audit_phase_parts = [], [], []
    offset = 0.0
    control = load_desorption_control(c)
    branch_policy = _branch_jacobian_policy(c)
    solver_policy = resolved_spatial_solver_policy(c)
    runtime_control = c.get("_runtime_control") or {}
    if isinstance(c.get("_runtime_control"), dict):
        c["_runtime_control"]["resolved_spatial_solver_policy"] = (
            solver_policy
        )
    recovery_trackers = []

    def record_recovery_tracker(tracker) -> None:
        if tracker is None:
            return
        recovery_trackers.append(tracker)
        recorder = getattr(material, "record_bdf_recovery_summary", None)
        if callable(recorder):
            recorder(tracker.as_dict())

    def recovery_method():
        if solver_policy["policy"] == SPATIAL_SOLVER_POLICY_RADAU:
            return "Radau", None
        return make_recovering_bdf(material, lay, model=model)

    def jacobian_for(rhs_value):
        material_cache = getattr(
            rhs_value, "_jacobian_material_cache", None
        )

        def evaluate_jacobian(time_value, state_value):
            if material_cache is None:
                return bounded_spatial_jacobian(
                    rhs_value,
                    time_value,
                    state_value,
                    layout=lay,
                    material=material,
                    audit=jacobian_audit,
                    branch_policy=branch_policy,
                )
            return bounded_spatial_jacobian_with_material_cache(
                rhs_value,
                time_value,
                state_value,
                layout=lay,
                material=material,
                material_cache=material_cache,
                audit=jacobian_audit,
                branch_policy=branch_policy,
            )

        return evaluate_jacobian

    def record_jacobian_audit() -> None:
        if isinstance(c.get("_runtime_control"), dict):
            c["_runtime_control"]["spatial_jacobian_audit"] = (
                jacobian_audit.as_dict()
            )

    def report_phase(event: str, state_value: np.ndarray) -> None:
        callback = runtime_control.get("progress_callback")
        if callback is None:
            return
        cycle_number = int(runtime_control.get("cycle_number", 1))
        callback(
            {
                "event": event,
                "cycle": cycle_number,
                "phase": phase,
                "elapsed_s": float(
                    runtime_control.get("cycle_start_s", 0.0)
                )
                + float(offset)
                + (
                    float(used)
                    if event == "phase_completed" and "used" in locals()
                    else 0.0
                ),
                "phase_elapsed_s": (
                    float(used)
                    if event == "phase_completed" and "used" in locals()
                    else 0.0
                ),
                "phase_duration_s": float(duration),
                "checkpoint_state": np.asarray(
                    state_value, dtype=float
                ).tolist(),
            }
        )

    for phase, requested_duration in zip(STEPS, durations):
        requested_duration = float(requested_duration)
        phase_horizon_truncated = False
        if max_total_duration_s is not None:
            remaining = float(max_total_duration_s) - offset
            if remaining <= 1e-12:
                break
            phase_horizon_truncated = (
                remaining < requested_duration - 1e-12
            )
            duration = min(requested_duration, remaining)
        else:
            duration = requested_duration
        if isinstance(c.get("_runtime_control"), dict):
            c["_runtime_control"]["phase_offset_s"] = float(offset)
            c["_runtime_control"]["phase_duration_s"] = float(duration)
        report_phase("phase_started", state)
        ev = cycle_rhs(c, material, g, phase, systems=systems, dryer=dryer,
                       utilities=utilities, model=model)
        if phase == "desorption" and control is not None and control.mode == "state_control":
            runtime = DesorptionControlRuntime(
                control=control,
                target_bed_K=c["cycle"]["regeneration_temperature_C"] + 273.15,
                start_product_mol=float(state[lay.product][0]),
            )
            initial_obs = ev(0.0, state)[1]
            runtime.observe(0.0, float(np.min(state[lay.T_bed])),
                            initial_obs["product_CO2_mol_s"], float(state[lay.product][0]))
            start_chunk = 0.0
            used = 0.0
            local_t_parts, local_y_parts = [], []
            while start_chunk < float(duration) - 1e-12:
                end_chunk = min(start_chunk + control.checkpoint_s, float(duration))
                events = _events(c, material, g, model, phase)
                chunk_start = float(start_chunk)

                def chunk_rhs(local_time, values):
                    derivative, _observables = ev(
                        chunk_start + local_time, values
                    )
                    chunk_rhs.last_directional_rate_deltas = getattr(
                        ev, "last_directional_rate_deltas", None
                    )
                    chunk_rhs._jacobian_material_cache = getattr(
                        ev, "_jacobian_material_cache", None
                    )
                    chunk_rhs._jacobian_cache_policy = getattr(
                        ev, "_jacobian_cache_policy", None
                    )
                    return derivative

                chunk_rhs._jacobian_material_cache = getattr(
                    ev, "_jacobian_material_cache", None
                )
                chunk_rhs._jacobian_cache_policy = getattr(
                    ev, "_jacobian_cache_policy", None
                )

                method, recovery_tracker = recovery_method()
                try:
                    sol = solve_ivp(
                        chunk_rhs,
                        (0.0, end_chunk - start_chunk),
                        state,
                        method=method,
                        rtol=c["solver"]["rtol"],
                        atol=_atol(model),
                        dense_output=True,
                        max_step=c["solver"]["max_step_s"],
                        events=events,
                        jac=jacobian_for(chunk_rhs),
                    )
                finally:
                    record_recovery_tracker(recovery_tracker)
                if not sol.success:
                    raise ExpectedScenarioError(f"desorption integration failed: {sol.message}")
                record_jacobian_audit()
                _raise_for_temperature_guard(events, sol)
                previous_product = float(state[lay.product][0])
                state = sol.y[:, -1].copy()
                dt = max(end_chunk - start_chunk, 1e-30)
                flux = max(float(state[lay.product][0]) - previous_product, 0.0) / dt
                runtime.observe(end_chunk, float(np.min(state[lay.T_bed])),
                                flux, float(state[lay.product][0]))
                local_t = np.linspace(0.0, dt, int(c["solver"]["samples_per_step"]))
                local_t_parts.append(start_chunk + local_t)
                local_y_parts.append(sol.sol(local_t).T)
                start_chunk = end_chunk
                used = end_chunk
                if runtime.should_stop(start_chunk):
                    break
            local_t = np.concatenate(local_t_parts)
            local_y = np.vstack(local_y_parts)
            audit_t = local_t.copy()
            audit_y = local_y.copy()
            check_inventory_rows(local_y, model, where=f"{phase} control samples")
            completion[phase] = {"event_fired": False, "at_entry": False,
                                 "state_control": True,
                                 "horizon_truncated": bool(
                                     phase_horizon_truncated
                                     and used >= duration - 1e-12
                                 )}
            t_parts.append(local_t + offset)
            y_parts.append(local_y)
            audit_t_parts.append(audit_t + offset)
            audit_y_parts.append(audit_y)
            audit_phase_parts.extend([phase] * len(audit_t))
            p_parts.extend([phase] * len(local_t))
            windows.append((phase, offset, offset + used))
            report_phase("phase_completed", state)
            offset += used
            if (
                max_total_duration_s is not None
                and offset >= float(max_total_duration_s) - 1e-12
            ):
                break
            continue
        events = _events(c, material, g, model, phase)
        skip = False
        runtime_weather = None
        weather_provider = (c.get("_runtime_control") or {}).get(
            "weather_provider"
        )
        if weather_provider is not None:
            control_state = c.get("_runtime_control") or {}
            runtime_weather = weather_provider(
                float(control_state.get("cycle_start_s", 0.0)) + offset
            )
        if phase == "cooling":
            ambient_C = (
                runtime_weather.temperature_C
                if runtime_weather is not None
                else c["ambient"]["temperature_C"]
            )
            end = ambient_C + 273.15 + c["cycle"]["cooling_approach_K"]
            skip = max(float(state[lay.T_w]), float(np.max(state[lay.T_bed]))) <= end
        if phase == "repressurization":
            ambient_pressure = (
                runtime_weather.pressure_Pa
                if runtime_weather is not None
                else c["ambient"]["pressure_Pa"]
            )
            skip = (
                float(np.min(state_pressures(state, model)))
                >= 0.995 * ambient_pressure
            )
        if skip:
            used = 0.0
            local_t = np.zeros(1)
            local_y = state[None, :].copy()
            audit_t, audit_y = local_t.copy(), local_y.copy()
            completion[phase] = {
                "event_fired": False,
                "at_entry": True,
                "horizon_truncated": False,
            }
        else:
            def phase_rhs(local_time, values):
                derivative, _observables = ev(local_time, values)
                phase_rhs.last_directional_rate_deltas = getattr(
                    ev, "last_directional_rate_deltas", None
                )
                phase_rhs._jacobian_material_cache = getattr(
                    ev, "_jacobian_material_cache", None
                )
                phase_rhs._jacobian_cache_policy = getattr(
                    ev, "_jacobian_cache_policy", None
                )
                return derivative

            phase_rhs._jacobian_material_cache = getattr(
                ev, "_jacobian_material_cache", None
            )
            phase_rhs._jacobian_cache_policy = getattr(
                ev, "_jacobian_cache_policy", None
            )

            method, recovery_tracker = recovery_method()
            try:
                sol = solve_ivp(
                    phase_rhs,
                    (0.0, float(duration)),
                    state,
                    method=method,
                    rtol=c["solver"]["rtol"],
                    atol=_atol(model),
                    dense_output=True,
                    max_step=c["solver"]["max_step_s"],
                    events=events,
                    jac=jacobian_for(phase_rhs),
                )
            finally:
                record_recovery_tracker(recovery_tracker)
            if not sol.success:
                raise ExpectedScenarioError(f"{phase} integration failed: {sol.message}")
            record_jacobian_audit()
            _raise_for_temperature_guard(events, sol)
            state = sol.y[:, -1]
            used = float(sol.t[-1])
            local_t = np.linspace(0.0, used, int(c["solver"]["samples_per_step"])) if used > 0 else np.zeros(1)
            local_y = sol.sol(local_t).T if used > 0 else state[None, :].copy()
            if used > 0:
                audit_t = np.unique(np.concatenate([sol.t, 0.5 * (sol.t[:-1] + sol.t[1:])]))
                audit_y = sol.sol(audit_t).T
            else:
                audit_t, audit_y = local_t.copy(), local_y.copy()
            completion[phase] = {
                "event_fired": bool(
                    any(
                        len(fired) > 0
                        for event, fired in zip(events, sol.t_events or ())
                        if getattr(event, "event_kind", None)
                        != "temperature_guard"
                    )
                ),
                "at_entry": False,
                "horizon_truncated": bool(
                    phase_horizon_truncated
                    and used >= duration - 1e-12
                ),
            }
        t_parts.append(local_t + offset)
        y_parts.append(local_y)
        audit_t_parts.append(audit_t + offset)
        audit_y_parts.append(audit_y)
        audit_phase_parts.extend([phase] * len(audit_t))
        check_inventory_rows(local_y, model, where=f"{phase} samples")
        p_parts.extend([phase] * len(local_t))
        windows.append((phase, offset, offset + used))
        report_phase("phase_completed", state)
        offset += used
        check_inventory(state, model, where=phase)
        if (
            max_total_duration_s is not None
            and offset >= float(max_total_duration_s) - 1e-12
        ):
            break
    if isinstance(c.get("_runtime_control"), dict) and recovery_trackers:
        c["_runtime_control"]["bdf_trial_recovery"] = merge_trackers(
            recovery_trackers
        )
    return (np.concatenate(t_parts), np.vstack(y_parts), np.array(p_parts),
            windows, completion, np.concatenate(audit_t_parts),
            np.vstack(audit_y_parts), np.array(audit_phase_parts))


def _state_scale(c: dict, material, model: CoupledModel, Ta: float) -> tuple[np.ndarray, list[str]]:
    lay = model.layout
    n_amb = c["ambient"]["pressure_Pa"] * model.cell_gas_volume_m3 / (R * Ta)
    vals = np.r_[
        np.full(2, 100.0), np.full(model.n_cells, 100.0), np.full(model.n_cells, 100.0),
        np.tile(material.storage_scale(), model.n_cells),
        np.repeat(n_amb, 4),
    ]
    names = ["T_h", "T_w"] + [f"T_bed_{j}" for j in range(model.n_cells)] \
        + [f"T_gas_{j}" for j in range(model.n_cells)] \
        + [
            f"q_{j}_{i}"
            for j in range(model.n_cells)
            for i in range(model.n_storage)
        ] \
        + [f"n_{j}_{i}" for j in range(model.n_cells) for i in range(4)]
    return vals, names


def run_cycle(c: dict, material, g: Geometry, *, return_audit: bool = False,
              raw_sink=None, summarize_fn=None):
    systems = build_systems(c)
    require_declared_thermal_conversion(c, systems.source)
    if build_dryer(c) is not None:
        raise ExpectedScenarioError(
            "Explicit feed-dryer operation is not qualified in the V21 preparation.")
    if c["conditioning"]["electric_J_per_kg_water_removed"] not in (None, 0.0):
        raise ExpectedScenarioError(
            "Nonzero legacy feed-conditioning energy is not qualified in the V21 preparation.")
    control = load_desorption_control(c)
    if control is not None and control.mode == "state_control":
        raise ExpectedScenarioError(
            "State-controlled desorption is not qualified in the V21 preparation; use fixed_timer.")
    dryer = build_dryer(c)
    utilities = build_utilities(c)
    model = build_model(c, material, g, systems=systems, utilities=utilities)
    lay = model.layout
    state = initial_state(c, material, g, model)
    Ta = c["ambient"]["temperature_C"] + 273.15
    scale, names = _state_scale(c, material, model, Ta)
    tolerance = float(c["cycle"]["css_relative_tolerance"])
    history = []
    converged = False
    binding = ("", float("inf"))
    previous_meters = None
    audit = None
    for number in range(1, int(c["cycle"]["maximum_cycles"]) + 1):
        initial = state.copy()
        time, states, phases, windows, completion, audit_t, audit_y, audit_p = one_cycle(
            c, material, g, initial, systems=systems, dryer=dryer,
            utilities=utilities, model=model)
        state = states[-1].copy()
        # Per-cycle increments; meters are reset at the start of one_cycle.
        meters = {
            "cycle_time_s": float(time[-1]),
            "CO2_product_mol": float(state[lay.product][0]),
            "water_product_mol": float(state[lay.product][1]),
            "vent_total_mol": float(state[lay.vent].sum()),
            "heater_energy_J": float(state[lay.E_heat]),
            "cooling_removed_J": float(state[lay.E_cool]),
            "fan_energy_J": float(state[lay.E_fan]),
            "vacuum_energy_J": float(state[lay.E_vac]),
            "conditioning_energy_J": float(state[lay.E_cond]),
            "ambient_loss_energy_J": float(state[lay.E_loss]),
            "net_boundary_enthalpy_J": float(state[lay.E_netflow]),
            "electric_boundary_J": float(state[lay.E_electric_total]),
        }
        for i, species in enumerate(SPECIES):
            meters[f"inlet_{species}_mol"] = float(state[lay.inlet][i])
            meters[f"vent_{species}_mol"] = float(state[lay.vent][i])
            meters[f"product_{species}_mol"] = float(state[lay.product][i])
        if lay.extended:
            for i, species in enumerate(SPECIES):
                meters[f"raw_{species}_mol"] = float(state[lay.raw][i])
            meters["condensate_mol"] = float(state[lay.condensate])
            meters["condensate_product_mol"] = float(state[lay.condensate_product])
            meters["source_delivered_J"] = float(state[lay.e_source])
            meters["source_thermal_input_J"] = float(state[lay.e_source_input])
            meters["source_auxiliary_J"] = float(state[lay.e_source_aux])
            meters["cooling_removed_train_J"] = float(state[lay.e_cooler])
            meters["cooling_auxiliary_J"] = float(state[lay.e_cooler_aux])
        floors = {k: 1e-9 for k in meters}
        for k in ("heater_energy_J", "cooling_removed_J", "fan_energy_J",
                  "vacuum_energy_J", "conditioning_energy_J",
                  "ambient_loss_energy_J", "net_boundary_enthalpy_J",
                  "electric_boundary_J",
                  "source_delivered_J", "source_thermal_input_J",
                  "source_auxiliary_J", "cooling_removed_train_J",
                  "cooling_auxiliary_J"):
            floors[k] = 1e-3
        floors["cycle_time_s"] = 1e-6
        change = {k: abs(v - previous_meters[k]) / max(abs(v), floors[k])
                  for k, v in meters.items()} if previous_meters is not None else {}
        state_change = np.abs(state[:lay.inlet.start] - initial[:lay.inlet.start]) / scale
        candidates = [(names[i], float(v)) for i, v in enumerate(state_change)]
        candidates += list(change.items())
        binding = max(candidates, key=lambda item: item[1]) if candidates else ("", 0.0)
        group_local = float(max([v for n, v in candidates
                                 if n.startswith(("T_", "q_", "n_"))] or [0.0]))
        group_carbon = float(max([v for n, v in candidates
                                  if n.startswith(("inlet_", "vent_", "product_", "raw_",
                                                   "condensate"))] or [0.0]))
        group_energy = float(max([v for n, v in candidates
                                   if n.endswith("_J")] or [0.0]))
        group_time = float(change.get("cycle_time_s", 0.0))
        group_max = max(group_local, group_carbon, group_energy, group_time)
        converged = bool(number >= c["cycle"]["minimum_cycles"] and group_max < tolerance)
        inv0 = energy_inventory(c, material, g, initial, model, systems=systems)
        inv1 = energy_inventory(c, material, g, state, model, systems=systems)
        source = float(state[lay.e_source]) if lay.extended else 0.0
        first_law = ((inv1["total_J"] - inv0["total_J"]) - state[lay.E_heat] - source
                     + state[lay.E_cool] + state[lay.E_loss] - state[lay.E_netflow])
        first_scale = abs(state[lay.E_heat]) + abs(source) + abs(state[lay.E_cool]) \
            + abs(state[lay.E_loss]) + abs(state[lay.E_netflow]) + 1.0
        relative = abs(first_law) / first_scale
        if relative > RUNTIME_FIRST_LAW_BACKSTOP:
            raise ExpectedScenarioError(f"First-law closure failed in cycle {number}: {relative:g}.")
        q0 = initial[lay.q].reshape(model.n_cells, model.n_storage)
        q1 = state[lay.q].reshape(model.n_cells, model.n_storage)
        n0 = initial[lay.n_gas].reshape(model.n_cells, 4)
        n1 = state[lay.n_gas].reshape(model.n_cells, 4)
        storage_species_delta = np.zeros(4)
        for storage_index, species_index in enumerate(
            material.species_for_storage
        ):
            storage_species_delta[int(species_index)] += float(
                np.sum(
                    model.cell_mass_kg
                    * (q1[:, storage_index] - q0[:, storage_index])
                )
            )
        residual = (storage_species_delta
                    + n1.sum(axis=0) - n0.sum(axis=0)
                    - state[lay.inlet] + state[lay.vent] + state[lay.product])
        condensate_vec = np.zeros(4)
        if lay.extended:
            condensate_vec[1] = state[lay.condensate]
            residual[1] += condensate_vec[1]
        throughput_each = (state[lay.inlet] + state[lay.vent]
                           + state[lay.product] + condensate_vec)
        species_relative = species_relative_closure(residual, throughput_each)
        mass_error = float(np.max(species_relative))
        if mass_error > 1e-5:
            raise ExpectedScenarioError(f"Species balance failed: {mass_error:g}.")
        history.append({"cycle": number, "css_group_local": group_local,
                        "css_group_carbon": group_carbon, "css_group_energy": group_energy,
                        "css_group_time": group_time,
                        "binding": binding[0], "binding_change": binding[1],
                        "first_law_relative": relative,
                        "species_relative": mass_error,
                        "species_relative_by_species": dict(zip(SPECIES, species_relative.tolist())),
                        "species_residual_mol": dict(zip(SPECIES, residual.tolist())),
                        "species_relative": mass_error, "converged": converged})
        audit = (time, states, phases, windows, completion, audit_t, audit_y, audit_p,
                 meters, history, model, state, initial, relative, mass_error)
        previous_meters = meters
        if converged:
            break
    if audit is None:
        raise ExpectedScenarioError("No cycle was integrated.")
    summary = finalize_cycle(c, material, g, audit, raw_sink=raw_sink,
                             summarize_fn=summarize_fn)
    if not return_audit:
        return summary
    return summary, _audit_payload(model, audit)


def finalize_cycle(c: dict, material, g: Geometry, audit, *, raw_sink=None,
                   summarize_fn=None):
    """Persist raw/core evidence before optional summary/resource assembly.

    The raw sink is called exactly once, before the summary function.  A
    summary/export defect therefore cannot require re-integration.
    """
    (time, states, phases, windows, completion, audit_t, audit_y, audit_p,
     meters, history, model, state, initial, relative, mass_error) = audit
    lay = model.layout
    systems = build_systems(c)
    payload = _audit_payload(model, audit)
    payload["core"] = {
        "schema": "coupled-through-thickness-raw-core-v21/1",
        "phase_windows": windows,
        "phase_completion": completion,
        "meters": meters,
        "history": history,
        "initial_energy_inventory": energy_inventory(c, material, g, initial, model),
        "final_energy_inventory": energy_inventory(c, material, g, state, model),
        "energy_first_law_relative_residual": relative,
        "maximum_relative_species_balance_error": mass_error,
        "final_state": np.asarray(state).tolist(),
        "initial_state": np.asarray(initial).tolist(),
    }
    if raw_sink is not None:
        raw_sink(payload)
    if summarize_fn is not None:
        return summarize_fn(c, material, g, audit)
    from .spatial_postcore_v1 import PostcoreAxes, assemble_summary
    axes = PostcoreAxes(
        export_time_s=np.asarray(time),
        export_state=np.asarray(states),
        export_phase=np.asarray(phases).astype(str),
        audit_time_s=np.asarray(audit_t),
        audit_state=np.asarray(audit_y),
        audit_phase=np.asarray(audit_p).astype(str),
        windows=windows,
        completion=completion,
        meters=meters,
        history=history,
        model=model,
        state=np.asarray(state),
        initial=np.asarray(initial),
        first_law_relative=float(relative),
        species_relative=float(mass_error),
    )
    return assemble_summary(c, material, g, axes)


def _audit_payload(model: CoupledModel, audit) -> dict:
    (time, states, phases, windows, completion, audit_t, audit_y, audit_p,
     meters, history, model, state, initial, relative, mass_error) = audit
    lay = model.layout
    arrays = {
        "audit_time_s": np.asarray(audit_t),
        "audit_state": np.asarray(audit_y),
        "audit_phase": np.asarray(audit_p).astype(str),
        "export_time_s": np.asarray(time),
        "export_state": np.asarray(states),
        "export_phase": np.asarray(phases).astype(str),
        "spatial_T_bed_K": np.asarray(audit_y)[:, lay.T_bed],
        "spatial_T_gas_K": np.asarray(audit_y)[:, lay.T_gas],
        "spatial_q": np.asarray(audit_y)[:, lay.q].reshape(
            len(audit_y), model.n_cells, model.n_storage
        ),
        "spatial_n_gas": np.asarray(audit_y)[:, lay.n_gas].reshape(len(audit_y), model.n_cells, 4),
        "spatial_cell_mass_kg": np.asarray(model.cell_mass_kg),
        "spatial_cell_centers_m": (np.arange(model.n_cells) + 0.5) * model.dx_m,
    }
    manifest = {
        "schema": "coupled-through-thickness-raw-audit-v21/1",
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in arrays.items()},
        "state_layout": lay.as_dict(),
        "model": model.as_dict(),
        "step_windows": windows,
        "completion": completion,
        "observable_keys": [],
        "basis": ("Last-cycle sampled arrays from the local-gas backend; no routing or aggregation "
                  "has been applied. No five-phase pilot is implied by this file."),
    }
    return {"arrays": arrays, "manifest": manifest}


def _observation_extrema(c: dict, material, g: Geometry, model: CoupledModel,
                         utilities, audit_t, audit_y, audit_p) -> dict:
    evaluators = {}
    peak_values = []
    pump_media_overrun = 0.0
    pump_inlet_temperature_max_K = None
    pump_suction_pressure_min = None
    pump_inlet_pressure_min = None
    pump_inlet_pressure_max = None
    control_pressure_min = None
    control_pressure_max = None
    opening_min = None
    opening_max = None
    pump_bank_count = None
    pressure_spread_max = 0.0
    wall_cooling_peak_W = 0.0
    wall_cooling_unmet_peak_W = 0.0
    cooler_shortfall_max_K = 0.0
    cooler_duty_peak_W = 0.0
    pump_curve_clamped_any = False
    pumped_observed = False
    instantaneous_complete = utilities is not None
    for index, (phase, sample) in enumerate(zip(audit_p, audit_y)):
        phase_name = str(phase)
        if phase_name not in evaluators:
            evaluators[phase_name] = cycle_rhs(c, material, g, phase_name, model=model)
        obs = evaluators[phase_name](float(audit_t[index]), sample)[1]
        if phase_name in PUMPED_PHASES:
            pumped_observed = True
            pump_media_overrun = max(
                pump_media_overrun, float(obs["pump_media_temperature_overrun_K"]))
            inlet_T = float(obs["pump_manifold_temperature_K"])
            pump_inlet_temperature_max_K = (
                inlet_T if pump_inlet_temperature_max_K is None
                else max(pump_inlet_temperature_max_K, inlet_T))
            psuction = float(obs["pump_suction_pressure_Pa"])
            pump_suction_pressure_min = (
                psuction if pump_suction_pressure_min is None
                else min(pump_suction_pressure_min, psuction))
            pin = float(obs["pump_inlet_pressure_Pa"])
            pump_inlet_pressure_min = (
                pin if pump_inlet_pressure_min is None
                else min(pump_inlet_pressure_min, pin))
            pump_inlet_pressure_max = (
                pin if pump_inlet_pressure_max is None
                else max(pump_inlet_pressure_max, pin))
            pcontrol = float(obs["pressure_control_pressure_Pa"])
            control_pressure_min = (
                pcontrol if control_pressure_min is None
                else min(control_pressure_min, pcontrol))
            control_pressure_max = (
                pcontrol if control_pressure_max is None
                else max(control_pressure_max, pcontrol))
            observed_opening = obs.get("pressure_control_opening")
            if observed_opening is not None:
                opening_min = (
                    float(observed_opening) if opening_min is None
                    else min(opening_min, float(observed_opening)))
                opening_max = (
                    float(observed_opening) if opening_max is None
                    else max(opening_max, float(observed_opening)))
            if obs.get("pump_bank_count") is not None:
                pump_bank_count = int(obs["pump_bank_count"])
            pressure_spread_max = max(
                pressure_spread_max, float(obs["channel_pressure_spread_Pa"]))
            pump_curve_clamped_any = bool(
                pump_curve_clamped_any or obs["pump_curve_clamped"])
        wall_cooling_peak_W = max(wall_cooling_peak_W, float(obs["cooling_W"]))
        wall_cooling_unmet_peak_W = max(
            wall_cooling_unmet_peak_W, float(obs["cooling_capacity_shortfall_W"]))
        cooler_shortfall_max_K = max(
            cooler_shortfall_max_K, float(obs["cooler_shortfall_K"]))
        cooler_duty_peak_W = max(cooler_duty_peak_W, float(obs["cooler_duty_W"]))
        if utilities is not None:
            utility_now = utilities.convert(
                wall_cooling_duty_W=2.0 * float(obs["cooling_W"]),
                prepump_cooling_duty_W=2.0 * float(obs["cooler_duty_W"]),
            )
            if utility_now["total_electric_W"] is None:
                instantaneous_complete = False
            else:
                peak_values.append(
                    2.0 * (float(obs["heater_W"]) + float(obs["fan_W"])
                           + float(obs["vacuum_W"])
                           + float(obs["source_auxiliary_W"])
                           + float(obs["cooler_auxiliary_W"]))
                    + float(utility_now["total_electric_W"]))
    return {
        "pumped_observed": pumped_observed,
        "pump_media_overrun_K": pump_media_overrun,
        "pump_inlet_temperature_max_K": pump_inlet_temperature_max_K,
        "pump_suction_pressure_min_Pa": pump_suction_pressure_min,
        "pump_inlet_pressure_min_Pa": pump_inlet_pressure_min,
        "pump_inlet_pressure_max_Pa": pump_inlet_pressure_max,
        "control_pressure_min_Pa": control_pressure_min,
        "control_pressure_max_Pa": control_pressure_max,
        "opening_min": opening_min,
        "opening_max": opening_max,
        "pump_bank_count": pump_bank_count,
        "pressure_control_observable": pressure_control_observable(c),
        "pressure_control_capacity_basis": CAPACITY_PRESSURE_BASIS,
        "channel_pressure_spread_Pa": pressure_spread_max,
        "wall_cooling_peak_W": wall_cooling_peak_W,
        "wall_cooling_unmet_peak_W": wall_cooling_unmet_peak_W,
        "cooler_shortfall_max_K": cooler_shortfall_max_K,
        "cooler_duty_peak_W": cooler_duty_peak_W,
        "pump_curve_clamped_any": pump_curve_clamped_any,
        "instantaneous_complete": bool(instantaneous_complete),
        "peak_watts": (float(max(peak_values)) if instantaneous_complete and peak_values else None),
    }


def summarize(c: dict, material, g: Geometry, audit, *,
              cycle_time_s_override: float | None = None) -> dict:
    (time, states, phases, windows, completion, audit_t, audit_y, audit_p,
     meters, history, model, state, initial, relative, mass_error) = audit
    lay = model.layout
    systems = build_systems(c)
    Ta = c["ambient"]["temperature_C"] + 273.15
    target_C = float(c["cycle"]["regeneration_temperature_C"])
    target_K = target_C + 273.15
    heating = audit_p == "desorption"
    cooling = audit_p == "cooling"
    Tbed = audit_y[:, lay.T_bed]
    Tgas = audit_y[:, lay.T_gas]
    temperature_advisory = material.data.get("temperature_advisory") or {}
    advisory_temperature_mode = (
        temperature_advisory.get("mode") == "record_actual_temperatures"
    )
    historical_range_K = temperature_advisory.get(
        "historical_evaluation_range_temperature_K"
    )
    P = np.vstack([state_pressures(y, model) for y in audit_y])
    pressure_spread = P.max(axis=1) - P.min(axis=1)
    local_n_min = float(np.min(audit_y[:, lay.n_gas]))
    local_q_min = float(np.min(audit_y[:, lay.q]))
    water_audit = water_activity_audit(
        Tgas,
        audit_y[:, lay.n_gas].reshape(len(audit_y), model.n_cells, 4),
        model,
        maximum_model_temperature_K=(
            float(material.data["maximum_bed_temperature_C"]) + 273.15),
    )
    pump_info = pump_export(c)
    utilities = build_utilities(c)
    obs_extrema = _observation_extrema(c, material, g, model, utilities,
                                        audit_t, audit_y, audit_p)
    peak_watts = obs_extrema["peak_watts"]
    instantaneous_complete = obs_extrema["instantaneous_complete"]
    pump_media_overrun = obs_extrema["pump_media_overrun_K"]
    pump_inlet_temperature_max_K = obs_extrema["pump_inlet_temperature_max_K"]
    pump_suction_pressure_min = obs_extrema["pump_suction_pressure_min_Pa"]
    pump_inlet_pressure_min = obs_extrema["pump_inlet_pressure_min_Pa"]
    pump_inlet_pressure_max = obs_extrema["pump_inlet_pressure_max_Pa"]
    control_pressure_min = obs_extrema["control_pressure_min_Pa"]
    control_pressure_max = obs_extrema["control_pressure_max_Pa"]
    opening_min = obs_extrema["opening_min"]
    opening_max = obs_extrema["opening_max"]
    observed_bank_count = obs_extrema["pump_bank_count"]
    pressure_spread_max = obs_extrema["channel_pressure_spread_Pa"]
    wall_cooling_peak_W = obs_extrema["wall_cooling_peak_W"]
    wall_cooling_unmet_peak_W = obs_extrema["wall_cooling_unmet_peak_W"]
    cooler_shortfall_max_K = obs_extrema["cooler_shortfall_max_K"]
    cooler_duty_peak_W = obs_extrema["cooler_duty_peak_W"]
    pump_curve_clamped_any = obs_extrema["pump_curve_clamped_any"]
    pumped_observed = obs_extrema["pumped_observed"]
    pumped = np.isin(phases, PUMPED_PHASES)
    max_pumped_gas_K = float(np.max(Tgas[pumped])) if pumped.any() else None
    media_limit_C = pump_info.get("media_temperature_limit_C")
    media_overrun = (None if max_pumped_gas_K is None or media_limit_C is None
                     else max(max_pumped_gas_K - (media_limit_C + 273.15), 0.0))
    if heating.any():
        hidx = np.flatnonzero(heating)
        end_idx = hidx[-1]
        coldest_end = float(np.min(Tbed[end_idx]))
        hottest_end = float(np.max(Tbed[end_idx]))
        coldest_peak = float(np.min(np.max(Tbed[heating], axis=0)))
        hottest_peak = float(np.max(Tbed[heating]))
    else:
        end_idx = 0
        coldest_end = hottest_end = coldest_peak = hottest_peak = float("nan")
    endpoint = Ta + c["cycle"]["cooling_approach_K"]
    if cooling.any():
        cidx = np.flatnonzero(cooling)[-1]
        cool_zone = float(np.max(Tbed[cidx]))
        cool_wall = float(states[cidx, lay.T_w])
        cool_end = max(cool_zone, cool_wall)
    else:
        cool_zone = cool_wall = cool_end = float("nan")
    shortfall = max(target_K - coldest_end, 0.0)
    cooling_shortfall = max(cool_end - endpoint, 0.0) if isfinite(cool_end) else float("nan")
    cooling_done = completion.get("cooling", {})
    cooling_pass = bool(isfinite(cooling_shortfall) and cooling_shortfall <= COOLING_TOLERANCE_K
                        and (cooling_done.get("event_fired", False)
                             or cooling_done.get("at_entry", False)))
    product = state[lay.product]
    full_product = 2.0 * product
    kg_co2 = float(full_product[0] * MW[0])
    cycle_time = float(time[-1] if cycle_time_s_override is None else cycle_time_s_override)
    kg_day = kg_co2 / max(cycle_time, 1e-12) * 86400.0
    max_first_law = max(float(row.get("first_law_relative", relative)) for row in history)
    max_species = max(float(row.get("species_relative", mass_error)) for row in history)
    condensate_vec = np.zeros(4)
    if lay.extended:
        condensate_vec[1] = state[lay.condensate]
    q_final = state[lay.q].reshape(model.n_cells, model.n_storage)
    q_initial = initial[lay.q].reshape(model.n_cells, model.n_storage)
    storage_species_delta = np.zeros(4)
    for storage_index, species_index in enumerate(material.species_for_storage):
        storage_species_delta[int(species_index)] += float(
            np.sum(
                model.cell_mass_kg
                * (q_final[:, storage_index] - q_initial[:, storage_index])
            )
        )
    final_species_residual = (2.0 * (
        storage_species_delta
        + state[lay.n_gas].reshape(model.n_cells, 4).sum(axis=0)
        - initial[lay.n_gas].reshape(model.n_cells, 4).sum(axis=0)
        - state[lay.inlet] + state[lay.vent] + state[lay.product] + condensate_vec))
    pressure_audit = P[heating] if heating.any() else P
    source_energy = 2.0 * float(state[lay.e_source]) if lay.extended else 0.0
    external_input_energy = (
        2.0 * float(state[lay.e_source_input]) if lay.extended else None
    )
    source_auxiliary_energy = (
        2.0 * float(state[lay.e_source_aux]) if lay.extended else None
    )
    source_diagnostic = source_execution_diagnostic(c, systems.source)
    desorption_duration = sum(
        end - start for phase, start, end in windows if phase == "desorption"
    )
    external_carrier_reconciled = True
    source_input_factor = (
        source_input_per_delivered_W(systems.source)
        if systems.source is not None else 0.0
    )
    if systems.source is not None:
        input_half = external_input_energy / 2.0
        expected_input_half = 0.5 * source_energy * source_input_factor
        expected_auxiliary = (
            systems.source.auxiliary_electric_W * desorption_duration
        )
        external_carrier_reconciled = bool(
            source_diagnostic["external_carrier_complete"]
            and np.isfinite(source_energy)
            and np.isfinite(external_input_energy)
            and np.isfinite(source_auxiliary_energy)
            and math.isclose(
                input_half, float(expected_input_half),
                rel_tol=1e-10, abs_tol=1e-5,
            )
            and math.isclose(
                source_auxiliary_energy, expected_auxiliary,
                rel_tol=1e-10, abs_tol=1e-5,
            )
        )
    external_carrier_complete = bool(
        source_diagnostic["external_carrier_complete"]
        and external_carrier_reconciled
    )
    summary = {
        "model": "coupled_through_thickness_v21",
        "v21_scenario": (
            dict(c["v21_scenario"]) if isinstance(c.get("v21_scenario"), dict)
            else None
        ),
        "v21_technology_profile_sha256": c.get(
            "v21_technology_profile_sha256"
        ),
        "pump_rate_mode": pump_flow_basis(c),
        "cycles_run": len(history),
        "cyclic_steady_state": bool(history[-1]["converged"]),
        "cyclic_steady_state_tolerance": float(c["cycle"]["css_relative_tolerance"]),
        "cyclic_steady_state_binding_quantity": history[-1]["binding"],
        "cyclic_steady_state_binding_change": history[-1]["binding_change"],
        "css_groups": {"local": history[-1]["css_group_local"],
                       "carbon_product": history[-1]["css_group_carbon"],
                       "energy_meter": history[-1]["css_group_energy"],
                       "cycle_time": history[-1]["css_group_time"]},
        "cycle_time_s": cycle_time,
        "CO2_product_mol": float(full_product[0]),
        "CO2_product_kg": kg_co2,
        "CO2_product_kg_per_day": kg_co2 / max(cycle_time, 1e-12) * 86400.0,
        "CO2_product_kg_per_kg_sorbent_day": (kg_co2 / max(cycle_time, 1e-12) * 86400.0
                                               / float(c["bed"]["dry_sorbent_mass_kg"])),
        "heater_energy_J": 2.0 * float(state[lay.E_heat]),
        "cooling_removed_J": 2.0 * float(state[lay.E_cool]),
        "fan_energy_J": 2.0 * float(state[lay.E_fan]),
        "vacuum_energy_J": 2.0 * float(state[lay.E_vac]),
        "conditioning_energy_J": 2.0 * float(state[lay.E_cond]),
        "ambient_loss_energy_J": 2.0 * float(state[lay.E_loss]),
        "net_boundary_enthalpy_J": 2.0 * float(state[lay.E_netflow]),
        "energy_first_law_residual_J": float(2.0 * (
            energy_inventory(c, material, g, state, model)["total_J"]
            - energy_inventory(c, material, g, initial, model)["total_J"]
            - state[lay.E_heat] - (0.5 * source_energy)
            + state[lay.E_cool] + state[lay.E_loss]
            - state[lay.E_netflow])),
        "energy_first_law_relative_residual": relative,
        "maximum_cycle_first_law_relative_residual": max_first_law,
        "energy_first_law_closed": bool(relative <= FIRST_LAW_RELATIVE_TOLERANCE),
        "regeneration_target_C": target_C,
        "desorption_peak_bed_temperature_C": hottest_peak - 273.15,
        "coldest_zone_peak_bed_temperature_C": coldest_peak - 273.15,
        "coldest_zone_end_bed_temperature_C": coldest_end - 273.15,
        "desorption_final_bed_temperature_C": coldest_end - 273.15,
        "desorption_final_hottest_C": hottest_end - 273.15,
        "heating_shortfall_K": float(shortfall),
        "heating_overshoot_K": max(hottest_peak - target_K, 0.0),
        "regeneration_pass": bool(shortfall <= 2.0),
        "cooling_endpoint_C": endpoint - 273.15,
        "cooling_max_zone_end_C": cool_zone - 273.15 if isfinite(cool_zone) else None,
        "cooling_wall_end_C": cool_wall - 273.15 if isfinite(cool_wall) else None,
        "cooling_shortfall_K": None if not isfinite(cooling_shortfall) else float(cooling_shortfall),
        "cooling_pass": cooling_pass,
            "peak_desorption_pressure_Pa": float(np.max(pressure_audit)),
        "minimum_pressure_Pa": float(np.min(P)),
        "maximum_pressure_Pa": float(np.max(P)),
        "final_pressure_Pa": float(np.min(P[-1])),
        "channel_pressure_spread_Pa_max": float(np.max(pressure_spread)),
        "channel_pressure_spread_Pa_final": float(pressure_spread[-1]),
        "local_inventory_min_mol": local_n_min,
        "local_inventory_min_mol_kg": local_q_min,
        "withdrawal_allocation": {
            "share_sum": float(np.sum(model.cell_flow_share)),
            "nonnegative": bool(np.all(model.cell_flow_share >= 0.0)),
            "feasible": bool(np.all(model.cell_flow_share >= 0.0)
                             and np.isclose(np.sum(model.cell_flow_share), 1.0)),
            "basis": "ideal equal-share minimum-suction manifold; hypothetical flow balancing",
        },
        "pressure_control_observations": {
            "observable": obs_extrema["pressure_control_observable"],
            "pump_rate_model": pump_flow_basis(c),
            "opening_basis": "global suction valve from selected local pressure",
            "capacity_pressure_basis": obs_extrema["pressure_control_capacity_basis"],
            "pump_bank_count": observed_bank_count,
            "pump_bank_evaluated_once": True,
            "minimum_local_suction_pressure_Pa": pump_suction_pressure_min,
            "minimum_control_pressure_Pa": control_pressure_min,
            "maximum_control_pressure_Pa": control_pressure_max,
            "minimum_opening": opening_min,
            "maximum_opening": opening_max,
            "minimum_pump_inlet_pressure_Pa": pump_inlet_pressure_min,
            "maximum_pump_inlet_pressure_Pa": pump_inlet_pressure_max,
        },
        "pump_limitations": {
                "maximum_pump_inlet_temperature_K": pump_inlet_temperature_max_K,
            "media_limit_C": media_limit_C,
                "media_temperature_overrun_K": pump_media_overrun,
                "minimum_pump_suction_pressure_Pa": pump_suction_pressure_min,
                "channel_pressure_spread_Pa": pressure_spread_max,
                "curve_clamped_any": pump_curve_clamped_any,
            "wet_hot_validation": "UNKNOWN",
        },
            "pump_observations": {
                "pumped_samples_observed": pumped_observed,
                "pump_rate_model": pump_flow_basis(c),
                "maximum_pump_inlet_temperature_K": pump_inlet_temperature_max_K,
                "media_temperature_overrun_K": pump_media_overrun,
                "minimum_suction_pressure_Pa": pump_suction_pressure_min,
                "pressure_control_observable": obs_extrema["pressure_control_observable"],
                "maximum_control_pressure_Pa": control_pressure_max,
                "minimum_control_pressure_Pa": control_pressure_min,
                "minimum_opening": opening_min,
                "maximum_opening": opening_max,
                "minimum_pump_inlet_pressure_Pa": pump_inlet_pressure_min,
                "maximum_pump_inlet_pressure_Pa": pump_inlet_pressure_max,
                "channel_pressure_spread_Pa": pressure_spread_max,
                "curve_clamped_any": pump_curve_clamped_any,
            },
            "wall_cooling_observations": {
                "peak_delivered_W": wall_cooling_peak_W,
                "peak_unmet_W": wall_cooling_unmet_peak_W,
                "declared_capacity_W": (None if utilities is None
                                        else utilities.wall_cooling_capacity_W()),
            },
            "conditioner_observations": {
                "policy": "pump_inlet_cooler_knockout_v1",
                "pressure_drop_Pa": 0.0,
                "line_holdup_mol": 0.0,
                "pump_running_input_W": 135.0,
                "peak_duty_W": cooler_duty_peak_W,
                "maximum_target_shortfall_K": cooler_shortfall_max_K,
                "configured": bool(systems.conditioner is not None),
            },
        "peak_bed_temperature_C": float(np.max(Tbed) - 273.15),
        "minimum_bed_temperature_C": float(np.min(Tbed) - 273.15),
        "temperature_observations": {
            "basis": (
                "sampled accepted solver states retained in this cycle; "
                "not an independent continuous-time certification"
            ),
            "minimum_bed_temperature_K": float(np.min(Tbed)),
            "maximum_bed_temperature_K": float(np.max(Tbed)),
            "maximum_bed_temperature_C": float(np.max(Tbed) - 273.15),
            "requested_regeneration_temperature_C": target_C,
            "historical_evaluation_range_K": historical_range_K,
            "historical_range_hard_stop": (
                False if advisory_temperature_mode else None
            ),
            "scenario_boundary_excursion_K": (
                max(
                    float(np.max(Tbed)) - float(historical_range_K[1]),
                    0.0,
                )
                if advisory_temperature_mode
                and isinstance(historical_range_K, list)
                and len(historical_range_K) == 2
                else None
            ),
            "status": (
                "actual_temperatures_recorded_no_advisory_stop"
                if advisory_temperature_mode
                else "legacy_configured_guard"
            ),
        },
        "source_execution": source_diagnostic,
        "external_source_delivered_heat_J": (
            source_energy if systems.source is not None else None
        ),
        "external_thermal_input_J": (
            external_input_energy if systems.source is not None else None
        ),
        "source_auxiliary_electricity_J": (
            source_auxiliary_energy if systems.source is not None else None
        ),
        "external_carrier_complete": external_carrier_complete,
        "external_carrier_reconciled": external_carrier_reconciled,
        "source_input_per_delivered": source_input_factor,
            "maximum_water_activity": water_audit["maximum_upper_bound"],
            "maximum_water_activity_basis": water_audit["basis"],
            "water_activity_audit": water_audit,
        "species_balance_residual_mol": dict(zip(SPECIES, final_species_residual.tolist())),
        "maximum_relative_species_balance_error": max_species,
        "last_cycle_relative_species_balance_error": mass_error,
        "species_closure": {
            "relative_by_species": history[-1].get("species_relative_by_species", {}),
            "residual_mol_by_species": history[-1].get("species_residual_mol", {}),
            "maximum_cycle_relative": max_species,
            "decision_threshold": 1e-8,
            "pass": bool(max_species <= 1e-8),
        },
        "flags": [],
        "numerical_operating_checks_passed": bool(
            history[-1]["converged"] and max_first_law <= 1e-8 and max_species <= 1e-8),
        "validity_scope": "Numerical conservation and configured-limit checks only; not experimental validation.",
        "geometry": g.as_dict(),
        "state_size": lay.size,
        "state_layout": lay.as_dict(),
        "spatial_model": model.as_dict(),
        "transport_assumptions": {
            "h_gs_W_m2_K": float(model.gas_solid_h_W_m2_K),
            "source_class": "analyst",
            "pressure_applicability": (
                "all modeled local pressures, including very low isolated cooling"),
            "transport_probe_domain": (
                "does not establish this constant at very low isolated-cooling pressure"),
            "not_calibration": True,
        },
        "phase_windows": [{"phase": p, "start_s": s, "end_s": e, "duration_s": e - s}
                          for p, s, e in windows],
        "phase_completion": completion,
        "history": history,
        "solver_policy": {
            "rtol": float(c["solver"]["rtol"]),
            "max_step_s": float(c["solver"]["max_step_s"]),
            "samples_per_step": int(c["solver"]["samples_per_step"]),
        },
        "numerical_qualification": {
            "conservation_css_known": True,
            "conservation_css_status": (
                "pass" if history[-1]["converged"] and max_first_law <= 1e-8
                and max_species <= 1e-8 else "fail"),
            "mesh_validation_known": False,
            "mesh_validation_status": "unknown",
            "mesh_note": "A single mesh cannot establish the N8/N16 mesh qualification.",
        },
        "model_limitations": ("Local-gas parallel-channel surrogate; zero transverse gas exchange; "
                              "no axial front, maldistribution, in-bed condensation, wall axial "
                              "conduction or measured transport calibration."),
    }
    # Full-train physical gates and complete model electricity. The utility
    # adapter is the unchanged old conversion; upstream drying and equipment
    # adequacy remain unknown.
    setpoint = float(c["cycle"]["regeneration_pressure_Pa"])
    p_gate = 1.05 * setpoint
    peak_p = float(summary["peak_desorption_pressure_Pa"])
    final_p = float(summary["final_pressure_Pa"])
    Patm = float(c["ambient"]["pressure_Pa"])
    material_max_C = float(material.data["maximum_bed_temperature_C"])
    water_known = bool(water_audit["known"])
    if not water_known:
        summary["maximum_water_activity"] = None
    cooling_window = sum(e - s for p, s, e in windows if p == "cooling")
    flow_drop = float(airflow(c, g)["pressure_drop_fraction_of_ambient"])
    pump_domain = pump_info.get("pressure_domain_Pa")
    pump_curve_known = pump_domain is not None
    physical_checks = {
        "regeneration_temperature_reached": {
            "known": True, "pass": bool(shortfall <= 2.0), "value": shortfall, "threshold": 2.0},
        "regeneration_temperature_overshoot": {
            "known": True, "pass": bool(summary["heating_overshoot_K"] <= BED_OVERSHOOT_GUARD_K),
            "value": summary["heating_overshoot_K"], "threshold": BED_OVERSHOOT_GUARD_K},
        "pump_pressure_target": {
            "known": True, "pass": bool(peak_p <= p_gate), "value": peak_p, "threshold": p_gate},
        "repressurization_pressure": {
            "known": True, "pass": bool(final_p >= 0.98 * Patm), "value": final_p,
            "threshold": 0.98 * Patm},
        "pump_media_temperature": {
            "known": bool(pumped_observed and media_limit_C is not None),
            "pass": bool(pump_media_overrun <= 0.0) if pumped_observed and media_limit_C is not None else None,
            "value": pump_media_overrun, "threshold": 0.0},
        "saturation": {
            "known": water_known,
            "pass": (bool(summary["maximum_water_activity"] <= 1.01) if water_known else None),
            "value": summary["maximum_water_activity"], "threshold": 1.01,
            "basis": water_audit["basis"],
            "audit": water_audit},
        "flow_model_domain": {
            "known": bool(np.isfinite(flow_drop)), "pass": bool(flow_drop <= 0.10),
            "value": flow_drop, "threshold": 0.10},
        "product_present": {
            "known": True, "pass": bool(kg_co2 > 1e-10), "value": kg_co2, "threshold": 1e-10},
        "cooldown_completed_on_time": {
            "known": True,
            "pass": bool(cooling_pass and cooling_window < c["cycle"]["cooling_max_s"] - 1e-6),
            "value": cooling_window, "threshold": c["cycle"]["cooling_max_s"]},
        "wall_cooling_capacity": {
            "known": bool(utilities is not None
                          and utilities.wall_cooling_capacity_W() is not None),
            "pass": bool(cooling_pass) if utilities is not None and
            utilities.wall_cooling_capacity_W() is not None else None,
            "value": wall_cooling_peak_W,
            "unmet_peak_W": wall_cooling_unmet_peak_W},
        "conditioner_cooling_target": {
            "known": bool(systems.conditioner is not None and pumped_observed),
            "pass": bool(cooler_shortfall_max_K <= 1.0)
            if systems.conditioner is not None and pumped_observed else None,
            "value": cooler_shortfall_max_K, "threshold": 1.0},
        "pump_curve_domain": {
            "known": bool(pump_curve_known and pumped_observed),
            "pass": bool(not pump_curve_clamped_any) if pump_curve_known and pumped_observed else None,
            "value": pump_curve_clamped_any},
    }
    for name, check in physical_checks.items():
        if check["known"] and check["pass"] is False:
            flag = {
                "regeneration_temperature_reached": "Coldest-zone regeneration endpoint is below the 2 K allowance.",
                "regeneration_temperature_overshoot": "Hot-zone heating overshoot exceeds the 5 K guard.",
                "pump_pressure_target": "Desorption pressure exceeds the 1.05x setpoint gate.",
                "repressurization_pressure": "Repressurization ends below 98% of ambient.",
                "pump_media_temperature": "Pump media temperature gate failed.",
                "saturation": "Water activity exceeds the 1.01 saturation screen.",
                "flow_model_domain": "Pressure-drop fraction exceeds 0.10.",
                "product_present": "No usable CO2 product.",
                "cooldown_completed_on_time": "Cooling did not complete before its cap.",
                "wall_cooling_capacity": "Wall-cooling capacity audit failed.",
                "conditioner_cooling_target": "Conditioner cooling target shortfall exceeds 1 K.",
                "pump_curve_domain": "Pump curve domain exceeded.",
            }[name]
            if flag not in summary["flags"]:
                summary["flags"].append(flag)
    known_checks = all(c["known"] for c in physical_checks.values())
    passed_checks = all(c["pass"] is True for c in physical_checks.values() if c["known"])
    summary["physical_gates"] = {
        "known": known_checks,
        "pass": bool(known_checks and passed_checks),
        "checks": physical_checks,
        "unknown_checks": [name for name, c in physical_checks.items() if not c["known"]],
        "material_limit_C": material_max_C,
        "material_limit_pass": (
            None
            if advisory_temperature_mode
            else bool(summary["peak_bed_temperature_C"] <= material_max_C)
        ),
        "material_limit_basis": (
            "temperature advisory records excursions; no evidenced physical limit"
            if advisory_temperature_mode
            else "configured scenario/material guard"
        ),
        "species_closure_pass": bool(max_species <= 1e-8),
    }
    if (
        not advisory_temperature_mode
        and summary["peak_bed_temperature_C"] > material_max_C
    ):
        summary["flags"].append("Material temperature limit exceeded.")
    utilities = build_utilities(c)
    cooling_duration = sum(e - s for p, s, e in windows if p == "cooling")
    pumped_duration = sum(e - s for p, s, e in windows if p in PUMPED_PHASES)
    wall_duty_J = 2.0 * float(state[lay.E_cool])
    prepump_duty_J = 2.0 * float(state[lay.e_cooler]) if lay.extended else 0.0
    utility_block = (None if utilities is None else utilities.cycle_energy(
        wall_cooling_duty_J=wall_duty_J,
        wall_cooling_on_duration_s=cooling_duration,
        prepump_cooling_duty_J=prepump_duty_J,
        prepump_cooling_on_duration_s=pumped_duration if prepump_duty_J > 0.0 else 0.0,
    ))
    electric_terms = {
        "electric_heater_J": 2.0 * float(state[lay.E_heat]),
        "electric_pump_J": 2.0 * float(state[lay.E_vac]),
        "electric_fan_J": 2.0 * float(state[lay.E_fan]),
        "electric_legacy_conditioning_J": 2.0 * float(state[lay.E_cond]),
        "electric_dryer_J": 0.0,
        "electric_source_auxiliary_J": 2.0 * float(state[lay.e_source_aux]) if lay.extended else 0.0,
        "electric_cooler_auxiliary_J": 2.0 * float(state[lay.e_cooler_aux]) if lay.extended else 0.0,
        "electric_utilities_J": None if utility_block is None else utility_block["total_electric_J"],
    }
    complete_electric = all(v is not None for v in electric_terms.values())
    total_electric = (float(sum(v for v in electric_terms.values() if v is not None))
                      if complete_electric else None)
    electric_meter_J = 2.0 * float(state[lay.E_electric_total])
    electric_scale = max(abs(total_electric) if total_electric is not None else 0.0,
                         abs(electric_meter_J), 1.0)
    integrated_reconciled = bool(
        complete_electric
        and abs(electric_meter_J - float(total_electric)) <= 1e-8 * electric_scale)
    carrier_electric = (None if total_electric is None
                        else float(total_electric - float(electric_terms["electric_utilities_J"] or 0.0)))
    carrier_reconciled = bool(
        total_electric is not None and carrier_electric is not None
        and abs(carrier_electric + float(electric_terms["electric_utilities_J"] or 0.0)
                - float(total_electric)) <= 1e-8 * electric_scale)
    summary["electricity"] = {
        "terms_J": electric_terms,
        "total_J": total_electric,
        "kWh_day": (None if total_electric is None
                    else total_electric / 3.6e6 * 86400.0 / max(cycle_time, 1e-12)),
        "complete": complete_electric,
        "integrated_complete": bool(complete_electric),
        "integrated_reconciled": integrated_reconciled,
        "integrated_boundary_meter_J": electric_meter_J,
        "instantaneous_complete": bool(instantaneous_complete),
        "peak_electric_W": peak_watts,
        "peak_envelope_W": 3000.0,
        "envelope_exceeded": (None if peak_watts is None else bool(peak_watts > 3000.0)),
        "kWh_kg_CO2": (None if total_electric is None or kg_co2 <= 1e-10
                       else total_electric / 3.6e6 / kg_co2),
        "electricity_kWh_kg_CO2": (None if total_electric is None or kg_co2 <= 1e-10
                                   else total_electric / 3.6e6 / kg_co2),
        "utility_adapter": utility_block,
        "upstream_drying_status": "UNKNOWN",
        "note": ("Complete model electricity is known only when every configured carrier is declared; "
                 "upstream drying and equipment/cost acceptance remain unknown."),
    }
    summary["plant"] = {
        "electricity_J": total_electric,
        "electricity_boundary_meter_J": electric_meter_J,
        "electricity_complete": complete_electric,
        "electricity_terms_J": electric_terms,
        "instantaneous_complete": bool(instantaneous_complete),
        "instantaneous_peak_W": peak_watts,
        "peak_envelope_W": 3000.0,
        "external_thermal_input_J": (
            external_input_energy if systems.source is not None else None
        ),
        "external_source_delivered_heat_J": (
            source_energy if systems.source is not None else None
        ),
        "source_auxiliary_electricity_J": (
            source_auxiliary_energy if systems.source is not None else None
        ),
    }
    summary["energy_carriers"] = {
        "electric_total_J": carrier_electric,
        "electric_utilities_J": electric_terms["electric_utilities_J"],
        "plant_electricity_J": total_electric,
        "carrier_relation_reconciled": carrier_reconciled,
        "external_thermal_input_J": (
            external_input_energy if systems.source is not None else None
        ),
        "external_source_delivered_heat_J": (
            source_energy if systems.source is not None else None
        ),
        "source_auxiliary_electricity_J": (
            source_auxiliary_energy if systems.source is not None else None
        ),
        "external_carrier_applicable": systems.source is not None,
        "external_carrier_complete": external_carrier_complete,
        "external_carrier_reconciled": external_carrier_reconciled,
        "external_carrier_status": (
            "not_applicable_by_electrical_profile"
            if systems.source is None
            else "pass" if external_carrier_complete else "fail"
        ),
        "separation": (
            "electricity, external thermal input, delivered heat and source "
            "auxiliary electricity are separate carriers"
        ),
    }
    prescribed_2pct = bool(
        abs(float(c["ambient"]["relative_humidity"]) - 0.02) <= 1e-12
        and build_dryer(c) is None
        and c["conditioning"]["electric_J_per_kg_water_removed"] in (None, 0.0))
    model_carrier_known = bool(
        complete_electric
        and integrated_reconciled
        and instantaneous_complete
        and carrier_reconciled
        and external_carrier_complete
        and prescribed_2pct
    )
    model_carrier_pass = bool(model_carrier_known and peak_watts is not None
                              and peak_watts <= 3000.0)
    summary["model_carrier_scope"] = {
        "scope": "prescribed_conditioned_2pct_rh",
        "applicable": prescribed_2pct,
        "known": model_carrier_known,
        "pass": model_carrier_pass,
        "integrated_reconciled": integrated_reconciled,
        "carrier_relation_reconciled": carrier_reconciled,
        "instantaneous_complete": bool(instantaneous_complete),
        "peak_envelope_exceeded": (None if peak_watts is None else bool(peak_watts > 3000.0)),
        "note": ("Model-carrier feasibility only; upstream drying is outside this scope. "
                 if prescribed_2pct else
                 "The configured feed is outside the prescribed 2% RH boundary, so model-carrier "
                 "feasibility is unknown."),
    }
    summary["real_resource_readiness"] = {
        "known": False,
        "status": "unknown",
        "reasons": ["upstream drying/availability/energy unknown",
                    "equipment selection and wet/hot adequacy unknown",
                    "complete CAPEX/OPEX unknown"],
    }
    summary["resource_gates"] = {
        "known": model_carrier_known,
        "status": "pass" if model_carrier_pass else ("fail" if model_carrier_known else "unknown"),
        "model_carrier_scope": summary["model_carrier_scope"],
        "real_resource_readiness": summary["real_resource_readiness"],
    }
    summary["material_status"] = material.data["status"]
    summary["material_empirical_status"] = (
        material.data.get("status")
        or "UNKNOWN_EVIDENCE_STATUS"
    )
    if summary["heating_shortfall_K"] > 2.0:
        summary["flags"].append("Coldest-zone regeneration endpoint is below the 2 K allowance.")
    if not cooling_pass:
        summary["flags"].append("Cooling endpoint not met by every zone and wall.")
    if (
        not advisory_temperature_mode
        and summary["peak_bed_temperature_C"] + 273.15
        > material.data["maximum_bed_temperature_C"] + 273.15
    ):
        summary["flags"].append("Material temperature limit exceeded.")
    if not summary["cyclic_steady_state"]:
        summary["flags"].append("Cyclic steady state not reached.")
    if relative > 1e-8:
        summary["flags"].append("First-law relative residual exceeds the 1e-8 decision gate.")
    return summary


def water_activity_audit(Tgas: np.ndarray, ng: np.ndarray, model: CoupledModel, *,
                         maximum_model_temperature_K: float) -> dict:
    """Conservative water-activity bound for saved local gas states.

    The Antoine closure is evaluated directly inside its 255.9-373 K domain.
    Above 373 K and within the declared model-temperature range, the
    denominator is held at p_sat(373 K).  Because saturation pressure is
    monotone increasing, this is an upper bound on p_H2O/p_sat(T_gas), not an
    extrapolated relative humidity.  States below the correlation domain or
    above the model range leave the gate unknown.
    """
    Tgas = np.asarray(Tgas, dtype=float)
    ng = np.asarray(ng, dtype=float)
    if Tgas.shape != ng.shape[:2]:
        raise ValueError("water-activity temperature and gas-inventory axes do not match")
    if ng.shape != (Tgas.shape[0], model.n_cells, 4):
        raise ValueError("water-activity gas inventory must have shape (samples, cells, 4)")
    upper_K = float(maximum_model_temperature_K)
    if not np.isfinite(upper_K) or upper_K < 373.0:
        raise ValueError("maximum model temperature must be finite and at least 373 K")
    direct_max = None
    hot_max = None
    direct_count = 0
    hot_count = 0
    below_count = 0
    above_count = 0
    nonfinite_count = 0
    saturation_at_upper = water_saturation_pressure_Pa(373.0)
    for i in range(Tgas.shape[0]):
        for j in range(model.n_cells):
            t = float(Tgas[i, j])
            if not np.isfinite(t):
                nonfinite_count += 1
                continue
            p_h2o = max(float(ng[i, j, 1]), 0.0) * R * t / model.cell_gas_volume_m3[j]
            if t < 255.9:
                below_count += 1
            elif t <= 373.0:
                value = p_h2o / water_saturation_pressure_Pa(t)
                direct_max = value if direct_max is None else max(direct_max, value)
                direct_count += 1
            elif t <= upper_K:
                value = p_h2o / saturation_at_upper
                hot_max = value if hot_max is None else max(hot_max, value)
                hot_count += 1
            else:
                above_count += 1
    supported = (
        Tgas.size > 0
        and below_count == 0
        and above_count == 0
        and nonfinite_count == 0
    )
    candidates = [v for v in (direct_max, hot_max) if v is not None]
    maximum = max(candidates) if supported and candidates else None
    return {
        "known": bool(supported and maximum is not None),
        "maximum_upper_bound": maximum,
        "direct_maximum": direct_max,
        "hot_maximum_upper_bound": hot_max,
        "direct_samples": direct_count,
        "hot_samples": hot_count,
        "below_domain_samples": below_count,
        "above_model_range_samples": above_count,
        "nonfinite_samples": nonfinite_count,
        "saturation_temperature_upper_bound_K": 373.0,
        "saturation_pressure_upper_bound_Pa": saturation_at_upper,
        "maximum_model_temperature_K": upper_K,
        "monotonic_saturation_assumption": True,
        "exact_relative_humidity_extrapolated": False,
        "basis": (
            "Direct Antoine evaluation in 255.9-373 K; conservative "
            "p_H2O/p_sat(373 K) upper bound above 373 K within the model range"),
        "wet_hot_validation": "UNKNOWN",
        "note": (
            "The hot result is a bound under the declared monotonic-saturation "
            "assumption, not an extrapolated exact RH or wet-material validation."),
    }


def result_payload(summary: dict, *, candidate_id: str, scenario_id: str,
                   identity: dict, actual_input_identity: dict | None = None) -> dict:
    conservation_pass = (
        summary["cyclic_steady_state"]
        and summary["maximum_cycle_first_law_relative_residual"] <= 1e-8
        and summary["species_closure"]["maximum_cycle_relative"] <= 1e-8
    )
    cold_pass = summary["regeneration_pass"] and summary["cooling_pass"]
    audits = {
        "numerical": {
            "known": bool(not conservation_pass),
            "status": "fail" if not conservation_pass else "unknown",
        },
        "conservation_css": {
            "known": True,
            "status": "pass" if conservation_pass else "fail",
            "maximum_cycle_first_law_relative_residual":
                summary["maximum_cycle_first_law_relative_residual"],
            "maximum_cycle_species_relative_residual":
                summary["species_closure"]["maximum_cycle_relative"],
        },
        "physical": {"known": bool(summary["physical_gates"]["known"]),
                     "status": ("pass" if summary["physical_gates"]["pass"]
                                else "fail" if summary["physical_gates"]["known"]
                                else "unknown")},
        "resource": {"known": bool(summary["resource_gates"]["known"]),
                     "status": ("pass" if summary["resource_gates"]["status"] == "pass"
                                else "fail" if summary["resource_gates"]["status"] == "fail"
                                else "unknown")},
        "real_resource_readiness": summary["real_resource_readiness"],
        "coldspot": {"known": True, "status": "pass" if cold_pass else "fail"},
        "allzone_css": {"known": bool(summary["cyclic_steady_state"]),
                        "status": "pass" if summary["cyclic_steady_state"] else "fail"},
    }
    return {
        "schema": "operating-window-coupled-result-v21/1",
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "identity": dict(identity),
        "recipe_identity": dict(identity),
        "actual_input_identity": dict(actual_input_identity or {}),
        "identity_kind_note": ("`recipe_identity` is the C candidate+scenario recipe hash; "
                               "`actual_input_identity` is the resolved runtime/source/material/"
                               "equipment/hash record and must be compared separately."),
        "audits": audits,
        "gates": {"failed_gate_names": [] if not summary["flags"] else summary["flags"]},
        "production_kg_day": summary["CO2_product_kg_per_day"],
        "energy": {"electricity_kWh_day": summary["electricity"]["kWh_day"],
                   "electricity_kWh_kg_CO2": summary["electricity"]["electricity_kWh_kg_CO2"],
                   "electricity_complete": summary["electricity"]["integrated_complete"],
                   "electricity_integrated_reconciled": summary["electricity"]["integrated_reconciled"],
                   "electricity_instantaneous_complete": summary["electricity"]["instantaneous_complete"],
                   "peak_electric_W": summary["electricity"]["peak_electric_W"],
                   "external_thermal_input_J": summary["external_thermal_input_J"],
                   "external_source_delivered_heat_J":
                       summary["external_source_delivered_heat_J"],
                   "source_auxiliary_electricity_J":
                       summary["source_auxiliary_electricity_J"],
                   "external_carrier_complete":
                       summary["external_carrier_complete"],
                   "external_carrier_reconciled":
                       summary["external_carrier_reconciled"]},
        "margins": {"heating_shortfall_K": summary["heating_shortfall_K"],
                    "cooling_shortfall_K": summary["cooling_shortfall_K"],
                    "regeneration_pass": summary["regeneration_pass"],
                    "cooling_pass": summary["cooling_pass"]},
        "phase_durations_s": summary["phase_windows"],
        "criteria_provenance": {
            "first_law_relative": 1e-8,
            "css_tolerance": 0.002,
            "regeneration_tolerance_K": 2.0,
            "cooling_endpoint_tolerance_K": 0.1,
            "solver": summary["solver_policy"],
            "architecture": "local_gas_parallel_channel",
        },
    }
