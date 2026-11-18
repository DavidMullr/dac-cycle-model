"""Transverse gas/solid adsorption model with parallel gas channels.

Each cell has a local gas inventory, solid loading, temperature and pressure.
Feed, leakage and withdrawal follow the declared channel shares; transverse gas
exchange is not resolved. Effective bulk conductivity supplies through-thickness
heat transfer, without counting gas conduction twice.

Heater and wall states, pressure control and shared cooling are coupled to the
material balances. An ideal pump-inlet cooler and knockout targets 55 C. Storage
supports one or two CO2 sites, water and optional inert adsorbates."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
import time

import numpy as np
from scipy.integrate import solve_ivp

from ..airflow import airflow
from .. import pump_profile
from ..common import (CP_GAS, MW, R, ExpectedScenarioError,
                     gas_enthalpy_mol,
                     gas_internal_energy_mol, pump_profile_for)
from ..pressure_control import (
    CAPACITY_PRESSURE_BASIS,
    pressure_control_observable,
    pressure_control_pressure,
)
from ..cycle_control import DesorptionControlRuntime, load_desorption_control
from ..feed_dryer import build_dryer
from ..geometry import Geometry
from ..systems import build_systems, Systems
from ..heat_source_contract import (
    HeatSourceContractError,
    require_declared_thermal_conversion,
)
from ..utilities import build_utilities
from ..pump_rates import MODES, rate_at_pressure
from .lumped_port.pump_conditioner import (
    evaluate_pump_conditioner,
    pump_conditioner_config,
)
from .spatial_jacobian import (
    SpatialJacobianAudit,
    bounded_spatial_jacobian,
)
from .bdf_trial_recovery import make_recovering_bdf, merge_trackers

STEPS = ("adsorption", "evacuation", "desorption", "cooling", "repressurization")
PUMPED_PHASES = ("evacuation", "desorption", "cooling")


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


def _pump_rates(c: dict, pressure_Pa: float, temperature_K: float,
                    *, profile=None) -> dict:
    """Total installed bank evaluated once.

    The default fixed-standard-reference interpretation treats the tabulated
    ordinate as volume at the pump profile's reference pressure/temperature.
    The actual-suction-volume alternative is retained only as an explicit
    scenario because the vendor curve does not settle that basis.
    """
    profile = pump_profile_for(c) if profile is None else profile
    if profile is None:
        raise ValueError("A reference pump profile is required for every spatial case.")
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
    pump = _pump_rates(
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
        raise ValueError("coupled_through_thickness section is required for the spatial engine.")
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
        raise HeatSourceContractError(
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
        raise ExpectedScenarioError("Explicit feed-dryer operation is not qualified in the spatial model.")
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
        storage_cp_rows = []
        storage_gradient_rows = []
        species_energy_for_storage_rows = []
        for j in range(model.n_cells):
            partials_Pa = ygas[j] * p[j]
            try:
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
                pump = _pump_rates(c, p_suction, T_mix, profile=profile)
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
        collecting = ((phase == "desorption" and _t >= purge_s)
                      or (phase == "cooling" and product_route == "all_desorption_and_cooling"))
        inlet = F[:, None] * y_feed[None, :] + L[:, None] * y_leak[None, :]
        derivative[lay.inlet] = inlet.sum(axis=0)
        if phase == "adsorption":
            derivative[lay.vent] = (W[:, None] * ygas).sum(axis=0)
            derivative[lay.product] = 0.0
        elif phase in PUMPED_PHASES:
            vapour = 0.5 * (stream.n_vapour_mol_s if stream is not None else w_total_full)
            composition = stream.vapour_composition if stream is not None else y_mix
            derivative[lay.product] = vapour * composition if collecting else 0.0
            derivative[lay.vent] = 0.0 if collecting else vapour * composition
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
                pump_eval = _pump_rates(c, stream.p_pump_Pa, stream.t_out_K, profile=profile)
            else:
                pump_eval = _pump_rates(c, p_suction, T_mix, profile=profile)
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
            derivative[lay.condensate_product] = cond if collecting else 0.0
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
            "Explicit feed-dryer operation is not qualified in the spatial model.")
    if c["conditioning"]["electric_J_per_kg_water_removed"] not in (None, 0.0):
        raise ExpectedScenarioError(
            "Nonzero upstream feed-conditioning energy is not qualified in the spatial model.")
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
    runtime_control = c.get("_runtime_control") or {}
    recovery_trackers = []

    def record_recovery_tracker(tracker) -> None:
        if tracker is None:
            return
        recovery_trackers.append(tracker)
        recorder = getattr(material, "record_bdf_recovery_summary", None)
        if callable(recorder):
            recorder(tracker.as_dict())

    def recovery_method():
        return make_recovering_bdf(material, lay, model=model)

    def jacobian_for(rhs_value):
        return lambda time_value, state_value: bounded_spatial_jacobian(
            rhs_value,
            time_value,
            state_value,
            layout=lay,
            material=material,
            audit=jacobian_audit,
        )

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
                    return ev(chunk_start + local_time, values)[0]

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
                return ev(local_time, values)[0]

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
