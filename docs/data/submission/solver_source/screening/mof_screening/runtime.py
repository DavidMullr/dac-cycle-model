"""Shared case translation and versioned spatial execution."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from ..common import MW, ExpectedScenarioError
from ..geometry import build_bare_geometry
from ..systems import build_systems
from ..utilities import build_utilities
from .contract import (
    CONFIG_ROOT,
    ROOT,
    SHARED_ACTUATOR_POLICY,
    MofScreeningError,
    canonical_sha256,
    effective_runtime_policy,
    effective_temperature_policy,
    file_sha256,
    is_humidity_candidate,
    runtime_wall_limit_s,
    temperature_guard_assessment,
)
from .evaluation import annotate_result
from .energy_balance import evaluate_energy_balance
from .materials import MaterialAdapter, build_material_adapter
from .young2021 import buck_saturation_pressure_Pa
from .spatial_jacobian import (
    BRANCH_POLICY_ENV,
    BRANCH_POLICY_V1,
    JACOBIAN_CACHE_POLICY_ENV,
    JACOBIAN_CACHE_POLICY_EXACT,
)
from .spatial_v1 import (
    PUMPED_PHASES,
    STEPS,
    build_model,
    energy_inventory,
    initial_state,
    one_cycle,
    resolved_spatial_solver_policy,
)
from .weather import load_weather_window
from .lumped_port.pressure_control import POLICY_VARIANT
from .lumped_port.pump_conditioner import enable_pump_conditioner
from .lumped_port.solver_start import DEFAULT_DEPLETION_FRACTION
from .lumped_port.weather_driver import run_weather_session


PROFILE_PATH = CONFIG_ROOT / "v21_spatial_profile.json"
KNF_PROFILE = "knf_n922_ste_50hz"
_CASE_RUNTIME_POLICY = object()
NUMERICAL_POLICY_SCHEMA = "mof-screening-numerical-policy/1"


def _resolved_spatial_numerical_policy() -> dict:
    """Resolve and validate the inherited spatial numerical policy."""

    configured = os.environ.get(BRANCH_POLICY_ENV)
    if configured not in {None, BRANCH_POLICY_V1}:
        raise MofScreeningError(
            "unknown spatial numerical policy: "
            f"{configured!r}; expected {BRANCH_POLICY_V1!r}",
            field=f"environment.{BRANCH_POLICY_ENV}",
        )
    branch_aware = configured == BRANCH_POLICY_V1
    configured_cache = os.environ.get(JACOBIAN_CACHE_POLICY_ENV)
    if configured_cache not in {None, JACOBIAN_CACHE_POLICY_EXACT}:
        raise MofScreeningError(
            "unknown Jacobian cache policy: "
            f"{configured_cache!r}; expected "
            f"{JACOBIAN_CACHE_POLICY_EXACT!r}",
            field=f"environment.{JACOBIAN_CACHE_POLICY_ENV}",
        )
    return {
        "schema": NUMERICAL_POLICY_SCHEMA,
        "policy": configured,
        "branch_aware_jacobian": branch_aware,
        "activation_source": (
            "environment" if branch_aware else "default"
        ),
        "environment_variable": BRANCH_POLICY_ENV,
        "jacobian_cache_policy": configured_cache,
        "jacobian_cache_activation_source": (
            "environment"
            if configured_cache == JACOBIAN_CACHE_POLICY_EXACT
            else "default"
        ),
        "jacobian_cache_environment_variable": (
            JACOBIAN_CACHE_POLICY_ENV
        ),
        "validated": True,
    }


def _emit_progress(
    callback, fraction: float, message: str, details: dict | None = None
) -> None:
    """Keep legacy two-argument callbacks usable with the richer contract."""
    if callback is None:
        return
    if details is None:
        callback(fraction, message)
        return
    try:
        callback(fraction, message, details)
    except TypeError as error:
        if "positional argument" not in str(error):
            raise
        callback(fraction, message)


def _candidate_humidity_diagnostic(adapter: MaterialAdapter) -> bool:
    return callable(
        getattr(adapter, "diagnostic_saturation_pressure_Pa", None)
    )


def _relative_humidity_basis(adapter: MaterialAdapter) -> str:
    if _candidate_humidity_diagnostic(adapter):
        return str(adapter.diagnostic_relative_humidity_basis)
    return "p_H2O / Buck_saturation_pressure(T_bed)"


def _saturation_diagnostic_law(adapter: MaterialAdapter) -> str:
    if _candidate_humidity_diagnostic(adapter):
        return str(adapter.diagnostic_saturation_law)
    return "Buck 1996 at T_bed"


def _resolved_wall_limit_s(case: dict, override) -> float | None:
    value = (
        runtime_wall_limit_s(case, legacy_default_s=300.0)
        if override is _CASE_RUNTIME_POLICY
        else override
    )
    if value is not None and float(value) <= 0.0:
        raise MofScreeningError(
            "max_wall_s must be positive when supplied",
            field="case.runtime_policy.time_limit_s",
        )
    return None if value is None else float(value)


def derived_geometry(case: dict) -> dict:
    equipment = case["equipment"]
    volume = (
        float(equipment["bed_face_area_m2"])
        * float(equipment["bed_thickness_m"])
    )
    bulk_density = float(equipment["dry_shaped_mass_kg"]) / volume
    if not 100.0 <= bulk_density <= 2000.0:
        raise MofScreeningError(
            "Dry mass, face area and thickness imply a bulk density outside "
            f"100-2000 kg/m3 (got {bulk_density:g})",
            field="case.equipment",
        )
    void = float(equipment["bed_void_fraction"])
    return {
        "volume_m3": volume,
        "bulk_density_kg_m3": bulk_density,
        "pellet_density_kg_m3": bulk_density / (1.0 - void),
    }


def _validate_weather_for_material(material: dict, weather_rows) -> None:
    if (
        material.get("water", {}).get("model") != "dry_only"
        or is_humidity_candidate(material)
    ):
        return
    if any(float(row.relative_humidity) != 0.0 for row in weather_rows):
        raise MofScreeningError(
            "dry-only material cannot run with nonzero weather relative humidity",
            field="case.weather.relative_humidity",
        )


def temperature_reporting(
    material: dict, case: dict, traces: list[dict], adapter_report: dict
) -> tuple[dict, dict, bool]:
    """Separate sampled accepted states from all material query temperatures."""

    advisory_temperature = material.get("temperature_advisory") or {}
    advisory_mode = (
        advisory_temperature.get("mode") == "record_actual_temperatures"
    )
    minimum_K = min(
        (row["bed_temperature_min_K"] for row in traces), default=None
    )
    maximum_K = max(
        (row["bed_temperature_max_K"] for row in traces), default=None
    )
    accepted = {
        "basis": (
            "sample extrema from retained accepted solver traces; "
            "display sampling is not continuous-time certification"
        ),
        "minimum_bed_temperature_K": minimum_K,
        "maximum_bed_temperature_K": maximum_K,
        "maximum_bed_temperature_C": (
            None if maximum_K is None else maximum_K - 273.15
        ),
        "requested_regeneration_temperature_C": float(
            case["cycle"]["regeneration_temperature_C"]
        ),
        "historical_evaluation_range_K": advisory_temperature.get(
            "historical_evaluation_range_temperature_K"
        ),
        "historical_range_hard_stop": False if advisory_mode else None,
        "temperature_clamped": False if advisory_mode else None,
        "raw_states_retained": True,
    }
    queries = deepcopy(
        adapter_report.get("material_query_temperature_observations")
    )
    return accepted, queries, advisory_mode


def phase_sequence_complete(windows, completion) -> bool:
    """A cycle is complete after all five phase states have been traversed."""
    traversed = tuple(
        row[0] if isinstance(row, (tuple, list)) else row["phase"]
        for row in windows
    )
    return traversed == STEPS and all(
        phase in completion
        and not bool(completion[phase].get("horizon_truncated", False))
        for phase in STEPS
    )


def build_spatial_config(material: dict, case: dict, weather) -> dict:
    if (
        material.get("water", {}).get("model") == "dry_only"
        and not is_humidity_candidate(material)
        and float(weather.relative_humidity) != 0.0
    ):
        raise MofScreeningError(
            "dry-only material cannot run with nonzero weather relative humidity",
            field="case.weather.relative_humidity",
        )
    profile = deepcopy(json.loads(PROFILE_PATH.read_text(encoding="utf-8")))
    profile["material_file"] = (
        f"materials/screening/{material['material_id']}.json"
    )
    geometry = derived_geometry(case)
    equipment = case["equipment"]
    cycle = case["cycle"]
    profile["ambient"] = {
        "temperature_C": float(weather.temperature_C),
        "pressure_Pa": float(weather.pressure_Pa),
        "relative_humidity": float(weather.relative_humidity),
        "co2_ppm_dry": float(weather.co2_ppm_dry),
        "water_mole_fraction": float(weather.water_mole_fraction),
        "water_vapour_partial_pressure_Pa": float(
            weather.prepared_buck_water_vapour_partial_Pa
        ),
        "water_pressure_law": "Buck 1996",
        "local_bed_water_pressure_law": (
            "Antoine <=373 K; IAPWS SR1-86(1992) above 373 K"
            if is_humidity_candidate(material)
            else (
                "IAPWS SR1-86(1992)"
                if material.get("equilibrium_family")
                in {
                    "young2021_hot_wadst_gab_extrapolated",
                    "young2021_cold_hot_wadst_gab_extrapolated",
                }
                else "Buck 1996"
            )
        ),
    }
    profile["bed"] = {
        "dry_sorbent_mass_kg": float(equipment["dry_shaped_mass_kg"]),
        "pellet_density_kg_m3": geometry["pellet_density_kg_m3"],
        "bed_void_fraction": float(equipment["bed_void_fraction"]),
        "particle_diameter_m": float(equipment["particle_diameter_m"]),
    }
    profile["geometry"] = {
        "kind": "cassette",
        "bed_thickness_m": float(equipment["bed_thickness_m"]),
        "metal_thickness_m": 0.0,
        "metal_density_kg_m3": 1.0,
        "metal_cp_J_kg_K": 900.0,
        "frame_mass_kg": float(equipment["wall_heat_capacity_J_K"])
        / 900.0,
        "extra_dead_volume_m3": 0.0005,
    }
    profile["airflow"].update(
        {
            "volume_flow_m3_s": float(equipment["airflow_m3_s"]),
            "fan_efficiency": 0.55,
            "air_viscosity_Pa_s": 0.0000185,
            "minor_loss_coefficient": 2.0,
        }
    )
    profile["thermal"].update(
        {
            "heater_power_W": float(equipment["heater_power_W"]),
            "setpoint_gain_W_K": 1000.0,
            "heater_wall_conductance_W_K": 20.0,
            "heater_heat_capacity_J_K": 120.0,
            "heater_temperature_margin_K": 40.0,
            "bed_control_gain_W_K": 40.0,
            "heater_ceiling_band_K": 5.0,
            "effective_bed_conductivity_W_m_K": float(
                equipment["effective_bed_conductivity_W_m_K"]
            ),
            "wall_contact_h_W_m2_K": 80.0,
            "heat_loss_U_W_m2_K": float(equipment["wall_loss_UA_W_K"])
            / max(2.0 * float(equipment["bed_face_area_m2"]), 1e-9),
            "cooling_h_W_m2_K": float(
                equipment["wall_cooling_UA_W_K"]
            )
            / max(float(equipment["bed_face_area_m2"]), 1e-9),
            "wall_heat_capacity_full_J_K": float(
                equipment["wall_heat_capacity_J_K"]
            ),
            "wall_cooling_UA_full_W_K": float(
                equipment["wall_cooling_UA_W_K"]
            ),
        }
    )
    profile["cycle"].update(
        {
            "adsorption_s": float(cycle["adsorption_s"]),
            "evacuation_s": float(cycle["evacuation_s"]),
            "desorption_s": float(cycle["desorption_s"]),
            "cooling_max_s": float(cycle["cooling_max_s"]),
            "cooling_approach_K": 0.5,
            "repressurization_max_s": float(
                cycle["repressurization_max_s"]
            ),
            "regeneration_temperature_C": float(
                cycle["regeneration_temperature_C"]
            ),
            "regeneration_pressure_Pa": float(
                cycle["regeneration_pressure_Pa"]
            ),
            "purge_s": float(cycle["purge_s"]),
            "minimum_cycles": 1,
            "maximum_cycles": 1,
            "css_relative_tolerance": 0.002,
            "desorption_control": {
                "schema": "desorption-control-v8/1",
                "mode": "fixed_timer",
                "minimum_duration_s": max(
                    float(cycle["desorption_s"]) - 1e-3,
                    1e-3,
                ),
                "checkpoint_s": 30.0,
                "bed_temperature_margin_K": 2.0,
                "tail_window_s": 300.0,
                "tail_rate_limit_mol_s": 0.0001,
                "tail_mass_fraction_of_product": 0.01,
                "require_all_tail_samples_below_rate": True,
            },
        }
    )
    profile["conditioning"] = {
        "electric_J_per_kg_water_removed": 0.0,
        "outlet_relative_humidity": None,
    }
    profile["collection"] = deepcopy(case["collection"])
    profile["pump"] = {
        "profile": KNF_PROFILE,
        "profile_file": "equipment/pump_profiles.json",
        "bank_count": 1,
        "bank_contract": "ideal_parallel_unit_curve",
        "pressure_control_observable": "maximum",
        "throttle_band_Pa": 500.0,
        "air_leak_Pa_m3_s": 1e-5,
    }
    profile["conditioner"] = {
        "cooling_target_C": 55.0,
        "max_cooling_duty_W": 2000.0,
        "cooling_auxiliary_electric_W": 0.0,
        "valve_max_conductance_mol_s_Pa": None,
        "line_molar_conductance_mol_s_Pa": None,
    }
    profile["utilities"] = {
        "wall_cooling": {
            "kind": "constant_cop",
            "cop": 3.0,
            "auxiliary_electric_W": 0.0,
            "max_cooling_capacity_W": 2000.0,
        },
        "prepump_cooling": {
            "kind": "constant_cop",
            "cop": 3.0,
            "auxiliary_electric_W": 0.0,
            "max_cooling_capacity_W": 2000.0,
        },
    }
    profile["coupled_through_thickness"].update(
        {
            "n_cells_half": int(case["solver"]["cells"]),
            "gas_solid_heat_transfer_W_m2_K": float(
                equipment["gas_solid_heat_transfer_W_m2_K"]
            ),
            "gas_solid_specific_area_m2_m3": 2400.0,
            "pump_flow_basis": "reference_fixed_inlet_speed",
            "transport_status": "EXPLICIT_ANALYST_ASSUMPTION",
        }
    )
    profile["solver"] = {
        "rtol": float(case["solver"]["rtol"]),
        "max_step_s": float(case["solver"]["max_step_s"]),
        "samples_per_step": int(case["solver"]["samples_per_phase"]),
    }
    profile["material_maximum_temperature_C"] = float(
        material["maximum_bed_temperature_C"]
    )
    profile["initial_state"] = {
        "basis": (
            "equilibrated"
            if case["initial_state"]["basis"]
            == "equilibrated_with_first_weather"
            else "clean_dry"
        )
    }
    profile["shared_geometry_derivation"] = {
        **geometry,
        "basis": "bed volume = dry mass / bulk density = face area x thickness",
        "pellet_density_basis": "bulk density / (1 - void fraction)",
    }
    profile["shared_equipment"] = {
        "schema": "mof-screening-shared-equipment/1",
        "dry_shaped_mass_kg": float(equipment["dry_shaped_mass_kg"]),
        "airflow_m3_s": float(equipment["airflow_m3_s"]),
        "heater_power_full_W": float(equipment["heater_power_W"]),
        "wall_heat_capacity_full_J_K": float(
            equipment["wall_heat_capacity_J_K"]
        ),
        "wall_loss_UA_full_W_K": float(equipment["wall_loss_UA_W_K"]),
        "wall_cooling_UA_full_W_K": float(
            equipment["wall_cooling_UA_W_K"]
        ),
        "combined_cooling_power_full_W": 2000.0,
        "cooling_cop": 3.0,
        "conditioner_target_C": 55.0,
        "pump_running_input_W": 135.0,
        "spatial_only": {
            "effective_bed_conductivity_W_m_K": float(
                equipment["effective_bed_conductivity_W_m_K"]
            ),
            "gas_solid_heat_transfer_W_m2_K": float(
                equipment["gas_solid_heat_transfer_W_m2_K"]
            ),
        },
        "lumped_applicability": {
            "effective_bed_conductivity_W_m_K": (
                "not applicable to the well-mixed lumped representation"
            ),
            "wall_cooling_UA_full_W_K": (
                "not applicable; the lumped representation uses the reviewed "
                "phase-level cooling request and shared 2 kW capacity"
            ),
            "gas_solid_heat_transfer_W_m2_K": (
                "not applicable to the well-mixed lumped representation"
            ),
        },
    }
    profile["operating_policy"] = deepcopy(SHARED_ACTUATOR_POLICY)
    profile["source_identity"] = {
        "spatial_control": "mof-screening-shared-actuators-v2",
        "spatial_engine": "screening/mof_screening/spatial_v1.py",
        "retained_parent": "screening/coupled_through_thickness_v21.py",
        "narrow_modifications": [
            "generalized site storage vector",
            "site energy and Cp from the shared material adapter",
            "reviewed ideal 55 C pump conditioner with zero pressure drop",
            "shared phase-level actuator availability and targets",
            "shared-case weather and geometry translation",
            "new source identity and result wrapper",
        ],
        "numerical_policy": _resolved_spatial_numerical_policy(),
        "solver_policy": resolved_spatial_solver_policy(profile),
    }
    profile = enable_pump_conditioner(
        profile, target_temperature_C=55.0
    )
    return profile


def source_binding() -> dict:
    """Bind results to executable hashes and their verified frozen closure."""
    from .preflight import execution_source_identity

    return execution_source_identity()


def _electric_terms(config: dict, state: np.ndarray, layout, windows) -> dict:
    utilities = build_utilities(config)
    cooling_duration = sum(
        end - start for phase, start, end in windows if phase == "cooling"
    )
    pumped_duration = sum(
        end - start for phase, start, end in windows if phase in PUMPED_PHASES
    )
    wall_duty = 2.0 * float(state[layout.E_cool])
    prepump_duty = (
        2.0 * float(state[layout.e_cooler]) if layout.extended else 0.0
    )
    utility_block = (
        None
        if utilities is None
        else utilities.cycle_energy(
            wall_cooling_duty_J=wall_duty,
            wall_cooling_on_duration_s=cooling_duration,
            prepump_cooling_duty_J=prepump_duty,
            prepump_cooling_on_duration_s=(
                pumped_duration if prepump_duty > 0.0 else 0.0
            ),
        )
    )
    terms = {
        "heater": 2.0 * float(state[layout.E_heat]),
        "pump": 2.0 * float(state[layout.E_vac]),
        "fan": 2.0 * float(state[layout.E_fan]),
        "legacy_conditioning": 2.0 * float(state[layout.E_cond]),
        "dryer": 0.0,
        "source_auxiliary": (
            2.0 * float(state[layout.e_source_aux])
            if layout.extended
            else 0.0
        ),
        "cooler_auxiliary": (
            2.0 * float(state[layout.e_cooler_aux])
            if layout.extended
            else 0.0
        ),
        "utilities": (
            None if utility_block is None else utility_block["total_electric_J"]
        ),
    }
    complete = all(value is not None for value in terms.values())
    total = (
        sum(float(value) for value in terms.values() if value is not None)
        if complete
        else None
    )
    return {
        "terms_J": terms,
        "total_J": total,
        "complete": complete,
        "utility_block": utility_block,
    }


def mass_closure(
    material: MaterialAdapter,
    model,
    initial: np.ndarray,
    final: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    layout = model.layout
    q0 = initial[layout.q].reshape(model.n_cells, model.n_storage)
    q1 = final[layout.q].reshape(model.n_cells, model.n_storage)
    n0 = initial[layout.n_gas].reshape(model.n_cells, 4)
    n1 = final[layout.n_gas].reshape(model.n_cells, 4)
    storage_delta = np.zeros(4)
    for storage_index, species_index in enumerate(material.species_for_storage):
        storage_delta[int(species_index)] += float(
            np.sum(
                model.cell_mass_kg
                * (q1[:, storage_index] - q0[:, storage_index])
            )
        )
    residual = (
        storage_delta
        + n1.sum(axis=0)
        - n0.sum(axis=0)
        - final[layout.inlet]
        + final[layout.vent]
        + final[layout.product]
    )
    condensate = np.zeros(4)
    if layout.extended:
        condensate[1] = final[layout.condensate]
        residual[1] += condensate[1]
    throughput = (
        final[layout.inlet]
        + final[layout.vent]
        + final[layout.product]
        + condensate
    )
    scale = np.maximum(
        np.maximum(np.abs(throughput), np.abs(final[layout.inlet])),
        1e-9,
    )
    return residual, np.abs(residual) / scale


def _trace_rows(
    model, times, states, phases, windows=(), *, material=None
) -> list[dict]:
    layout = model.layout
    phase_starts = {
        phase_name: float(start_s)
        for phase_name, start_s, _end_s in windows
    }
    rows = []
    for phase in (
        "adsorption",
        "evacuation",
        "desorption",
        "cooling",
        "repressurization",
    ):
        indices = np.flatnonzero(np.asarray(phases) == phase)
        if indices.size == 0:
            continue
        selected = indices[
            np.linspace(
                0,
                indices.size - 1,
                min(25, indices.size),
                dtype=int,
            )
        ]
        for index in selected:
            state = np.asarray(states[index], dtype=float)
            n_gas = state[layout.n_gas].reshape(model.n_cells, 4)
            Tgas = state[layout.T_gas]
            pressures = (
                n_gas.sum(axis=1)
                * 8.314462618
                * Tgas
                / model.cell_gas_volume_m3
            )
            local_water_partial = (
                np.maximum(n_gas[:, 1], 0.0)
                * 8.314462618
                * (
                    Tgas
                    if _candidate_humidity_diagnostic(material)
                    else state[layout.T_bed]
                )
                / model.cell_gas_volume_m3
            )
            local_rh = local_water_partial / np.array(
                [
                    (
                        material.diagnostic_saturation_pressure_Pa(
                            float(value)
                        )
                        if _candidate_humidity_diagnostic(material)
                        else buck_saturation_pressure_Pa(float(value))
                    )
                    for value in state[layout.T_bed]
                ]
            )
            q = state[layout.q].reshape(model.n_cells, model.n_storage)
            co2_loading = (
                float(np.mean(q[:, material.species_for_storage == 0]))
                if material is not None
                else float(np.mean(q[:, 0]))
            )
            water_loading = (
                float(np.mean(q[:, material.water_storage_index]))
                if material is not None
                else float(np.mean(q[:, : min(2, model.n_storage)]))
            )
            rows.append(
                {
                    "elapsed_s": float(times[index]),
                    "phase_elapsed_s": float(
                        times[index] - phase_starts.get(phase, 0.0)
                    ),
                    "phase": phase,
                    "pressure_Pa": float(np.max(pressures)),
                    "min_pressure_Pa": float(np.min(pressures)),
                    "pressure_spread_Pa": float(
                        np.max(pressures) - np.min(pressures)
                    ),
                    "bed_temperature_min_K": float(
                        np.min(state[layout.T_bed])
                    ),
                    "bed_temperature_mean_K": float(
                        np.mean(state[layout.T_bed])
                    ),
                    "bed_temperature_max_K": float(
                        np.max(state[layout.T_bed])
                    ),
                    "gas_temperature_mean_K": float(np.mean(Tgas)),
                    "wall_temperature_K": float(state[layout.T_w]),
                    "minimum_gas_inventory_mol": float(np.min(n_gas)),
                    "minimum_storage_mol_kg": float(
                        np.min(state[layout.q])
                    ),
                    "co2_loading_mean_mol_kg": co2_loading,
                    "water_loading_mean_mol_kg": water_loading,
                    "local_rh_min": float(np.min(local_rh)),
                    "local_rh_mean": float(np.mean(local_rh)),
                    "local_rh_max": float(np.max(local_rh)),
                    "local_relative_humidity_basis": (
                        _relative_humidity_basis(material)
                    ),
                    "saturation_diagnostic_law": (
                        _saturation_diagnostic_law(material)
                    ),
                }
            )
    return rows


def run_spatial(
    material: dict,
    case: dict,
    *,
    cancel_check=None,
    progress=None,
    max_wall_s=_CASE_RUNTIME_POLICY,
    source_identity: dict | None = None,
) -> dict:
    """Run whole spatial cycles through the retained V21 phase engine."""

    import time

    execution_source_identity = deepcopy(
        source_identity or source_binding()
    )
    numerical_policy = _resolved_spatial_numerical_policy()
    started = time.monotonic()
    wall_limit_s = _resolved_wall_limit_s(case, max_wall_s)
    runtime_policy = effective_runtime_policy(case)
    temperature_policy = effective_temperature_policy(case, material)
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    _validate_weather_for_material(material, weather.rows)
    horizon_s = float(case["weather"]["hours"]) * 3600.0
    first_config = build_spatial_config(material, case, weather.rows[0])
    solver_policy = resolved_spatial_solver_policy(first_config)
    geometry = build_bare_geometry(first_config)
    first_systems = build_systems(first_config)
    first_utilities = build_utilities(first_config)
    model = build_model(
        first_config,
        adapter,
        geometry,
        systems=first_systems,
        utilities=first_utilities,
    )
    state = initial_state(first_config, adapter, geometry, model)
    initial_snapshot = state.copy()
    initial_inventory = energy_inventory(
        first_config,
        adapter,
        geometry,
        initial_snapshot,
        model,
        systems=first_systems,
    )
    nominal_cycle_s = sum(
        float(case["cycle"][key])
        for key in (
            "adsorption_s",
            "evacuation_s",
            "desorption_s",
            "cooling_max_s",
            "repressurization_max_s",
        )
    )
    if horizon_s + 1e-9 < nominal_cycle_s:
        raise MofScreeningError(
            "screening horizon is shorter than one configured cycle",
            field="case.weather.hours",
        )
    traces = []
    cycles = []
    elapsed = 0.0
    product_total = np.zeros(4)
    vent_total = np.zeros(4)
    inlet_total = np.zeros(4)
    condensate_total = 0.0
    electric_total_J = 0.0
    electric_components = {
        "heater": 0.0,
        "pump": 0.0,
        "fan": 0.0,
        "conditioning": 0.0,
        "cooling": 0.0,
    }
    electric_complete = True
    max_energy_relative = 0.0
    max_species_relative = 0.0
    partial_final_cycle = False

    def report_engine_progress(payload):
        simulated_s = float(payload.get("elapsed_s", 0.0))
        event = payload.get("event", "integration_step")
        phase = payload.get("phase", "phase")
        cycle = int(payload.get("cycle", len(cycles) + 1))
        message = (
            f"spatial cycle {cycle} {phase} completed"
            if event == "phase_completed"
            else f"spatial cycle {cycle} {phase} at {simulated_s:.0f} s"
        )
        _emit_progress(
            progress,
            min(simulated_s / horizon_s, 0.99),
            message,
            {
                **payload,
                "engine": "spatial",
                "simulated_s": simulated_s,
                "runtime_policy": runtime_policy,
            },
        )

    while elapsed < horizon_s - 1e-9:
        if cancel_check is not None:
            cancel_check()
        if (
            wall_limit_s is not None
            and time.monotonic() - started > wall_limit_s
        ):
            raise MofScreeningError(
                f"screening exceeded the user-selected "
                f"{wall_limit_s:g} s execution time limit",
                status=504,
            )
        weather_row = weather.at(elapsed)
        config = first_config
        config["_runtime_control"] = {
            "cancel_check": cancel_check,
            "deadline_monotonic": (
                None
                if wall_limit_s is None
                else started + wall_limit_s
            ),
            "progress_callback": report_engine_progress,
            "temperature_guard_limit_K": (
                None
                if temperature_policy["enforcement_limit_C"] is None
                else float(temperature_policy["enforcement_limit_C"])
                + 273.15
            ),
            "weather_provider": weather.at,
            "cycle_start_s": elapsed,
            "cycle_number": len(cycles) + 1,
            "phase_offset_s": 0.0,
        }
        systems = build_systems(config)
        utilities = build_utilities(config)
        initial = state.copy()
        try:
            (
                time_axis,
                states,
                phases,
                windows,
                completion,
                audit_t,
                audit_y,
                audit_p,
            ) = one_cycle(
                config,
                adapter,
                geometry,
                initial,
                systems=systems,
                utilities=utilities,
                model=model,
                max_total_duration_s=min(
                    nominal_cycle_s, horizon_s - elapsed
                ),
            )
        except ExpectedScenarioError as error:
            if "declared wall-time cap" in str(error):
                raise MofScreeningError(
                    "screening reached the user-selected execution time limit",
                    status=504,
                ) from error
            raise
        except Exception as error:
            diagnostics = dict(
                getattr(error, "diagnostics", None) or {}
            )
            failure_report = getattr(adapter, "failure_report", None)
            if callable(failure_report):
                full_report = failure_report()
                if isinstance(full_report, dict):
                    diagnostics = {**diagnostics, **full_report}
            raise MofScreeningError(
                f"spatial cycle {len(cycles)+1} failed: {error}",
                status=422,
                diagnostics=diagnostics or None,
            ) from error
        if cancel_check is not None:
            cancel_check()
        final = states[-1].copy()
        cycle_time = float(time_axis[-1])
        cycle_complete = phase_sequence_complete(windows, completion)
        partial_final_cycle = partial_final_cycle or not cycle_complete
        product = final[model.layout.product].copy()
        vent = final[model.layout.vent].copy()
        inlet = final[model.layout.inlet].copy()
        condensate = (
            float(final[model.layout.condensate])
            if model.layout.extended
            else 0.0
        )
        product_total += product
        vent_total += vent
        inlet_total += inlet
        condensate_total += condensate
        electric = _electric_terms(config, final, model.layout, windows)
        if electric["complete"] and electric["total_J"] is not None:
            electric_total_J += float(electric["total_J"])
            terms = electric["terms_J"]
            for key in ("heater", "pump", "fan"):
                electric_components[key] += float(terms[key])
            electric_components["conditioning"] += float(
                terms["legacy_conditioning"]
                + terms["dryer"]
                + terms["source_auxiliary"]
                + terms["cooler_auxiliary"]
            )
            electric_components["cooling"] += float(terms["utilities"])
        else:
            electric_complete = False
        final_inventory = energy_inventory(
            config,
            adapter,
            geometry,
            final,
            model,
            systems=systems,
        )
        cycle_initial_inventory = energy_inventory(
            config,
            adapter,
            geometry,
            initial,
            model,
            systems=systems,
        )
        delivered_source = (
            float(final[model.layout.e_source])
            if model.layout.extended
            else 0.0
        )
        energy_boundary = (
            float(final[model.layout.E_heat])
            + delivered_source
            - float(final[model.layout.E_cool])
            - float(final[model.layout.E_loss])
            + float(final[model.layout.E_netflow])
        )
        energy_residual = (
            final_inventory["total_J"]
            - cycle_initial_inventory["total_J"]
            - energy_boundary
        )
        delta_inventory = (
            final_inventory["total_J"]
            - cycle_initial_inventory["total_J"]
        )
        gross_exchange = (
            abs(float(final[model.layout.E_heat]))
            + abs(delivered_source)
            + abs(float(final[model.layout.E_cool]))
            + abs(float(final[model.layout.E_loss]))
            + abs(float(final[model.layout.E_netflow]))
        )
        energy_balance = evaluate_energy_balance(
            energy_residual,
            delta_inventory,
            energy_boundary,
            gross_exchange,
        )
        energy_relative = energy_balance["gross_relative_residual"]
        _mass, species_relative = mass_closure(
            adapter, model, initial, final
        )
        species_max = float(np.max(species_relative))
        if not energy_balance["gross_pass"]:
            raise MofScreeningError(
                f"spatial cycle {len(cycles)+1} energy closure failed: "
                f"{energy_relative:g} > 1e-6 "
                f"(inventory change {final_inventory['total_J'] - cycle_initial_inventory['total_J']:g} J, "
                f"boundary {energy_boundary:g} J, residual {energy_residual:g} J)",
                status=422,
            )
        if species_max > 1e-5:
            raise MofScreeningError(
                f"spatial cycle {len(cycles)+1} species closure failed: "
                f"{species_max:g} > 1e-5",
                status=422,
            )
        max_energy_relative = max(max_energy_relative, energy_relative)
        max_species_relative = max(max_species_relative, species_max)
        sampled = _trace_rows(
            model,
            audit_t,
            audit_y,
            audit_p,
            windows,
            material=adapter,
        )
        for trace in sampled:
            trace["cycle"] = len(cycles) + 1
            trace["elapsed_s"] += elapsed
        traces.extend(sampled)
        phase_evidence = []
        for phase_name, phase_start, phase_end in windows:
            indices = np.flatnonzero(
                (np.asarray(time_axis) >= phase_start - 1e-9)
                & (np.asarray(time_axis) <= phase_end + 1e-9)
            )
            if indices.size:
                phase_evidence.append(
                    {
                        "phase": phase_name,
                        "start_s": phase_start,
                        "end_s": phase_end,
                        "state_start": np.asarray(
                            states[indices[0]]
                        ).tolist(),
                        "state_end": np.asarray(
                            states[indices[-1]]
                        ).tolist(),
                    }
                )
        cycles.append(
            {
                "cycle": len(cycles) + 1,
                "weather_interval_end_utc": weather_row.interval_end_utc,
                "cycle_time_s": cycle_time,
                "complete": cycle_complete,
                "phase_completion": completion,
                "windows": [
                    {"phase": p, "start_s": s, "end_s": e}
                    for p, s, e in windows
                ],
                "phase_evidence": phase_evidence,
                "product_CO2_mol_half": float(product[0]),
                "product_H2O_mol_half": float(product[1]),
                "condensate_mol_half": condensate,
                "electricity_J": electric["total_J"],
                "state_before": initial.tolist(),
                "state_after": final.tolist(),
                "meter_state_after_full_symmetry": {
                    "inlet_mol": (2.0 * final[model.layout.inlet]).tolist(),
                    "vent_mol": (2.0 * final[model.layout.vent]).tolist(),
                    "product_mol": (2.0 * final[model.layout.product]).tolist(),
                    "condensate_mol": 2.0 * condensate,
                    "heater_J": 2.0 * float(final[model.layout.E_heat]),
                    "fan_J": 2.0 * float(final[model.layout.E_fan]),
                    "pump_J": 2.0 * float(final[model.layout.E_vac]),
                    "cooling_removed_J": 2.0
                    * float(final[model.layout.E_cool]),
                    "ambient_loss_J": 2.0
                    * float(final[model.layout.E_loss]),
                    "net_boundary_enthalpy_J": 2.0
                    * float(final[model.layout.E_netflow]),
                },
                "energy_boundary_half_J": energy_boundary,
                "energy_residual_half_J": energy_residual,
                "energy_relative_residual": energy_relative,
                "energy_balance": energy_balance,
                "species_residual_half_mol": _mass.tolist(),
                "species_relative_residual": species_relative.tolist(),
            }
        )
        state = final
        elapsed += cycle_time
        if progress is not None:
            _emit_progress(
                progress,
                min(elapsed / horizon_s, 1.0),
                f"spatial cycle {len(cycles)} complete",
                {
                    "event": "cycle_completed",
                    "engine": "spatial",
                    "cycle": len(cycles),
                    "phase": None,
                    "elapsed_s": elapsed,
                    "simulated_s": elapsed,
                    "cycle_completed": True,
                    "cycles_completed": sum(
                        1 for cycle in cycles if cycle.get("complete")
                    ),
                    "runtime_policy": runtime_policy,
                    "checkpoint_state": final.tolist(),
                },
            )
    if not cycles:
        raise MofScreeningError(
            "no complete spatial cycle fits inside the requested horizon",
            field="case.weather.hours",
        )
    collected_kg = 2.0 * product_total[0] * MW[0]
    wet_total = 2.0 * float(product_total.sum())
    dry_total = 2.0 * float(
        product_total[0] + product_total[2] + product_total[3]
    )
    completed_cycle_count = sum(
        1 for cycle in cycles if cycle.get("complete")
    )
    desorption_rows = [
        row
        for row in traces
        if row["phase"] == "desorption"
        and int(row["cycle"]) <= completed_cycle_count
    ]
    cooling_rows = [
        row
        for row in traces
        if row["phase"] == "cooling"
        and int(row["cycle"]) <= completed_cycle_count
    ]
    repress_rows = [
        row
        for row in traces
        if row["phase"] == "repressurization"
        and int(row["cycle"]) <= completed_cycle_count
    ]
    cooling_end_weather = weather.at(
        float(cooling_rows[-1]["elapsed_s"])
    )
    repressurization_end_weather = weather.at(
        float(repress_rows[-1]["elapsed_s"])
    )
    target_temperature_K = (
        float(case["cycle"]["regeneration_temperature_C"]) + 273.15
    )
    target_pressure_Pa = float(
        case["cycle"]["regeneration_pressure_Pa"]
    )
    endpoint = {
        "regeneration_temperature_K": (
            desorption_rows[-1]["bed_temperature_min_K"]
            if desorption_rows
            else None
        ),
        "regeneration_temperature_target_K": target_temperature_K,
        "regeneration_temperature_pass": bool(
            desorption_rows
            and desorption_rows[-1]["bed_temperature_min_K"]
            >= target_temperature_K - 2.0
        ),
        "desorption_pressure_Pa": (
            desorption_rows[-1]["min_pressure_Pa"]
            if desorption_rows
            else None
        ),
        "desorption_pressure_target_Pa": target_pressure_Pa,
        "desorption_pressure_pass": bool(
            desorption_rows
            and desorption_rows[-1]["min_pressure_Pa"]
            <= 1.05 * target_pressure_Pa
        ),
        "cooling_temperature_K": (
            cooling_rows[-1]["bed_temperature_max_K"]
            if cooling_rows
            else None
        ),
        "cooling_temperature_target_K": (
            cooling_end_weather.temperature_C + 273.15 + 0.5
        ),
        "cooling_pass": bool(
            cooling_rows
            and cooling_rows[-1]["bed_temperature_max_K"]
            <= cooling_end_weather.temperature_C + 273.15 + 0.5
        ),
        "repressurization_pressure_Pa": (
            repress_rows[-1]["min_pressure_Pa"] if repress_rows else None
        ),
        "repressurization_pass": bool(
            repress_rows
            and repress_rows[-1]["min_pressure_Pa"]
            >= 0.98 * repressurization_end_weather.pressure_Pa
        ),
        "repressurization_pressure_target_Pa": (
            0.98 * repressurization_end_weather.pressure_Pa
        ),
    }
    endpoint["all_pass"] = all(
        endpoint[key]
        for key in (
            "regeneration_temperature_pass",
            "desorption_pressure_pass",
            "cooling_pass",
            "repressurization_pass",
        )
    )
    endpoint["legacy_interpretation"] = (
        "Achieved endpoint fields use the latest completed cycle. Any partial "
        "final-stage samples remain present in cycles, traces and the finite-"
        "horizon record and are reported separately from completed targets."
    )
    peak_temperature_K = max(
        (row["bed_temperature_max_K"] for row in traces), default=None
    )
    adapter_report = adapter.source_report()
    (
        accepted_temperature_report,
        material_query_temperature_report,
        advisory_temperature_mode,
    ) = temperature_reporting(
        material, case, traces, adapter_report
    )
    water_loading_trace = [
        row["water_loading_mean_mol_kg"]
        for row in traces
        if row.get("water_loading_mean_mol_kg") is not None
    ]
    local_rh_trace = [
        value
        for row in traces
        for value in (
            row.get("local_rh_min"),
            row.get("local_rh_mean"),
            row.get("local_rh_max"),
        )
        if value is not None
    ]
    water_burden = {
        "basis": "mol/kg dry material",
        "initial_loading_mol_kg": (
            water_loading_trace[0] if water_loading_trace else None
        ),
        "final_loading_mol_kg": (
            water_loading_trace[-1] if water_loading_trace else None
        ),
        "minimum_loading_mol_kg": (
            min(water_loading_trace) if water_loading_trace else None
        ),
        "maximum_loading_mol_kg": (
            max(water_loading_trace) if water_loading_trace else None
        ),
        "minimum_local_relative_humidity": (
            min(local_rh_trace) if local_rh_trace else None
        ),
        "maximum_local_relative_humidity": (
            max(local_rh_trace) if local_rh_trace else None
        ),
        "local_relative_humidity_basis": _relative_humidity_basis(adapter),
        "saturation_diagnostic_law": _saturation_diagnostic_law(adapter),
    }
    metrics = {
        "collected_CO2_kg_day": collected_kg * 86400.0 / max(elapsed, 1e-9),
        "collected_CO2_kg": collected_kg,
        "electricity_kWh_day": (
            electric_total_J / 3.6e6 * 86400.0 / max(elapsed, 1e-9)
            if electric_complete
            else None
        ),
        "electricity_kWh": (
            electric_total_J / 3.6e6 if electric_complete else None
        ),
        "electricity_kWh_kg_CO2": (
            electric_total_J / 3.6e6 / collected_kg
            if electric_complete and collected_kg > 0.0
            else None
        ),
        "wet_product_CO2_fraction": (
            float(2.0 * product_total[0] / wet_total)
            if wet_total > 0.0
            else None
        ),
        "dry_product_CO2_fraction": (
            float(2.0 * product_total[0] / dry_total)
            if dry_total > 0.0
            else None
        ),
        "water_condensed_mol": 2.0 * condensate_total,
        "water_vented_mol": 2.0 * float(vent_total[1]),
        "water_in_product_mol": 2.0 * float(product_total[1]),
        "water_loading_initial_mol_kg": water_burden[
            "initial_loading_mol_kg"
        ],
        "water_loading_final_mol_kg": water_burden["final_loading_mol_kg"],
        "water_loading_minimum_mol_kg": water_burden[
            "minimum_loading_mol_kg"
        ],
        "water_loading_maximum_mol_kg": water_burden[
            "maximum_loading_mol_kg"
        ],
        "local_relative_humidity_minimum": water_burden[
            "minimum_local_relative_humidity"
        ],
        "local_relative_humidity_maximum": water_burden[
            "maximum_local_relative_humidity"
        ],
        "peak_bed_temperature_K": peak_temperature_K,
        "electricity_components_kWh": {
            key: value / 3.6e6
            for key, value in electric_components.items()
        },
    }
    result = {
        "schema": "mof-screening-result/1",
        "case_label": case["label"],
        "engine": "spatial",
        "engine_identity": (
            "mof-screening-spatial-shared-actuators-target-aware-v3"
        ),
        "numerical_policy": numerical_policy,
        "spatial_solver_policy": solver_policy,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": execution_source_identity,
        "input_identity": {
            "material_sha256": canonical_sha256(material),
            "case_sha256": canonical_sha256(case),
            "combined_sha256": canonical_sha256(
                {"material": material, "case": case}
            ),
        },
        "material": {
            "material_id": material["material_id"],
            "name": material["name"],
            "evidence": material["evidence"],
            "synthetic": material["synthetic"],
            "shaped_material": material["shaped_material"],
            "assumptions": material["assumptions"],
            "validity": material["validity"],
            "uncertainty": material["uncertainty"],
            "provenance": material["provenance"],
            "adapter": adapter_report,
        },
        "material_package": material,
        "case": case,
        "weather": weather.provenance(),
        "operating_policy": deepcopy(case["operating_policy"]),
        "temperature_policy": temperature_policy,
        "temperature_guard_assessment": temperature_guard_assessment(
            material, case, peak_bed_temperature_K=peak_temperature_K
        ),
        "accepted_state_temperature_observations": (
            accepted_temperature_report
        ),
        "material_query_temperature_observations": (
            material_query_temperature_report
        ),
        "execution_runtime_policy": runtime_policy,
        "runtime": {
            "policy": runtime_policy,
            "elapsed_wall_s": float(time.monotonic() - started),
            "terminal_status": "completed",
            "automatic_wall_termination": (
                runtime_policy["mode"] == "time_limit"
            ),
        },
        "shared_equipment": deepcopy(first_config["shared_equipment"]),
        "geometry": {
            **derived_geometry(case),
            "retained_v21_geometry": geometry.as_dict(),
            "representation": (
                "transverse_symmetric_half_bed_parallel_channels"
            ),
            "axial_resolution": "not resolved",
            "manifold_redistribution": "not resolved",
        },
        "state_layout": model.layout.as_dict(),
        "horizon": {
            "requested_s": horizon_s,
            "simulated_s": elapsed,
            "complete": bool(abs(elapsed - horizon_s) <= 1e-6),
            "whole_cycle_basis": False,
            "cycles_started": len(cycles),
            "cycles_completed": completed_cycle_count,
            "partial_final_cycle": partial_final_cycle,
            "initial_state": case["initial_state"]["basis"],
            "cold_start": bool(case["initial_state"]["cold_start"]),
            "css_established": False,
        },
        "metrics": metrics,
        "water_burden": water_burden,
        "endpoint": endpoint,
        "checks": {
            "mass_relative_residual_by_species": [max_species_relative],
            "mass_relative_residual_basis": (
                "maximum_relative_residual_across_species"
            ),
            "mass_closure_pass": max_species_relative <= 1e-5,
            "energy_relative_residual": max_energy_relative,
            "energy_closure_pass": max_energy_relative <= 1e-6,
            "energy_legacy_net_relative_residual": max(
                (
                    cycle["energy_balance"][
                        "legacy_net_relative_residual"
                    ]
                    for cycle in cycles
                ),
                default=0.0,
            ),
            "energy_legacy_net_closure_pass": all(
                cycle["energy_balance"]["legacy_net_pass"]
                for cycle in cycles
            ),
            "energy_normalization": {
                "criterion": "gross_boundary_exchange",
                "relative_tolerance": 1e-6,
                "absolute_floor_J": 1.0,
                "basis": (
                    "max(abs(delta_inventory_J), "
                    "gross_boundary_exchange_J, 1 J)"
                ),
                "legacy_net_basis_retained": True,
            },
            "temperature_limit_pass": (
                None
                if advisory_temperature_mode
                else bool(
                    peak_temperature_K is not None
                    and peak_temperature_K
                    <= float(material["maximum_bed_temperature_C"]) + 273.15
                )
            ),
            "temperature_limit_basis": (
                "recorded scenario excursion; no evidenced physical limit"
                if advisory_temperature_mode
                else "configured material/scenario guard"
            ),
            "finite_observations": True,
        },
        "numerical_qualification": {
            "status": "not_established_for_this_run",
            "family_gate_relative": 0.002,
            "statement": (
                "A separate identical-input mesh and tighter-solver study "
                "is required; this is a screening diagnostic."
            ),
        },
        "initial_material_inventory": {
            "energy_J": initial_inventory["total_J"],
            "state_hash": canonical_sha256(initial_snapshot.tolist()),
        },
        "cycles": cycles,
        "traces": traces,
        "assumptions": {
            "engine": (
                "Versioned extension of retained V21 transverse symmetric "
                "half-bed parallel-channel model with local gas, solid, wall "
                "and heater states and generalized site storage."
            ),
            "hardware": (
                "KNF N922 50 Hz reference curve, 135 W running input, 2 kW "
                "heater, one combined 2 kW cooling budget at COP 3, and an "
                "ideal pump-inlet knockout targeting 55 C."
            ),
            "weather": weather.provenance()["interval_semantics"],
            "humidity": (
                "Humidity is gas forcing; the selected material adapter owns "
                "the water equilibrium law."
            ),
            "transport": (
                "Axial breakthrough and manifold redistribution are not resolved."
            ),
            "pressure_control": (
                "Reviewed V21 throttle controller uses the maximum local "
                "cell pressure and the common regeneration target."
            ),
            "electricity": (
                "Upstream drying and downstream compression are outside this boundary."
            ),
        },
        "failure": None,
    }
    return annotate_result(result)


def build_lumped_config(material: dict, case: dict, weather) -> dict:
    """Translate the shared case to the accepted finite-cycle lumped port."""

    config = build_spatial_config(material, case, weather)
    config["thermal"]["setpoint_gain_W_K"] = 1000.0
    config["utilities"] = {
        "wall_cooling": {
            "kind": "constant_cop",
            "cop": 3.0,
            "auxiliary_electric_W": 0.0,
            "max_cooling_capacity_W": 2000.0,
        }
    }
    config["cycle"]["adsorption_control"] = "fixed_total_moles"
    config["cycle"]["cooling_stop_offset_K"] = 0.5
    config["solver"]["initial_step_policy"] = (
        "positive_gas_depletion_bound_v1"
    )
    config["solver"]["initial_step_max_s"] = 5.0
    config["solver"]["initial_step_depletion_fraction"] = (
        DEFAULT_DEPLETION_FRACTION
    )
    config = enable_pump_conditioner(config, target_temperature_C=55.0)
    config["source_identity"]["lumped_engine"] = (
        "screening/mof_screening/lumped_port/engine.py"
    )
    config["source_identity"]["lumped_origin"] = (
        "Research/Matched_material_weather_2026-11-01/matched_weather/engine.py"
    )
    return config


def run_lumped(
    material: dict,
    case: dict,
    *,
    cancel_check=None,
    progress=None,
    max_wall_s=_CASE_RUNTIME_POLICY,
    checkpoint_path: Path | None = None,
    source_identity: dict | None = None,
) -> dict:
    """Run the reviewed chronological lumped engine to an exact horizon."""

    import time

    execution_source_identity = deepcopy(
        source_identity or source_binding()
    )
    started = time.monotonic()
    wall_limit_s = _resolved_wall_limit_s(case, max_wall_s)
    runtime_policy = effective_runtime_policy(case)
    temperature_policy = effective_temperature_policy(case, material)
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    _validate_weather_for_material(material, weather.rows)
    horizon_s = float(case["weather"]["hours"]) * 3600.0
    base_config = build_lumped_config(material, case, weather.rows[0])
    runtime_control = {
        "cancel_check": cancel_check,
        "deadline_monotonic": (
            None
            if wall_limit_s is None
            else started + wall_limit_s
        ),
        "temperature_guard_limit_K": (
            None
            if temperature_policy["enforcement_limit_C"] is None
            else float(temperature_policy["enforcement_limit_C"])
            + 273.15
        ),
    }
    checkpoint = Path(
        checkpoint_path
        or CONFIG_ROOT / "runtime" / f"screening_{uuid.uuid4().hex}.json"
    )
    source_paths = (
        ROOT / "screening/mof_screening/runtime.py",
        ROOT / "screening/mof_screening/materials.py",
        ROOT / "screening/mof_screening/young2021.py",
        ROOT / "screening/mof_screening/young2021_hot.py",
        ROOT / "screening/mof_screening/young2021_cold_hot.py",
        ROOT / "screening/mof_screening/lumped_port/engine.py",
        ROOT / "screening/mof_screening/lumped_port/weather_driver.py",
        ROOT / "screening/mof_screening/lumped_port/pressure_control.py",
        ROOT / "screening/mof_screening/lumped_port/pump_conditioner.py",
    )
    source_hashes = {
        path.relative_to(ROOT).as_posix(): file_sha256(path)
        for path in source_paths
    }
    job_id = uuid.uuid4().hex
    attempt_id = uuid.uuid4().hex

    def session_progress(payload):
        if cancel_check is not None:
            cancel_check()
        if (
            wall_limit_s is not None
            and time.monotonic() > started + wall_limit_s
        ):
            raise MofScreeningError(
                f"screening exceeded the user-selected "
                f"{wall_limit_s:g} s execution time limit",
                status=504,
            )
        if progress is not None:
            elapsed = float(payload.get("elapsed_s", 0.0))
            _emit_progress(
                progress,
                min(elapsed / horizon_s, 1.0),
                f"lumped {payload.get('phase', 'phase')} at {elapsed:.0f} s",
                {
                    **payload,
                    "engine": "lumped",
                    "simulated_s": elapsed,
                    "runtime_policy": runtime_policy,
                },
            )

    try:
        session = run_weather_session(
            adapter,
            weather_hours=list(weather.rows),
            base_config=base_config,
            job_id=job_id,
            attempt_id=attempt_id,
            restart_of_attempt_id=None,
            checkpoint_path=checkpoint,
            context={
                "source_release_sha256": canonical_sha256(source_hashes),
                "source_hashes": source_hashes,
                "mode": case["weather"]["mode"],
                "start_utc": case["weather"].get("start_utc"),
                "hours": case["weather"]["hours"],
            },
            p_water_mode="buck_explicit",
            rtol=float(case["solver"]["rtol"]),
            max_step_s=float(case["solver"]["max_step_s"]),
            maximum_cycles=None,
            progress_callback=session_progress,
            solver_method="BDF",
            use_bounded_jacobian=(
                adapter.equilibrium_family
                in {
                    "young2021_wadst_gab",
                    "young2021_hot_wadst_gab_extrapolated",
                    "young2021_cold_hot_wadst_gab_extrapolated",
                }
            ),
            control_variant=POLICY_VARIANT,
            pump_conditioner=False,
            runtime_control=runtime_control,
        )
    except MofScreeningError:
        raise
    except Exception as error:
        if "declared wall-time cap" in str(error):
            raise MofScreeningError(
                "screening reached the user-selected execution time limit",
                status=504,
            ) from error
        raise MofScreeningError(
            f"lumped weather session failed: {error}", status=422
        ) from error
    summary = session["summary"]
    elapsed = float(summary["elapsed_s"])
    product = np.asarray(summary["product_mol"], dtype=float)
    collected_kg = float(product[0] * MW[0])
    wet_total = float(product.sum())
    dry_total = float(product[0] + product[2] + product[3])
    phase_records = summary["phase_history"]
    traces = []
    for row in phase_records:
        traces.append(
            {
                "elapsed_s": float(row["end_elapsed_s"]),
                "phase_elapsed_s": float(row["duration_s"]),
                "phase": row["phase"],
                "cycle": int(row["cycle"]),
                "pressure_Pa": float(row["pressure_end_Pa"]),
                "min_pressure_Pa": float(row["pressure_end_Pa"]),
                "pressure_spread_Pa": 0.0,
                "bed_temperature_min_K": float(row["temperature_end_K"]),
                "bed_temperature_mean_K": float(row["temperature_end_K"]),
                "bed_temperature_max_K": float(row["temperature_end_K"]),
                "gas_temperature_mean_K": float(row["temperature_end_K"]),
                "wall_temperature_K": float(row["temperature_end_K"]),
                "co2_loading_mean_mol_kg": float(
                    row.get("co2_loading_end_mol_kg", 0.0)
                ),
                "water_loading_mean_mol_kg": float(
                    row.get("water_loading_end_mol_kg", 0.0)
                ),
                "local_rh_min": float(
                    row.get("local_relative_humidity_end", 0.0)
                ),
                "local_rh_mean": float(
                    row.get("local_relative_humidity_end", 0.0)
                ),
                "local_rh_max": float(
                    row.get("local_relative_humidity_end", 0.0)
                ),
                "local_relative_humidity_basis": row.get(
                    "local_relative_humidity_basis",
                    _relative_humidity_basis(adapter),
                ),
                "saturation_diagnostic_law": row.get(
                    "saturation_diagnostic_law",
                    _saturation_diagnostic_law(adapter),
                ),
            }
        )
    completed_cycles = int(summary["completed_cycles"])
    cycles_started = max(
        (int(row["cycle"]) for row in phase_records), default=0
    )
    partial_final_cycle = bool(
        elapsed < horizon_s - 1e-6 or cycles_started > completed_cycles
    )
    desorption_endpoints = [
        row["endpoint"]
        for row in reversed(phase_records)
        if row["phase"] == "desorption"
        and int(row["cycle"]) <= completed_cycles
    ]
    evacuation_endpoints = [
        row["endpoint"]
        for row in reversed(phase_records)
        if row["phase"] == "evacuation"
        and int(row["cycle"]) <= completed_cycles
    ]
    cooling_endpoints = [
        row["endpoint"]
        for row in reversed(phase_records)
        if row["phase"] == "cooling"
        and int(row["cycle"]) <= completed_cycles
    ]
    repressurization_endpoints = [
        row["endpoint"]
        for row in reversed(phase_records)
        if row["phase"] == "repressurization"
        and int(row["cycle"]) <= completed_cycles
    ]
    desorption_endpoint = (
        desorption_endpoints[0] if desorption_endpoints else {}
    )
    evacuation_endpoint = (
        evacuation_endpoints[0] if evacuation_endpoints else {}
    )
    cooling_endpoint = cooling_endpoints[0] if cooling_endpoints else {}
    repressurization_endpoint = (
        repressurization_endpoints[0] if repressurization_endpoints else {}
    )
    endpoint = {
        "violations": summary["endpoint_violations"],
        "temperature_violation_count": summary[
            "temperature_endpoint_violation_count"
        ],
        "pressure_violation_count": summary[
            "pressure_endpoint_violation_count"
        ],
        "legacy_interpretation": (
            "violations and physical_endpoint_violation_count retain the "
            "raw upstream history, including partial stages. all_pass now "
            "uses requested targets from the latest completed cycle. None of "
            "these endpoint fields is a hard-limit or conservation verdict."
        ),
        "regeneration_temperature_K": desorption_endpoint.get(
            "temperature_end_K"
        ),
        "regeneration_temperature_target_K": desorption_endpoint.get(
            "temperature_target_K"
        ),
        "regeneration_temperature_pass": desorption_endpoint.get(
            "temperature_ok"
        ),
        "desorption_pressure_Pa": evacuation_endpoint.get(
            "pressure_end_Pa"
        ),
        "desorption_pressure_target_Pa": evacuation_endpoint.get(
            "pressure_target_Pa"
        ),
        "desorption_pressure_pass": evacuation_endpoint.get("pressure_ok"),
        "cooling_temperature_K": cooling_endpoint.get("temperature_end_K"),
        "cooling_temperature_target_K": cooling_endpoint.get(
            "cooling_event_target_K",
            cooling_endpoint.get("temperature_target_K"),
        ),
        "cooling_pass": cooling_endpoint.get(
            "cooling_event_ok", cooling_endpoint.get("temperature_ok")
        ),
        "repressurization_pressure_Pa": repressurization_endpoint.get(
            "pressure_end_Pa"
        ),
        "repressurization_pressure_target_Pa": (
            repressurization_endpoint.get("pressure_target_Pa")
        ),
        "repressurization_pass": repressurization_endpoint.get(
            "pressure_ok"
        ),
    }
    endpoint["all_pass"] = all(
        endpoint.get(key) is True
        for key in (
            "regeneration_temperature_pass",
            "desorption_pressure_pass",
            "cooling_pass",
            "repressurization_pass",
        )
    )
    water_loading_trace = [
        row["water_loading_mean_mol_kg"]
        for row in traces
        if row.get("water_loading_mean_mol_kg") is not None
    ]
    local_rh_trace = [
        value
        for row in traces
        for value in (
            row.get("local_rh_min"),
            row.get("local_rh_mean"),
            row.get("local_rh_max"),
        )
        if value is not None
    ]
    water_burden = {
        "basis": "mol/kg dry material",
        "initial_loading_mol_kg": (
            water_loading_trace[0] if water_loading_trace else None
        ),
        "final_loading_mol_kg": (
            water_loading_trace[-1] if water_loading_trace else None
        ),
        "minimum_loading_mol_kg": (
            min(water_loading_trace) if water_loading_trace else None
        ),
        "maximum_loading_mol_kg": (
            max(water_loading_trace) if water_loading_trace else None
        ),
        "minimum_local_relative_humidity": (
            min(local_rh_trace) if local_rh_trace else None
        ),
        "maximum_local_relative_humidity": (
            max(local_rh_trace) if local_rh_trace else None
        ),
        "local_relative_humidity_basis": _relative_humidity_basis(adapter),
        "saturation_diagnostic_law": _saturation_diagnostic_law(adapter),
    }
    metrics = {
        "collected_CO2_kg_day": summary["gross_collected_co2_kg_day"],
        "collected_CO2_kg": collected_kg,
        "electricity_kWh_day": (
            float(summary["electricity_J"])
            * 86400.0
            / 3.6e6
            / max(elapsed, 1.0)
        ),
        "electricity_kWh": float(summary["electricity_J"]) / 3.6e6,
        "electricity_kWh_kg_CO2": (
            float(summary["electricity_J"]) / 3.6e6 / collected_kg
            if collected_kg > 0.0
            else None
        ),
        "wet_product_CO2_fraction": summary["wet_basis_co2_fraction"],
        "dry_product_CO2_fraction": summary["dry_purity"],
        "water_condensed_mol": float(
            np.asarray(summary.get("liquid_drain_mol") or [0, 0, 0, 0])[1]
        ),
        "water_vented_mol": float(
            np.asarray(
                summary.get("conditioned_vent_mol") or summary["vent_mol"]
            )[1]
        ),
        "water_in_product_mol": float(product[1]),
        "water_loading_initial_mol_kg": water_burden[
            "initial_loading_mol_kg"
        ],
        "water_loading_final_mol_kg": water_burden["final_loading_mol_kg"],
        "water_loading_minimum_mol_kg": water_burden[
            "minimum_loading_mol_kg"
        ],
        "water_loading_maximum_mol_kg": water_burden[
            "maximum_loading_mol_kg"
        ],
        "local_relative_humidity_minimum": water_burden[
            "minimum_local_relative_humidity"
        ],
        "local_relative_humidity_maximum": water_burden[
            "maximum_local_relative_humidity"
        ],
        "peak_bed_temperature_K": max(
            (row["temperature_end_K"] for row in phase_records), default=None
        ),
        "electricity_components_kWh": {
            "heater": float(summary["delivered_heat_J"]) / 3.6e6,
            "pump": float(summary["pump_electric_J"]) / 3.6e6,
            "fan": float(summary["fan_electric_J"]) / 3.6e6,
            "cooling": (
                float(summary["cooling_removed_J"])
                + float(summary["conditioner_cooling_removed_J"])
            )
            / 3.6e6
            / 3.0,
        },
    }
    result = {
        "schema": "mof-screening-result/1",
        "case_label": case["label"],
        "engine": "lumped",
        "engine_identity": (
            "mof-screening-lumped-reviewed-port-target-aware-v2"
        ),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": execution_source_identity,
        "input_identity": {
            "material_sha256": canonical_sha256(material),
            "case_sha256": canonical_sha256(case),
            "combined_sha256": canonical_sha256(
                {"material": material, "case": case}
            ),
        },
        "material": {
            "material_id": material["material_id"],
            "name": material["name"],
            "evidence": material["evidence"],
            "synthetic": material["synthetic"],
            "shaped_material": material["shaped_material"],
            "assumptions": material["assumptions"],
            "validity": material["validity"],
            "uncertainty": material["uncertainty"],
            "provenance": material["provenance"],
            "adapter": adapter.source_report(),
        },
        "material_package": material,
        "case": case,
        "weather": weather.provenance(),
        "operating_policy": deepcopy(case["operating_policy"]),
        "temperature_policy": temperature_policy,
        "temperature_guard_assessment": temperature_guard_assessment(
            material,
            case,
            peak_bed_temperature_K=metrics.get("peak_bed_temperature_K"),
        ),
        "execution_runtime_policy": runtime_policy,
        "runtime": {
            "policy": runtime_policy,
            "elapsed_wall_s": float(time.monotonic() - started),
            "terminal_status": "completed",
            "automatic_wall_termination": (
                runtime_policy["mode"] == "time_limit"
            ),
        },
        "shared_equipment": deepcopy(base_config["shared_equipment"]),
        "geometry": {
            **derived_geometry(case),
            "representation": "well_mixed_finite_cycle_lumped",
            "axial_resolution": "not applicable",
            "manifold_redistribution": "not resolved",
        },
        "state_layout": {
            "model": "reviewed_finite_cycle_lumped_port",
            "material_sites": adapter.n_sites,
            "storage_states": list(adapter.storage_labels),
            "state_size": int(summary["state_size"]),
        },
        "horizon": {
            "requested_s": horizon_s,
            "simulated_s": elapsed,
            "complete": bool(abs(elapsed - horizon_s) <= 1e-6),
            "whole_cycle_basis": False,
            "cycles_started": cycles_started,
            "cycles_completed": completed_cycles,
            "partial_final_cycle": partial_final_cycle,
            "initial_state": case["initial_state"]["basis"],
            "cold_start": bool(case["initial_state"]["cold_start"]),
            "css_established": False,
        },
        "metrics": metrics,
        "water_burden": water_burden,
        "endpoint": endpoint,
        "checks": {
            "mass_relative_residual_by_species": [
                float(summary["mass_closure_relative_error"])
            ],
            "mass_relative_residual_basis": (
                "maximum_relative_residual_across_species"
            ),
            "mass_closure_pass": bool(
                float(summary["mass_closure_relative_error"]) <= 1e-5
            ),
            "energy_residual_J": summary["energy_residual_J"],
            "energy_relative_residual": summary[
                "energy_closure_relative_error"
            ],
            "energy_closure_pass": bool(
                float(summary["energy_closure_relative_error"]) <= 1e-6
            ),
            "energy_legacy_net_relative_residual": summary.get(
                "energy_legacy_net_relative_residual"
            ),
            "energy_legacy_net_closure_pass": summary.get(
                "energy_legacy_net_closure_pass"
            ),
            "energy_normalization": {
                "criterion": "gross_boundary_exchange",
                "relative_tolerance": 1e-6,
                "absolute_floor_J": 1.0,
                "basis": summary.get("energy_normalization"),
                "legacy_net_basis_retained": True,
            },
            "temperature_limit_pass": not bool(
                summary.get("flags", {}).get(
                    "temperature_support_excursion", False
                )
            ),
            "finite_observations": True,
        },
        "numerical_qualification": {
            "status": "not_established_for_this_run",
            "family_gate_relative": 0.002,
            "statement": (
                "The reviewed lumped engine is its own representation; no "
                "spatial equivalence or CSS qualification is inferred."
            ),
        },
        "raw_model_evidence": {
            "checkpoint_path": session["checkpoint_path"],
            "checkpoint_sha256": session["checkpoint_sha256"],
            "summary": summary,
        },
        "cycles": [
            {
                "cycle": int(row["cycle"]),
                "phase": row["phase"],
                "start_elapsed_s": row["start_elapsed_s"],
                "end_elapsed_s": row["end_elapsed_s"],
                "duration_s": row["duration_s"],
                "endpoint": row["endpoint"],
                "energy_closure_relative_error": row[
                    "energy_closure_relative_error"
                ],
            }
            for row in phase_records
        ],
        "traces": traces,
        "assumptions": {
            "engine": (
                "Portable attribution-preserving port of the accepted "
                "matched_weather finite-cycle lumped engine with hourly "
                "forcing segmented inside phases."
            ),
            "hardware": (
                "KNF N922 50 Hz reference curve, 135 W running input, 2 kW "
                "heater, combined 2 kW cooling at COP 3 and ideal 55 C "
                "pump-inlet knockout."
            ),
            "weather": weather.provenance()["interval_semantics"],
            "transport": "Well-mixed gas and solid state; no spatial gradient.",
            "pressure_control": (
                "Accepted pressure-regulated target-inlet control with "
                "adaptive cooling to ambient plus 0.5 K."
            ),
            "collection": (
                "All desorption and cooling product after zero initial purge."
            ),
            "electricity": (
                "Upstream drying and downstream compression are outside this boundary."
            ),
        },
        "failure": (
            None
            if summary.get("conservation_pass")
            else "closure_or_endpoint_failure"
        ),
    }
    return annotate_result(result)
