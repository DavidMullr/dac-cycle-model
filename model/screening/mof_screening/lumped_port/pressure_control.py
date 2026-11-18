"""Opt-in pressure-regulated adsorption and adaptive-cooling policy."""
from __future__ import annotations

from copy import deepcopy


FIXED_CONTROL = "fixed_total_moles"
PRESSURE_CONTROL = "pressure_regulated_target_inlet"
POLICY_VARIANT = "pressure_regulated_adaptive_cooling_v1"
POLICY_SCHEMA = "matched-material-pressure-control/1"
PRESSURE_RESPONSE_TAU_S = 0.5
COOLING_STOP_OFFSET_K = 0.5
COOLING_MAX_S = 7200.0


def apply_control_variant(config: dict, variant: str | None) -> dict:
    """Return a config with one explicitly selected control policy."""
    if variant in (None, FIXED_CONTROL):
        return deepcopy(config)
    if variant != POLICY_VARIANT:
        raise ValueError(f"unknown pressure-control variant: {variant}")
    result = deepcopy(config)
    result.setdefault("cycle", {}).update(
        {
            "adsorption_control": PRESSURE_CONTROL,
            "pressure_response_tau_s": PRESSURE_RESPONSE_TAU_S,
            "pressure_target": "current_inlet_pressure",
            "cooling_stop_offset_K": COOLING_STOP_OFFSET_K,
        }
    )
    result["cycle"].setdefault("cooling_max_s", COOLING_MAX_S)
    return result


def control_identity(config: dict) -> dict:
    """Checkpoint identity for the RHS and cooling policy."""
    cycle = config["cycle"]
    return {
        "schema": POLICY_SCHEMA,
        "adsorption_control": cycle.get("adsorption_control", FIXED_CONTROL),
        "pressure_response_tau_s": cycle.get("pressure_response_tau_s"),
        "pressure_target": cycle.get("pressure_target"),
        "cooling_stop_offset_K": cycle.get("cooling_stop_offset_K"),
        "cooling_max_s": float(cycle["cooling_max_s"]),
    }


def cycle_duration_accounting(
    phase_history: list[dict],
    completed_cycles: int,
    elapsed_s: float,
) -> dict:
    """Separate completed-cycle durations from an unfinished final tail."""
    durations = []
    for cycle_index in range(int(completed_cycles)):
        cycle_rows = phase_history[cycle_index * 5 : (cycle_index + 1) * 5]
        if len(cycle_rows) != 5:
            raise ValueError("incomplete phase-history cycle")
        durations.append(
            float(cycle_rows[-1]["end_elapsed_s"])
            - float(cycle_rows[0]["start_elapsed_s"])
        )
    completed_end_s = (
        float(phase_history[int(completed_cycles) * 5 - 1]["end_elapsed_s"])
        if completed_cycles
        else 0.0
    )
    return {
        "completed_cycle_durations_s": durations,
        "mean_completed_cycle_duration_s": (
            float(sum(durations) / len(durations)) if durations else None
        ),
        "unfinished_tail_s": float(elapsed_s) - completed_end_s,
        "completed_span_s": completed_end_s,
    }


def policy_document() -> dict:
    return {
        "schema": POLICY_SCHEMA,
        "variant": POLICY_VARIANT,
        "adsorption": {
            "mode": PRESSURE_CONTROL,
            "pressure_target": "current_inlet_pressure",
            "pressure_response_tau_s": PRESSURE_RESPONSE_TAU_S,
            "outlet_flow_rule": (
                "F_out=F0+(n/T)*Tdot; negative required flow is rejected"
            ),
            "energy_capacity": "Ca+n*R",
        },
        "cooling": {
            "stop_target": "current_inlet_temperature+0.5 K",
            "maximum_duration_s": COOLING_MAX_S,
            "legacy_endpoint_tolerance_K": 1.0,
        },
        "preserved_limits": {
            "heater_power_W": 2000.0,
            "cooling_power_W": 2000.0,
            "wall_cooling_cop": 3.0,
            "pump_ultimate_pressure_Pa": 11000.0,
            "pump_media_temperature_limit_C": 60.0,
            "regeneration_temperature_C": 99.85,
            "regeneration_temperature_K": 373.0,
            "regeneration_pressure_Pa": 20000.0,
            "adsorption_s": 1800.0,
            "desorption_s": 5400.0,
        },
        "claim_limit": (
            "conditional control-model screen; no hardware, material, annual, "
            "wet-mixture, or product-drying validation"
        ),
    }
