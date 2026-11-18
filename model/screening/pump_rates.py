"""Pump flow and power at the declared inlet state."""

from __future__ import annotations
import numpy as np
from . import pump_profile
from .common import R

REFERENCE_FIXED_MOLAR = "reference_fixed_molar"

REFERENCE_FIXED_INLET_SPEED = "reference_fixed_inlet_speed"

ACTUAL_SUCTION_STRESS_TEST = "actual_suction_stress_test"

MODES = (REFERENCE_FIXED_MOLAR, REFERENCE_FIXED_INLET_SPEED, ACTUAL_SUCTION_STRESS_TEST)

MODEL_EQUATIONS = {
    REFERENCE_FIXED_MOLAR:
        "n_dot = b Q(P) P_r / (R T_r); fixed-reference molar-flow assumption",
    REFERENCE_FIXED_INLET_SPEED:
        "S_ref(P) = b Q(P) P_r / P; n_dot = S_ref(P) P / (R T_i); "
        "fixed-speed temperature-transfer assumption",
    ACTUAL_SUCTION_STRESS_TEST:
        "n_dot = b Q(P) P / (R T_i); weaker-supported actual-suction-volume ordinate stress test",
}

def _finite(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value}.")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative; got {value}.")
    return value

def _bank(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("bank_count must be a positive integer.")
    return value

def _mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Unknown vacuum rate model {mode!r}; choose from {MODES}.")
    return mode

def rate_at_pressure(profile: pump_profile.PumpProfile, pressure_Pa: float,
                     inlet_temperature_K: float, *, bank_count: int = 1,
                     mode: str = REFERENCE_FIXED_MOLAR) -> dict:
    """Evaluate one of the three authorised molar-rate models at a pump inlet state.

    ``Q(P)`` is the existing digitized curve.  The profile's own reference pressure and
    temperature are used; no 101325 Pa assumption is embedded in this module.  The return
    payload separates the plotted/reference-condition flow, pressure-normalised speed and
    actual inlet speed so the latter cannot be silently reported as a reference speed.
    """
    mode = _mode(mode)
    pressure = _finite(pressure_Pa, "pressure_Pa", nonnegative=True)
    temperature = _finite(inlet_temperature_K, "inlet_temperature_K", positive=True)
    bank = _bank(bank_count)
    reference_pressure = _finite(profile.reference_pressure_Pa, "profile.reference_pressure_Pa", positive=True)
    reference_temperature = _finite(profile.reference_temperature_K, "profile.reference_temperature_K", positive=True)

    plotted_flow = float(profile.reference_flow_m3_s_at(pressure)) * bank
    if pressure <= profile.ultimate_pressure_Pa:
        pressure_regime = "at_or_below_ultimate"
    elif profile.curve_domain_exceeded(pressure):
        pressure_regime = "above_map_top_held"
    else:
        pressure_regime = "inside_curve"

    pressure_normalised_speed = (
        plotted_flow * reference_pressure / pressure if pressure > 0.0 and plotted_flow > 0.0 else 0.0
    )
    if mode == REFERENCE_FIXED_MOLAR:
        molar_flow = plotted_flow * reference_pressure / (R * reference_temperature)
        actual_volume_interpretation = "reference_condition_Q_at_profile_P_ref_T_ref"
    elif mode == REFERENCE_FIXED_INLET_SPEED:
        molar_flow = pressure_normalised_speed * pressure / (R * temperature)
        actual_volume_interpretation = "reference_condition_Q_transferred_at_fixed_inlet_speed"
    else:
        molar_flow = plotted_flow * pressure / (R * temperature)
        actual_volume_interpretation = "plotted_Q_interpreted_at_actual_P_and_T"

    actual_inlet_speed = molar_flow * R * temperature / pressure if pressure > 0.0 else 0.0
    equivalent_reference_condition_flow = (
        molar_flow * R * reference_temperature / reference_pressure
    )
    return {
        "mode": mode,
        "equation": MODEL_EQUATIONS[mode],
        "pressure_Pa": pressure,
        "inlet_temperature_K": temperature,
        "bank_count": bank,
        "profile_name": profile.name,
        "profile_reference_pressure_Pa": reference_pressure,
        "profile_reference_temperature_K": reference_temperature,
        "plotted_curve_volumetric_flow_m3_s": plotted_flow,
        "equivalent_reference_condition_volumetric_flow_m3_s":
            float(equivalent_reference_condition_flow),
        "plotted_reference_condition_flow_m3_s":
            float(plotted_flow) if mode != ACTUAL_SUCTION_STRESS_TEST else None,
        "pressure_normalised_reference_flow_speed_m3_s": pressure_normalised_speed,
        "actual_inlet_volumetric_speed_m3_s": float(actual_inlet_speed),
        "molar_flow_mol_s": float(molar_flow),
        "flow_interpretation": actual_volume_interpretation,
        "pressure_regime": pressure_regime,
        "upper_domain_exceeded": bool(profile.curve_domain_exceeded(pressure)),
        "below_or_at_ultimate": bool(pressure <= profile.ultimate_pressure_Pa),
        "electric_power_W": float(profile.electric_power_W(pressure)) * bank,
        "electric_basis": profile.electrical_basis,
        "media_temperature_overrun_K": float(profile.media_temperature_overrun_K(temperature)),
    }
