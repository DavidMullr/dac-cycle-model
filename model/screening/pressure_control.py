"""Pressure observable used by the global suction-valve controller.

The controller selects minimum or maximum local pressure. Pump capacity is
evaluated once for the installed bank at its actual conditioned inlet state."""
from __future__ import annotations

import numpy as np

DEFAULT_PRESSURE_CONTROL_OBSERVABLE = "minimum"
PRESSURE_CONTROL_OBSERVABLES = ("minimum", "maximum")
CAPACITY_PRESSURE_BASIS = "minimum_local_suction_then_conditioned_pump_inlet"


def pressure_control_observable(config: dict) -> str:
    value = (config.get("pump") or {}).get(
        "pressure_control_observable", DEFAULT_PRESSURE_CONTROL_OBSERVABLE)
    if value not in PRESSURE_CONTROL_OBSERVABLES:
        raise ValueError(
            "pump.pressure_control_observable must be one of "
            f"{PRESSURE_CONTROL_OBSERVABLES}; got {value!r}")
    return value


def pressure_control_pressure(cell_pressures_Pa, observable: str) -> float:
    if observable not in PRESSURE_CONTROL_OBSERVABLES:
        raise ValueError(
            "pressure-control observable must be one of "
            f"{PRESSURE_CONTROL_OBSERVABLES}; got {observable!r}")
    values = np.asarray(cell_pressures_Pa, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("pressure-control inputs must be a non-empty finite array")
    return float(np.max(values) if observable == "maximum" else np.min(values))
