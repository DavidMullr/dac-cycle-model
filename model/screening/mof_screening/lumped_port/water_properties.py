"""Shared physical water-vapor diagnostic and conditioner property closure."""
from __future__ import annotations

import math

from ...common import water_saturation_pressure_Pa


def buck_saturation_pressure_Pa(temperature_C: float) -> float:
    """Buck (1996) liquid-water saturation pressure from Celsius input."""
    temperature = float(temperature_C)
    if not math.isfinite(temperature):
        raise ValueError("Buck temperature must be finite")
    return 611.21 * math.exp(
        (18.678 - temperature / 234.5)
        * temperature
        / (257.14 + temperature)
    )


def physical_saturation_pressure_Pa(
    temperature_K: float,
) -> tuple[float, str]:
    """Buck at or below 50 C; Antoine correlation above 50 C."""
    temperature = float(temperature_K)
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    temperature_C = temperature - 273.15
    if temperature_C <= 50.0:
        return (
            buck_saturation_pressure_Pa(temperature_C),
            "Buck 1996",
        )
    return water_saturation_pressure_Pa(temperature), (
        "screening.common NIST/Stull Antoine"
    )
