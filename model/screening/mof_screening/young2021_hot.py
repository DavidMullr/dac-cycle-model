"""Explicit 25-120 C extrapolation of the unchanged Young 2021 coefficients.

This module is separate from the strict Young 2021 package. It changes only
the saturation-pressure correlation to IAPWS and omits the 70 C validation
guard. Every equilibrium coefficient, LDF rate and process heat remains the
one in the original package. Hot binary, rate and energy behavior are not
experimentally validated.
"""
from __future__ import annotations

import math

from . import young2021


EVIDENCE_SUPPORT_TEMPERATURE_K = (298.15, 343.15)
EVALUATION_ENVELOPE_TEMPERATURE_K = (298.15, 393.15)
EVALUATION_ENVELOPE_RELATIVE_HUMIDITY = (0.0, 0.55)
EVALUATION_ENVELOPE_CO2_PRESSURE_PA = (0.0, 100_000.0)
APPROVED_LDF_RATES_S_INV = (0.003, 0.0086)
APPROVED_PROCESS_HEATS_J_MOL = (70_000.0, 46_000.0)

WATER_CRITICAL_TEMPERATURE_K = 647.096
WATER_CRITICAL_PRESSURE_PA = 22.064e6


def iapws_saturation_pressure_Pa(temperature_K: float) -> float:
    """Return the IAPWS SR1-86(1992) liquid-vapour saturation pressure."""

    temperature_K = float(temperature_K)
    if not math.isfinite(temperature_K):
        raise ValueError("temperature_K must be finite")
    if not 273.16 <= temperature_K < WATER_CRITICAL_TEMPERATURE_K:
        raise ValueError(
            "temperature_K outside IAPWS saturation support "
            "[273.16, 647.096)"
        )
    theta = 1.0 - temperature_K / WATER_CRITICAL_TEMPERATURE_K
    coefficients = (
        -7.85951783,
        1.84408259,
        -11.7866497,
        22.6807411,
        -15.9618719,
        1.80122502,
    )
    powers = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
    exponent = (
        WATER_CRITICAL_TEMPERATURE_K
        / temperature_K
        * sum(
            coefficient * theta**power
            for coefficient, power in zip(coefficients, powers)
        )
    )
    pressure_Pa = WATER_CRITICAL_PRESSURE_PA * math.exp(exponent)
    if not math.isfinite(pressure_Pa) or pressure_Pa <= 0.0:
        raise ValueError("IAPWS saturation pressure is not finite and positive")
    return pressure_Pa


def local_relative_humidity(
    temperature_K: float, water_partial_Pa: float
) -> float:
    water_partial_Pa = float(water_partial_Pa)
    if not math.isfinite(water_partial_Pa) or water_partial_Pa < 0.0:
        raise ValueError("water_partial_Pa must be finite and nonnegative")
    rh = water_partial_Pa / iapws_saturation_pressure_Pa(float(temperature_K))
    if not math.isfinite(rh) or not 0.0 <= rh < 1.0:
        raise ValueError(
            "water_partial_Pa / saturation_pressure must be finite in [0, 1)"
        )
    return rh


def validate_evaluation_envelope(
    temperature_K: float, co2_Pa: float, rh: float
) -> None:
    values = (temperature_K, co2_Pa, rh)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("hot extrapolation inputs must be finite")
    if not (
        EVALUATION_ENVELOPE_TEMPERATURE_K[0] - 1e-9
        <= float(temperature_K)
        <= EVALUATION_ENVELOPE_TEMPERATURE_K[1] + 1e-9
    ):
        raise ValueError(
            "temperature_K outside the explicit 25-120 C evaluation envelope"
        )
    if not 0.0 <= float(co2_Pa) <= 100_000.0 + 1e-9:
        raise ValueError("co2_Pa outside the explicit evaluation envelope")
    if not 0.0 <= float(rh) <= 0.55 + 1e-12:
        raise ValueError(
            "relative_humidity outside the explicit 0-0.55 evaluation envelope"
        )


def gab_water_loading(relative_humidity: float, temperature_K: float) -> float:
    return young2021.gab_water_loading(
        float(relative_humidity), float(temperature_K)
    )


def equilibrium(
    temperature_K: float,
    co2_Pa: float,
    water_partial_Pa: float,
    *,
    reference_temperature_K: float = 298.15,
    water_loading_mol_kg: float | None = None,
) -> tuple[float, float, dict]:
    """Evaluate conditional hot WADST equilibrium from explicit H2O partial."""

    local_rh = local_relative_humidity(temperature_K, water_partial_Pa)
    validate_evaluation_envelope(temperature_K, co2_Pa, local_rh)
    gab_target = gab_water_loading(local_rh, temperature_K)
    water_loading = (
        gab_target
        if water_loading_mol_kg is None
        else float(water_loading_mol_kg)
    )
    co2_loading, wet_weight = young2021.co2_loading_from_water(
        float(temperature_K),
        float(co2_Pa),
        water_loading,
        reference_temperature_K=reference_temperature_K,
    )
    return co2_loading, water_loading, {
        "local_relative_humidity": local_rh,
        "water_saturation_pressure_Pa": iapws_saturation_pressure_Pa(
            float(temperature_K)
        ),
        "wet_site_weight": wet_weight,
        "gab_water_target_mol_kg": gab_target,
        "water_loading_source": (
            "GAB target"
            if water_loading_mol_kg is None
            else "explicit evolving inventory"
        ),
        "evidence_support_temperature_K": list(
            EVIDENCE_SUPPORT_TEMPERATURE_K
        ),
        "evaluation_envelope_temperature_K": list(
            EVALUATION_ENVELOPE_TEMPERATURE_K
        ),
        "hot_binary_validated": False,
        "hot_rate_validated": False,
        "hot_energy_validated": False,
    }


def source_report() -> dict:
    return {
        "classification": "CONDITIONAL_HOT_NUMERICAL_EXTRAPOLATION",
        "source_doi": "10.1039/D1EE01272J",
        "measured_coupled_support_temperature_K": list(
            EVIDENCE_SUPPORT_TEMPERATURE_K
        ),
        "evaluation_envelope_temperature_K": list(
            EVALUATION_ENVELOPE_TEMPERATURE_K
        ),
        "evaluation_envelope_relative_humidity": list(
            EVALUATION_ENVELOPE_RELATIVE_HUMIDITY
        ),
        "temperature_evidence_status": (
            "measured coupled behavior supported through 343.15 K; "
            "343.15-393.15 K is explicitly unvalidated numerical evaluation"
        ),
        "saturation_pressure_law": "IAPWS SR1-86(1992)",
        "young_toth_coefficients_unchanged": True,
        "young_gab_coefficients_unchanged": True,
        "stored_water_to_co2_coupling": True,
        "co2_to_water_coupling": False,
        "hot_binary_validated": False,
        "hot_rate_validated": False,
        "hot_energy_validated": False,
        "hot_accuracy_bound_available": False,
        "wilkins_110C_constraint_mol_kg": {
            "p_H2O_range_Pa": [15_700.0, 92_700.0],
            "status": "diagnostic constraint, not spliced into Young specimen",
        },
    }
