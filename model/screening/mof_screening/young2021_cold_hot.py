"""Named 20-120 C conditional continuation of unchanged Young 2021 laws.

This provider exists only for the approved cold-transient + hot paired
scenario. The measured coupled support remains 25-70 C. Values below 25 C and
above 70 C are assumed, unvalidated correlation continuations.
"""
from __future__ import annotations

import math

from . import young2021
from . import young2021_hot


EVIDENCE_SUPPORT_TEMPERATURE_K = (298.15, 343.15)
EVALUATION_ENVELOPE_TEMPERATURE_K = (293.15, 393.15)
EVALUATION_ENVELOPE_RELATIVE_HUMIDITY = (0.0, 0.55)
EVALUATION_ENVELOPE_CO2_PRESSURE_PA = (0.0, 100_000.0)
TEMPERATURE_ROUNDOFF_TOLERANCE_K = 1e-9
RELATIVE_HUMIDITY_ROUNDOFF_TOLERANCE = 1e-12
CO2_PRESSURE_ROUNDOFF_TOLERANCE_PA = 1e-9

# This is the audited upper bound hard-coded in the original implementation:
# GAB(0.55, 298.15 K). It is an implementation bound, not a directly measured
# experimental loading limit.
ORIGINAL_IMPLEMENTATION_WATER_CAP_MOL_KG = (
    young2021.gab_water_loading(0.55, 298.15)
)
# Self-consistent maximum of the unchanged GAB formula over the approved
# 20-120 C / RH<=0.55 envelope.
GAB_ENVELOPE_MAX_TEMPERATURE_K = 302.7029459003439
GAB_ENVELOPE_MAX_MOL_KG = 4.135780869459574
WATER_EVALUATION_CAP_MOL_KG = 4.136
WATER_LOADING_EVALUATION_TOLERANCE_MOL_KG = 1e-9


class ColdHotDomainError(ValueError):
    """A named material-domain failure with query details attached."""

    def __init__(self, message: str, *, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def iapws_saturation_pressure_Pa(temperature_K: float) -> float:
    return young2021_hot.iapws_saturation_pressure_Pa(float(temperature_K))


def local_relative_humidity(
    temperature_K: float, water_partial_Pa: float
) -> float:
    return young2021_hot.local_relative_humidity(
        float(temperature_K), float(water_partial_Pa)
    )


def gab_water_loading(
    relative_humidity: float, temperature_K: float
) -> float:
    return young2021.gab_water_loading(
        float(relative_humidity), float(temperature_K)
    )


def validate_evaluation_envelope(
    temperature_K: float,
    co2_Pa: float,
    relative_humidity: float,
) -> None:
    values = (temperature_K, co2_Pa, relative_humidity)
    if not all(math.isfinite(float(value)) for value in values):
        raise ColdHotDomainError(
            "cold+hot extrapolation inputs must be finite",
            diagnostics={
                "temperature_K": float(temperature_K),
                "co2_Pa": float(co2_Pa),
                "relative_humidity": float(relative_humidity),
            },
        )
    lower_K, upper_K = EVALUATION_ENVELOPE_TEMPERATURE_K
    if not (
        lower_K - TEMPERATURE_ROUNDOFF_TOLERANCE_K
        <= float(temperature_K)
        <= upper_K + TEMPERATURE_ROUNDOFF_TOLERANCE_K
    ):
        raise ColdHotDomainError(
            "temperature_K outside the named 20-120 C cold+hot "
            "conditional evaluation envelope",
            diagnostics={
                "temperature_K": float(temperature_K),
                "allowed_temperature_K": [lower_K, upper_K],
                "domain": "temperature",
            },
        )
    lower_co2, upper_co2 = EVALUATION_ENVELOPE_CO2_PRESSURE_PA
    if not (
        lower_co2 - CO2_PRESSURE_ROUNDOFF_TOLERANCE_PA
        <= float(co2_Pa)
        <= upper_co2 + CO2_PRESSURE_ROUNDOFF_TOLERANCE_PA
    ):
        raise ColdHotDomainError(
            "co2_Pa outside the named cold+hot evaluation envelope",
            diagnostics={
                "co2_Pa": float(co2_Pa),
                "allowed_co2_Pa": [lower_co2, upper_co2],
                "domain": "co2_partial_pressure",
            },
        )
    lower_rh, upper_rh = EVALUATION_ENVELOPE_RELATIVE_HUMIDITY
    if not (
        lower_rh
        <= float(relative_humidity)
        <= upper_rh + RELATIVE_HUMIDITY_ROUNDOFF_TOLERANCE
    ):
        raise ColdHotDomainError(
            "relative_humidity outside the named 0-0.55 evaluation envelope",
            diagnostics={
                "relative_humidity": float(relative_humidity),
                "allowed_relative_humidity": [lower_rh, upper_rh],
                "domain": "relative_humidity",
            },
        )


def normalize_water_loading_for_evaluation(
    water_loading_mol_kg: float,
) -> tuple[float, dict]:
    """Return an evaluation-only water loading; never mutate ODE storage."""

    raw = float(water_loading_mol_kg)
    if not math.isfinite(raw):
        raise ColdHotDomainError(
            "water_loading_mol_kg must be finite",
            diagnostics={"raw_water_loading_mol_kg": raw},
        )
    tolerance = WATER_LOADING_EVALUATION_TOLERANCE_MOL_KG
    cap = WATER_EVALUATION_CAP_MOL_KG
    if raw < -tolerance:
        raise ColdHotDomainError(
            "water_loading_mol_kg below cold+hot evaluation tolerance",
            diagnostics={
                "raw_water_loading_mol_kg": raw,
                "lower_tolerance_mol_kg": -tolerance,
                "domain": "water_loading_lower",
            },
        )
    if raw > cap + tolerance:
        raise ColdHotDomainError(
            "water_loading_mol_kg above cold+hot evaluation cap",
            diagnostics={
                "raw_water_loading_mol_kg": raw,
                "evaluation_cap_mol_kg": cap,
                "upper_tolerance_mol_kg": tolerance,
                "domain": "water_loading_upper",
            },
        )
    normalized = min(max(raw, 0.0), cap)
    return normalized, {
        "raw_water_loading_mol_kg": raw,
        "evaluation_water_loading_mol_kg": normalized,
        "lower_normalized": raw < 0.0,
        "upper_normalized": raw > cap,
        "normalization_delta_mol_kg": normalized - raw,
        "evaluation_tolerance_mol_kg": tolerance,
        "stored_inventory_mutated": False,
    }


def _co2_loading_from_water_evaluation(
    temperature_K: float,
    co2_Pa: float,
    water_loading_mol_kg: float,
    *,
    reference_temperature_K: float,
) -> tuple[float, float, dict]:
    normalized, normalization = normalize_water_loading_for_evaluation(
        water_loading_mol_kg
    )
    dry = young2021.toth_loading(
        float(co2_Pa),
        float(temperature_K),
        young2021.YOUNG2021_DRY,
        float(reference_temperature_K),
    )
    wet = young2021.toth_loading(
        float(co2_Pa),
        float(temperature_K),
        young2021.YOUNG2021_WET,
        float(reference_temperature_K),
    )
    weight = young2021.wet_site_weight(normalized)
    return (
        (1.0 - weight) * dry + weight * wet,
        weight,
        normalization,
    )


def equilibrium(
    temperature_K: float,
    co2_Pa: float,
    water_partial_Pa: float,
    *,
    reference_temperature_K: float = 298.15,
    water_loading_mol_kg: float | None = None,
) -> tuple[float, float, dict]:
    local_rh = local_relative_humidity(temperature_K, water_partial_Pa)
    validate_evaluation_envelope(temperature_K, co2_Pa, local_rh)
    gab_target = young2021.gab_water_loading(local_rh, temperature_K)
    raw_water = (
        gab_target
        if water_loading_mol_kg is None
        else float(water_loading_mol_kg)
    )
    co2_loading, wet_weight, normalization = (
        _co2_loading_from_water_evaluation(
            temperature_K,
            co2_Pa,
            raw_water,
            reference_temperature_K=reference_temperature_K,
        )
    )
    return co2_loading, raw_water, {
        "local_relative_humidity": local_rh,
        "water_saturation_pressure_Pa": (
            iapws_saturation_pressure_Pa(temperature_K)
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
        "water_loading_normalization": normalization,
        "cold_hot_conditional": True,
        "measured_cold_validation": False,
        "measured_hot_coupling_validation": False,
        "hot_rate_validated": False,
        "hot_energy_validated": False,
    }


def source_report() -> dict:
    return {
        "classification": (
            "CONDITIONAL_COLD_TRANSIENT_AND_HOT_NUMERICAL_EXTRAPOLATION"
        ),
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
        "cold_extension_assumed": True,
        "cold_extension_measured_validation": False,
        "cold_extension_basis": (
            "saved evacuation RHS physically cooled toward and below 25 C; "
            "the same published correlation is continued to 20 C without "
            "cold data or a Low-to-Young sample transfer"
        ),
        "hot_extension_status": (
            "unvalidated 70-120 C conditional numerical evaluation"
        ),
        "original_implementation_water_cap_mol_kg": (
            ORIGINAL_IMPLEMENTATION_WATER_CAP_MOL_KG
        ),
        "original_cap_distinction": (
            "implementation cap GAB(0.55,25 C), not a directly measured "
            "experimental maximum"
        ),
        "gab_envelope_max_temperature_K": (
            GAB_ENVELOPE_MAX_TEMPERATURE_K
        ),
        "gab_envelope_max_mol_kg": GAB_ENVELOPE_MAX_MOL_KG,
        "water_evaluation_cap_mol_kg": WATER_EVALUATION_CAP_MOL_KG,
        "water_loading_evaluation_tolerance_mol_kg": (
            WATER_LOADING_EVALUATION_TOLERANCE_MOL_KG
        ),
        "normalization_scope": (
            "CO2 target helper evaluation only; raw inventory and energy "
            "remain unchanged"
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
    }
