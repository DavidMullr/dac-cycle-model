"""Temperature-advisory evaluation of unchanged Young 2021 material laws.

This provider records actual temperature excursions and exposes them as
unvalidated extrapolation. It has no discretionary temperature ceiling. Only
genuine constitutive-input and equation-domain failures are rejected.
"""
from __future__ import annotations

import math

from . import young2021
from . import young2021_hot


MEASURED_COUPLED_SUPPORT_TEMPERATURE_K = (298.15, 343.15)
HISTORICAL_EVALUATION_RANGE_TEMPERATURE_K = (293.15, 393.15)
REQUESTED_REGENERATION_TEMPERATURE_C = 116.0
IAPWS_TEMPERATURE_SUPPORT_K = (
    273.16,
    young2021_hot.WATER_CRITICAL_TEMPERATURE_K,
)
WATER_LOADING_NEGATIVE_NUMERICAL_TOLERANCE_MOL_KG = 1e-9


class AdvisoryMaterialDomainError(ValueError):
    """A precise constitutive or equation-domain failure."""

    def __init__(self, message: str, *, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def iapws_saturation_pressure_Pa(temperature_K: float) -> float:
    return young2021_hot.iapws_saturation_pressure_Pa(
        float(temperature_K)
    )


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


def _finite_nonnegative(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AdvisoryMaterialDomainError(
            f"{field} must be finite",
            diagnostics={field: number, "domain": f"{field}_finite"},
        )
    if number < 0.0:
        raise AdvisoryMaterialDomainError(
            f"{field} must be nonnegative",
            diagnostics={field: number, "domain": f"{field}_nonnegative"},
        )
    return number


def validate_equation_inputs(
    temperature_K: float,
    co2_Pa: float,
    water_partial_Pa: float,
    water_loading_mol_kg: float,
) -> tuple[float, float, float, float]:
    """Validate only equations and branches, never an advisory temperature."""

    temperature = float(temperature_K)
    if not math.isfinite(temperature):
        raise AdvisoryMaterialDomainError(
            "temperature_K must be finite",
            diagnostics={"temperature_K": temperature, "domain": "temperature_finite"},
        )
    lower, upper = IAPWS_TEMPERATURE_SUPPORT_K
    if not lower <= temperature < upper:
        raise AdvisoryMaterialDomainError(
            "temperature_K outside the genuine IAPWS saturation equation domain",
            diagnostics={
                "temperature_K": temperature,
                "allowed_temperature_K": [lower, upper],
                "domain": "iapws_saturation_temperature",
            },
        )
    co2 = _finite_nonnegative(co2_Pa, "co2_Pa")
    water_partial = _finite_nonnegative(
        water_partial_Pa, "water_partial_Pa"
    )
    raw_water = float(water_loading_mol_kg)
    if not math.isfinite(raw_water):
        raise AdvisoryMaterialDomainError(
            "water_loading_mol_kg must be finite",
            diagnostics={
                "raw_water_loading_mol_kg": raw_water,
                "domain": "water_loading_finite",
            },
        )
    tolerance = WATER_LOADING_NEGATIVE_NUMERICAL_TOLERANCE_MOL_KG
    if raw_water < -tolerance:
        raise AdvisoryMaterialDomainError(
            "water_loading_mol_kg is below the numerical nonnegative branch",
            diagnostics={
                "raw_water_loading_mol_kg": raw_water,
                "lower_tolerance_mol_kg": -tolerance,
                "domain": "water_loading_nonnegative",
            },
        )
    return temperature, co2, water_partial, raw_water


def normalize_water_loading_for_evaluation(
    water_loading_mol_kg: float,
) -> tuple[float, dict]:
    """Clamp only a sub-1e-9 numerical residual; raw inventory is unchanged."""

    raw = float(water_loading_mol_kg)
    if not math.isfinite(raw):
        raise AdvisoryMaterialDomainError(
            "water_loading_mol_kg must be finite",
            diagnostics={"raw_water_loading_mol_kg": raw},
        )
    tolerance = WATER_LOADING_NEGATIVE_NUMERICAL_TOLERANCE_MOL_KG
    if raw < -tolerance:
        raise AdvisoryMaterialDomainError(
            "water_loading_mol_kg is below the numerical nonnegative branch",
            diagnostics={
                "raw_water_loading_mol_kg": raw,
                "lower_tolerance_mol_kg": -tolerance,
                "domain": "water_loading_nonnegative",
            },
        )
    evaluated = max(raw, 0.0)
    return evaluated, {
        "raw_water_loading_mol_kg": raw,
        "evaluation_water_loading_mol_kg": evaluated,
        "lower_normalized": raw < 0.0,
        "normalization_delta_mol_kg": evaluated - raw,
        "evaluation_tolerance_mol_kg": tolerance,
        "stored_inventory_mutated": False,
    }


def equilibrium(
    temperature_K: float,
    co2_Pa: float,
    water_partial_Pa: float,
    *,
    reference_temperature_K: float = 298.15,
    water_loading_mol_kg: float | None = None,
) -> tuple[float, float, dict]:
    """Evaluate unchanged Young equations with no advisory temperature guard."""

    temperature, co2_partial, water_partial, checked_water = (
        validate_equation_inputs(
            temperature_K,
            co2_Pa,
            water_partial_Pa,
            (
                0.0
                if water_loading_mol_kg is None
                else float(water_loading_mol_kg)
            ),
        )
    )
    local_rh = local_relative_humidity(
        temperature, water_partial
    )
    gab_target = young2021.gab_water_loading(local_rh, temperature)
    target_water = (
        gab_target if water_loading_mol_kg is None else checked_water
    )
    normalized_water, normalization = (
        normalize_water_loading_for_evaluation(target_water)
    )
    dry = young2021.toth_loading(
        co2_partial,
        temperature,
        young2021.YOUNG2021_DRY,
        float(reference_temperature_K),
    )
    wet = young2021.toth_loading(
        co2_partial,
        temperature,
        young2021.YOUNG2021_WET,
        float(reference_temperature_K),
    )
    weight = young2021.wet_site_weight(normalized_water)
    co2 = (1.0 - weight) * dry + weight * wet
    water = float(target_water) if water_loading_mol_kg is not None else gab_target
    outputs = (float(co2), float(water), float(gab_target), float(weight))
    if not all(math.isfinite(value) for value in outputs):
        raise AdvisoryMaterialDomainError(
            "Young2021 advisory equilibrium produced a nonfinite result",
            diagnostics={
                "temperature_K": temperature,
                "co2_Pa": co2_partial,
                "water_partial_Pa": water_partial,
                "raw_water_loading_mol_kg": target_water,
                "outputs": list(outputs),
                "domain": "young2021_equilibrium_finite",
            },
        )
    return float(co2), float(water), {
        "local_relative_humidity": local_rh,
        "water_saturation_pressure_Pa": (
            iapws_saturation_pressure_Pa(temperature)
        ),
        "wet_site_weight": float(weight),
        "gab_water_target_mol_kg": float(gab_target),
        "water_loading_source": (
            "GAB target"
            if water_loading_mol_kg is None
            else "explicit evolving inventory"
        ),
        "water_loading_normalization": normalization,
        "measured_coupled_support_temperature_K": list(
            MEASURED_COUPLED_SUPPORT_TEMPERATURE_K
        ),
        "historical_evaluation_range_temperature_K": list(
            HISTORICAL_EVALUATION_RANGE_TEMPERATURE_K
        ),
        "reference_regeneration_temperature_C": (
            REQUESTED_REGENERATION_TEMPERATURE_C
        ),
        "temperature_policy": (
            "record excursions; no advisory or requested-temperature ceiling"
        ),
        "cold_hot_conditional": True,
        "measured_hot_coupling_validation": False,
        "hot_rate_validated": False,
        "hot_energy_validated": False,
    }


def source_report() -> dict:
    return {
        "classification": "TEMPERATURE_ADVISORY_CONDITIONAL_EXTRAPOLATION",
        "source_doi": "10.1039/D1EE01272J",
        "measured_coupled_support_temperature_K": list(
            MEASURED_COUPLED_SUPPORT_TEMPERATURE_K
        ),
        "historical_evaluation_range_temperature_K": list(
            HISTORICAL_EVALUATION_RANGE_TEMPERATURE_K
        ),
        "reference_regeneration_temperature_C": (
            REQUESTED_REGENERATION_TEMPERATURE_C
        ),
        "historical_range_enforced": False,
        "reference_temperature_enforced": False,
        "temperature_policy": (
            "record accepted and query temperatures; reject only genuine "
            "constitutive or equation-domain failures"
        ),
        "iapws_temperature_support_K": list(IAPWS_TEMPERATURE_SUPPORT_K),
        "iapws_critical_temperature_exclusive": True,
        "relative_humidity_domain": [
            0.0,
            1.0,
        ],
        "relative_humidity_upper_exclusive": True,
        "co2_partial_pressure_domain": [0.0, None],
        "water_loading_domain": [
            -WATER_LOADING_NEGATIVE_NUMERICAL_TOLERANCE_MOL_KG,
            None,
        ],
        "water_loading_upper_bound_known": False,
        "physical_degradation_limit_C": None,
        "evidenced_physical_constraints": [],
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
