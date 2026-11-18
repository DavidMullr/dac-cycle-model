"""Conditional Young et al. (2021) coupled water-CO2 equilibrium law.

This module is a portable native implementation of the published WADST and
GAB fits. It is used only inside the explicitly declared support domain and
does not import research artefacts at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

R = 8.31446261815324


@dataclass(frozen=True)
class TothParameters:
    q_inf_0_mol_kg: float
    chi: float
    b_0_Pa_inv: float
    minus_delta_h_0_J_mol: float
    tau_0: float
    alpha: float


@dataclass(frozen=True)
class GabParameters:
    q_monolayer_mol_kg: float
    c_constant_J_mol: float
    d_constant_K_inv: float
    f_constant_J_mol: float
    g_constant_J_mol_K: float
    e_10_intercept_J_mol: float
    e_10_slope_J_mol_K: float


YOUNG2021_DRY = TothParameters(
    q_inf_0_mol_kg=4.86,
    chi=0.0,
    b_0_Pa_inv=2.85e-21,
    minus_delta_h_0_J_mol=117_798.0,
    tau_0=0.209,
    alpha=0.523,
)
YOUNG2021_WET = TothParameters(
    q_inf_0_mol_kg=9.035,
    chi=0.0,
    b_0_Pa_inv=1.23e-18,
    minus_delta_h_0_J_mol=203_687.0,
    tau_0=0.053,
    alpha=0.053,
)
YOUNG2021_GAB = GabParameters(
    q_monolayer_mol_kg=3.63,
    c_constant_J_mol=47_110.0,
    d_constant_K_inv=0.023744,
    f_constant_J_mol=57_706.0,
    g_constant_J_mol_K=-47.814,
    e_10_intercept_J_mol=57_220.0,
    e_10_slope_J_mol_K=-44.38,
)
CRITICAL_WATER_LOADING_MOL_KG = 1.532
SUPPORTED_TEMPERATURE_K = (298.15, 343.15)
SUPPORTED_RELATIVE_HUMIDITY = (0.0, 0.55)
SUPPORTED_CO2_PRESSURE_PA = (0.0, 100_000.0)
TEMPERATURE_ROUNDOFF_TOLERANCE_K = 1e-9
RELATIVE_HUMIDITY_ROUNDOFF_TOLERANCE = 1e-12
CO2_PRESSURE_ROUNDOFF_TOLERANCE_PA = 1e-9


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_domain(temperature_K: float, co2_Pa: float, rh: float) -> None:
    temperature_K = _finite(temperature_K, "temperature_K")
    co2_Pa = _finite(co2_Pa, "co2_Pa")
    rh = _finite(rh, "relative_humidity")
    if (
        temperature_K
        < SUPPORTED_TEMPERATURE_K[0] - TEMPERATURE_ROUNDOFF_TOLERANCE_K
        or temperature_K
        > SUPPORTED_TEMPERATURE_K[1] + TEMPERATURE_ROUNDOFF_TOLERANCE_K
    ):
        raise ValueError(
            f"temperature_K outside Young2021 support "
            f"[{SUPPORTED_TEMPERATURE_K[0]}, {SUPPORTED_TEMPERATURE_K[1]}]"
        )
    if rh < 0.0 or (
        rh
        > SUPPORTED_RELATIVE_HUMIDITY[1]
        + RELATIVE_HUMIDITY_ROUNDOFF_TOLERANCE
    ):
        raise ValueError(
            f"relative_humidity outside Young2021 audited support "
            f"[0, {SUPPORTED_RELATIVE_HUMIDITY[1]}]"
        )
    if co2_Pa < 0.0 or (
        co2_Pa
        > SUPPORTED_CO2_PRESSURE_PA[1]
        + CO2_PRESSURE_ROUNDOFF_TOLERANCE_PA
    ):
        raise ValueError(
            f"co2_Pa outside Young2021 support "
            f"[{SUPPORTED_CO2_PRESSURE_PA[0]}, {SUPPORTED_CO2_PRESSURE_PA[1]}]"
        )


def buck_saturation_pressure_Pa(temperature_K: float) -> float:
    """Buck (1996) saturation pressure used by Young et al. SI eq. on p.17."""

    temperature_C = float(temperature_K) - 273.15
    return 611.21 * math.exp(
        (18.678 - temperature_C / 234.5)
        * temperature_C
        / (257.14 + temperature_C)
    )


def toth_loading(
    pressure_Pa: float,
    temperature_K: float,
    parameters: TothParameters,
    reference_temperature_K: float,
) -> float:
    pressure_Pa = max(float(pressure_Pa), 0.0)
    if pressure_Pa == 0.0:
        return 0.0
    # Table 2/4 b0 values are rescaled for the adjusted Eq. 3 affinity.
    # They are not affinities at the 298.15 K reference temperature.
    exponent = parameters.minus_delta_h_0_J_mol / (R * temperature_K)
    b = parameters.b_0_Pa_inv * math.exp(
        max(-700.0, min(700.0, exponent))
    )
    q_inf = parameters.q_inf_0_mol_kg * math.exp(
        parameters.chi
        * (1.0 - temperature_K / reference_temperature_K)
    )
    tau = parameters.tau_0 + parameters.alpha * (
        1.0 - reference_temperature_K / temperature_K
    )
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("Toth exponent must be finite and positive")
    ratio = b * pressure_Pa
    scaled = ratio**tau
    denominator = (1.0 + scaled) ** (1.0 / tau)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Toth denominator is not finite and positive")
    return q_inf * ratio / denominator


def gab_water_loading(relative_humidity: float, temperature_K: float) -> float:
    """Evaluate the published GAB expression at local bed conditions."""

    relative_humidity = _finite(relative_humidity, "relative_humidity")
    temperature_K = _finite(temperature_K, "temperature_K")
    if not 0.0 <= relative_humidity < 1.0:
        raise ValueError("relative_humidity must be in [0, 1)")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if relative_humidity == 0.0:
        return 0.0
    e_10 = (
        YOUNG2021_GAB.e_10_slope_J_mol_K * temperature_K
        + YOUNG2021_GAB.e_10_intercept_J_mol
    )
    e_1 = (
        YOUNG2021_GAB.c_constant_J_mol
        - math.exp(YOUNG2021_GAB.d_constant_K_inv * temperature_K)
    )
    e_2_9 = (
        YOUNG2021_GAB.f_constant_J_mol
        + YOUNG2021_GAB.g_constant_J_mol_K * temperature_K
    )
    c_affinity = math.exp((e_1 - e_10) / (R * temperature_K))
    k_affinity = math.exp((e_2_9 - e_10) / (R * temperature_K))
    denominator = (1.0 - k_affinity * relative_humidity) * (
        1.0 + (c_affinity - 1.0) * k_affinity * relative_humidity
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("GAB denominator is not finite and positive")
    return (
        YOUNG2021_GAB.q_monolayer_mol_kg
        * k_affinity
        * c_affinity
        * relative_humidity
        / denominator
    )


def wet_site_weight(water_loading_mol_kg: float) -> float:
    water_loading_mol_kg = _finite(
        water_loading_mol_kg, "water_loading_mol_kg"
    )
    if water_loading_mol_kg < 0.0:
        raise ValueError("water_loading_mol_kg must be nonnegative")
    if water_loading_mol_kg == 0.0:
        return 0.0
    return math.exp(-CRITICAL_WATER_LOADING_MOL_KG / water_loading_mol_kg)


def local_relative_humidity(
    temperature_K: float, water_partial_Pa: float
) -> float:
    water_partial_Pa = float(water_partial_Pa)
    if not math.isfinite(water_partial_Pa) or water_partial_Pa < 0.0:
        raise ValueError("water_partial_Pa must be finite and nonnegative")
    p_sat = buck_saturation_pressure_Pa(float(temperature_K))
    if not math.isfinite(p_sat) or p_sat <= 0.0:
        raise ValueError("water saturation pressure is not finite and positive")
    return water_partial_Pa / p_sat


def co2_loading_from_water(
    temperature_K: float,
    co2_Pa: float,
    water_loading_mol_kg: float,
    *,
    reference_temperature_K: float = 298.15,
) -> tuple[float, float]:
    """Return WADST CO2 loading and wet-site weight at explicit q_H2O."""

    water_loading_mol_kg = _finite(
        water_loading_mol_kg, "water_loading_mol_kg"
    )
    if not 0.0 <= water_loading_mol_kg <= 4.131251972115188:
        raise ValueError("water_loading_mol_kg outside audited support")
    dry = toth_loading(
        co2_Pa,
        temperature_K,
        YOUNG2021_DRY,
        reference_temperature_K,
    )
    wet = toth_loading(
        co2_Pa,
        temperature_K,
        YOUNG2021_WET,
        reference_temperature_K,
    )
    weight = wet_site_weight(water_loading_mol_kg)
    return (
        (1.0 - weight) * dry + weight * wet,
        weight,
    )


def equilibrium(
    temperature_K: float,
    co2_Pa: float,
    water_partial_Pa: float,
    *,
    reference_temperature_K: float = 298.15,
    water_loading_mol_kg: float | None = None,
) -> tuple[float, float, dict]:
    """Return CO2/water loadings and local-condition diagnostics."""

    local_rh = local_relative_humidity(temperature_K, water_partial_Pa)
    validate_domain(temperature_K, co2_Pa, local_rh)
    gab_target = gab_water_loading(local_rh, temperature_K)
    water_loading = (
        gab_target
        if water_loading_mol_kg is None
        else float(water_loading_mol_kg)
    )
    co2_loading, weight = co2_loading_from_water(
        temperature_K,
        co2_Pa,
        water_loading,
        reference_temperature_K=reference_temperature_K,
    )
    p_sat = buck_saturation_pressure_Pa(float(temperature_K))
    return co2_loading, water_loading, {
        "local_relative_humidity": local_rh,
        "water_saturation_pressure_Pa": p_sat,
        "wet_site_weight": weight,
        "gab_water_target_mol_kg": gab_target,
        "water_loading_source": (
            "GAB target"
            if water_loading_mol_kg is None
            else "explicit evolving inventory"
        ),
    }
