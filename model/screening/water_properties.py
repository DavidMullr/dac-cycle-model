"""Bounded bulk-water property relations for the heating/conditioning train.

Saturation pressure and temperature use the IAPWS SR1-86(1992) / Wagner-Pruss auxiliary
equation for the saturation line of ordinary water, i.e.

    ln(p_sat / p_c) = (T_c / T) * sum_i a_i * theta**n_i,   theta = 1 - T / T_c

with T_c = 647.096 K, p_c = 22.064 MPa and the six published coefficients below. The relation
is bounded to the triple point <= T <= critical point and raises outside that range instead of
extrapolating. Source: IAPWS SR1-86(1992), https://www.iapws.org/relguide/Supp-sat.html
(the same coefficients appear in the NIST review table of IAPWS vapor-pressure correlations,
https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=935265).

Enthalpy closure
----------------
The vapour enthalpy is the same ideal-gas convention the bed model already uses for water
vapour, h_g(T) = CP_GAS[water] * (T - T_ref) + R * T_ref, so the conditioner audit shares one
reference with the bed gas balance. The liquid enthalpy is defined by one documented latent
heat anchor and two constant heat capacities (a Kirchhoff-consistent closure):

    hfg(373.15 K) = 2256.66 kJ/kg   (Spirax Sarco dry saturated steam table TI-GCM-01, 1.01325 bar)
    hfg(T)        = hfg_anchor + (CP_GAS[water] - Cp_liquid) * (T - T_anchor)
    h_l(T)        = h_g(T) - hfg(T)

Cp_liquid = 4180 J/(kg K) is **bulk liquid water** and is deliberately independent of the
hypothetical adsorbed-phase heat capacities in the material file. Because the two constant heat
capacities are used consistently, the conditioner enthalpy balance closes exactly in this
closure, while hfg(T) itself deviates from the tabulated steam values by a few percent at
150-165 C; that deviation is reported by `closure_deviation_report` rather than hidden.

The closure is bounded to 273.16-473.15 K (0.01-200 C), the range over which the two constant
heat capacities keep the latent-heat deviation within the few percent reported by
`accuracy_report`. Outside it these functions raise instead of extrapolating, so an almost-critical
steam specification cannot be accepted while still claiming a small latent-heat error. This is a
screening closure for bulk utility water; it is not an ice/steam property library, and it does not
change the bed's own water data.

Both the anchor and the comparison values come from published saturated-steam tables:
Spirax Sarco TI-GCM-01 (https://content.spiraxsarco.com/-/media/spiraxsarco/international/documents/en/ti/dry-saturated-ti-gcm-01-en.ashx)
and GEMU saturated-steam pressure table
(https://www.gemu-group.com/media/5a/8b/ea/1773173332/bs_p_090_steam_pressure_table_water_en.pdf).

This module is for bulk water in the jacket, the cooler and the separator. It does not model
condensation inside the adsorbent, dissolved gases or direct steam injection.
"""
from __future__ import annotations

import numpy as np

from .common import CP_GAS, MW, R

CRITICAL_TEMPERATURE_K = 647.096
CRITICAL_PRESSURE_PA = 22.064e6
TRIPLE_POINT_TEMPERATURE_K = 273.16
#: Approved screening range of the constant-Cp liquid/vapour enthalpy closure (0.01-200 C).
CLOSURE_MIN_TEMPERATURE_K = 273.16
CLOSURE_MAX_TEMPERATURE_K = 473.15

# IAPWS SR1-86(1992) saturation-line coefficients, paired with the exponents used in the release.
WAGNER_COEFFICIENTS = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)
WAGNER_EXPONENTS = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)

WATER_MOLAR_MASS_KG_MOL = float(MW[1])
WATER_CP_LIQUID_J_KG_K = 4180.0
WATER_CP_LIQUID_J_MOL_K = WATER_CP_LIQUID_J_KG_K * WATER_MOLAR_MASS_KG_MOL
WATER_CP_VAPOUR_J_MOL_K = float(CP_GAS[1])          # the model's screening ideal-gas constant

LATENT_HEAT_ANCHOR_TEMPERATURE_K = 373.15
LATENT_HEAT_ANCHOR_J_KG = 2256.66e3                  # Spirax Sarco TI-GCM-01 at 1.01325 bar abs
LATENT_HEAT_ANCHOR_J_MOL = LATENT_HEAT_ANCHOR_J_KG * WATER_MOLAR_MASS_KG_MOL

# Published table anchors used to state the accuracy of the implemented relations.
# (T_C, p_sat_Pa, source) - GEMU absolute-pressure table and Spirax Sarco TI-GCM-01.
SATURATION_TABLE_ANCHORS = (
    (40.0, 7381.0, "GEMU table interpolated between 0.05 and 0.1 bar abs"),
    (60.1, 20000.0, "GEMU table, 0.2 bar abs row"),
    (100.0, 101325.0, "Spirax Sarco TI-GCM-01, 1.01325 bar abs row"),
    (151.8, 500000.0, "GEMU table, 5 bar abs row"),
    (165.0, 700000.0, "GEMU table, 7 bar abs row; Spirax 6 bar g = 7.013 bar abs row"),
)

# (T_C, hfg_J_kg, source) - Spirax Sarco TI-GCM-01 (absolute page up to 2.6 bar, gauge page above).
LATENT_HEAT_TABLE_ANCHORS = (
    (69.1, 2335.28e3, "Spirax TI-GCM-01, 0.30 bar abs"),
    (100.0, 2256.66e3, "Spirax TI-GCM-01, 1.01325 bar abs"),
    (143.8, 2133.24e3, "Spirax TI-GCM-01, 3.0 bar abs"),
    (165.1, 2065.72e3, "Spirax TI-GCM-01, 7.013 bar abs (6.0 bar g row)"),
    (200.0, 1940.7e3, "Spirax TI-GCM-01 dry saturated table, 200 C row (15.5 bar abs); hfg = h_g - h_f"),
)


def _check_closure_temperature(temperature_K: float, name: str) -> float:
    """Bound the constant-Cp enthalpy closure to its documented screening range."""
    temperature = float(temperature_K)
    if not CLOSURE_MIN_TEMPERATURE_K <= temperature <= CLOSURE_MAX_TEMPERATURE_K:
        raise ValueError(f"The constant-Cp water enthalpy closure ({name}) is defined for "
                         f"{CLOSURE_MIN_TEMPERATURE_K} K <= T <= {CLOSURE_MAX_TEMPERATURE_K} K "
                         f"(0.01-200 C); got {temperature:.3f} K. It does not extrapolate towards ice or "
                         "towards the critical point.")
    return temperature


def saturation_pressure_Pa(temperature_K: float) -> float:
    """Saturation pressure of ordinary water on the IAPWS SR1-86(1992) saturation line."""
    temperature = float(temperature_K)
    if not TRIPLE_POINT_TEMPERATURE_K <= temperature <= CRITICAL_TEMPERATURE_K:
        raise ValueError(f"Saturation relation is defined for {TRIPLE_POINT_TEMPERATURE_K} K <= T <= "
                         f"{CRITICAL_TEMPERATURE_K} K; got {temperature:.3f} K.")
    theta = 1.0 - temperature / CRITICAL_TEMPERATURE_K
    series = sum(a * theta ** n for a, n in zip(WAGNER_COEFFICIENTS, WAGNER_EXPONENTS))
    return CRITICAL_PRESSURE_PA * float(np.exp(CRITICAL_TEMPERATURE_K / temperature * series))


def saturation_temperature_K(pressure_Pa: float) -> float:
    """Invert the saturation line by bisection on the monotone pressure relation."""
    pressure = float(pressure_Pa)
    if not saturation_pressure_Pa(TRIPLE_POINT_TEMPERATURE_K) <= pressure <= CRITICAL_PRESSURE_PA:
        raise ValueError(f"Saturation temperature is only defined between the triple-point pressure "
                         f"({saturation_pressure_Pa(TRIPLE_POINT_TEMPERATURE_K):.1f} Pa) and the critical "
                         f"pressure {CRITICAL_PRESSURE_PA:.0f} Pa; got {pressure:.1f} Pa.")
    low, high = TRIPLE_POINT_TEMPERATURE_K, CRITICAL_TEMPERATURE_K
    for _ in range(200):
        middle = 0.5 * (low + high)
        if saturation_pressure_Pa(middle) < pressure:
            low = middle
        else:
            high = middle
        if high - low < 1e-9:
            break
    return 0.5 * (low + high)


def latent_heat_J_mol(temperature_K: float) -> float:
    """Latent heat of vaporisation in J/mol from the documented constant-Cp closure."""
    temperature = _check_closure_temperature(temperature_K, "latent heat")
    return (LATENT_HEAT_ANCHOR_J_MOL
            + (WATER_CP_VAPOUR_J_MOL_K - WATER_CP_LIQUID_J_MOL_K)
            * (temperature - LATENT_HEAT_ANCHOR_TEMPERATURE_K))


def vapour_enthalpy_J_mol(temperature_K: float, reference_temperature_K: float) -> float:
    """Vapour enthalpy in the same convention the bed gas balance uses for water."""
    temperature = _check_closure_temperature(temperature_K, "vapour enthalpy")
    return WATER_CP_VAPOUR_J_MOL_K * (temperature - reference_temperature_K) + R * reference_temperature_K


def liquid_enthalpy_J_mol(temperature_K: float, reference_temperature_K: float) -> float:
    """Saturated-liquid enthalpy consistent with `vapour_enthalpy_J_mol` and the closure above."""
    return vapour_enthalpy_J_mol(temperature_K, reference_temperature_K) - latent_heat_J_mol(temperature_K)


def dew_point_C(water_partial_pressure_Pa: float) -> float:
    """Dew point of a water-vapour partial pressure, or NaN below the triple-point pressure."""
    partial = float(water_partial_pressure_Pa)
    if partial <= saturation_pressure_Pa(TRIPLE_POINT_TEMPERATURE_K):
        return float("nan")
    return saturation_temperature_K(partial) - 273.15


def accuracy_report() -> dict:
    """Largest deviation of the implemented relations from the published table anchors."""
    pressure_error = max(abs(saturation_pressure_Pa(t_c + 273.15) - p) / p
                         for t_c, p, _ in SATURATION_TABLE_ANCHORS)
    temperature_error = max(abs(saturation_temperature_K(p) - (t_c + 273.15))
                            for t_c, p, _ in SATURATION_TABLE_ANCHORS)
    latent_error = max(abs(latent_heat_J_mol(t_c + 273.15) / WATER_MOLAR_MASS_KG_MOL - hfg) / hfg
                       for t_c, hfg, _ in LATENT_HEAT_TABLE_ANCHORS)
    return {
        "saturation_pressure_max_relative_error": float(pressure_error),
        "saturation_temperature_max_absolute_error_K": float(temperature_error),
        "latent_heat_closure_max_relative_error": float(latent_error),
        "latent_heat_closure_anchors_C": [t_c for t_c, _, _ in LATENT_HEAT_TABLE_ANCHORS],
        "enthalpy_closure_range_K": [CLOSURE_MIN_TEMPERATURE_K, CLOSURE_MAX_TEMPERATURE_K],
        "enthalpy_closure_range_C": [CLOSURE_MIN_TEMPERATURE_K - 273.15, CLOSURE_MAX_TEMPERATURE_K - 273.15],
        "saturation_source": "IAPWS SR1-86(1992) / Wagner-Pruss auxiliary saturation equation",
        "table_sources": ["Spirax Sarco TI-GCM-01 dry saturated steam tables",
                          "GEMU saturated steam pressure table"],
        "enthalpy_closure": ("h_g from the model's ideal-gas water vapour Cp, h_l = h_g - hfg with "
                             "hfg anchored at 100 C (2256.66 kJ/kg) and constant Cp_liquid = 4180 J/(kg K); "
                             "bulk liquid properties are independent of the adsorbed-phase heat capacities"),
        "note": ("The saturation line itself is a primary-source relation; the latent-heat closure is a "
                 "screening approximation whose deviation from the tables is reported above and which is "
                 "bounded to the stated 0.01-200 C range instead of extrapolating."),
    }
