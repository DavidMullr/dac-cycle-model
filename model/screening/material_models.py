"""Equilibrium families for measured materials, for the shared material interface.

The independent-species `independent_langmuir_sips` family in `screening/common.py` is a two-parameter-per-species
surrogate calibrated to one point. A measured isotherm has a shape that a single Langmuir may not
reproduce, so this module adds one family that keeps the *measured* shape and the existing energy
inventory at the same time:

    q_CO2(T, p) = q_ref( p * exp( Q / R * (1/T - 1/T_ref) ) )

`q_ref` is a monotone reference curve measured at `T_ref` (PCHIP in ln p, with a documented Henry
extension towards zero pressure). The exponential factor is the activity shift of a *constant*
isosteric heat Q, which is the same ansatz as the independent-species Kirchhoff form with dCp_CO2 = 0. That matters:
the cycle's energy inventory is built from `heats_at(T)`, so with dCp_CO2 = 0 it returns exactly Q at
every temperature and no loading-dependent term appears that the inventory does not model. A material
that wants the author's constant-71 kJ/mol shift convention therefore must set the CO2 adsorbed-phase
heat capacity to CP_GAS/MW, and this class refuses to load otherwise rather than silently mixing the
author convention with our Kirchhoff scenario.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .common import CP_GAS, MW, R, adsorbate_molar_cp, water_saturation_pressure_Pa

FAMILY_NAME = "activity_shift_reference_isotherm"
#: The adsorbed-phase heat capacity that makes dCp_CO2 exactly zero (constant isosteric heat).
CO2_CONSTANT_HEAT_CP_J_KG_K = float(CP_GAS[0] / MW[0])
#: Support policies. Absent policy means strict: a query outside the measured curve window is refused.
STRICT_POLICY = "strict"
NAMED_EXTRAPOLATION_POLICY = "named_extrapolation_scenario"


def reference_curve_loading(pressure_ref_Pa, curve_pressure_Pa, curve_loading_mol_kg):
    """The one reference-curve evaluator: monotone PCHIP in ln p, shared by fit and deployed model.

    Below the measured window the physical zero-pressure limit is the Henry line through the first
    measured point; above it the curve is held at its last measured loading. Both extensions are
    reported by the caller, and neither is silently treated as measured.
    """
    pressures = np.asarray(curve_pressure_Pa, dtype=float)
    loadings = np.asarray(curve_loading_mol_kg, dtype=float)
    shape = PchipInterpolator(np.log(pressures), loadings, extrapolate=False)
    p = np.atleast_1d(np.asarray(pressure_ref_Pa, dtype=float))
    out = np.zeros_like(p)
    positive = p > 0
    if np.any(positive):
        inside = (p > pressures[0]) & (p <= pressures[-1])
        if np.any(inside):
            out[inside] = shape(np.log(p[inside]))
        below = positive & (p <= pressures[0])
        if np.any(below):
            out[below] = loadings[0] / pressures[0] * p[below]
        above = p > pressures[-1]
        if np.any(above):
            out[above] = loadings[-1]
    return out


class ActivityShiftReferenceIsotherm:
    """Measured CO2 reference curve with a constant-isosteric-heat activity shift.

    `heats` keeps the reference enthalpy array exactly like the independent-species family, `heats_at(T)` returns
    Q for CO2 at every temperature, and the other three species keep the configured Kirchhoff
    behaviour so nothing about water, nitrogen or oxygen changes.
    """

    def __init__(self, data: dict, kinetic_spec=None, particle_diameter_m: float | None = None):
        self.data = data
        self.heats = np.array([data[s]["heat_J_mol"] for s in ("co2", "water", "nitrogen", "oxygen")])
        self.cp_adsorbed = adsorbate_molar_cp(data)
        self.kinetic_spec = kinetic_spec
        self.particle_diameter_m = particle_diameter_m
        curve = data["co2"].get("reference_curve")
        if not isinstance(curve, dict):
            raise ValueError(f"A {FAMILY_NAME} material needs co2.reference_curve with pressure_Pa and "
                             f"loading_mol_kg arrays.")
        pressures = np.asarray(curve["pressure_Pa"], dtype=float)
        loadings = np.asarray(curve["loading_mol_kg"], dtype=float)
        if pressures.ndim != 1 or pressures.size < 3 or pressures.size != loadings.size:
            raise ValueError("co2.reference_curve needs at least three paired pressure/loading entries.")
        if not (np.all(np.isfinite(pressures)) and np.all(np.isfinite(loadings))):
            raise ValueError("co2.reference_curve entries must be finite.")
        if np.any(pressures <= 0) or np.any(loadings < 0):
            raise ValueError("co2.reference_curve pressures must be positive and loadings nonnegative.")
        order = np.argsort(pressures)
        self.curve_pressure = pressures[order]
        self.curve_loading = loadings[order]
        if np.any(np.diff(self.curve_loading) < 0):
            raise ValueError("co2.reference_curve loading must be monotone non-decreasing in pressure; "
                             "a non-monotone measured curve cannot be used as an equilibrium isotherm.")
        if not np.allclose(np.diff(self.curve_pressure) > 0, True):
            raise ValueError("co2.reference_curve pressures must be strictly increasing.")
        # PCHIP on ln p: shape-preserving, so the interpolant cannot overshoot between measured points.
        self._logp = np.log(self.curve_pressure)
        self._shape = PchipInterpolator(self._logp, self.curve_loading, extrapolate=False)
        self._henry_slope = float(self.curve_loading[0] / self.curve_pressure[0])
        self.delta_cp = CP_GAS - self.cp_adsorbed
        if "temperature_K" not in curve:
            raise ValueError("co2.reference_curve must state the temperature_K it was measured at: the CO2 "
                             "pressure mapping uses that temperature, and the global reference temperature "
                             "stays the template's so inherited non-CO2 properties are untouched.")
        self.curve_temperature_K = float(curve["temperature_K"])
        co2_policy = data["co2"].get("support") or {}
        self.support_policy = co2_policy.get("policy", STRICT_POLICY)
        self.support_scenario = co2_policy.get("scenario_name", "")
        measured_range = co2_policy.get("measured_temperature_range_K")
        self.measured_temperature_range_K = (float(measured_range[0]), float(measured_range[1])) \
            if measured_range else None
        # Bounded reporting: the right-hand side can call this thousands of times per solve, so the
        # model keeps counters and an envelope instead of retaining a record per evaluation.
        self.extrapolation_counter = 0
        self.extrapolation_envelope: dict = {"min_temperature_K": None, "max_temperature_K": None,
                                             "min_mapped_pressure_Pa": None, "max_mapped_pressure_Pa": None}
        if abs(float(self.delta_cp[0])) > 1.0e-9:
            raise ValueError(
                f"{FAMILY_NAME} requires dCp_CO2 = 0 so the constant-isosteric-heat shift matches the energy "
                f"inventory; set adsorbed_phase_cp_J_kg_K[0] = {CO2_CONSTANT_HEAT_CP_J_KG_K!r} "
                f"(got {self.cp_adsorbed[0]!r}, dCp_CO2 = {float(self.delta_cp[0])!r} J/mol/K).")
        self.b_water_ref = 1 / (data["water"]["half_loading_relative_humidity_at_reference"]
                                * water_saturation_pressure_Pa(data["reference_temperature_K"]))

    # -- interface used by the kinetics binder ------------------------------------------------
    def with_kinetics(self, kinetic_spec, particle_diameter_m: float):
        return ActivityShiftReferenceIsotherm(self.data, kinetic_spec, float(particle_diameter_m))

    def without_kinetics(self):
        return ActivityShiftReferenceIsotherm(self.data)

    def kinetic_scale(self, temperature_K: float) -> np.ndarray:
        if self.kinetic_spec is None:
            return np.ones(len(self.data["adsorption_rate_s_inv"]))
        return self.kinetic_spec.scale(temperature_K, self.particle_diameter_m)

    def log_activity_shift(self, temperature_K: float) -> np.ndarray:
        """ln(b(T)/b_ref) per species; CO2 uses the constant-heat shift (dCp_CO2 = 0 by construction)."""
        reference = self.data["reference_temperature_K"]
        return (self.heats / R * (1.0 / temperature_K - 1.0 / reference)
                + self.delta_cp / R * (1.0 - reference / temperature_K - np.log(temperature_K / reference)))

    def heats_at(self, temperature_K: float) -> np.ndarray:
        return self.heats + self.delta_cp * (temperature_K - self.data["reference_temperature_K"])

    # -- the measured shape -------------------------------------------------------------------
    def mapped_pressure(self, temperature_K: float, pressure_Pa):
        """p_ref = p * exp(Q/R (1/T - 1/T_curve)): the curve-temperature pressure of the same loading."""
        shift = float(self.heats[0]) / R * (1.0 / temperature_K - 1.0 / self.curve_temperature_K)
        return np.maximum(pressure_Pa, 0.0) * float(np.exp(shift))

    def reference_loading(self, pressure_ref_Pa):
        """The cached evaluator: same PCHIP knots as `reference_curve_loading`, built once per material.

        The solver calls this on every right-hand-side evaluation, so the interpolation object is created
        in `__init__` and never rebuilt here. The numbers are identical to the shared fitter's evaluator
        because both use PCHIP over the same measured nodes.
        """
        p = np.atleast_1d(np.asarray(pressure_ref_Pa, dtype=float))
        out = np.zeros_like(p)
        positive = p > 0
        if np.any(positive):
            inside = (p > self.curve_pressure[0]) & (p <= self.curve_pressure[-1])
            if np.any(inside):
                out[inside] = self._shape(np.log(p[inside]))
            below = positive & (p <= self.curve_pressure[0])
            if np.any(below):
                out[below] = self.curve_loading[0] / self.curve_pressure[0] * p[below]
            above = p > self.curve_pressure[-1]
            if np.any(above):
                out[above] = self.curve_loading[-1]
        return out

    def support_of(self, temperature_K: float, pressure_Pa: float) -> dict:
        """Classify a CO2 query by temperature and mapped pressure, not by the mapped pressure alone."""
        mapped = float(self.mapped_pressure(temperature_K, pressure_Pa))
        low, high = float(self.curve_pressure[0]), float(self.curve_pressure[-1])
        if pressure_Pa <= 0.0:
            classification = "zero_pressure"
        elif np.isclose(mapped, low, rtol=1e-12) or np.isclose(mapped, high, rtol=1e-12):
            classification = "measured_endpoint"
        elif low < mapped <= high:
            classification = "inside_curve"
        elif mapped < low:
            classification = "henry_extension"
        else:
            classification = "above_curve"
        if self.measured_temperature_range_K is None:
            temperature_supported = True
        else:
            temperature_supported = (self.measured_temperature_range_K[0] - 0.5 <= temperature_K
                                     <= self.measured_temperature_range_K[1] + 0.5)
        return {"mapped_reference_pressure_Pa": mapped, "curve_domain_Pa": [low, high],
                "curve_temperature_K": self.curve_temperature_K,
                "classification": classification,
                "outside_curve": classification in ("henry_extension", "above_curve"),
                "temperature_supported": temperature_supported,
                "measured_temperature_range_K": list(self.measured_temperature_range_K)
                if self.measured_temperature_range_K else None,
                "policy": self.support_policy, "scenario_name": self.support_scenario}

    def _record_extrapolation(self, support: dict, temperature_K: float) -> None:
        """Bounded count and envelope: no per-evaluation retention on a solver right-hand side."""
        self.extrapolation_counter += 1
        envelope = self.extrapolation_envelope
        updates = (("min_temperature_K", temperature_K, min), ("max_temperature_K", temperature_K, max),
                   ("min_mapped_pressure_Pa", support["mapped_reference_pressure_Pa"], min),
                   ("max_mapped_pressure_Pa", support["mapped_reference_pressure_Pa"], max))
        for key, value, reducer in updates:
            envelope[key] = value if envelope[key] is None else reducer(envelope[key], value)

    def extrapolation_summary(self) -> dict:
        return {"evaluations_outside_support": self.extrapolation_counter,
                "envelope": dict(self.extrapolation_envelope), "policy": self.support_policy,
                "scenario_name": self.support_scenario}

    def equilibrium(self, temperature_K: float, partial_pressures_Pa: np.ndarray) -> np.ndarray:
        if temperature_K <= 0:
            raise ValueError("Absolute temperature must be positive.")
        p = np.maximum(partial_pressures_Pa, 0)
        support = self.support_of(temperature_K, float(p[0]))
        # Curve endpoints and the physical zero-pressure limit are supported; a Henry extension below the
        # measured window and anything above it are not, and neither is a temperature outside the measured
        # range. All reasons are listed, not only the first one.
        reasons = []
        if support["outside_curve"]:
            reasons.append(f"the CO2 query maps to {support['mapped_reference_pressure_Pa']:.6g} Pa, outside "
                           f"the measured curve window "
                           f"[{support['curve_domain_Pa'][0]:.6g}, {support['curve_domain_Pa'][1]:.6g}] Pa")
        if not support["temperature_supported"]:
            reasons.append(f"temperature {temperature_K:g} K is outside the measured range "
                           f"{support['measured_temperature_range_K']}")
        if reasons and self.support_policy != NAMED_EXTRAPOLATION_POLICY:
            raise ValueError(
                f"CO2 query at T={temperature_K:g} K, p={float(p[0]):g} Pa is outside the measured support: "
                + "; ".join(reasons)
                + f". The default policy is strict; declare co2.support.policy = "
                f"{NAMED_EXTRAPOLATION_POLICY!r} with a scenario_name to evaluate an explicitly named "
                f"extrapolation, or use a material whose measured window covers this query.")
        if reasons:
            self._record_extrapolation(support, temperature_K)
        q_co2 = float(np.atleast_1d(self.reference_loading(
            self.mapped_pressure(temperature_K, float(p[0]))))[0])
        # Non-CO2 species keep the template's global reference temperature and caloric defaults, so the
        # material comparison changes CO2 only.
        other_shift = np.exp(np.clip(self.log_activity_shift(temperature_K)[1:], -100, 100))
        x_water = self.b_water_ref * other_shift[0] * p[1]
        exponent = self.data["water"]["sips_exponent"]
        water_fraction = 1 / (1 + np.exp(np.clip(-exponent * np.log(max(x_water, 1e-100)), -700, 700))) \
            if p[1] > 0 else 0.0
        q_water = self.data["water"]["saturation_mol_kg"] * water_fraction
        q_co2 /= 1 + self.data["co2_water_penalty_per_mol_kg"] * q_water
        return np.array([q_co2, q_water,
                         self.data["nitrogen"]["henry_mol_kg_Pa_at_reference"] * other_shift[1] * p[2],
                         self.data["oxygen"]["henry_mol_kg_Pa_at_reference"] * other_shift[2] * p[3]])

    def support_report(self, temperature_K: float, pressure_Pa: float) -> dict:
        """Where the query sits relative to the measured window and the transformed curve domain."""
        mapped = float(self.mapped_pressure(temperature_K, pressure_Pa))
        return {
            "mapped_reference_pressure_Pa": mapped,
            "pressure_extrapolated": bool(mapped < self.curve_pressure[0] or mapped > self.curve_pressure[-1]),
            "curve_domain_Pa": [float(self.curve_pressure[0]), float(self.curve_pressure[-1])],
            "curve_temperature_K": self.curve_temperature_K,
            "policy": self.support_policy,
        }

    def rates(self, temperature_K: float, partial_pressures_Pa: np.ndarray, loading: np.ndarray) -> np.ndarray:
        delta = self.equilibrium(temperature_K, partial_pressures_Pa) - loading
        k = np.where(delta >= 0, self.data["adsorption_rate_s_inv"], self.data["desorption_rate_s_inv"])
        return k * self.kinetic_scale(temperature_K) * delta

    #: Reference-temperature name used by reports.
    @property
    def reference_temperature_K(self) -> float:
        return float(self.data["reference_temperature_K"])
