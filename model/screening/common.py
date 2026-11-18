"""Shared material contract, configuration and SI-unit gas properties."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = 8.314462618
SPECIES = ("CO2", "H2O", "N2", "O2")
MW = np.array([0.0440095, 0.01801528, 0.0280134, 0.0319988])
CP_GAS = np.array([37.5, 34.0, 29.2, 29.4])  # J/(mol K), screening constants


class ExpectedScenarioError(RuntimeError):
    """Deliberate failure of a configured scenario, not a programming error.

    Raised for operating-limit, conservation, inventory, first-law and integration failures
    that a scenario sweep is expected to report per case. Anything else (a plain RuntimeError,
    ValueError, KeyError or TypeError) is left to propagate so real defects are never hidden
    as scenario outcomes.
    """


def gas_internal_energy_mol(temperature_K, reference_temperature_K: float) -> np.ndarray:
    """Ideal-gas molar internal energy u_g = (Cp - R) * (T - Tref)."""
    return (CP_GAS - R) * (temperature_K - reference_temperature_K)


def gas_enthalpy_mol(temperature_K, reference_temperature_K: float) -> np.ndarray:
    """Ideal-gas molar enthalpy h_g = Cp * (T - Tref) + R * Tref."""
    return CP_GAS * (temperature_K - reference_temperature_K) + R * reference_temperature_K


def adsorbate_molar_cp(material_data: dict) -> np.ndarray:
    """Adsorbed-phase molar heat capacity Cp_a = MW * adsorbed_phase_cp_J_kg_K."""
    return MW * np.asarray(material_data["adsorbed_phase_cp_J_kg_K"])


def adsorbate_internal_energy_mol(temperature_K, reference_temperature_K: float, heats_reference: np.ndarray,
                                  cp_adsorbed: np.ndarray) -> np.ndarray:
    """Adsorbed-phase molar internal energy u_a = Cp_a*(T-Tref) + R*Tref - Q_ref."""
    return cp_adsorbed * (temperature_K - reference_temperature_K) + R * reference_temperature_K - heats_reference


def water_saturation_pressure_Pa(temperature_K: float) -> float:
    """NIST/Stull Antoine correlation, 255.9-373 K; pressure converted bar -> Pa."""
    if not 255.9 <= temperature_K <= 373.0:
        raise ValueError("Water vapour-pressure correlation is valid only at 255.9-373 K.")
    return 1e5 * 10 ** (4.6543 - 1435.264 / (temperature_K - 64.848))


def feed_composition(config: dict, *, conditioned: bool = True) -> np.ndarray:
    """Mole fractions of the gas the train delivers to the bed (conditioned) or of raw ambient air.

    With an explicit `dryer` section the conditioned composition is the dryer's resolved outlet
    composition, so every caller that asks for the feed receives the same processed stream the cycle
    is fed. The unconditioned branch never builds the dryer: the vessel leak and any raw comparison
    keep the ambient composition. The only supported edge is this function calling `build_dryer`;
    `feed_dryer` never calls back here, which is what keeps the direction acyclic.
    """
    if conditioned:
        from .feed_dryer import build_dryer
        dryer = build_dryer(config)
        if dryer is not None:
            return np.array(dryer.y_conditioned, dtype=float)
    a = config["ambient"]
    rh = a["relative_humidity"]
    if conditioned and config["conditioning"]["outlet_relative_humidity"] is not None:
        rh = config["conditioning"]["outlet_relative_humidity"]
    water = rh * water_saturation_pressure_Pa(a["temperature_C"] + 273.15) / a["pressure_Pa"]
    co2_dry = a["co2_ppm_dry"] * 1e-6
    # Argon is lumped with nitrogen. CO2 is specified on a dry-gas basis.
    dry = np.array([co2_dry, 0.0, (1 - co2_dry) * 0.7905, (1 - co2_dry) * 0.2095])
    return dry * (1 - water) + np.array([0, water, 0, 0])




def pump_profile_for(config: dict):
    """Resolve the configured pump profile (local import keeps `common` free of a module cycle)."""
    from . import pump_profile
    return pump_profile.profile_for(config)


def preload_pump_profile(config: dict) -> None:
    """Read and validate a tabulated pump profile once, before any integration starts."""
    pump_profile_for(config)




class MaterialModel(Protocol):
    data: dict
    heats: np.ndarray

    def equilibrium(self, temperature_K: float, partial_pressures_Pa: np.ndarray) -> np.ndarray: ...
    def rates(self, temperature_K: float, partial_pressures_Pa: np.ndarray, loading: np.ndarray) -> np.ndarray: ...
    def heats_at(self, temperature_K: float) -> np.ndarray: ...


class IndependentLangmuirSips:
    """Deliberately hypothetical surrogate, calibrated to a single dry CO2 point.

    Temperature dependence uses a Kirchhoff-consistent van't Hoff form so that the
    sorption enthalpy and the isotherm share one set of heat capacities:

        Cp_a      = MW * adsorbed_phase_cp_J_kg_K        (adsorbed-phase molar Cp)
        dCp       = CP_GAS - Cp_a
        Q(T)      = Q_ref + dCp * (T - T_ref)            (sorption enthalpy magnitude)
        ln(b/bref) = Q_ref/R * (1/T - 1/T_ref)
                     + dCp/R * (1 - T_ref/T - ln(T/T_ref))

    Both terms vanish at T_ref, so the calibrated reference point is preserved
    exactly. `heats` stays the reference set Q_ref; `heats_at(T)` is the working
    enthalpy. The configured adsorbed-phase Cp values are engineering assumptions,
    and the liquid-like water value (4180 J/kg/K) is the dominant one; it is kept,
    not replaced by a gas value.
    """

    def __init__(self, data: dict, kinetic_spec=None, particle_diameter_m: float | None = None):
        self.data = data
        self.heats = np.array([data[s]["heat_J_mol"] for s in ("co2", "water", "nitrogen", "oxygen")])
        self.cp_adsorbed = adsorbate_molar_cp(data)
        #: Optional opt-in kinetic scaling (see screening/kinetics.py). None keeps the default
        #: constant reference rates exactly as calibrated.
        self.kinetic_spec = kinetic_spec
        self.particle_diameter_m = particle_diameter_m
        co2 = data["co2"]
        self.b_co2_ref = co2["reference_loading_mol_kg"] / (
            co2["reference_partial_pressure_Pa"] * (co2["saturation_mol_kg"] - co2["reference_loading_mol_kg"]))
        self.b_water_ref = 1 / (data["water"]["half_loading_relative_humidity_at_reference"] * water_saturation_pressure_Pa(data["reference_temperature_K"]))

    def with_kinetics(self, kinetic_spec, particle_diameter_m: float):
        """New instance carrying the scaling; the receiver and its shared data stay untouched."""
        return IndependentLangmuirSips(self.data, kinetic_spec, float(particle_diameter_m))

    def without_kinetics(self):
        """New unbound instance: used when the configuration asks for the default model."""
        return IndependentLangmuirSips(self.data)

    def kinetic_scale(self, temperature_K: float) -> np.ndarray:
        """Per-species multiplier on the reference rate constants (all ones on the default path)."""
        if self.kinetic_spec is None:
            return np.ones(len(self.data["adsorption_rate_s_inv"]))
        return self.kinetic_spec.scale(temperature_K, self.particle_diameter_m)

    def log_activity_shift(self, temperature_K: float) -> np.ndarray:
        """ln(b(T)/b_ref) for each independent isotherm (Kirchhoff-consistent)."""
        reference = self.data["reference_temperature_K"]
        delta_cp = CP_GAS - self.cp_adsorbed
        return (self.heats / R * (1.0 / temperature_K - 1.0 / reference)
                + delta_cp / R * (1.0 - reference / temperature_K - np.log(temperature_K / reference)))

    def heats_at(self, temperature_K: float) -> np.ndarray:
        """Working sorption enthalpy magnitude Q(T) = Q_ref + dCp*(T - T_ref)."""
        return self.heats + (CP_GAS - self.cp_adsorbed) * (temperature_K - self.data["reference_temperature_K"])

    def equilibrium(self, temperature_K: float, partial_pressures_Pa: np.ndarray) -> np.ndarray:
        if temperature_K <= 0:
            raise ValueError("Absolute temperature must be positive.")
        p = np.maximum(partial_pressures_Pa, 0)
        factors = np.exp(np.clip(self.log_activity_shift(temperature_K), -100, 100))
        x = self.b_co2_ref * factors[0] * p[0]
        q_co2 = self.data["co2"]["saturation_mol_kg"] * x / (1 + x)
        x_water = self.b_water_ref * factors[1] * p[1]
        # Stable Hill/Sips evaluation, including the exactly dry limit.
        exponent = self.data["water"]["sips_exponent"]
        log_ratio = exponent * np.log(max(x_water, 1e-100))
        water_fraction = 1 / (1 + np.exp(np.clip(-log_ratio, -700, 700))) if p[1] > 0 else 0.0
        q_water = self.data["water"]["saturation_mol_kg"] * water_fraction
        q_co2 /= 1 + self.data["co2_water_penalty_per_mol_kg"] * q_water
        return np.array([q_co2, q_water,
            self.data["nitrogen"]["henry_mol_kg_Pa_at_reference"] * factors[2] * p[2],
            self.data["oxygen"]["henry_mol_kg_Pa_at_reference"] * factors[3] * p[3]])

    def rates(self, temperature_K: float, partial_pressures_Pa: np.ndarray, loading: np.ndarray) -> np.ndarray:
        delta = self.equilibrium(temperature_K, partial_pressures_Pa) - loading
        k = np.where(delta >= 0, self.data["adsorption_rate_s_inv"], self.data["desorption_rate_s_inv"])
        return k * self.kinetic_scale(temperature_K) * delta


def make_material(data: dict, kinetic_spec=None, particle_diameter_m: float | None = None) -> "MaterialModel":
    """Resolve the configured equilibrium family to its model implementation.

    The measured-reference family lives in `screening/material_models.py` and is imported lazily, so the
    default family path stays free of an import cycle and a configuration that does not use the new
    family cannot be affected by it.
    """
    families = {"independent_langmuir_sips": IndependentLangmuirSips}
    name = data["equilibrium_family"]
    if name not in families:
        from .material_models import FAMILY_NAME, ActivityShiftReferenceIsotherm
        families[FAMILY_NAME] = ActivityShiftReferenceIsotherm
    if name not in families:
        raise ValueError("Unsupported equilibrium family. Add and validate a MaterialModel implementation before using it.")
    return families[name](data, kinetic_spec, particle_diameter_m)


def co2_loading_scale(material_data: dict) -> float:
    """Characteristic CO2 loading scale of a material, whichever family it uses.

    The cycle needs a scale for its cyclic-steady-state comparison, not a fitted saturation. The default
    family declares one; the measured-reference family reaches its measured curve maximum. Using the
    curve maximum keeps the family from being discarded by a comparison that only knows the default key.
    """
    co2 = material_data.get("co2") or {}
    if "saturation_mol_kg" in co2:
        return float(co2["saturation_mol_kg"])
    curve = co2.get("reference_curve") or {}
    loadings = curve.get("loading_mol_kg")
    if not loadings:
        raise ValueError("Material has neither co2.saturation_mol_kg nor a co2.reference_curve to scale "
                         "the CO2 state comparison.")
    return float(max(loadings))
