"""Material laws shared by the spatial and lumped engines.

Adapters provide equilibrium, kinetic rates, storage/species mapping and caloric
properties. Independent species use Kirchhoff-consistent thermodynamics;
site-resolved packages keep each site's loading and adsorption energy explicit.
Scientific sources and applicability limits belong to each material package."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..common import water_saturation_pressure_Pa
from ..material_models import ActivityShiftReferenceIsotherm
from . import young2021
from . import young2021_advisory_temperature
from . import young2021_cold_hot
from . import young2021_hot
from .contract import (
    COMPATIBLE_EXTRAPOLATION_POLICY,
    NAMED_EXTRAPOLATION_POLICY,
    STRICT_POLICY,
    SUPPORTED_REFERENCE_SUPPORT_POLICIES,
    MofScreeningError,
)


R = 8.31446261815324
MW = np.array([0.0440095, 0.01801528, 0.0280134, 0.0319988])
CP_GAS = np.array([37.5, 34.0, 29.2, 29.4])
SPECIES = ("CO2", "H2O", "N2", "O2")
SCALE_AWARE_NUMERICAL_POLICY = "scale_aware_5atol_v1"
SCALE_AWARE_COMPONENT_ATOL_MOL_KG = (1e-9, 1e-9)
SCALE_AWARE_BAND_MULTIPLIER = 5.0


def buck_saturation_pressure_Pa(temperature_C: float) -> float:
    """Buck (1996) saturation pressure over liquid water."""
    return 611.21 * math.exp(
        (18.678 - temperature_C / 234.5) * temperature_C / (257.14 + temperature_C)
    )


class MaterialAdapter:
    """Common material-law interface consumed by both cycle representations."""

    material_id: str
    name: str
    equilibrium_family: str
    reference_temperature_K: float
    maximum_bed_temperature_C: float
    solid_cp_J_kg_K: float
    storage_labels: tuple[str, ...]
    species_for_storage: np.ndarray
    species_labels: tuple[str, ...] = SPECIES

    @property
    def n_storage(self) -> int:
        return len(self.storage_labels)

    @property
    def water_storage_index(self) -> int:
        return self.storage_labels.index("H2O")

    @property
    def n_sites(self) -> int:
        return int(
            sum(1 for species in self.species_for_storage if species == 0)
        )

    @property
    def site_species(self) -> tuple[int, ...]:
        return (0,) * self.n_sites

    @property
    def maximum_bed_temperature_K(self) -> float:
        return self.maximum_bed_temperature_C + 273.15

    @property
    def water_scale_mol_kg(self) -> float:
        return float(self.storage_scale()[self.water_storage_index])

    def site_loadings(
        self, temperature_K: float, partial_pressures_Pa: np.ndarray
    ) -> np.ndarray:
        return self.equilibrium(temperature_K, partial_pressures_Pa)[
            self.species_for_storage == 0
        ]

    def site_rates(
        self,
        temperature_K: float,
        partial_pressures_Pa: np.ndarray,
        site_loading: np.ndarray,
        water_loading: float,
        inert_loading: np.ndarray,
    ) -> np.ndarray:
        storage = np.empty(self.n_storage, dtype=float)
        storage[self.species_for_storage == 0] = site_loading
        storage[self.water_storage_index] = water_loading
        inert_indices = np.flatnonzero(
            np.isin(self.species_for_storage, (2, 3))
        )
        if inert_indices.size:
            storage[inert_indices] = inert_loading
        rates, _ = self.rates(
            temperature_K, partial_pressures_Pa, storage
        )
        output = np.zeros(self.n_sites + 3, dtype=float)
        output[: self.n_sites] = rates[self.species_for_storage == 0]
        output[self.n_sites] = rates[self.water_storage_index]
        if inert_indices.size:
            output[self.n_sites + 1 :] = rates[inert_indices]
        return output

    def absolute_energies_J_mol(self, temperature_K: float) -> np.ndarray:
        output = np.zeros(self.n_sites + 3, dtype=float)
        output[: self.n_storage] = self.absolute_energy_J_mol(temperature_K)
        return output

    def storage_energy_terms(
        self, temperature_K: float, storage: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """Return specific energy, its T derivative and loading gradient.

        Quantities are on a dry-material basis:
        ``e_ads`` in J/kg, ``Cp_ads`` in J/(kg K), and ``h`` in J/mol.
        The default uses a linear material energy potential.
        """
        loading = np.asarray(storage, dtype=float)
        if loading.shape != (self.n_storage,):
            raise ValueError(
                f"storage_energy_terms expected {self.n_storage} loadings"
            )
        if not np.all(np.isfinite(loading)):
            raise ValueError("storage loadings must be finite")
        unit_energy = np.asarray(
            self.absolute_energy_J_mol(float(temperature_K)), dtype=float
        )
        unit_cp = np.asarray(self.component_cp_molar_J_mol_K(), dtype=float)
        return (
            float(loading @ unit_energy),
            float(loading @ unit_cp),
            unit_energy.copy(),
        )

    def component_cp_J_mol_K(self) -> np.ndarray:
        output = np.zeros(self.n_sites + 3, dtype=float)
        output[: self.n_storage] = self.component_cp_molar_J_mol_K()
        return output

    def equilibrium(self, temperature_K: float, partials_Pa: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return storage-rate vector and species adsorption-rate (4,)."""
        raise NotImplementedError

    def absolute_energy_J_mol(self, temperature_K: float) -> np.ndarray:
        raise NotImplementedError

    def component_cp_molar_J_mol_K(self) -> np.ndarray:
        raise NotImplementedError

    def storage_scale(self) -> np.ndarray:
        raise NotImplementedError

    def source_report(self) -> dict:
        raise NotImplementedError


@dataclass
class IndependentLangmuirSipsAdapter(MaterialAdapter):
    """Independent-site dry CO2 and water laws with Kirchhoff closure."""

    data: Mapping
    storage_labels = ("CO2", "H2O", "N2", "O2")
    species_for_storage = np.arange(4, dtype=int)

    def __post_init__(self) -> None:
        self.material_id = str(self.data["material_id"])
        self.name = str(self.data["name"])
        self.equilibrium_family = str(self.data["equilibrium_family"])
        self.reference_temperature_K = float(self.data["reference_temperature_K"])
        self.maximum_bed_temperature_C = float(
            self.data["maximum_bed_temperature_C"]
        )
        self.solid_cp_J_kg_K = float(self.data["solid_cp_J_kg_K"])
        self._heats_reference = np.array(
            [
                self.data["co2"]["heat_J_mol"],
                self.data["water"]["heat_J_mol"],
                self.data["nitrogen"]["heat_J_mol"],
                self.data["oxygen"]["heat_J_mol"],
            ],
            dtype=float,
        )
        self._cp_molar = MW * np.asarray(
            self.data["adsorbed_phase_cp_J_kg_K"], dtype=float
        )
        co2 = self.data["co2"]
        self._b_co2_ref = co2["reference_loading_mol_kg"] / (
            co2["reference_partial_pressure_Pa"]
            * (co2["saturation_mol_kg"] - co2["reference_loading_mol_kg"])
        )
        self._b_water_ref = 1.0 / (
            self.data["water"]["half_loading_relative_humidity_at_reference"]
            * water_saturation_pressure_Pa(self.reference_temperature_K)
        )
        self._ad_rate = np.asarray(
            self.data["kinetics"]["adsorption_rate_s_inv"], dtype=float
        )
        self._de_rate = np.asarray(
            self.data["kinetics"]["desorption_rate_s_inv"], dtype=float
        )

    def _log_activity_shift(self, temperature_K: float) -> np.ndarray:
        reference = self.reference_temperature_K
        delta_cp = CP_GAS - self._cp_molar
        return (
            self._heats_reference / R * (1.0 / temperature_K - 1.0 / reference)
            + delta_cp
            / R
            * (
                1.0
                - reference / temperature_K
                - math.log(temperature_K / reference)
            )
        )

    def equilibrium(self, temperature_K: float, partials_Pa: np.ndarray) -> np.ndarray:
        p = np.maximum(np.asarray(partials_Pa, dtype=float), 0.0)
        factors = np.exp(np.clip(self._log_activity_shift(temperature_K), -100.0, 100.0))
        x_co2 = self._b_co2_ref * factors[0] * p[0]
        co2 = self.data["co2"]
        loading_co2 = co2["saturation_mol_kg"] * x_co2 / (1.0 + x_co2)
        x_water = self._b_water_ref * factors[1] * p[1]
        water = self.data["water"]
        if p[1] > 0.0:
            log_ratio = water["sips_exponent"] * math.log(max(x_water, 1e-100))
            fraction = 1.0 / (
                1.0 + math.exp(float(np.clip(-log_ratio, -700.0, 700.0)))
            )
        else:
            fraction = 0.0
        loading_water = water["saturation_mol_kg"] * fraction
        penalty = float(self.data.get("co2_water_penalty_per_mol_kg", 0.0))
        loading_co2 /= 1.0 + penalty * loading_water
        return np.array(
            [
                loading_co2,
                loading_water,
                self.data["nitrogen"]["henry_mol_kg_Pa_at_reference"]
                * factors[2]
                * p[2],
                self.data["oxygen"]["henry_mol_kg_Pa_at_reference"]
                * factors[3]
                * p[3],
            ],
            dtype=float,
        )

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self.equilibrium(temperature_K, partials_Pa)
        delta = target - np.asarray(storage, dtype=float)
        rates = np.where(delta >= 0.0, self._ad_rate, self._de_rate) * delta
        return rates, rates

    def absolute_energy_J_mol(self, temperature_K: float) -> np.ndarray:
        return (
            self._cp_molar * (temperature_K - self.reference_temperature_K)
            + R * self.reference_temperature_K
            - self._heats_reference
        )

    def component_cp_molar_J_mol_K(self) -> np.ndarray:
        return self._cp_molar.copy()

    def storage_scale(self) -> np.ndarray:
        return np.array(
            [
                self.data["co2"]["saturation_mol_kg"],
                self.data["water"]["saturation_mol_kg"],
                max(
                    float(
                        self.data["nitrogen"][
                            "henry_mol_kg_Pa_at_reference"
                        ]
                    )
                    * 100000.0,
                    0.05,
                ),
                max(
                    float(
                        self.data["oxygen"][
                            "henry_mol_kg_Pa_at_reference"
                        ]
                    )
                    * 21000.0,
                    0.05,
                ),
            ],
            dtype=float,
        )

    def source_report(self) -> dict:
        return {
            "adapter": "independent_langmuir_sips_portable",
            "source_family": self.equilibrium_family,
            "co2_states": 1,
            "species_states": ["CO2", "H2O", "N2", "O2"],
            "caloric_closure": "Kirchhoff-consistent reference heats",
            "water_mixture": "independent unary water law with optional empirical CO2 penalty",
            "runtime_research_imports": False,
        }


@dataclass
class DryActivityShiftReferenceEvaluator:
    """CO2 reference-curve evaluator with no physical water state variables."""

    data: Mapping

    def __post_init__(self) -> None:
        curve = self.data["co2"]["reference_curve"]
        pressure = np.asarray(curve["pressure_Pa"], dtype=float)
        loading = np.asarray(curve["loading_mol_kg"], dtype=float)
        order = np.argsort(pressure)
        self.curve_pressure = pressure[order]
        self.curve_loading = loading[order]
        self._shape = PchipInterpolator(
            np.log(self.curve_pressure), self.curve_loading, extrapolate=False
        )
        self.curve_temperature_K = float(curve["temperature_K"])
        self.heats = np.array(
            [
                float(self.data["co2"]["heat_J_mol"]),
                0.0,
                float(self.data["nitrogen"]["heat_J_mol"]),
                float(self.data["oxygen"]["heat_J_mol"]),
            ],
            dtype=float,
        )
        self.cp_adsorbed = np.asarray(
            self.data["adsorbed_phase_cp_J_kg_K"], dtype=float
        ) * MW
        self.delta_cp = CP_GAS - self.cp_adsorbed
        support = self.data["co2"].get("support") or {}
        self.support_policy = str(support.get("policy", STRICT_POLICY))
        if self.support_policy not in {
            STRICT_POLICY,
            NAMED_EXTRAPOLATION_POLICY,
            COMPATIBLE_EXTRAPOLATION_POLICY,
        }:
            raise ValueError(
                "co2.support.policy must be one of "
                f"{SUPPORTED_REFERENCE_SUPPORT_POLICIES!r}; got "
                f"{self.support_policy!r}"
            )
        self.support_scenario_name = str(
            support.get("scenario_name", "")
        )
        measured = support.get("measured_temperature_range_K")
        self.measured_temperature_range_K = (
            (float(measured[0]), float(measured[1])) if measured else None
        )
        self.reference_temperature_K = float(
            self.data["reference_temperature_K"]
        )

    def log_activity_shift(self, temperature_K: float) -> np.ndarray:
        reference = self.reference_temperature_K
        return (
            self.heats / R * (1.0 / temperature_K - 1.0 / reference)
            + self.delta_cp
            / R
            * (
                1.0
                - reference / temperature_K
                - math.log(temperature_K / reference)
            )
        )

    def mapped_pressure(self, temperature_K: float, pressure_Pa):
        shift = float(self.heats[0]) / R * (
            1.0 / temperature_K - 1.0 / self.curve_temperature_K
        )
        return np.maximum(pressure_Pa, 0.0) * float(np.exp(shift))

    def reference_loading(self, pressure_ref_Pa):
        pressure = np.atleast_1d(np.asarray(pressure_ref_Pa, dtype=float))
        output = np.zeros_like(pressure)
        positive = pressure > 0.0
        inside = (
            positive
            & (pressure > self.curve_pressure[0])
            & (pressure <= self.curve_pressure[-1])
        )
        if np.any(inside):
            output[inside] = self._shape(np.log(pressure[inside]))
        below = positive & (pressure <= self.curve_pressure[0])
        if np.any(below):
            output[below] = (
                self.curve_loading[0] / self.curve_pressure[0] * pressure[below]
            )
        output[pressure > self.curve_pressure[-1]] = self.curve_loading[-1]
        return output

    def support_of(self, temperature_K: float, pressure_Pa: float) -> dict:
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
        temperature_supported = bool(
            self.measured_temperature_range_K is None
            or (
                self.measured_temperature_range_K[0] - 0.5
                <= temperature_K
                <= self.measured_temperature_range_K[1] + 0.5
            )
        )
        return {
            "mapped_reference_pressure_Pa": mapped,
            "curve_domain_Pa": [low, high],
            "curve_temperature_K": self.curve_temperature_K,
            "classification": classification,
            "outside_curve": classification in ("henry_extension", "above_curve"),
            "temperature_supported": temperature_supported,
            "measured_temperature_range_K": (
                list(self.measured_temperature_range_K)
                if self.measured_temperature_range_K
                else None
            ),
            "policy": self.support_policy,
            "scenario_name": self.support_scenario_name,
        }

    def equilibrium(
        self, temperature_K: float, partial_pressures_Pa: np.ndarray
    ) -> np.ndarray:
        if temperature_K <= 0.0:
            raise ValueError("Absolute temperature must be positive.")
        partials = np.maximum(
            np.asarray(partial_pressures_Pa, dtype=float), 0.0
        )
        support = self.support_of(temperature_K, float(partials[0]))
        reasons = []
        if support["outside_curve"]:
            reasons.append(
                f"mapped CO2 pressure {support['mapped_reference_pressure_Pa']:.6g} Pa "
                "is outside the measured curve window"
            )
        if not support["temperature_supported"]:
            reasons.append(
                f"temperature {temperature_K:g} K is outside the measured range"
            )
        if reasons and self.support_policy not in {
            NAMED_EXTRAPOLATION_POLICY,
            COMPATIBLE_EXTRAPOLATION_POLICY,
        }:
            raise ValueError(
                f"CO2 query at T={temperature_K:g} K, p={partials[0]:g} Pa "
                "is outside the measured support: "
                + "; ".join(reasons)
                + f". Diagnostic policy: {self.support_policy!r}."
            )
        q_co2 = float(
            self.reference_loading(
                self.mapped_pressure(temperature_K, float(partials[0]))
            )[0]
        )
        shift = np.exp(
            np.clip(self.log_activity_shift(temperature_K)[1:], -100, 100)
        )
        return np.array(
            [
                q_co2,
                0.0,
                float(
                    self.data["nitrogen"]["henry_mol_kg_Pa_at_reference"]
                )
                * shift[1]
                * partials[2],
                float(
                    self.data["oxygen"]["henry_mol_kg_Pa_at_reference"]
                )
                * shift[2]
                * partials[3],
            ]
        )


@dataclass
class ActivityShiftReferenceAdapter(MaterialAdapter):
    """Adapter for the measured-curve activity-shift family."""

    data: Mapping
    storage_labels = ("CO2", "H2O", "N2", "O2")
    species_for_storage = np.arange(4, dtype=int)

    def __post_init__(self) -> None:
        self.material_id = str(self.data["material_id"])
        self.name = str(self.data["name"])
        self.equilibrium_family = str(self.data["equilibrium_family"])
        self.reference_temperature_K = float(
            self.data["reference_temperature_K"]
        )
        self.maximum_bed_temperature_C = float(
            self.data["maximum_bed_temperature_C"]
        )
        self.solid_cp_J_kg_K = float(self.data["solid_cp_J_kg_K"])
        self.dry_only_water = self.data["water"].get("model") == "dry_only"
        if self.dry_only_water:
            self.model = DryActivityShiftReferenceEvaluator(dict(self.data))
        else:
            self.model = ActivityShiftReferenceIsotherm(dict(self.data))
        self._heats_reference = self.model.heats.copy()
        self._cp_molar = self.model.cp_adsorbed.copy()
        self._ad_rate = np.asarray(
            self.data["kinetics"]["adsorption_rate_s_inv"], dtype=float
        )
        self._de_rate = np.asarray(
            self.data["kinetics"]["desorption_rate_s_inv"], dtype=float
        )

    def equilibrium(
        self, temperature_K: float, partials_Pa: np.ndarray
    ) -> np.ndarray:
        output = self.model.equilibrium(temperature_K, partials_Pa)
        if self.dry_only_water:
            output[1] = 0.0
        return output

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self.equilibrium(temperature_K, partials_Pa)
        delta = target - np.asarray(storage, dtype=float)
        rates = np.where(delta >= 0.0, self._ad_rate, self._de_rate) * delta
        return rates, rates

    def absolute_energy_J_mol(self, temperature_K: float) -> np.ndarray:
        output = (
            self._cp_molar * (temperature_K - self.reference_temperature_K)
            + R * self.reference_temperature_K
            - self._heats_reference
        )
        if self.dry_only_water:
            output[1] = 0.0
        return output

    def component_cp_molar_J_mol_K(self) -> np.ndarray:
        return self._cp_molar.copy()

    def storage_scale(self) -> np.ndarray:
        curve = self.data["co2"]["reference_curve"]["loading_mol_kg"]
        return np.array(
            [
                max(float(value) for value in curve),
                (
                    self.data["water"]["solver_normalizer_mol_kg"]
                    if self.dry_only_water
                    else self.data["water"]["saturation_mol_kg"]
                ),
                max(
                    self.data["nitrogen"][
                        "henry_mol_kg_Pa_at_reference"
                    ]
                    * 100000.0,
                    0.05,
                ),
                max(
                    self.data["oxygen"][
                        "henry_mol_kg_Pa_at_reference"
                    ]
                    * 21000.0,
                    0.05,
                ),
            ],
            dtype=float,
        )

    def source_report(self) -> dict:
        penalty = float(
            self.data.get("co2_water_penalty_per_mol_kg", 0.0)
        )
        return {
            "adapter": "activity_shift_reference_portable",
            "source_family": self.equilibrium_family,
            "co2_states": 1,
            "water_states": 0 if self.dry_only_water else 1,
            "co2_reference_curve_preserved": True,
            "constant_isosteric_heat_shift": True,
            "water_mode": "dry_only" if self.dry_only_water else "unary_sips",
            "physical_water_storage_mol_kg": (
                0.0 if self.dry_only_water else None
            ),
            "independent_component_storage": True,
            "co2_water_competition_enabled": bool(
                not self.dry_only_water and penalty != 0.0
            ),
            "moisture_sorption_heat_included": (not self.dry_only_water),
            "water_sorption_rate_validated": False,
            "hot_desorption_rate_validated": False,
            "co2_support_policy": getattr(
                self.model, "support_policy", None
            ),
            "co2_support_scenario_name": getattr(
                self.model, "support_scenario_name", None
            ),
            "runtime_research_imports": False,
        }


@dataclass
class SplitRateReferenceAdapter(MaterialAdapter):
    """Two kinetic CO2 reservoirs sharing one reference thermodynamics law."""

    data: Mapping
    storage_labels = (
        "CO2_fast",
        "CO2_slow",
        "H2O",
        "N2",
        "O2",
    )
    species_for_storage = np.array([0, 0, 1, 2, 3], dtype=int)

    def __post_init__(self) -> None:
        self.material_id = str(self.data["material_id"])
        self.name = str(self.data["name"])
        self.equilibrium_family = str(self.data["equilibrium_family"])
        self.reference_temperature_K = float(
            self.data["reference_temperature_K"]
        )
        self.maximum_bed_temperature_C = float(
            self.data["maximum_bed_temperature_C"]
        )
        self.solid_cp_J_kg_K = float(self.data["solid_cp_J_kg_K"])
        self._fractions = np.asarray(
            self.data["co2"]["reservoir_fractions"], dtype=float
        )
        cp_j_kg = np.asarray(
            self.data["adsorbed_phase_cp_J_kg_K"], dtype=float
        )
        self._cp_molar = np.array(
            [
                cp_j_kg[0] * MW[0],
                cp_j_kg[1] * MW[0],
                cp_j_kg[2] * MW[1],
                cp_j_kg[3] * MW[2],
                cp_j_kg[4] * MW[3],
            ],
            dtype=float,
        )
        self._heats_reference = np.array(
            [
                self.data["co2"]["heat_J_mol"],
                self.data["co2"]["heat_J_mol"],
                0.0,
                self.data["nitrogen"]["heat_J_mol"],
                self.data["oxygen"]["heat_J_mol"],
            ],
            dtype=float,
        )
        model_data = dict(self.data)
        model_data["adsorbed_phase_cp_J_kg_K"] = [
            cp_j_kg[0],
            0.0,
            cp_j_kg[3],
            cp_j_kg[4],
        ]
        self.model = DryActivityShiftReferenceEvaluator(model_data)
        self._ad_rate = np.asarray(
            self.data["kinetics"]["adsorption_rate_s_inv"], dtype=float
        )
        self._de_rate = np.asarray(
            self.data["kinetics"]["desorption_rate_s_inv"], dtype=float
        )

    def equilibrium(
        self, temperature_K: float, partials_Pa: np.ndarray
    ) -> np.ndarray:
        total = self.model.equilibrium(temperature_K, partials_Pa)
        return np.array(
            [
                self._fractions[0] * total[0],
                self._fractions[1] * total[0],
                0.0,
                total[2],
                total[3],
            ],
            dtype=float,
        )

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self.equilibrium(temperature_K, partials_Pa)
        delta = target - np.asarray(storage, dtype=float)
        rates = np.where(delta >= 0.0, self._ad_rate, self._de_rate) * delta
        species_rates = np.array(
            [rates[0] + rates[1], rates[2], rates[3], rates[4]],
            dtype=float,
        )
        return rates, species_rates

    def absolute_energy_J_mol(self, temperature_K: float) -> np.ndarray:
        output = (
            self._cp_molar * (temperature_K - self.reference_temperature_K)
            + R * self.reference_temperature_K
            - self._heats_reference
        )
        output[self.water_storage_index] = 0.0
        return output

    def component_cp_molar_J_mol_K(self) -> np.ndarray:
        return self._cp_molar.copy()

    def storage_scale(self) -> np.ndarray:
        curve = self.data["co2"]["reference_curve"]["loading_mol_kg"]
        capacity = max(float(value) for value in curve)
        return np.array(
            [
                capacity * self._fractions[0],
                capacity * self._fractions[1],
                self.data["water"]["solver_normalizer_mol_kg"],
                max(
                    self.data["nitrogen"][
                        "henry_mol_kg_Pa_at_reference"
                    ]
                    * 100000.0,
                    0.05,
                ),
                max(
                    self.data["oxygen"][
                        "henry_mol_kg_Pa_at_reference"
                    ]
                    * 21000.0,
                    0.05,
                ),
            ],
            dtype=float,
        )

    def source_report(self) -> dict:
        return {
            "adapter": "split_rate_reference_portable",
            "source_family": self.equilibrium_family,
            "co2_states": 2,
            "co2_reservoir_order": ["fast", "slow"],
            "co2_reservoir_fractions": self._fractions.tolist(),
            "co2_reference_curve_preserved": True,
            "shared_thermodynamic_curve": True,
            "separate_kinetic_inventories": True,
            "water_mode": "dry_only",
            "physical_water_storage_mol_kg": 0.0,
            "co2_support_policy": self.model.support_policy,
            "co2_support_scenario_name": self.model.support_scenario_name,
            "runtime_research_imports": False,
        }


@dataclass
class SiteResolvedAdapter(MaterialAdapter):
    """Two-site CO2 adapter with site-resolved inventories and energies."""

    data: Mapping
    storage_labels = ("CO2_site_1", "CO2_site_2", "H2O")
    species_for_storage = np.array([0, 0, 1], dtype=int)

    def __post_init__(self) -> None:
        self.material_id = str(self.data["material_id"])
        self.name = str(self.data["name"])
        self.equilibrium_family = str(self.data["equilibrium_family"])
        self.reference_temperature_K = float(self.data["reference_temperature_K"])
        self.maximum_bed_temperature_C = float(
            self.data["maximum_bed_temperature_C"]
        )
        self.solid_cp_J_kg_K = float(self.data["solid_cp_J_kg_K"])
        co2 = self.data["co2"]
        self._capacity = np.asarray(co2["site_capacity_mol_kg"], dtype=float)
        self._affinity_prefactor = np.asarray(
            co2["site_affinity_m3_mol"], dtype=float
        )
        self._site_u = np.asarray(co2["site_internal_energy_J_mol"], dtype=float)
        water = self.data["water"]
        self._water_measurement_T = float(water["measurement_temperature_K"])
        self._water_reference_T = float(water["reference_temperature_K"])
        self._water_heat = float(water["sorption_heat_J_mol"])
        points = np.asarray(water["points"], dtype=float)
        self._water_rh = points[:, 0]
        self._water_loading = points[:, 1]
        self._water_interpolator = PchipInterpolator(
            self._water_rh, self._water_loading, extrapolate=False
        )
        self._water_outside_calls = 0
        cp_j_kg = np.asarray(
            self.data["adsorbed_phase_cp_J_kg_K"], dtype=float
        )
        self._cp_molar = np.r_[
            MW[0] * cp_j_kg[0],
            MW[0] * cp_j_kg[1],
            MW[1] * cp_j_kg[2],
        ]
        self._u_reference = np.r_[self._site_u, -self._water_heat]
        self._ad_rate = np.asarray(
            self.data["kinetics"]["adsorption_rate_s_inv"], dtype=float
        )
        self._de_rate = np.asarray(
            self.data["kinetics"]["desorption_rate_s_inv"], dtype=float
        )

    def site_loadings(self, temperature_K: float, co2_pressure_Pa: float) -> np.ndarray:
        concentration = max(float(co2_pressure_Pa), 0.0) / (R * temperature_K)
        # Each site has its own exponent. Keep the operation explicit so a
        # common scalar affine cannot collapse the two source states.
        b = self._affinity_prefactor * np.exp(
            np.clip(-self._site_u / (R * temperature_K), -700.0, 700.0)
        )
        ratio = b * concentration
        return self._capacity * ratio / (1.0 + ratio)

    def _water_loading_at_reference_rh(self, rh: float) -> float:
        clipped = min(max(float(rh), 0.0), 1.0)
        if not math.isclose(clipped, rh, rel_tol=0.0, abs_tol=1e-14):
            self._water_outside_calls += 1
        return float(self._water_interpolator(clipped))

    def water_equilibrium(
        self, temperature_K: float, water_partial_Pa: float
    ) -> float:
        if water_partial_Pa <= 0.0:
            return 0.0
        reference_pressure = water_partial_Pa * math.exp(
            self._water_heat
            / R
            * (1.0 / temperature_K - 1.0 / self._water_measurement_T)
        )
        rh = reference_pressure / water_saturation_pressure_Pa(
            self._water_measurement_T
        )
        return self._water_loading_at_reference_rh(rh)

    def equilibrium(self, temperature_K: float, partials_Pa: np.ndarray) -> np.ndarray:
        sites = self.site_loadings(temperature_K, float(partials_Pa[0]))
        water = self.water_equilibrium(temperature_K, float(partials_Pa[1]))
        return np.r_[sites, water]

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self.equilibrium(temperature_K, partials_Pa)
        delta = target - np.asarray(storage, dtype=float)
        rates = np.where(delta >= 0.0, self._ad_rate, self._de_rate) * delta
        species_rates = np.array(
            [rates[0] + rates[1], rates[2], 0.0, 0.0], dtype=float
        )
        return rates, species_rates

    def absolute_energy_J_mol(self, temperature_K: float) -> np.ndarray:
        site_energy = self._site_u + (CP_GAS[0] - R) * (
            temperature_K - self.reference_temperature_K
        )
        water_energy = (
            (CP_GAS[1] - R) * (temperature_K - self.reference_temperature_K)
            + R * temperature_K
            - self._water_heat
        )
        return np.r_[site_energy, water_energy]

    def component_cp_molar_J_mol_K(self) -> np.ndarray:
        return self._cp_molar.copy()

    def storage_scale(self) -> np.ndarray:
        return np.r_[self._capacity, float(self._water_loading[-1])]

    def source_report(self) -> dict:
        return {
            "adapter": "site_resolved_concentration_langmuir_portable",
            "source_family": self.equilibrium_family,
            "co2_states": 2,
            "site_internal_energy_J_mol": self._site_u.tolist(),
            "caloric_closure": "site inventories and measured/reference-shifted water share one energy inventory",
            "water_mixture": "single-temperature PCHIP activity transfer; conditional water model",
            "water_outside_domain_calls": self._water_outside_calls,
            "runtime_research_imports": False,
        }


@dataclass
class Young2021WadstGabAdapter(MaterialAdapter):
    """One water inventory coupled into the Young2021 WADST CO2 target."""

    data: Mapping
    storage_labels = ("CO2", "H2O")
    species_for_storage = np.array([0, 1], dtype=int)

    def __post_init__(self) -> None:
        self.material_id = str(self.data["material_id"])
        self.name = str(self.data["name"])
        self.equilibrium_family = str(self.data["equilibrium_family"])
        self.reference_temperature_K = float(self.data["reference_temperature_K"])
        self.maximum_bed_temperature_C = float(
            self.data["maximum_bed_temperature_C"]
        )
        self.solid_cp_J_kg_K = float(self.data["solid_cp_J_kg_K"])
        self._heats_reference = np.array(
            [self.data["co2"]["heat_J_mol"], self.data["water"]["heat_J_mol"]],
            dtype=float,
        )
        # Native CP_GAS keeps adsorbate and gas bookkeeping on one convention.
        self._cp_molar = CP_GAS[:2].copy()
        self._ad_rate = np.asarray(
            self.data["kinetics"]["adsorption_rate_s_inv"], dtype=float
        )
        self._de_rate = np.asarray(
            self.data["kinetics"]["desorption_rate_s_inv"], dtype=float
        )
        self._local_rh_count = 0
        self._local_rh_min = math.inf
        self._local_rh_max = -math.inf

    def equilibrium(
        self, temperature_K: float, partials_Pa: np.ndarray
    ) -> np.ndarray:
        co2, water, details = young2021.equilibrium(
            float(temperature_K),
            float(partials_Pa[0]),
            float(partials_Pa[1]),
            reference_temperature_K=self.reference_temperature_K,
        )
        local_rh = float(details["local_relative_humidity"])
        self._local_rh_count += 1
        self._local_rh_min = min(self._local_rh_min, local_rh)
        self._local_rh_max = max(self._local_rh_max, local_rh)
        return np.array([co2, water], dtype=float)

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        loading = np.asarray(storage, dtype=float)
        if loading.shape != (2,):
            raise ValueError("Young2021 storage must contain CO2 and H2O")
        if not np.all(np.isfinite(loading)) or np.any(loading < -1e-9):
            raise ValueError("Young2021 loading must be finite and nonnegative")
        local_rh = young2021.local_relative_humidity(
            float(temperature_K), float(partials_Pa[1])
        )
        young2021.validate_domain(
            float(temperature_K), float(partials_Pa[0]), local_rh
        )
        water_target = young2021.gab_water_loading(
            local_rh, float(temperature_K)
        )
        co2_target, wet_weight = young2021.co2_loading_from_water(
            float(temperature_K),
            float(partials_Pa[0]),
            float(loading[1]),
            reference_temperature_K=self.reference_temperature_K,
        )
        delta = np.array(
            [co2_target - loading[0], water_target - loading[1]],
            dtype=float,
        )
        rates = np.where(
            delta >= 0.0, self._ad_rate, self._de_rate
        ) * delta
        self._local_rh_count += 1
        self._local_rh_min = min(self._local_rh_min, local_rh)
        self._local_rh_max = max(self._local_rh_max, local_rh)
        self._last_rate_details = {
            "co2_target_mol_kg": co2_target,
            "water_target_mol_kg": water_target,
            "wet_site_weight": wet_weight,
            "water_loading_source": "explicit evolving inventory",
        }
        species_rates = np.array(
            [rates[0], rates[1], 0.0, 0.0], dtype=float
        )
        return rates, species_rates

    def absolute_energy_J_mol(self, temperature_K: float) -> np.ndarray:
        return (
            self._cp_molar * (temperature_K - self.reference_temperature_K)
            + R * self.reference_temperature_K
            - self._heats_reference
        )

    def component_cp_molar_J_mol_K(self) -> np.ndarray:
        return self._cp_molar.copy()

    def storage_scale(self) -> np.ndarray:
        maximum_water = young2021.gab_water_loading(
            young2021.SUPPORTED_RELATIVE_HUMIDITY[1],
            young2021.SUPPORTED_TEMPERATURE_K[0],
        )
        return np.array(
            [young2021.YOUNG2021_WET.q_inf_0_mol_kg, maximum_water],
            dtype=float,
        )

    def source_report(self) -> dict:
        return {
            "adapter": "young2021_wadst_gab_conditional",
            "source_family": self.equilibrium_family,
            "source_doi": "10.1039/D1EE01272J",
            "co2_states": 1,
            "water_states": 1,
            "water_to_co2_coupling": True,
            "co2_to_water_coupling": False,
            "relative_humidity_basis": "local p_H2O / p_sat(T_bed)",
            "relative_humidity_use": (
                "continuous conditional interpolation within RH 0-0.55; "
                "30% and 55% are source conditions, not independent validation"
            ),
            "supported_temperature_K": list(young2021.SUPPORTED_TEMPERATURE_K),
            "supported_relative_humidity": list(
                young2021.SUPPORTED_RELATIVE_HUMIDITY
            ),
            "supported_co2_pressure_Pa": list(
                young2021.SUPPORTED_CO2_PRESSURE_PA
            ),
            "maximum_observed_local_relative_humidity": (
                self._local_rh_max if self._local_rh_count else None
            ),
            "minimum_observed_local_relative_humidity": (
                self._local_rh_min if self._local_rh_count else None
            ),
            "local_relative_humidity_evaluation_count": self._local_rh_count,
            "kinetics": {
                "equal_adsorption_desorption": bool(
                    np.array_equal(self._ad_rate, self._de_rate)
                ),
                "water_dependent_modulation": False,
                "humid_co2_transfer_validated": False,
                "hot_vacuum_transfer_validated": False,
            },
            "last_rate_details": getattr(self, "_last_rate_details", None),
            "caloric_closure": {
                "basis": "native constant CP_GAS with 70/46 kJ/mol process heats",
                "reciprocal_wadst_energy": False,
            },
            "runtime_research_imports": False,
        }


@dataclass
class Young2021HotWadstGabAdapter(Young2021WadstGabAdapter):
    """Conditional 25-120 C extrapolation with the shared spatial contract."""

    def equilibrium(
        self, temperature_K: float, partials_Pa: np.ndarray
    ) -> np.ndarray:
        co2, water, _details = young2021_hot.equilibrium(
            float(temperature_K),
            float(partials_Pa[0]),
            float(partials_Pa[1]),
            reference_temperature_K=self.reference_temperature_K,
        )
        local_rh = young2021_hot.local_relative_humidity(
            float(temperature_K), float(partials_Pa[1])
        )
        self._local_rh_count += 1
        self._local_rh_min = min(self._local_rh_min, local_rh)
        self._local_rh_max = max(self._local_rh_max, local_rh)
        return np.array([co2, water], dtype=float)

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        loading = np.asarray(storage, dtype=float)
        if loading.shape != (2,):
            raise ValueError("hot Young2021 storage must contain CO2 and H2O")
        if not np.all(np.isfinite(loading)) or np.any(loading < -1e-9):
            raise ValueError("hot Young2021 loading must be finite and nonnegative")
        local_rh = young2021_hot.local_relative_humidity(
            float(temperature_K), float(partials_Pa[1])
        )
        young2021_hot.validate_evaluation_envelope(
            float(temperature_K), float(partials_Pa[0]), local_rh
        )
        water_target = young2021_hot.gab_water_loading(
            local_rh, float(temperature_K)
        )
        co2_target, wet_weight = young2021.co2_loading_from_water(
            float(temperature_K),
            float(partials_Pa[0]),
            float(loading[1]),
            reference_temperature_K=self.reference_temperature_K,
        )
        delta = np.array(
            [co2_target - loading[0], water_target - loading[1]],
            dtype=float,
        )
        rates = np.where(
            delta >= 0.0, self._ad_rate, self._de_rate
        ) * delta
        self._local_rh_count += 1
        self._local_rh_min = min(self._local_rh_min, local_rh)
        self._local_rh_max = max(self._local_rh_max, local_rh)
        self._last_rate_details = {
            "co2_target_mol_kg": co2_target,
            "water_target_mol_kg": water_target,
            "wet_site_weight": wet_weight,
            "water_loading_source": "explicit evolving inventory",
            "classification": "unvalidated hot numerical extrapolation",
        }
        return rates, np.array(
            [rates[0], rates[1], 0.0, 0.0], dtype=float
        )

    def storage_scale(self) -> np.ndarray:
        maximum_water = young2021_hot.gab_water_loading(
            young2021_hot.EVALUATION_ENVELOPE_RELATIVE_HUMIDITY[1],
            young2021_hot.EVALUATION_ENVELOPE_TEMPERATURE_K[0],
        )
        return np.array(
            [young2021.YOUNG2021_WET.q_inf_0_mol_kg, maximum_water],
            dtype=float,
        )

    def temperature_probe_bounds_K(self) -> tuple[float, float]:
        """Numerical probes must stay within the declared evaluation envelope."""

        return tuple(young2021_hot.EVALUATION_ENVELOPE_TEMPERATURE_K)

    def source_report(self) -> dict:
        approved_rates = np.array(
            young2021_hot.APPROVED_LDF_RATES_S_INV, dtype=float
        )
        rates_unchanged = bool(
            np.array_equal(self._ad_rate, approved_rates)
            and np.array_equal(self._de_rate, approved_rates)
        )
        heats_unchanged = bool(
            tuple(float(value) for value in self._heats_reference)
            == tuple(young2021_hot.APPROVED_PROCESS_HEATS_J_MOL)
        )
        return {
            **super().source_report(),
            **young2021_hot.source_report(),
            "adapter": "young2021_hot_wadst_gab_extrapolated",
            "supported_temperature_K": list(
                young2021_hot.EVALUATION_ENVELOPE_TEMPERATURE_K
            ),
            "measured_coupled_support_temperature_K": list(
                young2021_hot.EVIDENCE_SUPPORT_TEMPERATURE_K
            ),
            "supported_relative_humidity": list(
                young2021_hot.EVALUATION_ENVELOPE_RELATIVE_HUMIDITY
            ),
            "relative_humidity_basis": (
                "local p_H2O / IAPWS p_sat(T_bed)"
            ),
            "evidence_support_status": (
                "measured Young coupled support ends at 70 C; "
                "the 70-120 C output is a conditional numerical evaluation"
            ),
            "original_strict_domain_unchanged": True,
            "temperature_probe_bounds_K": list(
                self.temperature_probe_bounds_K()
            ),
            "young_ldf_rates_unchanged": rates_unchanged,
            "young_process_heats_unchanged": heats_unchanged,
            "source_reproduction_complete": bool(
                rates_unchanged and heats_unchanged
            ),
            "lumped_engine_withheld": True,
            "runtime_research_imports": False,
        }


@dataclass
class Young2021ColdHotWadstGabAdapter(Young2021WadstGabAdapter):
    """Named 20-120 C conditional continuation with explicit evaluation rules."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._cold_hot_evaluation_count = 0
        self._water_raw_min_mol_kg = math.inf
        self._water_raw_max_mol_kg = -math.inf
        self._water_evaluation_time_min_mol_kg = math.inf
        self._water_evaluation_max_mol_kg = -math.inf
        self._water_lower_normalization_count = 0
        self._water_upper_normalization_count = 0
        self._domain_failure_context = None

    def _observe_evaluation(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        local_rh: float,
        water_loading: float,
        normalization: dict,
    ) -> None:
        self._cold_hot_evaluation_count += 1
        raw = float(normalization["raw_water_loading_mol_kg"])
        evaluated = float(
            normalization["evaluation_water_loading_mol_kg"]
        )
        self._water_raw_min_mol_kg = min(self._water_raw_min_mol_kg, raw)
        self._water_raw_max_mol_kg = max(self._water_raw_max_mol_kg, raw)
        self._water_evaluation_time_min_mol_kg = min(
            self._water_evaluation_time_min_mol_kg, evaluated
        )
        self._water_evaluation_max_mol_kg = max(
            self._water_evaluation_max_mol_kg, evaluated
        )
        self._water_lower_normalization_count += int(
            bool(normalization["lower_normalized"])
        )
        self._water_upper_normalization_count += int(
            bool(normalization["upper_normalized"])
        )
        self._local_rh_count += 1
        self._local_rh_min = min(self._local_rh_min, float(local_rh))
        self._local_rh_max = max(self._local_rh_max, float(local_rh))
        self._last_rate_details = {
            "temperature_K": float(temperature_K),
            "partials_Pa": np.asarray(partials_Pa, dtype=float).tolist(),
            "local_relative_humidity": float(local_rh),
            **normalization,
        }

    def _augment_domain_error(self, error: Exception, **context) -> None:
        diagnostics = getattr(error, "diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics = {
            **diagnostics,
            **context,
            "adapter": "young2021_cold_hot_wadst_gab_extrapolated",
            "last_rate_details": self._last_rate_details,
        }
        try:
            error.diagnostics = diagnostics
        except Exception:
            pass

    def equilibrium(
        self, temperature_K: float, partials_Pa: np.ndarray
    ) -> np.ndarray:
        try:
            co2, water, details = young2021_cold_hot.equilibrium(
                float(temperature_K),
                float(partials_Pa[0]),
                float(partials_Pa[1]),
                reference_temperature_K=self.reference_temperature_K,
            )
        except Exception as error:
            self._augment_domain_error(
                error,
                temperature_K=float(temperature_K),
                partials_Pa=np.asarray(
                    partials_Pa, dtype=float
                ).tolist(),
                call="equilibrium",
            )
            raise
        self._observe_evaluation(
            float(temperature_K),
            np.asarray(partials_Pa, dtype=float),
            float(details["local_relative_humidity"]),
            water,
            details["water_loading_normalization"],
        )
        return np.array([co2, water], dtype=float)

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        loading = np.asarray(storage, dtype=float)
        if loading.shape != (2,):
            raise ValueError(
                "cold+hot Young2021 storage must contain CO2 and H2O"
            )
        if not np.all(np.isfinite(loading)):
            raise ValueError(
                "cold+hot Young2021 loading must be finite"
            )
        try:
            local_rh = young2021_cold_hot.local_relative_humidity(
                float(temperature_K), float(partials_Pa[1])
            )
            young2021_cold_hot.validate_evaluation_envelope(
                float(temperature_K),
                float(partials_Pa[0]),
                local_rh,
            )
            water_target = young2021_cold_hot.gab_water_loading(
                local_rh, float(temperature_K)
            )
            co2_target, wet_weight, normalization = (
                young2021_cold_hot._co2_loading_from_water_evaluation(
                    float(temperature_K),
                    float(partials_Pa[0]),
                    float(loading[1]),
                    reference_temperature_K=self.reference_temperature_K,
                )
            )
        except Exception as error:
            self._augment_domain_error(
                error,
                temperature_K=float(temperature_K),
                partials_Pa=np.asarray(
                    partials_Pa, dtype=float
                ).tolist(),
                raw_storage_mol_kg=loading.tolist(),
                call="rates",
            )
            raise
        delta = np.array(
            [co2_target - loading[0], water_target - loading[1]],
            dtype=float,
        )
        rates = np.where(
            delta >= 0.0, self._ad_rate, self._de_rate
        ) * delta
        self._observe_evaluation(
            float(temperature_K),
            np.asarray(partials_Pa, dtype=float),
            local_rh,
            float(loading[1]),
            normalization,
        )
        self._last_rate_details.update(
            {
                "co2_target_mol_kg": co2_target,
                "water_target_mol_kg": water_target,
                "wet_site_weight": wet_weight,
                "raw_co2_delta_mol_kg": float(delta[0]),
                "raw_water_delta_mol_kg": float(delta[1]),
                "water_loading_source": "explicit evolving inventory",
                "classification": (
                    "named unvalidated cold-transient and hot numerical "
                    "extrapolation"
                ),
            }
        )
        return rates, np.array(
            [rates[0], rates[1], 0.0, 0.0], dtype=float
        )

    def storage_scale(self) -> np.ndarray:
        return np.array(
            [
                young2021.YOUNG2021_WET.q_inf_0_mol_kg,
                young2021_cold_hot.WATER_EVALUATION_CAP_MOL_KG,
            ],
            dtype=float,
        )

    def storage_probe_bounds_mol_kg(self) -> dict:
        tolerance = (
            young2021_cold_hot
            .WATER_LOADING_EVALUATION_TOLERANCE_MOL_KG
        )
        return {
            "lower": [
                -tolerance,
                -tolerance,
            ],
            "upper": [
                young2021.YOUNG2021_WET.q_inf_0_mol_kg,
                young2021_cold_hot.WATER_EVALUATION_CAP_MOL_KG
                + tolerance,
            ],
            "policy": (
                "component-aware deterministic probes; evaluation-only "
                "normalization does not mutate the state vector"
            ),
        }

    def temperature_probe_bounds_K(self) -> tuple[float, float]:
        return tuple(
            young2021_cold_hot
            .EVALUATION_ENVELOPE_TEMPERATURE_K
        )

    def record_failure_context(self, context: dict) -> None:
        self._domain_failure_context = dict(context)

    def failure_report(self) -> dict | None:
        if self._domain_failure_context is None:
            return None
        report = dict(self._domain_failure_context)
        vector = report.get(
            "state_vector", report.get("rejected_state_vector")
        )
        if isinstance(vector, list):
            encoded = json.dumps(
                vector,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            report["state_json_sha256"] = sha256(encoded).hexdigest()
        return report

    def source_report(self) -> dict:
        super_report = super().source_report()
        approved_rates = np.array(
            young2021_hot.APPROVED_LDF_RATES_S_INV, dtype=float
        )
        rates_unchanged = bool(
            np.array_equal(self._ad_rate, approved_rates)
            and np.array_equal(self._de_rate, approved_rates)
        )
        heats_unchanged = bool(
            tuple(float(value) for value in self._heats_reference)
            == tuple(young2021_hot.APPROVED_PROCESS_HEATS_J_MOL)
        )
        return {
            **super_report,
            **young2021_cold_hot.source_report(),
            "adapter": "young2021_cold_hot_wadst_gab_extrapolated",
            "supported_temperature_K": list(
                young2021_cold_hot
                .EVALUATION_ENVELOPE_TEMPERATURE_K
            ),
            "supported_relative_humidity": list(
                young2021_cold_hot
                .EVALUATION_ENVELOPE_RELATIVE_HUMIDITY
            ),
            "relative_humidity_basis": (
                "local p_H2O / IAPWS p_sat(T_bed)"
            ),
            "temperature_probe_bounds_K": list(
                self.temperature_probe_bounds_K()
            ),
            "storage_probe_bounds_mol_kg": (
                self.storage_probe_bounds_mol_kg()
            ),
            "young_ldf_rates_unchanged": rates_unchanged,
            "young_process_heats_unchanged": heats_unchanged,
            "source_reproduction_complete": bool(
                rates_unchanged and heats_unchanged
            ),
            "water_evaluation_observations": {
                "count": self._cold_hot_evaluation_count,
                "raw_min_mol_kg": (
                    None
                    if self._water_raw_min_mol_kg == math.inf
                    else self._water_raw_min_mol_kg
                ),
                "raw_max_mol_kg": (
                    None
                    if self._water_raw_max_mol_kg == -math.inf
                    else self._water_raw_max_mol_kg
                ),
                "evaluated_min_mol_kg": (
                    None
                    if self._water_evaluation_time_min_mol_kg == math.inf
                    else self._water_evaluation_time_min_mol_kg
                ),
                "evaluated_max_mol_kg": (
                    None
                    if self._water_evaluation_max_mol_kg == -math.inf
                    else self._water_evaluation_max_mol_kg
                ),
                "lower_normalization_count": (
                    self._water_lower_normalization_count
                ),
                "upper_normalization_count": (
                    self._water_upper_normalization_count
                ),
                "stored_inventory_mutated": False,
            },
            "lumped_engine_withheld": True,
            "runtime_research_imports": False,
        }


@dataclass
class Young2021AdvisoryWadstGabAdapter(Young2021WadstGabAdapter):
    """Young equations with recorded temperature excursions and branch checks."""

    def __post_init__(self) -> None:
        super().__post_init__()
        policy = self.data.get("water_evaluation_policy", {}).get(
            "numerical_tolerance_policy", {}
        )
        self._numerical_policy = policy.get("name", "strict")
        if self._numerical_policy not in {"strict", SCALE_AWARE_NUMERICAL_POLICY}:
            raise MofScreeningError(
                f"unsupported advisory numerical policy: {self._numerical_policy}",
                field="material.water_evaluation_policy.numerical_tolerance_policy",
            )
        self._numerical_warning_call_count = 0
        self._numerical_hard_rejection_count = 0
        self._numerical_raw_min_mol_kg = math.inf
        self._last_numerical_excursion = None
        self._bdf_recovered_trials = []
        self._bdf_recovery_summaries = []
        self._recovered_material_query_count = 0
        self._temperature_query_count = 0
        self._temperature_rate_query_count = 0
        self._temperature_equilibrium_query_count = 0
        self._temperature_query_min_K = math.inf
        self._temperature_query_max_K = -math.inf
        self._failed_material_query_count = 0
        self._last_material_query_error = None
        self._domain_failure_context = None
        self._advisory_evaluation_count = 0
        self._water_raw_min_mol_kg = math.inf
        self._water_raw_max_mol_kg = -math.inf
        self._water_lower_normalization_count = 0

    def _observe_query(
        self, kind: str, temperature_K: float, partials_Pa: np.ndarray
    ) -> None:
        self._temperature_query_count += 1
        if kind == "rates":
            self._temperature_rate_query_count += 1
        elif kind == "equilibrium":
            self._temperature_equilibrium_query_count += 1
        temperature = float(temperature_K)
        if math.isfinite(temperature):
            self._temperature_query_min_K = min(
                self._temperature_query_min_K, temperature
            )
            self._temperature_query_max_K = max(
                self._temperature_query_max_K, temperature
            )

    def _observe_evaluation(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        local_rh: float,
        raw_water_loading: float,
        normalization: dict,
    ) -> None:
        self._advisory_evaluation_count += 1
        self._water_raw_min_mol_kg = min(
            self._water_raw_min_mol_kg, float(raw_water_loading)
        )
        self._water_raw_max_mol_kg = max(
            self._water_raw_max_mol_kg, float(raw_water_loading)
        )
        self._water_lower_normalization_count += int(
            bool(normalization["lower_normalized"])
        )
        self._local_rh_count += 1
        self._local_rh_min = min(self._local_rh_min, float(local_rh))
        self._local_rh_max = max(self._local_rh_max, float(local_rh))
        self._last_rate_details = {
            "temperature_K": float(temperature_K),
            "partials_Pa": np.asarray(partials_Pa, dtype=float).tolist(),
            "local_relative_humidity": float(local_rh),
            **normalization,
        }

    def _augment_domain_error(
        self, error: Exception, *, kind: str, **context
    ) -> None:
        self._failed_material_query_count += 1
        self._last_material_query_error = str(error)
        diagnostics = getattr(error, "diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics = {
            **diagnostics,
            **context,
            "adapter": "young2021_advisory_wadst_gab",
            "call": kind,
            "temperature_advisory": True,
            "temperature_policy_stopped_query": False,
            "numerical_policy": self._numerical_policy_status(),
            "last_rate_details": getattr(self, "_last_rate_details", None),
        }
        try:
            error.diagnostics = diagnostics
        except Exception:
            pass

    def equilibrium(
        self, temperature_K: float, partials_Pa: np.ndarray
    ) -> np.ndarray:
        partials = np.asarray(partials_Pa, dtype=float)
        self._observe_query("equilibrium", temperature_K, partials)
        try:
            co2, water, details = (
                young2021_advisory_temperature.equilibrium(
                    float(temperature_K),
                    float(partials[0]),
                    float(partials[1]),
                    reference_temperature_K=self.reference_temperature_K,
                )
            )
        except Exception as error:
            self._augment_domain_error(
                error,
                kind="equilibrium",
                temperature_K=float(temperature_K),
                partials_Pa=partials.tolist(),
            )
            raise
        self._observe_evaluation(
            float(temperature_K),
            partials,
            float(details["local_relative_humidity"]),
            float(water),
            details["water_loading_normalization"],
        )
        return np.array([co2, water], dtype=float)

    def _numerical_policy_status(self) -> dict:
        scale_aware = (
            self._numerical_policy == SCALE_AWARE_NUMERICAL_POLICY
        )
        return {
            "schema": "advisory-numerical-tolerance-policy/1",
            "name": self._numerical_policy,
            "component_atol_mol_kg": list(
                SCALE_AWARE_COMPONENT_ATOL_MOL_KG
            ),
            "warning_tolerance_mol_kg": (
                -SCALE_AWARE_COMPONENT_ATOL_MOL_KG[0]
            ),
            "continuation_band_multiplier": (
                SCALE_AWARE_BAND_MULTIPLIER if scale_aware else 1.0
            ),
            "continuation_boundary_mol_kg": (
                -SCALE_AWARE_BAND_MULTIPLIER
                * SCALE_AWARE_COMPONENT_ATOL_MOL_KG[0]
                if scale_aware
                else -SCALE_AWARE_COMPONENT_ATOL_MOL_KG[0]
            ),
            "evaluation_floor_scope": (
                "constitutive evaluation argument only"
                if scale_aware
                else "sub-atol water normalization"
            ),
            "raw_state_inventory_rate_delta_energy_mutated": False,
            "warning_rate_call_count": self._numerical_warning_call_count,
            "hard_rejection_count": self._numerical_hard_rejection_count,
            "raw_negative_min_mol_kg": (
                None
                if self._numerical_raw_min_mol_kg == math.inf
                else self._numerical_raw_min_mol_kg
            ),
            "last_warning": getattr(
                self, "_last_numerical_excursion", None
            ),
        }

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        partials = np.asarray(partials_Pa, dtype=float)
        loading = np.asarray(storage, dtype=float)
        self._observe_query("rates", temperature_K, partials)
        if loading.shape != (2,):
            raise ValueError(
                "temperature-advisory Young2021 storage must contain CO2 and H2O"
            )
        if not np.all(np.isfinite(loading)):
            error = ValueError(
                "temperature-advisory Young2021 loading must be finite"
            )
            self._augment_domain_error(
                error,
                kind="rates",
                temperature_K=float(temperature_K),
                partials_Pa=partials.tolist(),
                raw_storage_mol_kg=loading.tolist(),
            )
            raise error
        strict_tolerance = (
            young2021_advisory_temperature
            .WATER_LOADING_NEGATIVE_NUMERICAL_TOLERANCE_MOL_KG
        )
        scale_aware = (
            self._numerical_policy == SCALE_AWARE_NUMERICAL_POLICY
        )
        continuation = (
            SCALE_AWARE_BAND_MULTIPLIER
            * SCALE_AWARE_COMPONENT_ATOL_MOL_KG[1]
            if scale_aware
            else strict_tolerance
        )
        negative_min = float(np.min(loading))
        if negative_min < 0.0:
            self._numerical_raw_min_mol_kg = min(
                self._numerical_raw_min_mol_kg, negative_min
            )
        warning_mask = np.logical_and(
            loading < -strict_tolerance,
            loading >= -continuation,
        )
        if scale_aware and np.any(warning_mask):
            self._numerical_warning_call_count += 1
            self._last_numerical_excursion = {
                "temperature_K": float(temperature_K),
                "raw_storage_mol_kg": loading.tolist(),
                "warning_components": [
                    SPECIES[index] for index in np.flatnonzero(warning_mask)
                ],
                "warning_boundary_mol_kg": -strict_tolerance,
                "continuation_boundary_mol_kg": -continuation,
                "continued": True,
            }
        if np.any(loading < -continuation):
            self._numerical_hard_rejection_count += 1
            error = ValueError(
                (
                    "temperature-advisory Young2021 storage components must be "
                    "nonnegative within the configured numerical continuation band"
                    if scale_aware
                    else (
                        "temperature-advisory Young2021 storage components must be "
                        "nonnegative within the 1e-9 mol/kg numerical tolerance"
                    )
                )
            )
            self._augment_domain_error(
                error,
                kind="rates",
                temperature_K=float(temperature_K),
                partials_Pa=partials.tolist(),
                raw_storage_mol_kg=loading.tolist(),
                storage_components=["CO2", "H2O"],
                negative_tolerance_mol_kg=-strict_tolerance,
                continuation_boundary_mol_kg=-continuation,
            )
            raise error
        try:
            evaluation_loading = (
                np.maximum(loading, 0.0) if scale_aware else loading
            )
            co2_target, _water_target, details = (
                young2021_advisory_temperature.equilibrium(
                    float(temperature_K),
                    float(partials[0]),
                    float(partials[1]),
                    reference_temperature_K=self.reference_temperature_K,
                    water_loading_mol_kg=float(evaluation_loading[1]),
                )
            )
            local_rh = float(details["local_relative_humidity"])
            water_target = young2021.gab_water_loading(
                local_rh, float(temperature_K)
            )
            delta = np.array(
                [co2_target - loading[0], water_target - loading[1]],
                dtype=float,
            )
            rates = np.where(
                delta >= 0.0, self._ad_rate, self._de_rate
            ) * delta
            if not np.all(np.isfinite(rates)):
                raise ValueError(
                    "temperature-advisory Young2021 rate is nonfinite"
                )
        except Exception as error:
            self._augment_domain_error(
                error,
                kind="rates",
                temperature_K=float(temperature_K),
                partials_Pa=partials.tolist(),
                raw_storage_mol_kg=loading.tolist(),
            )
            raise
        normalization = details["water_loading_normalization"]
        self._observe_evaluation(
            float(temperature_K),
            partials,
            local_rh,
            float(loading[1]),
            normalization,
        )
        self._last_rate_details.update(
            {
                "co2_target_mol_kg": float(co2_target),
                "water_target_mol_kg": float(water_target),
                "wet_site_weight": float(details["wet_site_weight"]),
                "raw_co2_delta_mol_kg": float(delta[0]),
                "raw_water_delta_mol_kg": float(delta[1]),
                "water_loading_source": "explicit evolving inventory",
                "evaluation_water_loading_mol_kg": float(
                    evaluation_loading[1]
                ),
                "numerical_tolerance_policy": (
                    self._numerical_policy_status()
                ),
                "classification": (
                    "temperature-advisory conditional extrapolation; "
                    "accepted state follows the configured equation path"
                ),
            }
        )
        return rates, np.array(
            [rates[0], rates[1], 0.0, 0.0], dtype=float
        )

    def storage_scale(self) -> np.ndarray:
        return np.array(
            [
                young2021.YOUNG2021_WET.q_inf_0_mol_kg,
                young2021_cold_hot.WATER_EVALUATION_CAP_MOL_KG,
            ],
            dtype=float,
        )

    def temperature_probe_bounds_K(self) -> tuple[float, float]:
        return tuple(
            young2021_advisory_temperature.IAPWS_TEMPERATURE_SUPPORT_K
        )

    def record_failure_context(self, context: dict) -> None:
        self._domain_failure_context = dict(context)

    def acknowledge_recovered_trial(self, context: dict) -> None:
        """Clear only the trial failure that the BDF wrapper recovered."""
        self._bdf_recovered_trials.append(dict(context))
        self._recovered_material_query_count += 1
        self._domain_failure_context = None
        self._last_material_query_error = None

    def record_bdf_recovery_summary(self, summary: dict) -> None:
        self._bdf_recovery_summaries.append(dict(summary))

    def failure_report(self) -> dict | None:
        if self._domain_failure_context is None:
            return None
        report = dict(self._domain_failure_context)
        vector = report.get(
            "state_vector", report.get("rejected_state_vector")
        )
        if isinstance(vector, list):
            encoded = json.dumps(
                vector,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            report["state_json_sha256"] = sha256(encoded).hexdigest()
        return report

    def source_report(self) -> dict:
        base = Young2021WadstGabAdapter.source_report(self)
        approved_rates = np.array(
            young2021_hot.APPROVED_LDF_RATES_S_INV, dtype=float
        )
        rates_unchanged = bool(
            np.array_equal(self._ad_rate, approved_rates)
            and np.array_equal(self._de_rate, approved_rates)
        )
        heats_unchanged = bool(
            tuple(float(value) for value in self._heats_reference)
            == tuple(young2021_hot.APPROVED_PROCESS_HEATS_J_MOL)
        )
        return {
            **base,
            **young2021_advisory_temperature.source_report(),
            "adapter": "young2021_advisory_wadst_gab",
            "supported_temperature_K": list(
                young2021_advisory_temperature
                .MEASURED_COUPLED_SUPPORT_TEMPERATURE_K
            ),
            "temperature_probe_bounds_K": list(
                self.temperature_probe_bounds_K()
            ),
            "young_ldf_rates_unchanged": rates_unchanged,
            "young_process_heats_unchanged": heats_unchanged,
            "source_reproduction_complete": bool(
                rates_unchanged and heats_unchanged
            ),
            "material_query_temperature_observations": {
                "basis": (
                    "all successful and attempted material calls, including "
                    "accepted ODE states and solver/Jacobian trial queries; "
                    "not the accepted-state peak"
                ),
                "count": self._temperature_query_count,
                "rates_call_count": self._temperature_rate_query_count,
                "equilibrium_call_count": (
                    self._temperature_equilibrium_query_count
                ),
                "minimum_finite_temperature_K": (
                    None
                    if self._temperature_query_min_K == math.inf
                    else self._temperature_query_min_K
                ),
                "maximum_finite_temperature_K": (
                    None
                    if self._temperature_query_max_K == -math.inf
                    else self._temperature_query_max_K
                ),
                "failed_query_count": self._failed_material_query_count,
                "recovered_query_count": (
                    self._recovered_material_query_count
                ),
                "terminal_failed_query_count": max(
                    self._failed_material_query_count
                    - self._recovered_material_query_count,
                    0,
                ),
                "last_error": self._last_material_query_error,
            },
            "advisory_evaluation_observations": {
                "count": self._advisory_evaluation_count,
                "raw_water_min_mol_kg": (
                    None
                    if self._water_raw_min_mol_kg == math.inf
                    else self._water_raw_min_mol_kg
                ),
                "raw_water_max_mol_kg": (
                    None
                    if self._water_raw_max_mol_kg == -math.inf
                    else self._water_raw_max_mol_kg
                ),
                "lower_normalization_count": (
                    self._water_lower_normalization_count
                ),
                "stored_inventory_mutated": False,
            },
            "last_rate_details": getattr(
                self, "_last_rate_details", None
            ),
            "numerical_tolerance_policy": (
                self._numerical_policy_status()
            ),
            "bdf_trial_recovery": {
                "declared_policy": self.data.get(
                    "water_evaluation_policy", {}
                ).get("bdf_trial_recovery"),
                "recovered_trial_count": len(
                    self._bdf_recovered_trials
                ),
                "recovered_trials": list(self._bdf_recovered_trials),
                "solver_summaries": list(self._bdf_recovery_summaries),
                "terminal_failure_context": (
                    None
                    if self._domain_failure_context is None
                    else "retained_current_terminal_context"
                ),
            },
            "water_loading_domain": [
                -(
                    SCALE_AWARE_BAND_MULTIPLIER
                    * SCALE_AWARE_COMPONENT_ATOL_MOL_KG[1]
                    if self._numerical_policy
                    == SCALE_AWARE_NUMERICAL_POLICY
                    else young2021_advisory_temperature
                    .WATER_LOADING_NEGATIVE_NUMERICAL_TOLERANCE_MOL_KG
                ),
                None,
            ],
            "lumped_engine_withheld": True,
            "runtime_research_imports": False,
        }


def build_material_adapter(data: Mapping) -> MaterialAdapter:
    extension = data.get("humidity_candidate")
    if isinstance(extension, Mapping):
        # The native dispatch must receive only the original dry contract.
        # The wrapper re-exposes the opt-in candidate identity.
        from .stored_water_shift import StoredWaterShiftAdapter

        from .contract import load_humidity_candidate_origin

        dry_data = load_humidity_candidate_origin(extension)
        base_adapter = build_material_adapter(dry_data)
        return StoredWaterShiftAdapter(
            package=data,
            extension=extension,
            base_adapter=base_adapter,
            dry_base_package=dry_data,
        )
    family = data.get("equilibrium_family")
    if family == "purolite_a110_rh_curves_gab":
        from .purolite import PuroliteAdapter
        return PuroliteAdapter(data)
    if family == "independent_langmuir_sips":
        return IndependentLangmuirSipsAdapter(data)
    if family == "site_resolved_concentration_langmuir":
        return SiteResolvedAdapter(data)
    if family == "activity_shift_reference_isotherm":
        return ActivityShiftReferenceAdapter(data)
    if family == "split_rate_activity_shift_reference_isotherm":
        return SplitRateReferenceAdapter(data)
    if family == "young2021_wadst_gab":
        return Young2021WadstGabAdapter(data)
    if family == "young2021_hot_wadst_gab_extrapolated":
        return Young2021HotWadstGabAdapter(data)
    if family == "young2021_cold_hot_wadst_gab_extrapolated":
        return Young2021ColdHotWadstGabAdapter(data)
    if family == "young2021_advisory_wadst_gab":
        return Young2021AdvisoryWadstGabAdapter(data)
    raise MofScreeningError(
        f"unsupported material adapter: {family!r}",
        field="material.equilibrium_family",
    )


def partial_pressures_Pa(
    pressure_Pa: float, dry_co2_ppm: float, water_mole_fraction: float
) -> np.ndarray:
    water = min(max(float(water_mole_fraction), 0.0), 1.0)
    dry = np.array(
        [
            dry_co2_ppm * 1e-6,
            0.0,
            (1.0 - dry_co2_ppm * 1e-6) * 0.7905,
            (1.0 - dry_co2_ppm * 1e-6) * 0.2095,
        ],
        dtype=float,
    )
    y = dry * (1.0 - water) + np.array([0.0, water, 0.0, 0.0])
    return y * pressure_Pa
