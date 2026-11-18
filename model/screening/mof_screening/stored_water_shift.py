"""Opt-in stored-water adapter for conditional humid screening candidates.

This module is intentionally self-contained. Candidate water nodes and
coefficients are materialized in their native packages; no runtime path reads
from the Research workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..common import water_saturation_pressure_Pa as antoine_saturation_pressure_Pa
from ..water_properties import saturation_pressure_Pa as iapws_saturation_pressure_Pa
from .contract import MofScreeningError


R = 8.31446261815324
MW_WATER = 0.01801528
TINY_NEGATIVE_WATER_MOL_KG = 1e-9


def saturation_pressure_Pa(temperature_K: float) -> float:
    """Use the retained Antoine law through 373 K and IAPWS above it."""
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("water saturation temperature must be finite and positive")
    if 255.9 <= temperature <= 373.0:
        return float(antoine_saturation_pressure_Pa(temperature))
    if 373.0 < temperature < 647.096:
        return float(iapws_saturation_pressure_Pa(temperature))
    raise ValueError(
        "water saturation is unsupported outside 255.9 K <= T < 647.096 K; "
        f"got {temperature:g} K"
    )


def _finite_nonnegative_vector(values: Any, *, length: int, field: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain finite numbers") from error
    if (
        vector.shape != (length,)
        or not np.all(np.isfinite(vector))
        or np.any(vector < 0.0)
    ):
        raise ValueError(
            f"{field} must be a finite, nonnegative vector of length {length}"
        )
    return vector


@dataclass(frozen=True)
class StoredWaterLaw:
    """Materialized unary water target and its evidence-bearing assumptions."""

    kind: str
    reference_temperature_K: float
    heat_J_mol: float
    adsorbed_cp_J_kg_K: float
    adsorption_rate_s_inv: float
    desorption_rate_s_inv: float
    evidence_class: str
    provenance: Mapping[str, Any]
    assumptions: tuple[str, ...]
    support_activity: tuple[float, float] | None
    extrapolation_policy: str | None
    anchor_activity: float | None = None
    anchor_loading_mol_kg: float | None = None
    saturation_mol_kg: float | None = None
    half_activity: float | None = None
    exponent: float | None = None
    activity_nodes: tuple[float, ...] = ()
    loading_nodes_mol_kg: tuple[float, ...] = ()
    temperatures_K: tuple[float, ...] = ()
    activity_node_rows: tuple[tuple[float, ...], ...] = ()
    loading_table_mol_kg: tuple[tuple[float, ...], ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StoredWaterLaw":
        kind = str(data["kind"])
        assumptions = tuple(
            str(value).strip()
            for value in data.get("assumptions", ())
            if str(value).strip()
        )
        provenance = data["provenance"]
        raw_activity_nodes = data.get("activity_nodes", ())
        activity_node_rows = (
            tuple(tuple(float(value) for value in row) for row in raw_activity_nodes)
            if raw_activity_nodes and isinstance(raw_activity_nodes[0], (list, tuple))
            else ()
        )
        return cls(
            kind=kind,
            reference_temperature_K=float(data["reference_temperature_K"]),
            heat_J_mol=float(data["heat_J_mol"]),
            adsorbed_cp_J_kg_K=float(data["adsorbed_cp_J_kg_K"]),
            adsorption_rate_s_inv=float(data["adsorption_rate_s_inv"]),
            desorption_rate_s_inv=float(data["desorption_rate_s_inv"]),
            evidence_class=str(data["evidence_class"]),
            provenance=provenance,
            assumptions=assumptions,
            support_activity=(
                tuple(float(value) for value in data["support_activity"])
                if data.get("support_activity") is not None
                else None
            ),
            extrapolation_policy=data.get("extrapolation_policy"),
            anchor_activity=(
                float(data["anchor_activity"])
                if data.get("anchor_activity") is not None
                else None
            ),
            anchor_loading_mol_kg=(
                float(data["anchor_loading_mol_kg"])
                if data.get("anchor_loading_mol_kg") is not None
                else None
            ),
            saturation_mol_kg=(
                float(data["saturation_mol_kg"])
                if data.get("saturation_mol_kg") is not None
                else None
            ),
            half_activity=(
                float(data["half_activity"])
                if data.get("half_activity") is not None
                else None
            ),
            exponent=(
                float(data["exponent"]) if data.get("exponent") is not None else None
            ),
            activity_nodes=tuple(
                float(value)
                for value in (() if activity_node_rows else raw_activity_nodes)
            ),
            loading_nodes_mol_kg=tuple(
                float(value) for value in data.get("loading_nodes_mol_kg", ())
            ),
            temperatures_K=tuple(
                float(value) for value in data.get("temperatures_K", ())
            ),
            activity_node_rows=activity_node_rows,
            loading_table_mol_kg=tuple(
                tuple(float(value) for value in row)
                for row in data.get("loading_table_mol_kg", ())
            ),
        )

    @property
    def storage_scale_mol_kg(self) -> float:
        if self.kind == "sips_anchor":
            assert self.saturation_mol_kg is not None
            return float(self.saturation_mol_kg)
        if self.kind == "linear_one_node":
            assert (
                self.anchor_activity is not None
                and self.anchor_loading_mol_kg is not None
            )
            return float(self.anchor_loading_mol_kg / self.anchor_activity)
        if self.kind == "pchip_nodes":
            return float(max(self.loading_nodes_mol_kg))
        if self.kind == "temperature_activity_table":
            return float(max(max(row) for row in self.loading_table_mol_kg))
        raise ValueError(f"unsupported stored-water law kind: {self.kind!r}")

    def _curve_loading(
        self, activity: float, activity_nodes: np.ndarray, loadings: np.ndarray
    ) -> float:
        lower = float(activity_nodes[0])
        upper = float(activity_nodes[-1])
        if lower <= activity <= upper:
            return float(
                PchipInterpolator(activity_nodes, loadings, extrapolate=False)(activity)
            )
        if activity == 0.0 and lower == 0.0:
            return float(loadings[0])
        if (
            activity > upper
            and bool(self.extrapolation_policy)
            and self.extrapolation_policy.startswith("linear_extension")
        ):
            slope = (loadings[-1] - loadings[-2]) / (
                activity_nodes[-1] - activity_nodes[-2]
            )
            return float(loadings[-1] + slope * (activity - upper))
        raise ValueError(
            f"{self.kind} water activity {activity:g} is outside support "
            f"[{lower:g}, {upper:g}]"
        )

    def loading(self, temperature_K: float, water_partial_Pa: float) -> float:
        temperature = float(temperature_K)
        pressure = float(water_partial_Pa)
        if (
            not math.isfinite(temperature)
            or temperature <= 0.0
            or not math.isfinite(pressure)
            or pressure < 0.0
        ):
            raise ValueError("water-law temperature and pressure must be finite")
        activity = pressure / saturation_pressure_Pa(temperature)
        if not 0.0 <= activity <= 1.0:
            raise ValueError(
                "water activity is outside the mathematical liquid-water "
                f"domain [0, 1]: {activity!r}"
            )
        if self.kind == "sips_anchor":
            assert (
                self.half_activity is not None
                and self.saturation_mol_kg is not None
                and self.exponent is not None
            )
            if activity == 0.0:
                value = 0.0
            else:
                ratio = (activity / self.half_activity) ** self.exponent
                value = self.saturation_mol_kg * ratio / (1.0 + ratio)
        elif self.kind == "linear_one_node":
            assert (
                self.anchor_activity is not None
                and self.anchor_loading_mol_kg is not None
            )
            value = self.anchor_loading_mol_kg / self.anchor_activity * activity
        elif self.kind == "pchip_nodes":
            value = self._curve_loading(
                activity,
                np.asarray(self.activity_nodes, dtype=float),
                np.asarray(self.loading_nodes_mol_kg, dtype=float),
            )
        elif self.kind == "temperature_activity_table":
            temperatures = np.asarray(self.temperatures_K, dtype=float)
            if temperature <= temperatures[0]:
                lower_index = upper_index = 0
                weight = 0.0
            elif temperature >= temperatures[-1]:
                lower_index = upper_index = len(temperatures) - 1
                weight = 0.0
            else:
                upper_index = int(
                    np.searchsorted(temperatures, temperature, side="right")
                )
                lower_index = upper_index - 1
                weight = (temperature - temperatures[lower_index]) / (
                    temperatures[upper_index] - temperatures[lower_index]
                )
            lower = self._curve_loading(
                activity,
                np.asarray(self.activity_node_rows[lower_index], dtype=float),
                np.asarray(self.loading_table_mol_kg[lower_index], dtype=float),
            )
            upper = self._curve_loading(
                activity,
                np.asarray(self.activity_node_rows[upper_index], dtype=float),
                np.asarray(self.loading_table_mol_kg[upper_index], dtype=float),
            )
            value = (1.0 - weight) * lower + weight * upper
        else:
            raise ValueError(f"unsupported stored-water law kind: {self.kind!r}")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("stored-water law produced a nonfinite or negative target")
        return float(value)

    def report(self) -> dict:
        return {
            "kind": self.kind,
            "evidence_class": self.evidence_class,
            "support_activity": (
                list(self.support_activity)
                if self.support_activity is not None
                else None
            ),
            "source_temperature_support_K": (
                list(self.temperatures_K)
                if self.temperatures_K
                else [self.reference_temperature_K]
            ),
            "extrapolation_policy": self.extrapolation_policy,
            "provenance": dict(self.provenance),
            "assumptions": list(self.assumptions),
            "materialized_at_runtime": True,
            "runtime_research_imports": False,
        }


@dataclass(frozen=True)
class ChemicalPotentialShift:
    """Stored-water-dependent shift of the dry CO2 effective pressure."""

    coefficients: tuple[float, ...]
    normalizer_mol_kg: float
    temperature_exponent: float
    reference_temperature_K: float
    calibration_status: str
    holdout_status: str
    assumptions: tuple[str, ...]
    provenance: Mapping[str, Any]

    def multiplier(self, temperature_K: float, water_loading_mol_kg: float) -> float:
        if (
            not math.isfinite(temperature_K)
            or temperature_K <= 0.0
            or not math.isfinite(water_loading_mol_kg)
            or water_loading_mol_kg < 0.0
        ):
            raise ValueError("chemical-potential shift state is invalid")
        fraction = water_loading_mol_kg / self.normalizer_mol_kg
        log_shift = sum(
            coefficient * fraction**power
            for power, coefficient in enumerate(self.coefficients, start=1)
        )
        log_shift *= (
            self.reference_temperature_K / temperature_K
        ) ** self.temperature_exponent
        if not math.isfinite(log_shift) or abs(log_shift) > 700.0:
            raise ValueError("chemical-potential shift is nonfinite or overflows")
        return float(math.exp(log_shift))


@dataclass(frozen=True)
class WaterActivatedTemperatureLangmuirCoupling:
    """Bounded apparent CO2 increment from stored water and temperature."""

    added_capacity_mol_kg: float
    water_normalizer_mol_kg: float
    activation_exponent: float
    half_pressure_Pa: float
    co2_heat_J_mol: float
    reference_temperature_K: float
    calibration_status: str
    holdout_status: str
    assumptions: tuple[str, ...]
    provenance: Mapping[str, Any]

    def added_loading(
        self,
        temperature_K: float,
        co2_partial_Pa: float,
        water_loading_mol_kg: float,
    ) -> float:
        if (
            not math.isfinite(temperature_K)
            or temperature_K <= 0.0
            or not math.isfinite(co2_partial_Pa)
            or co2_partial_Pa < 0.0
            or not math.isfinite(water_loading_mol_kg)
            or water_loading_mol_kg < 0.0
        ):
            raise ValueError("activated-capacity coupling state is invalid")
        fraction = water_loading_mol_kg / self.water_normalizer_mol_kg
        activation = 1.0 - math.exp(-(fraction**self.activation_exponent))
        if co2_partial_Pa == 0.0:
            return 0.0
        log_affinity = math.log(1.0 / self.half_pressure_Pa) + (
            self.co2_heat_J_mol
            / R
            * (1.0 / temperature_K - 1.0 / self.reference_temperature_K)
        )
        log_fraction_argument = log_affinity + math.log(co2_partial_Pa)
        if log_fraction_argument >= 0.0:
            inverse = math.exp(-log_fraction_argument)
            pressure_fraction = 1.0 / (1.0 + inverse)
        else:
            numerator = math.exp(log_fraction_argument)
            pressure_fraction = numerator / (1.0 + numerator)
        value = self.added_capacity_mol_kg * activation * pressure_fraction
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("activated-capacity coupling produced an invalid value")
        return float(value)


def _coupling_from_mapping(
    spec: Mapping[str, Any] | None,
) -> ChemicalPotentialShift | WaterActivatedTemperatureLangmuirCoupling | None:
    if spec is None:
        raise ValueError("co2_coupling must include an explicit decision")
    kind = str(spec["kind"])
    if kind == "unsupported":
        return None
    assumptions = tuple(str(value) for value in spec.get("assumptions", ()))
    if kind in {"water_fraction_quadratic", "water_fraction_linear"}:
        return ChemicalPotentialShift(
            coefficients=tuple(float(value) for value in spec["coefficients"]),
            normalizer_mol_kg=float(spec["normalizer_mol_kg"]),
            temperature_exponent=float(spec["temperature_exponent"]),
            reference_temperature_K=float(spec["reference_temperature_K"]),
            calibration_status=str(spec["calibration_status"]),
            holdout_status=str(spec["holdout_status"]),
            assumptions=assumptions,
            provenance=spec["provenance"],
        )
    if kind == "water_activated_temperature_langmuir_added_target":
        return WaterActivatedTemperatureLangmuirCoupling(
            added_capacity_mol_kg=float(spec["added_capacity_mol_kg"]),
            water_normalizer_mol_kg=float(spec["water_normalizer_mol_kg"]),
            activation_exponent=float(spec["activation_exponent"]),
            half_pressure_Pa=float(spec["half_pressure_Pa"]),
            co2_heat_J_mol=float(spec["co2_heat_J_mol"]),
            reference_temperature_K=float(spec["reference_temperature_K"]),
            calibration_status=str(spec["calibration_status"]),
            holdout_status=str(spec["holdout_status"]),
            assumptions=assumptions,
            provenance=spec["provenance"],
        )
    raise ValueError(f"unsupported CO2 coupling kind: {kind!r}")


class StoredWaterShiftAdapter:
    """Delegate dry physics while evolving an explicit stored-water inventory."""

    def __init__(
        self,
        *,
        package: Mapping[str, Any],
        extension: Mapping[str, Any],
        base_adapter: Any,
        dry_base_package: Mapping[str, Any],
    ) -> None:
        self.package = package
        self.data = dry_base_package
        self.extension = extension
        self.base_adapter = base_adapter
        self.material_id = str(package["material_id"])
        self.name = str(package["name"])
        self.equilibrium_family = str(base_adapter.equilibrium_family)
        self.reference_temperature_K = float(base_adapter.reference_temperature_K)
        self.maximum_bed_temperature_C = float(base_adapter.maximum_bed_temperature_C)
        self.solid_cp_J_kg_K = float(base_adapter.solid_cp_J_kg_K)
        self.storage_labels = tuple(base_adapter.storage_labels)
        self.species_for_storage = np.asarray(
            base_adapter.species_for_storage, dtype=int
        ).copy()
        self.species_labels = tuple(base_adapter.species_labels)
        self.water_index = self.storage_labels.index("H2O")
        self._negative_water_constitutive_calls = 0
        water_spec = extension["water_law"]
        self.water_law = (
            None
            if water_spec["kind"] == "native"
            else StoredWaterLaw.from_mapping(water_spec)
        )
        self.coupling_spec = extension["co2_coupling"]
        self.coupling = _coupling_from_mapping(self.coupling_spec)
        kinetics = self.data["kinetics"]
        self._rates_ad = _finite_nonnegative_vector(
            kinetics["adsorption_rate_s_inv"],
            length=self.n_storage,
            field="material.kinetics.adsorption_rate_s_inv",
        )
        self._rates_de = _finite_nonnegative_vector(
            kinetics["desorption_rate_s_inv"],
            length=self.n_storage,
            field="material.kinetics.desorption_rate_s_inv",
        )
        if tuple(extension["storage"]["labels"]) != self.storage_labels:
            raise MofScreeningError(
                "humidity_candidate.storage_labels differs from the dry "
                "adapter storage order",
                field="material.humidity_candidate.storage_labels",
            )

    @property
    def n_storage(self) -> int:
        return len(self.storage_labels)

    @property
    def n_sites(self) -> int:
        return int(sum(1 for species in self.species_for_storage if species == 0))

    @property
    def site_species(self) -> tuple[int, ...]:
        return (0,) * self.n_sites

    @property
    def maximum_bed_temperature_K(self) -> float:
        return self.maximum_bed_temperature_C + 273.15

    @property
    def water_scale_mol_kg(self) -> float:
        return float(self.storage_scale()[self.water_index])

    @property
    def water_storage_index(self) -> int:
        return self.water_index

    def _native_water_target(
        self, temperature_K: float, water_partial_Pa: float
    ) -> float:
        saturation = saturation_pressure_Pa(temperature_K)
        activity = float(water_partial_Pa) / saturation
        if not 0.0 <= activity <= 1.0:
            raise ValueError(
                "water activity is outside the mathematical liquid-water "
                f"domain [0, 1]: {activity!r}"
            )
        if hasattr(self.base_adapter, "_water_measurement_T"):
            measurement_temperature = float(self.base_adapter._water_measurement_T)
            water_heat = float(self.base_adapter._water_heat)
            reference_pressure = float(water_partial_Pa) * math.exp(
                water_heat
                / R
                * (1.0 / float(temperature_K) - 1.0 / measurement_temperature)
            )
            reference_activity = reference_pressure / saturation_pressure_Pa(
                measurement_temperature
            )
            if not 0.0 <= reference_activity <= 1.0:
                raise ValueError(
                    "water reference activity is outside the retained PCHIP "
                    f"support [0, 1]: {reference_activity!r}; "
                    f"actual activity={activity!r}, T={temperature_K:g} K"
                )
        if hasattr(self.base_adapter, "water_equilibrium"):
            return float(
                self.base_adapter.water_equilibrium(
                    float(temperature_K), float(water_partial_Pa)
                )
            )
        model = getattr(self.base_adapter, "model", None)
        if (
            model is not None
            and hasattr(model, "b_water_ref")
            and hasattr(model, "log_activity_shift")
        ):
            shift = float(model.log_activity_shift(float(temperature_K))[1])
            x_water = model.b_water_ref * math.exp(shift) * float(water_partial_Pa)
            if x_water <= 0.0:
                return 0.0
            exponent = float(self.data["water"]["sips_exponent"])
            log_ratio = exponent * math.log(x_water)
            fraction = 1.0 / (1.0 + math.exp(float(np.clip(-log_ratio, -700.0, 700.0))))
            return float(self.data["water"]["saturation_mol_kg"] * fraction)
        raise ValueError("native dry adapter has no independently evaluable water law")

    def _water_target(self, temperature_K: float, water_partial_Pa: float) -> float:
        if self.water_law is None:
            return self._native_water_target(temperature_K, water_partial_Pa)
        return self.water_law.loading(temperature_K, water_partial_Pa)

    def _target_from_coupling_loading(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        water_equilibrium_target: float,
        coupling_water_loading: float,
    ) -> np.ndarray:
        shifted = np.asarray(partials_Pa, dtype=float).copy()
        if (
            isinstance(self.coupling, ChemicalPotentialShift)
            and coupling_water_loading > 0.0
        ):
            shifted[0] *= self.coupling.multiplier(
                temperature_K, coupling_water_loading
            )
        if hasattr(self.base_adapter, "_water_measurement_T") and hasattr(
            self.base_adapter, "site_loadings"
        ):
            # CALF's dry nonwater target can be evaluated directly. Do not call
            # its independent water PCHIP merely to overwrite that target with
            # the candidate's materialized water law.
            target = np.asarray(
                self.base_adapter.site_loadings(temperature_K, float(shifted[0])),
                dtype=float,
            )
            target = np.r_[target, water_equilibrium_target]
        else:
            target = np.asarray(
                self.base_adapter.equilibrium(temperature_K, shifted),
                dtype=float,
            ).copy()
        if isinstance(
            self.coupling,
            WaterActivatedTemperatureLangmuirCoupling,
        ):
            target[0] += self.coupling.added_loading(
                temperature_K,
                float(partials_Pa[0]),
                coupling_water_loading,
            )
        target[self.water_index] = water_equilibrium_target
        if (
            target.shape != (self.n_storage,)
            or not np.all(np.isfinite(target))
            or np.any(target < -1e-12)
        ):
            raise ValueError(
                "humid target must be finite, nonnegative and in native storage order"
            )
        return target

    def equilibrium(self, temperature_K: float, partials_Pa: np.ndarray) -> np.ndarray:
        temperature = float(temperature_K)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_K must be finite and positive")
        partials = _finite_nonnegative_vector(
            partials_Pa, length=4, field="partial_pressures_Pa"
        )
        water_target = self._water_target(temperature, float(partials[1]))
        return self._target_from_coupling_loading(
            temperature,
            partials,
            water_target,
            water_target,
        )

    def site_loadings(
        self, temperature_K: float, partial_pressures_Pa: np.ndarray
    ) -> np.ndarray:
        return self.equilibrium(temperature_K, partial_pressures_Pa)[
            self.species_for_storage == 0
        ]

    def rates(
        self,
        temperature_K: float,
        partials_Pa: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        temperature = float(temperature_K)
        loadings = np.asarray(storage, dtype=float)
        if loadings.shape != (self.n_storage,) or not np.all(np.isfinite(loadings)):
            raise ValueError("storage loadings must be finite native-order values")
        partials = _finite_nonnegative_vector(
            partials_Pa, length=4, field="partial_pressures_Pa"
        )
        water_target = self._water_target(temperature, float(partials[1]))
        raw_water = float(loadings[self.water_index])
        if raw_water < -TINY_NEGATIVE_WATER_MOL_KG:
            raise ValueError(
                "stored water is below the explicit constitutive numerical "
                f"tolerance {-TINY_NEGATIVE_WATER_MOL_KG:g} mol/kg"
            )
        coupling_water = max(raw_water, 0.0)
        if raw_water < 0.0:
            self._negative_water_constitutive_calls += 1
        target = self._target_from_coupling_loading(
            temperature,
            partials,
            water_target,
            coupling_water,
        )
        delta = target - loadings
        rate = np.where(delta >= 0.0, self._rates_ad, self._rates_de) * delta
        if self.water_law is not None:
            water_rate = (
                self.water_law.adsorption_rate_s_inv
                if delta[self.water_index] >= 0.0
                else self.water_law.desorption_rate_s_inv
            )
            rate[self.water_index] = water_rate * delta[self.water_index]
        species_rates = np.zeros(4, dtype=float)
        for index, species in enumerate(self.species_for_storage):
            species_rates[int(species)] += rate[index]
        if not np.all(np.isfinite(rate)) or not np.all(np.isfinite(species_rates)):
            raise ValueError("material rate or species conservation is nonfinite")
        return rate, species_rates

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
        storage[self.water_index] = water_loading
        inert_indices = np.flatnonzero(np.isin(self.species_for_storage, (2, 3)))
        if inert_indices.size:
            storage[inert_indices] = inert_loading
        rates, _ = self.rates(temperature_K, partial_pressures_Pa, storage)
        output = np.zeros(self.n_sites + 3, dtype=float)
        output[: self.n_sites] = rates[self.species_for_storage == 0]
        output[self.n_sites] = rates[self.water_index]
        if inert_indices.size:
            output[self.n_sites + 1 :] = rates[inert_indices]
        return output

    def absolute_energy_J_mol(self, temperature_K: float) -> np.ndarray:
        temperature = float(temperature_K)
        energy = np.asarray(
            self.base_adapter.absolute_energy_J_mol(temperature),
            dtype=float,
        ).copy()
        if self.water_law is not None:
            cp_molar = MW_WATER * self.water_law.adsorbed_cp_J_kg_K
            energy[self.water_index] = (
                cp_molar * (temperature - self.water_law.reference_temperature_K)
                + R * self.water_law.reference_temperature_K
                - self.water_law.heat_J_mol
            )
        if energy.shape != (self.n_storage,) or not np.all(np.isfinite(energy)):
            raise ValueError("absolute adsorbed energy is nonfinite")
        return energy

    def absolute_energies_J_mol(self, temperature_K: float) -> np.ndarray:
        output = np.zeros(self.n_sites + 3, dtype=float)
        output[: self.n_storage] = self.absolute_energy_J_mol(temperature_K)
        return output

    def component_cp_molar_J_mol_K(self) -> np.ndarray:
        cp_molar = np.asarray(
            self.base_adapter.component_cp_molar_J_mol_K(), dtype=float
        ).copy()
        if self.water_law is not None:
            cp_molar[self.water_index] = MW_WATER * self.water_law.adsorbed_cp_J_kg_K
        return cp_molar

    def component_cp_J_mol_K(self) -> np.ndarray:
        output = np.zeros(self.n_sites + 3, dtype=float)
        output[: self.n_storage] = self.component_cp_molar_J_mol_K()
        return output

    def storage_energy_terms(
        self, temperature_K: float, storage: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        loading = np.asarray(storage, dtype=float)
        if loading.shape != (self.n_storage,) or not np.all(np.isfinite(loading)):
            raise ValueError("storage energy inputs must be finite")
        unit_energy = self.absolute_energy_J_mol(float(temperature_K))
        unit_cp = self.component_cp_molar_J_mol_K()
        return (
            float(loading @ unit_energy),
            float(loading @ unit_cp),
            unit_energy.copy(),
        )

    def storage_scale(self) -> np.ndarray:
        scale = np.asarray(self.base_adapter.storage_scale(), dtype=float).copy()
        if self.water_law is not None:
            scale[self.water_index] = self.water_law.storage_scale_mol_kg
        if (
            scale.shape != (self.n_storage,)
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("storage scale must be finite and positive")
        return scale

    def temperature_probe_bounds_K(self) -> tuple[float, float]:
        return (255.9, 647.096)

    def diagnostic_saturation_pressure_Pa(self, temperature_K: float) -> float:
        """Return the candidate law for accepted-state diagnostics only."""
        return saturation_pressure_Pa(temperature_K)

    @property
    def diagnostic_saturation_law(self) -> str:
        return "Antoine 255.9-373.0 K; IAPWS SR1-86(1992) " "373.0-647.096 K"

    @property
    def diagnostic_relative_humidity_basis(self) -> str:
        return (
            "p_H2O = n_H2O R T_gas / V_cell for spatial traces; "
            "p_H2O / candidate p_sat(T_bed)"
        )

    def _coupling_report(self) -> dict:
        if isinstance(self.coupling, ChemicalPotentialShift):
            return {
                "kind": "stored_water_dependent_effective_pressure_shift",
                "coefficients": list(self.coupling.coefficients),
                "normalizer_mol_kg": self.coupling.normalizer_mol_kg,
                "temperature_exponent": self.coupling.temperature_exponent,
                "calibration_status": self.coupling.calibration_status,
                "holdout_status": self.coupling.holdout_status,
                "provenance": dict(self.coupling.provenance),
                "assumptions": list(self.coupling.assumptions),
            }
        if isinstance(
            self.coupling,
            WaterActivatedTemperatureLangmuirCoupling,
        ):
            return {
                "kind": ("water_activated_temperature_langmuir_added_target"),
                "added_capacity_mol_kg": (self.coupling.added_capacity_mol_kg),
                "reference_added_target_mol_kg": self.coupling_spec[
                    "reference_added_target_mol_kg"
                ],
                "water_normalizer_mol_kg": (self.coupling.water_normalizer_mol_kg),
                "activation_exponent": (self.coupling.activation_exponent),
                "half_pressure_Pa": self.coupling.half_pressure_Pa,
                "co2_heat_J_mol": self.coupling.co2_heat_J_mol,
                "co2_heat_sensitivity_J_mol": list(
                    self.coupling_spec["co2_heat_sensitivity_J_mol"]
                ),
                "bounded_temperature_langmuir_affinity": True,
                "calibration_status": (self.coupling.calibration_status),
                "holdout_status": self.coupling.holdout_status,
                "provenance": dict(self.coupling.provenance),
                "assumptions": list(self.coupling.assumptions),
            }
        return {
            **dict(self.coupling_spec),
            "effective_coupling_active": False,
        }

    def source_report(self) -> dict:
        extension = self.extension
        return {
            "candidate_id": extension["candidate_id"],
            "material_id": self.material_id,
            "dry_origin_material_id": extension["dry_origin_material_id"],
            "dry_origin_package_sha256": extension["dry_origin_package_sha256"],
            "dry_origin_package_canonical_sha256": extension[
                "dry_origin_package_canonical_sha256"
            ],
            "evidence_tier": extension["evidence_tier"],
            "native_adapter": type(self.base_adapter).__name__,
            "native_base_source_report": self.base_adapter.source_report(),
            "storage_labels": list(self.storage_labels),
            "species_for_storage": self.species_for_storage.tolist(),
            "dry_limit": extension["dry_limit"]["definition"],
            "dry_limit_state": (
                "Exact native delegation only when stored water and external "
                "water partial pressure are both zero; stored water survives "
                "a dry gas sweep until desorption removes it."
            ),
            "rate_coupling_basis": (
                (
                    "actual stored-water inventory controls the one-way CO2 "
                    "coupling; the gas-derived water target controls only the "
                    "water-rate target"
                )
                if self.coupling is not None
                else (
                    "No water-to-CO2 response is modeled. "
                    "Stored water and the gas-derived water target affect "
                    "their own storage and transport terms only."
                )
            ),
            "native_water_clip_policy": (
                "Existing native water laws are evaluated in their reference "
                "activity coordinate before delegation. Out-of-support input "
                "is rejected rather than silently clipped."
            ),
            "tiny_negative_water_constitutive_tolerance_mol_kg": (
                -TINY_NEGATIVE_WATER_MOL_KG
            ),
            "negative_water_constitutive_calls": (
                self._negative_water_constitutive_calls
            ),
            "raw_state_inventory_rate_delta_energy_mutated": False,
            "water": (
                {
                    "kind": "native",
                    "source_report": self.base_adapter.source_report(),
                }
                if self.water_law is None
                else self.water_law.report()
            ),
            "co2_coupling": self._coupling_report(),
            "co2_coupling_explicit": True,
            "saturation_pressure_branching": {
                "ambient_cold": (
                    "screening.common NIST/Stull Antoine, " "255.9-373.0 K"
                ),
                "hot": "IAPWS SR1-86(1992), 373.0-647.096 K",
                "below_255_9_K": "hard unsupported domain",
            },
            "reciprocal_thermodynamics": bool(extension["reciprocal_thermodynamics"]),
            "validated_humid_cycle": bool(extension["validated_humid_cycle"]),
            "hot_co2_rich_vacuum_isotherm_closed": False,
            "assumptions": list(extension["assumptions"]),
            "limitations": list(extension["limitations"]),
            "provenance": dict(extension["provenance"]),
            "runtime_research_imports": False,
        }
