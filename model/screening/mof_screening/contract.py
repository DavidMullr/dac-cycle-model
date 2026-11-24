"""Shared material, weather, equipment, cycle, and result contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..material_models import NAMED_EXTRAPOLATION_POLICY, STRICT_POLICY
from .young2021_hot import (
    APPROVED_LDF_RATES_S_INV,
    APPROVED_PROCESS_HEATS_J_MOL,
)
from .windows_io import read_text_with_retry


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "config" / "mof_screening"
MATERIAL_ROOT = ROOT / "materials" / "screening"
WEATHER_PATH = MATERIAL_ROOT / "pully_hourly_2023_2025_normalized.csv"
TEMPERATURE_REGISTRY_PATH = CONFIG_ROOT / "temperature_registry.json"

MATERIAL_SCHEMA = "mof-screening-material/1"
CASE_SCHEMA = "mof-screening-case/1"
PREVIEW_SCHEMA = "mof-screening-preview/1"
RESULT_SCHEMA = "mof-screening-result/1"
RUNTIME_POLICY_SCHEMA = "mof-screening-runtime-policy/1"
HUMIDITY_CANDIDATE_SCHEMA = "mof-screening-humidity-candidate/1"

SPATIAL = "spatial"
LUMPED = "lumped"
FIXED_WEATHER = "fixed"
OFFICIAL_WEATHER = "official"
MANUAL_STOP = "manual_stop"
TIME_LIMIT = "time_limit"
TEMPERATURE_POLICY_SCHEMA = "mof-screening-temperature-policy/1"
STOP_AT_SCENARIO_GUARD = "stop_at_scenario_guard"
RECORD_SCENARIO_EXCURSIONS = "record_scenario_excursions"

SUPPORTED_FAMILIES = (
    "purolite_a110_rh_curves_gab",
    "independent_langmuir_sips",
    "activity_shift_reference_isotherm",
    "split_rate_activity_shift_reference_isotherm",
    "site_resolved_concentration_langmuir",
    "young2021_wadst_gab",
    "young2021_hot_wadst_gab_extrapolated",
    "young2021_cold_hot_wadst_gab_extrapolated",
    "young2021_advisory_wadst_gab",
)

COMPATIBLE_EXTRAPOLATION_POLICY = "named_extrapolation"
SUPPORTED_REFERENCE_SUPPORT_POLICIES = (
    STRICT_POLICY,
    NAMED_EXTRAPOLATION_POLICY,
    COMPATIBLE_EXTRAPOLATION_POLICY,
)

SHARED_ACTUATOR_POLICY = {
    "schema": "mof-screening-shared-operating-policy/1",
    "source": (
        "Reviewed portable lumped phase requests, mapped to spatial wall and "
        "heater nodes with finite transfer."
    ),
    "phases": {
        "adsorption": {
            "heater": "enabled",
            "cooler": "enabled",
            "target": "current ambient temperature",
            "request": "setpoint",
            "gain_W_K": 1000.0,
        },
        "evacuation": {
            "heater": "enabled",
            "cooler": "enabled",
            "target": None,
            "request": "zero_net_bed_energy_imbalance",
            "gain_W_K": None,
        },
        "desorption": {
            "heater": "enabled",
            "cooler": "disabled",
            "target": "regeneration temperature",
            "request": "positive_setpoint",
            "gain_W_K": 5000.0,
        },
        "cooling": {
            "heater": "disabled",
            "cooler": "enabled",
            "target": "current ambient temperature",
            "request": "positive_sensible_cooling",
            "gain_W_K": 1000.0,
        },
        "repressurization": {
            "heater": "enabled",
            "cooler": "enabled",
            "target": "current ambient temperature",
            "request": "setpoint",
            "gain_W_K": 1000.0,
        },
    },
    "hardware": {
        "heater_power_W": 2000.0,
        "combined_cooling_power_W": 2000.0,
        "cooling_cop": 3.0,
        "pump_running_input_W": 135.0,
        "conditioner_target_C": 55.0,
    },
    "conditioner": {
        "phase_availability": ["evacuation", "desorption", "cooling"],
        "policy": "pump_inlet_cooler_knockout_v1",
        "allocation": "conditioner_first_then_bed",
        "pressure_drop_Pa": 0.0,
        "line_holdup_mol": 0.0,
    },
    "representation_difference": {
        "spatial": (
            "phase requests act through finite wall/heater transfer and local "
            "through-thickness states"
        ),
        "lumped": "phase requests act on one well-mixed bed temperature",
        "pressure_control": (
            "spatial retains its local-pressure valve control; lumped retains "
            "the reviewed target-inlet pressure control"
        ),
    },
}


class MofScreeningError(ValueError):
    """A user-facing contract, scenario, or execution error."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        field: str | None = None,
        diagnostics: dict | None = None,
    ):
        super().__init__(message)
        self.status = int(status)
        self.field = field
        self.diagnostics = diagnostics


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json_strict(path: str | Path) -> Any:
    def reject(token: str):
        raise MofScreeningError(
            f"{Path(path).name}: non-finite JSON value {token} is not allowed",
            field="material",
        )

    try:
        return json.loads(
            read_text_with_retry(path),
            parse_constant=reject,
        )
    except FileNotFoundError as error:
        raise MofScreeningError(
            f"file not found: {Path(path).name}", status=404
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MofScreeningError(f"{Path(path).name}: invalid JSON: {error}") from error


def _mapping(value: Any, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise MofScreeningError(f"{field} must be an object", field=field)
    return value


def _string(mapping: Mapping, key: str, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MofScreeningError(
            f"{field}.{key} is required and must be text", field=f"{field}.{key}"
        )
    return value


def _number(
    mapping: Mapping,
    key: str,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MofScreeningError(
            f"{field}.{key} is required and must be a finite number",
            field=f"{field}.{key}",
        )
    number = float(value)
    if not math.isfinite(number):
        raise MofScreeningError(f"{field}.{key} must be finite", field=f"{field}.{key}")
    if minimum is not None:
        invalid = number <= minimum if exclusive_minimum else number < minimum
        if invalid:
            relation = ">" if exclusive_minimum else ">="
            raise MofScreeningError(
                f"{field}.{key} must be {relation} {minimum:g}",
                field=f"{field}.{key}",
            )
    if maximum is not None and number > maximum:
        raise MofScreeningError(
            f"{field}.{key} must be <= {maximum:g}", field=f"{field}.{key}"
        )
    return number


def _integer(
    mapping: Mapping,
    key: str,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MofScreeningError(
            f"{field}.{key} must be an integer", field=f"{field}.{key}"
        )
    if not minimum <= value <= maximum:
        raise MofScreeningError(
            f"{field}.{key} must be between {minimum} and {maximum}",
            field=f"{field}.{key}",
        )
    return value


def _boolean(mapping: Mapping, key: str, field: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise MofScreeningError(
            f"{field}.{key} must be a JSON boolean", field=f"{field}.{key}"
        )
    return value


def _string_list(
    mapping: Mapping,
    key: str,
    field: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a nonempty list"
        raise MofScreeningError(
            f"{field}.{key} must be {qualifier} of text",
            field=f"{field}.{key}",
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise MofScreeningError(
            f"{field}.{key} must contain only nonempty text",
            field=f"{field}.{key}",
        )
    return [item.strip() for item in value]


def _finite_number_list(
    value: Any,
    field: str,
    *,
    minimum_length: int,
) -> list[float]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise MofScreeningError(
            f"{field} must be a list with at least {minimum_length} numbers",
            field=field,
        )
    numbers = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MofScreeningError(
                f"{field}[{index}] must be a finite number",
                field=f"{field}[{index}]",
            )
        number = float(item)
        if not math.isfinite(number):
            raise MofScreeningError(
                f"{field}[{index}] must be finite",
                field=f"{field}[{index}]",
            )
        numbers.append(number)
    return numbers


def is_humidity_candidate(material: Mapping) -> bool:
    extension = material.get("humidity_candidate")
    return bool(
        isinstance(extension, Mapping)
        and extension.get("schema") == HUMIDITY_CANDIDATE_SCHEMA
    )


def _humidity_origin_path(dry_origin: str) -> Path:
    if dry_origin in {".", ".."} or any(char in dry_origin for char in ("/", "\\", ":")):
        raise MofScreeningError(
            "invalid dry origin material id",
            field="material.humidity_candidate.dry_origin_material_id",
        )
    origin_stems = {
        "calf20_nguyen2024_80_20_shaped": "calf20_shaped",
        "hypothetical_mmen_like": "hypothetical_mmen_like",
        "synthetic_independent_demo": "synthetic_independent_demo",
    }
    name = f"{origin_stems.get(dry_origin, dry_origin)}.json"
    active = MATERIAL_ROOT / name
    if active.is_file():
        return active
    return MATERIAL_ROOT / "source_evidence" / name


def _candidate_scientific_base_view(material: Mapping) -> dict:
    """Return dry scientific fields, excluding candidate annotations."""
    excluded = {
        "material_id",
        "name",
        "evidence",
        "provenance",
        "humidity_candidate",
    }
    return {key: value for key, value in material.items() if key not in excluded}


def load_humidity_candidate_origin(extension: Mapping) -> dict:
    """Load and validate the exact dry package named by an extension."""
    dry_origin = _string(
        extension,
        "dry_origin_material_id",
        "humidity_candidate",
    )
    path = _humidity_origin_path(dry_origin)
    if not path.is_file():
        raise MofScreeningError(
            "humidity_candidate dry origin package is missing",
            field="material.humidity_candidate.dry_origin_material_id",
        )
    return validate_material(read_json_strict(path))


def validate_humidity_candidate_extension(
    material: Mapping,
    *,
    require_origin_files: bool = True,
) -> dict:
    """Validate the opt-in extension without changing the dry package law."""
    extension = deepcopy(
        dict(
            _mapping(material.get("humidity_candidate"), "material.humidity_candidate")
        )
    )
    field = "material.humidity_candidate"
    if extension.get("schema") != HUMIDITY_CANDIDATE_SCHEMA:
        raise MofScreeningError(
            f"{field}.schema must be {HUMIDITY_CANDIDATE_SCHEMA!r}",
            field=f"{field}.schema",
        )
    for key in (
        "candidate_id",
        "evidence_tier",
        "dry_origin_material_id",
        "dry_origin_package_sha256",
        "dry_origin_package_canonical_sha256",
    ):
        _string(extension, key, field)
    material_id = _string(material, "material_id", "material")
    dry_origin = str(extension["dry_origin_material_id"])
    if material_id == dry_origin:
        raise MofScreeningError(
            "a humidity candidate requires a distinct material_id",
            field=f"{field}.dry_origin_material_id",
        )
    if extension["reciprocal_thermodynamics"] is not False:
        raise MofScreeningError(
            "humidity candidates in this release require "
            "reciprocal_thermodynamics=false",
            field=f"{field}.reciprocal_thermodynamics",
        )
    if extension["validated_humid_cycle"] is not False:
        raise MofScreeningError(
            "humidity candidates in this release require "
            "validated_humid_cycle=false",
            field=f"{field}.validated_humid_cycle",
        )
    origin_path = _humidity_origin_path(dry_origin)
    if require_origin_files:
        if not origin_path.is_file():
            raise MofScreeningError(
                "humidity_candidate dry origin package is missing",
                field=f"{field}.dry_origin_material_id",
            )
        if file_sha256(origin_path) != extension["dry_origin_package_sha256"]:
            raise MofScreeningError(
                "humidity_candidate dry origin raw hash differs",
                field=f"{field}.dry_origin_package_sha256",
            )
        origin = read_json_strict(origin_path)
        if canonical_sha256(origin) != extension["dry_origin_package_canonical_sha256"]:
            raise MofScreeningError(
                "humidity_candidate dry origin canonical hash differs",
                field=f"{field}.dry_origin_package_canonical_sha256",
            )
        candidate_base = _candidate_scientific_base_view(material)
        origin_base = _candidate_scientific_base_view(origin)
        if semantic_json_sha256(candidate_base) != semantic_json_sha256(origin_base):
            raise MofScreeningError(
                "humidity_candidate scientific fields differ from the exact "
                "bound dry origin; only candidate identity, evidence and "
                "provenance annotations may differ",
                field="material.humidity_candidate.dry_origin_material_id",
            )
    dry_limit = _mapping(extension.get("dry_limit"), f"{field}.dry_limit")
    if dry_limit.get("definition") != (
        "EXACT_NATIVE_DELEGATION_REQUIRES_ZERO_STORED_AND_EXTERNAL_WATER"
    ):
        raise MofScreeningError(
            "dry_limit.definition is not the reviewed exact-delegation contract",
            field=f"{field}.dry_limit.definition",
        )
    if (
        _number(
            dry_limit,
            "stored_water_mol_kg",
            f"{field}.dry_limit",
            minimum=0.0,
        )
        != 0.0
    ):
        raise MofScreeningError(
            "dry_limit.stored_water_mol_kg must be exactly zero",
            field=f"{field}.dry_limit.stored_water_mol_kg",
        )
    if (
        _number(
            dry_limit,
            "external_water_partial_pressure_Pa",
            f"{field}.dry_limit",
            minimum=0.0,
        )
        != 0.0
    ):
        raise MofScreeningError(
            "dry_limit.external_water_partial_pressure_Pa must be exactly zero",
            field=(f"{field}.dry_limit.external_water_partial_pressure_Pa"),
        )

    storage = _mapping(extension.get("storage"), f"{field}.storage")
    labels = storage.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) < 3
        or any(not isinstance(item, str) or not item for item in labels)
        or len(set(labels)) != len(labels)
    ):
        raise MofScreeningError(
            "storage.labels must be a unique nonempty native-order string list",
            field=f"{field}.storage.labels",
        )
    _string(storage, "contract", f"{field}.storage")
    family = _string(material, "equilibrium_family", "material")
    expected_labels = (
        ("CO2_fast", "CO2_slow", "H2O", "N2", "O2")
        if family == "split_rate_activity_shift_reference_isotherm"
        else (
            ("CO2_site_1", "CO2_site_2", "H2O")
            if family == "site_resolved_concentration_langmuir"
            else ("CO2", "H2O", "N2", "O2")
        )
    )
    if tuple(labels) != expected_labels:
        raise MofScreeningError(
            "storage.labels must exactly match the native family order",
            field=f"{field}.storage.labels",
        )
    if storage["contract"] != ", ".join(expected_labels):
        raise MofScreeningError(
            "storage.contract must exactly describe native storage labels",
            field=f"{field}.storage.contract",
        )

    water = _mapping(extension.get("water_law"), f"{field}.water_law")
    water_kind = _string(water, "kind", f"{field}.water_law")
    if water_kind not in {
        "native",
        "sips_anchor",
        "linear_one_node",
        "pchip_nodes",
        "temperature_activity_table",
    }:
        raise MofScreeningError(
            "water_law.kind is unsupported",
            field=f"{field}.water_law.kind",
        )
    _string(water, "evidence_class", f"{field}.water_law")
    _string_list(water, "assumptions", f"{field}.water_law")
    if not isinstance(water.get("provenance"), Mapping) or not water["provenance"]:
        raise MofScreeningError(
            "water_law.provenance must be a nonempty object",
            field=f"{field}.water_law.provenance",
        )
    if water_kind != "native":
        for key in (
            "reference_temperature_K",
            "heat_J_mol",
            "adsorbed_cp_J_kg_K",
            "adsorption_rate_s_inv",
            "desorption_rate_s_inv",
        ):
            _number(
                water,
                key,
                f"{field}.water_law",
                minimum=250.0 if key == "reference_temperature_K" else 0.0,
                exclusive_minimum=key
                in {
                    "reference_temperature_K",
                    "adsorbed_cp_J_kg_K",
                    "adsorption_rate_s_inv",
                    "desorption_rate_s_inv",
                },
            )
        support = water.get("support_activity")
        if (
            not isinstance(support, list)
            or len(support) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in support
            )
            or not 0.0 <= float(support[0]) < float(support[1]) <= 1.0
        ):
            raise MofScreeningError(
                "water_law.support_activity must be an increasing [low, high] "
                "pair within [0,1]",
                field=f"{field}.water_law.support_activity",
            )
        if water_kind == "sips_anchor":
            anchor = _number(
                water,
                "anchor_activity",
                f"{field}.water_law",
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            )
            loading = _number(
                water,
                "anchor_loading_mol_kg",
                f"{field}.water_law",
                minimum=0.0,
                exclusive_minimum=True,
            )
            saturation = _number(
                water,
                "saturation_mol_kg",
                f"{field}.water_law",
                minimum=loading,
                exclusive_minimum=True,
            )
            _number(
                water,
                "half_activity",
                f"{field}.water_law",
                minimum=0.0,
                exclusive_minimum=True,
            )
            _number(
                water,
                "exponent",
                f"{field}.water_law",
                minimum=0.0,
                exclusive_minimum=True,
            )
            expected_half = anchor * (saturation / loading - 1.0)
            if not math.isclose(
                float(water["half_activity"]),
                expected_half,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise MofScreeningError(
                    "water_law.half_activity does not match the retained anchor",
                    field=f"{field}.water_law.half_activity",
                )
        elif water_kind == "linear_one_node":
            _number(
                water,
                "anchor_activity",
                f"{field}.water_law",
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            )
            _number(
                water,
                "anchor_loading_mol_kg",
                f"{field}.water_law",
                minimum=0.0,
                exclusive_minimum=True,
            )
        elif water_kind == "pchip_nodes":
            activities = _finite_number_list(
                water.get("activity_nodes"),
                f"{field}.water_law.activity_nodes",
                minimum_length=3,
            )
            loadings = _finite_number_list(
                water.get("loading_nodes_mol_kg"),
                f"{field}.water_law.loading_nodes_mol_kg",
                minimum_length=3,
            )
            if (
                len(activities) != len(loadings)
                or activities[0] != 0.0
                or activities[-1] > 1.0
                or any(b <= a for a, b in zip(activities, activities[1:]))
                or any(value < 0.0 for value in loadings)
                or max(loadings) <= 0.0
            ):
                raise MofScreeningError(
                    "pchip_nodes must define sorted activity nodes starting at "
                    "zero with nonnegative loadings",
                    field=f"{field}.water_law.activity_nodes",
                )
            _string(water, "extrapolation_policy", f"{field}.water_law")
        else:
            temperatures = _finite_number_list(
                water.get("temperatures_K"),
                f"{field}.water_law.temperatures_K",
                minimum_length=2,
            )
            activity_rows = water.get("activity_nodes")
            table = water.get("loading_table_mol_kg")
            if (
                not isinstance(activity_rows, list)
                or not isinstance(table, list)
                or len(temperatures) != len(table)
                or any(b <= a for a, b in zip(temperatures, temperatures[1:]))
                or len(activity_rows) != len(temperatures)
            ):
                raise MofScreeningError(
                    "temperature_activity_table has incompatible temperature, "
                    "activity or row dimensions",
                    field=f"{field}.water_law.loading_table_mol_kg",
                )
            maximum = 0.0
            for row_index, row in enumerate(table):
                activities = _finite_number_list(
                    activity_rows[row_index],
                    (f"{field}.water_law.activity_nodes" f"[{row_index}]"),
                    minimum_length=3,
                )
                loadings = _finite_number_list(
                    row,
                    f"{field}.water_law.loading_table_mol_kg[{row_index}]",
                    minimum_length=3,
                )
                if (
                    len(loadings) != len(activities)
                    or activities[0] != 0.0
                    or activities[-1] > 1.0
                    or any(b <= a for a, b in zip(activities, activities[1:]))
                    or any(value < 0.0 for value in loadings)
                ):
                    raise MofScreeningError(
                        "temperature table rows must match activity nodes and "
                        "contain nonnegative loadings",
                        field=(
                            f"{field}.water_law.loading_table_mol_kg" f"[{row_index}]"
                        ),
                    )
                maximum = max(maximum, max(loadings))
            if maximum <= 0.0:
                raise MofScreeningError(
                    "temperature table needs a positive loading",
                    field=f"{field}.water_law.loading_table_mol_kg",
                )
            _string(water, "extrapolation_policy", f"{field}.water_law")

    coupling = _mapping(extension.get("co2_coupling"), f"{field}.co2_coupling")
    coupling_kind = _string(coupling, "kind", f"{field}.co2_coupling")
    _string_list(coupling, "assumptions", f"{field}.co2_coupling")
    if (
        not isinstance(coupling.get("provenance"), Mapping)
        or not coupling["provenance"]
    ):
        raise MofScreeningError(
            "co2_coupling.provenance must be a nonempty object",
            field=f"{field}.co2_coupling.provenance",
        )
    if coupling_kind == "unsupported":
        _string(coupling, "decision", f"{field}.co2_coupling")
    elif coupling_kind in {
        "water_fraction_quadratic",
        "water_fraction_linear",
    }:
        _finite_number_list(
            coupling.get("coefficients"),
            f"{field}.co2_coupling.coefficients",
            minimum_length=1,
        )
        for key in (
            "normalizer_mol_kg",
            "temperature_exponent",
            "reference_temperature_K",
        ):
            _number(
                coupling,
                key,
                f"{field}.co2_coupling",
                minimum=0.0,
                exclusive_minimum=key
                in {"normalizer_mol_kg", "reference_temperature_K"},
            )
        for key in ("calibration_status", "holdout_status"):
            _string(coupling, key, f"{field}.co2_coupling")
    elif coupling_kind == "water_activated_temperature_langmuir_added_target":
        for key in (
            "added_capacity_mol_kg",
            "reference_added_target_mol_kg",
            "water_normalizer_mol_kg",
            "activation_exponent",
            "half_pressure_Pa",
            "co2_heat_J_mol",
            "reference_temperature_K",
        ):
            _number(
                coupling,
                key,
                f"{field}.co2_coupling",
                minimum=0.0,
                exclusive_minimum=True,
            )
        sensitivities = _finite_number_list(
            coupling.get("co2_heat_sensitivity_J_mol"),
            f"{field}.co2_coupling.co2_heat_sensitivity_J_mol",
            minimum_length=3,
        )
        if (
            len(sensitivities) != 3
            or any(value <= 0.0 for value in sensitivities)
            or any(
                later <= earlier
                for earlier, later in zip(
                    sensitivities,
                    sensitivities[1:],
                )
            )
            or not any(
                math.isclose(
                    value,
                    float(coupling["co2_heat_J_mol"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for value in sensitivities
            )
        ):
            raise MofScreeningError(
                "co2_coupling.co2_heat_sensitivity_J_mol must contain "
                "three increasing sensitivities including the deployed value",
                field=f"{field}.co2_coupling.co2_heat_sensitivity_J_mol",
            )
        if water_kind == "sips_anchor" and not math.isclose(
            float(coupling["water_normalizer_mol_kg"]),
            float(water["anchor_loading_mol_kg"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MofScreeningError(
                "water-activated Langmuir coupling must normalize stored "
                "water at the declared proxy anchor",
                field=f"{field}.co2_coupling.water_normalizer_mol_kg",
            )
        for key in ("calibration_status", "holdout_status"):
            _string(coupling, key, f"{field}.co2_coupling")
    else:
        raise MofScreeningError(
            "co2_coupling.kind is unsupported",
            field=f"{field}.co2_coupling.kind",
        )

    _string_list(extension, "assumptions", field)
    _string_list(extension, "limitations", field)
    _string_list(extension, "missing_evidence", field)
    ranges = _mapping(extension.get("parameter_ranges"), f"{field}.parameter_ranges")
    if not ranges:
        raise MofScreeningError(
            "parameter_ranges must be a nonempty object",
            field=f"{field}.parameter_ranges",
        )
    if (
        not isinstance(extension.get("provenance"), Mapping)
        or not extension["provenance"]
    ):
        raise MofScreeningError(
            "provenance must be a nonempty object",
            field=f"{field}.provenance",
        )
    _mapping(extension.get("support_envelope"), f"{field}.support_envelope")
    return extension


def default_runtime_policy() -> dict:
    """Return the app's non-expiring execution policy."""
    return {
        "schema": RUNTIME_POLICY_SCHEMA,
        "mode": MANUAL_STOP,
        "time_limit_s": None,
        "classification": "operational_runtime_control",
        "physical_operating_condition": False,
    }


def normalize_runtime_policy(document: Any) -> dict:
    """Validate an explicit runtime control without mixing it into physics."""
    policy = dict(_mapping(document, "case.runtime_policy"))
    allowed = {
        "schema",
        "mode",
        "time_limit_s",
        "classification",
        "physical_operating_condition",
    }
    unknown = sorted(set(policy) - allowed)
    if unknown:
        raise MofScreeningError(
            "case.runtime_policy contains unsupported fields: " + ", ".join(unknown),
            field="case.runtime_policy",
        )
    if policy.get("schema") != RUNTIME_POLICY_SCHEMA:
        raise MofScreeningError(
            f"case.runtime_policy.schema must be {RUNTIME_POLICY_SCHEMA!r}",
            field="case.runtime_policy.schema",
        )
    mode = _string(policy, "mode", "case.runtime_policy")
    if mode not in (MANUAL_STOP, TIME_LIMIT):
        raise MofScreeningError(
            "case.runtime_policy.mode must be 'manual_stop' or 'time_limit'",
            field="case.runtime_policy.mode",
        )
    if mode == MANUAL_STOP:
        if policy.get("time_limit_s") not in (None, ""):
            raise MofScreeningError(
                "manual_stop does not accept a time_limit_s",
                field="case.runtime_policy.time_limit_s",
            )
        time_limit_s = None
    else:
        time_limit_s = _number(
            policy,
            "time_limit_s",
            "case.runtime_policy",
            minimum=0.0,
            exclusive_minimum=True,
        )
    if policy.get("classification") not in (None, "operational_runtime_control"):
        raise MofScreeningError(
            "case.runtime_policy.classification must be "
            "'operational_runtime_control'",
            field="case.runtime_policy.classification",
        )
    if policy.get("physical_operating_condition", False) is not False:
        raise MofScreeningError(
            "case.runtime_policy is operational and cannot claim to be a "
            "physical operating condition",
            field="case.runtime_policy.physical_operating_condition",
        )
    return {
        **default_runtime_policy(),
        "mode": mode,
        "time_limit_s": time_limit_s,
    }


def effective_runtime_policy(case: Mapping) -> dict:
    """Return the stored policy, or the app default for a reference case."""
    if "runtime_policy" not in case:
        return default_runtime_policy()
    return normalize_runtime_policy(case["runtime_policy"])


def runtime_wall_limit_s(
    case: Mapping, *, default_wall_limit_s: float | None = 300.0
) -> float | None:
    """Resolve wall termination from explicit case policy.

    A case without a policy retains the original direct-caller default.
    New app cases always carry an explicit policy.
    """

    if "runtime_policy" not in case:
        return default_wall_limit_s
    policy = effective_runtime_policy(case)
    return None if policy["mode"] == MANUAL_STOP else float(policy["time_limit_s"])


def default_temperature_policy() -> dict:
    return {
        "schema": TEMPERATURE_POLICY_SCHEMA,
        "mode": STOP_AT_SCENARIO_GUARD,
    }


def normalize_temperature_policy(document: Any) -> dict:
    policy = dict(_mapping(document, "case.temperature_policy"))
    mode = _string(policy, "mode", "case.temperature_policy")
    if policy.get("schema") != TEMPERATURE_POLICY_SCHEMA:
        raise MofScreeningError(
            f"case.temperature_policy.schema must be " f"{TEMPERATURE_POLICY_SCHEMA!r}",
            field="case.temperature_policy.schema",
        )
    if mode not in (STOP_AT_SCENARIO_GUARD, RECORD_SCENARIO_EXCURSIONS):
        raise MofScreeningError(
            "case.temperature_policy.mode must be "
            "'stop_at_scenario_guard' or 'record_scenario_excursions'",
            field="case.temperature_policy.mode",
        )
    return {**default_temperature_policy(), "mode": mode}


def _temperature_registry() -> dict:
    payload = read_json_strict(TEMPERATURE_REGISTRY_PATH)
    if payload.get("schema") != "mof-screening-temperature-registry/2":
        raise MofScreeningError(
            "temperature registry schema is invalid",
            field="temperature_registry.schema",
        )
    if payload.get("active") is not True:
        raise MofScreeningError(
            "temperature registry is not active",
            field="temperature_registry.active",
        )
    records = payload.get("records")
    if not isinstance(records, dict) or not records:
        raise MofScreeningError(
            "temperature registry records are missing",
            field="temperature_registry.records",
        )
    semantic_records = payload.get("semantic_records")
    if not isinstance(semantic_records, dict) or not semantic_records:
        raise MofScreeningError(
            "temperature registry semantic index is missing",
            field="temperature_registry.semantic_records",
        )
    return payload


def material_temperature_guard(material: Mapping) -> dict:
    package_hash = canonical_sha256(material)
    semantic_hash = exact_registry_semantic_sha256(material)
    registry = _temperature_registry()
    reviewed_canonical_hash = registry["semantic_records"].get(semantic_hash)
    record = (
        registry["records"].get(reviewed_canonical_hash)
        if isinstance(reviewed_canonical_hash, str)
        else None
    )
    if record is None:
        return {
            "matched": False,
            "package_canonical_sha256": package_hash,
            "package_semantic_sha256": semantic_hash,
            "identity_basis": "exact_rational_json_value_v1",
            "material_id": material.get("material_id"),
            "classification": "strict_unregistered_guard",
            "declared_guard_C": float(material["maximum_bed_temperature_C"]),
            "physical_degradation_limit_C": None,
            "basis": (
                "Exact normalized-package registry match absent; the package "
                "scalar remains a strict execution guard."
            ),
            "review_record": None,
        }
    if record.get("material_id") != material.get("material_id"):
        raise MofScreeningError(
            "temperature registry material identity mismatch",
            field="temperature_registry.records",
        )
    if not math.isclose(
        float(record["declared_guard_C"]),
        float(material["maximum_bed_temperature_C"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise MofScreeningError(
            "temperature registry guard differs from the normalized package",
            field="temperature_registry.records",
        )
    return {
        **record,
        "matched": True,
        "package_canonical_sha256": package_hash,
        "package_reviewed_canonical_sha256": record["package_canonical_sha256"],
        "package_semantic_sha256": semantic_hash,
        "identity_basis": "exact_rational_json_value_v1",
    }


def effective_temperature_policy(case: Mapping, material: Mapping) -> dict:
    requested = (
        normalize_temperature_policy(case["temperature_policy"])
        if "temperature_policy" in case
        else default_temperature_policy()
    )
    guard = material_temperature_guard(material)
    scenario_guard = float(guard["declared_guard_C"])
    physical_limit = guard.get("physical_degradation_limit_C")
    scenario_class = guard.get("classification") in {
        "unqualified_scenario_guard",
    }
    if requested["mode"] == RECORD_SCENARIO_EXCURSIONS and not scenario_class:
        raise MofScreeningError(
            "record_scenario_excursions requires an exact reviewed package "
            "record classified as an unqualified scenario guard",
            field="case.temperature_policy.mode",
        )
    if physical_limit is not None:
        physical_limit = float(physical_limit)
    if requested["mode"] == RECORD_SCENARIO_EXCURSIONS:
        enforcement_limit = physical_limit
    else:
        limits = [scenario_guard]
        if physical_limit is not None:
            limits.append(physical_limit)
        enforcement_limit = min(limits)
    return {
        "schema": TEMPERATURE_POLICY_SCHEMA,
        "mode": requested["mode"],
        "registry_matched": bool(guard["matched"]),
        "registry_classification": guard["classification"],
        "scenario_guard_C": scenario_guard,
        "physical_degradation_limit_C": physical_limit,
        "enforcement_limit_C": enforcement_limit,
        "scenario_excursion_permitted": (
            requested["mode"] == RECORD_SCENARIO_EXCURSIONS and scenario_class
        ),
        "physical_limit_known": physical_limit is not None,
        "evidence_gap": (
            None
            if physical_limit is not None
            else "No evidenced physical degradation limit is recorded."
        ),
        "basis": guard.get("basis"),
        "review_record": guard.get("review_record"),
        "package_canonical_sha256": guard["package_canonical_sha256"],
        "package_reviewed_canonical_sha256": guard.get(
            "package_reviewed_canonical_sha256"
        ),
        "package_semantic_sha256": guard["package_semantic_sha256"],
        "identity_basis": guard["identity_basis"],
    }


def temperature_guard_assessment(
    material: Mapping,
    case: Mapping,
    *,
    peak_bed_temperature_K: float | None,
) -> dict:
    policy = effective_temperature_policy(case, material)
    if peak_bed_temperature_K is None:
        return {
            "policy": policy,
            "peak_bed_temperature_C": None,
            "scenario_guard_excursion_C": None,
            "physical_degradation_limit_pass": None,
            "status": "not_available",
        }
    peak_C = float(peak_bed_temperature_K) - 273.15
    excursion_C = max(peak_C - policy["scenario_guard_C"], 0.0)
    physical_limit = policy["physical_degradation_limit_C"]
    physical_pass = (
        None if physical_limit is None else bool(peak_C <= float(physical_limit))
    )
    if excursion_C > 0.0 and policy["scenario_excursion_permitted"]:
        status = "scenario_excursion_recorded"
    elif excursion_C > 0.0:
        status = "strict_guard_excursion"
    else:
        status = "within_scenario_guard"
    return {
        "policy": policy,
        "peak_bed_temperature_C": peak_C,
        "scenario_guard_excursion_C": excursion_C,
        "physical_degradation_limit_pass": physical_pass,
        "physical_degradation_limit_known": (policy["physical_limit_known"]),
        "status": status,
        "statement": (
            "Scenario guard outcomes are execution/evidence classifications. "
            "An excursion with no evidenced physical degradation limit is not "
            "a physical hard-limit pass."
        ),
    }


def _positive_array(
    mapping: Mapping,
    key: str,
    field: str,
    *,
    length: int,
    allow_zero: bool = False,
) -> list[float]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise MofScreeningError(
            f"{field}.{key} must contain exactly {length} numbers",
            field=f"{field}.{key}",
        )
    values = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MofScreeningError(
                f"{field}.{key}[{index}] must be a number",
                field=f"{field}.{key}[{index}]",
            )
        number = float(item)
        invalid = (
            not math.isfinite(number)
            or number < 0.0
            or (not allow_zero and number == 0.0)
        )
        if invalid:
            condition = "nonnegative" if allow_zero else "positive"
            raise MofScreeningError(
                f"{field}.{key}[{index}] must be finite and {condition}",
                field=f"{field}.{key}[{index}]",
            )
        values.append(number)
    return values


def _physical_material_view(material: Mapping) -> dict:
    """Fields that determine a physical result and therefore source claims."""
    keys = (
        "equilibrium_family",
        "reference_temperature_K",
        "maximum_bed_temperature_C",
        "solid_cp_J_kg_K",
        "co2_water_penalty_per_mol_kg",
        "co2",
        "water",
        "nitrogen",
        "oxygen",
        "adsorbed_phase_cp_J_kg_K",
        "kinetics",
    )
    return {key: material.get(key) for key in keys}


def physical_material_sha256(material: Mapping) -> str:
    return canonical_sha256(_physical_material_view(material))


def _evidence_baseline_view(material: Mapping) -> dict:
    """Identity, sample/shaping declarations and physical laws behind proof."""
    return {
        "material_id": material.get("material_id"),
        "name": material.get("name"),
        "synthetic": material.get("synthetic"),
        "shaped_material": material.get("shaped_material"),
        "physical": _physical_material_view(material),
    }


def _semantic_json_value(value: Any) -> Any:
    """Normalize JSON numbers so int 850 and float 850.0 compare identically."""
    if isinstance(value, bool):
        return ["__boolean__", value]
    if isinstance(value, (int, float)):
        return ["__number__", float(value)]
    if isinstance(value, Mapping):
        return [
            "__object__",
            [
                [str(key), _semantic_json_value(item)]
                for key, item in sorted(value.items())
            ],
        ]
    if isinstance(value, (list, tuple)):
        return ["__array__", [_semantic_json_value(item) for item in value]]
    return value


def semantic_json_sha256(value: Any) -> str:
    """Hash JSON with integer and equivalent float values treated equally."""
    return canonical_sha256(_semantic_json_value(value))


def _exact_json_identity(value: Any) -> Any:
    """Preserve full JSON identity while equating equal numeric values."""
    if value is None:
        return ["__null__"]
    if isinstance(value, bool):
        return ["__boolean__", value]
    if isinstance(value, int):
        return ["__number__", str(value), "1"]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("registry identity does not accept nonfinite numbers")
        numerator, denominator = value.as_integer_ratio()
        return ["__number__", str(numerator), str(denominator)]
    if isinstance(value, str):
        return ["__string__", value]
    if isinstance(value, Mapping):
        return [
            "__object__",
            [
                [str(key), _exact_json_identity(item)]
                for key, item in sorted(value.items())
            ],
        ]
    if isinstance(value, (list, tuple)):
        return ["__array__", [_exact_json_identity(item) for item in value]]
    raise ValueError(f"registry identity does not accept {type(value).__name__}")


def exact_registry_semantic_sha256(value: Any) -> str:
    """Hash all normalized JSON values with exact numeric-value equality."""
    return canonical_sha256(_exact_json_identity(value))


def _evidence_baseline_sha256(material: Mapping) -> str:
    return canonical_sha256(_semantic_json_value(_evidence_baseline_view(material)))


def _original_loaded_evidence(evidence: Mapping) -> dict:
    """Return the first non-annotation evidence object, avoiding nested wrappers."""
    original = evidence
    while (
        original.get("status") == "USER_EDITED_ASSUMPTION"
        and original.get("edit_detected") is True
        and isinstance(original.get("original_evidence"), Mapping)
    ):
        original = original["original_evidence"]
    return deepcopy(dict(original))


def _mark_builtin_edits(material: dict) -> dict:
    builtins = {
        "purolite_a110_humid": MATERIAL_ROOT / "purolite_a110_humid.json",
        "hypothetical_mmen_like": MATERIAL_ROOT / "hypothetical_mmen_like.json",
        "calf20_nguyen2024_80_20_shaped": MATERIAL_ROOT / "calf20_shaped.json",
        "synthetic_independent_demo": MATERIAL_ROOT / "synthetic_independent_demo.json",
        "mmen_mg2_dobpdc_mcdonald2012_conditional": (
            MATERIAL_ROOT / "mmen_mg2_dobpdc_mcdonald2012_conditional.json"
        ),
        "lewatit_vp_oc_1065_low2023_conditional": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_low2023_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_hot116_conditional": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_hot116_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_cold_hot116_conditional": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_cold_hot116_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_advisory_temperature_conditional": (
            MATERIAL_ROOT
            / "lewatit_vp_oc_1065_young2021_advisory_temperature_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_dry_bdf_recovery_v1": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_dry_bdf_recovery_v1.json"
        ),
        "mmen_mg2_hob_powder_2024_dry_conditional": (
            MATERIAL_ROOT / "mmen_mg2_hob_powder_2024_dry_conditional.json"
        ),
        "mmen_mg2_hob_pvdf_2024_dry_conditional": (
            MATERIAL_ROOT / "mmen_mg2_hob_pvdf_2024_dry_conditional.json"
        ),
        "mof808_lys_2024_25c_dry_conditional": (
            MATERIAL_ROOT / "mof808_lys_2024_25c_dry_conditional.json"
        ),
        "mof808_tapa_2024_25c_dry_conditional": (
            MATERIAL_ROOT / "mof808_tapa_2024_25c_dry_conditional.json"
        ),
        "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional": (
            MATERIAL_ROOT / "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional.json"
        ),
        "cof1000_2025_25c_split_rate_dry_conditional": (
            MATERIAL_ROOT / "cof1000_2025_25c_split_rate_dry_conditional.json"
        ),
        "cof999_2024_25c_dry_conditional": (
            MATERIAL_ROOT / "cof999_2024_25c_dry_conditional.json"
        ),
        "mil101_pei800_30_glycerin_2026_25c_dry_conditional": (
            MATERIAL_ROOT / "mil101_pei800_30_glycerin_2026_25c_dry_conditional.json"
        ),
        "mil101_pei800_30_2026_25c_dry_conditional": (
            MATERIAL_ROOT / "mil101_pei800_30_2026_25c_dry_conditional.json"
        ),
    }
    provenance = material.setdefault("provenance", {})
    attribution = provenance.get("loaded_builtin")
    attributed_id = (
        attribution.get("material_id") if isinstance(attribution, Mapping) else None
    )
    baseline_id = (
        attributed_id
        if isinstance(attributed_id, str) and attributed_id in builtins
        else material.get("material_id")
    )
    baseline_path = builtins.get(baseline_id)
    if baseline_path is None or not baseline_path.is_file():
        return material
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    original_physical_sha = physical_material_sha256(baseline)
    current_physical_sha = physical_material_sha256(material)
    baseline_sha = _evidence_baseline_sha256(baseline)
    current_baseline_sha = _evidence_baseline_sha256(material)
    policy_edit_detected = False
    if baseline_id == "lewatit_vp_oc_1065_young2021_dry_bdf_recovery_v1":
        policy_edit_detected = canonical_sha256(
            _semantic_json_value(
                {
                    "water_evaluation_policy": material.get("water_evaluation_policy"),
                    "temperature_advisory": material.get("temperature_advisory"),
                }
            )
        ) != canonical_sha256(
            _semantic_json_value(
                {
                    "water_evaluation_policy": baseline.get("water_evaluation_policy"),
                    "temperature_advisory": baseline.get("temperature_advisory"),
                }
            )
        )
    evidence = material["evidence"]
    already_annotated = bool(
        evidence.get("status") == "USER_EDITED_ASSUMPTION"
        and evidence.get("edit_detected") is True
        and isinstance(evidence.get("original_evidence"), Mapping)
    )
    if current_baseline_sha == baseline_sha and not policy_edit_detected:
        if already_annotated:
            # This is an immutable preview made by the superseded comparison.
            # Preserve its exact bytes/hash so the already-issued preview can
            # still be consumed and inspected.
            return material
        material["evidence"]["loaded_source_sha256"] = original_physical_sha
        provenance["loaded_builtin"] = {
            "schema": "mof-screening-loaded-builtin/1",
            "material_id": baseline_id,
            "name": baseline.get("name"),
            "package_file": baseline_path.name,
            "baseline_sha256": baseline_sha,
        }
        return material
    original_evidence = _original_loaded_evidence(evidence)
    provenance["loaded_builtin"] = {
        "schema": "mof-screening-loaded-builtin/1",
        "material_id": baseline_id,
        "name": baseline.get("name"),
        "package_file": baseline_path.name,
        "baseline_sha256": baseline_sha,
    }
    material["evidence"] = {
        "status": "USER_EDITED_ASSUMPTION",
        "basis": (
            "The physical coefficients differ from the loaded built-in "
            "package. This preview is an explicitly user-edited assumption, "
            "not the original source-transfer or hypothetical evidence."
        ),
        "edit_detected": True,
        "loaded_physical_sha256": original_physical_sha,
        "edited_physical_sha256": current_physical_sha,
        "loaded_baseline_sha256": baseline_sha,
        "edited_baseline_sha256": current_baseline_sha,
        "original_evidence": original_evidence,
    }
    provenance["input_edit"] = {
        "class": "user-edited assumption",
        "loaded_baseline_material_id": baseline_id,
        "loaded_baseline_name": baseline.get("name"),
        "loaded_baseline_sha256": baseline_sha,
        "edited_baseline_sha256": current_baseline_sha,
        "loaded_physical_sha256": original_physical_sha,
        "edited_physical_sha256": current_physical_sha,
        "original_evidence_preserved_under": "material.evidence.original_evidence",
    }
    edit_assumption = (
        "The material identity, sample/shaping declaration or one or more "
        "physical coefficients were edited after loading a built-in package; "
        "results no longer inherit that package's source evidence status."
    )
    if edit_assumption not in material.setdefault("assumptions", []):
        material["assumptions"].append(edit_assumption)
    return material


def validate_material(document: Any) -> dict:
    """Validate a portable material package without silently filling gaps."""

    material = deepcopy(dict(_mapping(document, "material")))
    if material.get("schema") != MATERIAL_SCHEMA:
        raise MofScreeningError(
            f"material.schema must be {MATERIAL_SCHEMA!r}",
            field="material.schema",
        )
    _string(material, "material_id", "material")
    _string(material, "name", "material")
    family = _string(material, "equilibrium_family", "material")
    if family not in SUPPORTED_FAMILIES:
        raise MofScreeningError(
            "material.equilibrium_family is unsupported; add a bounded adapter "
            f"for {family!r} before running it",
            field="material.equilibrium_family",
        )
    if "humidity_candidate" in material:
        validate_humidity_candidate_extension(material)
    evidence = _mapping(material.get("evidence"), "material.evidence")
    _string(evidence, "status", "material.evidence")
    _string(evidence, "basis", "material.evidence")
    if type(material.get("synthetic")) is not bool:
        raise MofScreeningError(
            "material.synthetic must be a JSON boolean", field="material.synthetic"
        )
    shaped = _mapping(material.get("shaped_material"), "material.shaped_material")
    _string(shaped, "basis", "material.shaped_material")
    _string(shaped, "binder_support", "material.shaped_material")
    _number(
        material,
        "reference_temperature_K",
        "material",
        minimum=250.0,
        maximum=450.0,
    )
    _number(
        material,
        "maximum_bed_temperature_C",
        "material",
        minimum=40.0,
        maximum=500.0,
    )
    _number(
        material, "solid_cp_J_kg_K", "material", minimum=0.0, exclusive_minimum=True
    )

    co2 = _mapping(material.get("co2"), "material.co2")
    water = _mapping(material.get("water"), "material.water")
    if (
        family == "split_rate_activity_shift_reference_isotherm"
        and water.get("model") != "dry_only"
    ):
        raise MofScreeningError(
            "split-rate material requires material.water.model == 'dry_only'; "
            "humid or missing water modes are not silently ignored",
            field="material.water.model",
        )
    dry_only_water = (
        family
        in (
            "activity_shift_reference_isotherm",
            "split_rate_activity_shift_reference_isotherm",
        )
        and water.get("model") == "dry_only"
    )
    if family in (
        "independent_langmuir_sips",
        "activity_shift_reference_isotherm",
        "split_rate_activity_shift_reference_isotherm",
    ):
        if family == "independent_langmuir_sips":
            for key in (
                "saturation_mol_kg",
                "reference_loading_mol_kg",
                "reference_partial_pressure_Pa",
                "heat_J_mol",
            ):
                _number(
                    co2,
                    key,
                    "material.co2",
                    minimum=0.0,
                    exclusive_minimum=key != "heat_J_mol",
                )
            if float(co2["reference_loading_mol_kg"]) >= float(
                co2["saturation_mol_kg"]
            ):
                raise MofScreeningError(
                    "material.co2.reference_loading_mol_kg must be below saturation_mol_kg",
                    field="material.co2.reference_loading_mol_kg",
                )
        else:
            _number(co2, "heat_J_mol", "material.co2", minimum=0.0)
            _number(
                material,
                "co2_water_penalty_per_mol_kg",
                "material",
                minimum=0.0,
            )
            curve = _mapping(co2.get("reference_curve"), "material.co2.reference_curve")
            pressures = curve.get("pressure_Pa")
            loadings = curve.get("loading_mol_kg")
            if not (
                isinstance(pressures, list)
                and isinstance(loadings, list)
                and len(pressures) == len(loadings)
                and len(pressures) >= 3
            ):
                raise MofScreeningError(
                    "material.co2.reference_curve needs at least three paired "
                    "pressure_Pa and loading_mol_kg entries",
                    field="material.co2.reference_curve",
                )
            _number(
                curve,
                "temperature_K",
                "material.co2.reference_curve",
                minimum=250.0,
                maximum=450.0,
            )
            pressure_values = []
            loading_values = []
            for index, (pressure, loading) in enumerate(zip(pressures, loadings)):
                if (
                    isinstance(pressure, bool)
                    or isinstance(loading, bool)
                    or not isinstance(pressure, (int, float))
                    or not isinstance(loading, (int, float))
                    or not math.isfinite(float(pressure))
                    or not math.isfinite(float(loading))
                    or float(pressure) <= 0.0
                    or float(loading) < 0.0
                ):
                    raise MofScreeningError(
                        "material.co2.reference_curve entries require finite "
                        "positive pressure_Pa and nonnegative loading_mol_kg",
                        field=f"material.co2.reference_curve[{index}]",
                    )
                pressure_values.append(float(pressure))
                loading_values.append(float(loading))
            if any(b <= a for a, b in zip(pressure_values, pressure_values[1:])):
                raise MofScreeningError(
                    "material.co2.reference_curve pressure_Pa values must be "
                    "strictly increasing",
                    field="material.co2.reference_curve.pressure_Pa",
                )
            if any(b < a for a, b in zip(loading_values, loading_values[1:])):
                raise MofScreeningError(
                    "material.co2.reference_curve loading_mol_kg values must "
                    "be monotone nondecreasing",
                    field="material.co2.reference_curve.loading_mol_kg",
                )
            if max(loading_values) <= 0.0:
                raise MofScreeningError(
                    "material.co2.reference_curve needs a positive loading; "
                    "an all-zero curve has no defined adapter storage scale",
                    field="material.co2.reference_curve.loading_mol_kg",
                )
            if family == "split_rate_activity_shift_reference_isotherm":
                reservoir_order = co2.get("reservoir_order")
                if reservoir_order != ["fast", "slow"]:
                    raise MofScreeningError(
                        "split-rate material.co2.reservoir_order must be "
                        "exactly ['fast', 'slow']",
                        field="material.co2.reservoir_order",
                    )
                fractions = _positive_array(
                    co2,
                    "reservoir_fractions",
                    "material.co2",
                    length=2,
                    allow_zero=False,
                )
                if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                    raise MofScreeningError(
                        "material.co2.reservoir_fractions must sum to 1",
                        field="material.co2.reservoir_fractions",
                    )
            support = co2.get("support") or {}
            if not isinstance(support, Mapping):
                raise MofScreeningError(
                    "material.co2.support must be an object",
                    field="material.co2.support",
                )
            support_policy = support.get("policy", STRICT_POLICY)
            if support_policy not in SUPPORTED_REFERENCE_SUPPORT_POLICIES:
                raise MofScreeningError(
                    "material.co2.support.policy must be one of "
                    + ", ".join(
                        repr(value) for value in SUPPORTED_REFERENCE_SUPPORT_POLICIES
                    ),
                    field="material.co2.support.policy",
                )
        if dry_only_water:
            if float(material["co2_water_penalty_per_mol_kg"]) != 0.0:
                raise MofScreeningError(
                    "dry-only material requires "
                    "co2_water_penalty_per_mol_kg to be exactly zero",
                    field="material.co2_water_penalty_per_mol_kg",
                )
            _number(
                water,
                "solver_normalizer_mol_kg",
                "material.water",
                minimum=0.0,
                exclusive_minimum=True,
            )
            physical_storage = water.get("physical_storage_mol_kg")
            if (
                isinstance(physical_storage, bool)
                or not isinstance(physical_storage, (int, float))
                or float(physical_storage) != 0.0
            ):
                raise MofScreeningError(
                    "material.water.physical_storage_mol_kg must be exactly 0 "
                    "for dry-only water",
                    field="material.water.physical_storage_mol_kg",
                )
            _string(water, "basis", "material.water")
        else:
            for key in (
                "saturation_mol_kg",
                "half_loading_relative_humidity_at_reference",
                "sips_exponent",
                "heat_J_mol",
            ):
                _number(
                    water,
                    key,
                    "material.water",
                    minimum=0.0,
                    exclusive_minimum=key != "heat_J_mol",
                )
            if (
                not 0.0
                < float(water["half_loading_relative_humidity_at_reference"])
                <= 1.0
            ):
                raise MofScreeningError(
                    "material.water.half_loading_relative_humidity_at_reference "
                    "must be in (0, 1]",
                    field=(
                        "material.water." "half_loading_relative_humidity_at_reference"
                    ),
                )
        for gas in ("nitrogen", "oxygen"):
            block = _mapping(material.get(gas), f"material.{gas}")
            for key in ("henry_mol_kg_Pa_at_reference", "heat_J_mol"):
                _number(block, key, f"material.{gas}", minimum=0.0)
        n_ads = 5 if family == "split_rate_activity_shift_reference_isotherm" else 4
    elif family == "purolite_a110_rh_curves_gab":
        if material["reference_temperature_K"] != 298.15:
            raise MofScreeningError("The Purolite reference curves require 298.15 K",
                                    field="material.reference_temperature_K")
        for key, block in (("co2", co2), ("water", water)):
            _number(block, "heat_J_mol", f"material.{key}", minimum=0., exclusive_minimum=True)
        rows = co2.get("curves")
        if not isinstance(rows, list) or len(rows) != 4:
            raise MofScreeningError("Expected four Purolite humidity curves", field="material.co2.curves")
        for i, (row, rh) in enumerate(zip(rows, (0., 0.30, 0.55, 0.80))):
            field = f"material.co2.curves[{i}]"
            row = _mapping(row, field)
            if _number(row, "relative_humidity", field) != rh:
                raise MofScreeningError("Unexpected Purolite RH node", field=field)
            p = row.get("pressure_Pa")
            if not isinstance(p, list) or len(p) < 2:
                raise MofScreeningError("Expected at least two pressure points", field=field)
            p = _positive_array(row, "pressure_Pa", field, length=len(p))
            q = _positive_array(row, "loading_mol_kg", field, length=len(p))
            if any(b <= a for a, b in zip(p, p[1:])) or any(b < a for a, b in zip(q, q[1:])):
                raise MofScreeningError("Expected increasing pressures and nondecreasing loadings", field=field)
        n_ads = 2
    elif family in (
        "young2021_wadst_gab",
        "young2021_hot_wadst_gab_extrapolated",
        "young2021_cold_hot_wadst_gab_extrapolated",
        "young2021_advisory_wadst_gab",
    ):
        _number(co2, "heat_J_mol", "material.co2", minimum=0.0)
        support = co2.get("support")
        if not isinstance(support, Mapping):
            raise MofScreeningError(
                "material.co2.support must be an object",
                field="material.co2.support",
            )
        ranges = {
            "temperature_K": (298.15, 343.15),
            "relative_humidity": (0.0, 0.55),
            "co2_partial_pressure_Pa": (0.0, 100000.0),
        }
        for key, expected in ranges.items():
            value = support.get(key)
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in value
                )
                or not math.isclose(float(value[0]), expected[0], abs_tol=1e-12)
                or not math.isclose(float(value[1]), expected[1], abs_tol=1e-12)
            ):
                raise MofScreeningError(
                    f"material.co2.support.{key} must be {list(expected)}",
                    field=f"material.co2.support.{key}",
                )
        if support.get("relative_humidity_basis") != ("local p_H2O / p_sat(T_bed)"):
            raise MofScreeningError(
                "material.co2.support.relative_humidity_basis must use "
                "local p_H2O / p_sat(T_bed)",
                field="material.co2.support.relative_humidity_basis",
            )
        expected_water_model = {
            "young2021_wadst_gab": "young2021_gab",
            "young2021_hot_wadst_gab_extrapolated": (
                "young2021_gab_iapws_hot_extrapolated"
            ),
            "young2021_cold_hot_wadst_gab_extrapolated": (
                "young2021_gab_iapws_cold_hot_extrapolated"
            ),
            "young2021_advisory_wadst_gab": (
                "young2021_gab_iapws_temperature_advisory"
            ),
        }[family]
        if water.get("model") != expected_water_model:
            raise MofScreeningError(
                f"material.water.model must be {expected_water_model!r}",
                field="material.water.model",
            )
        if family == "young2021_cold_hot_wadst_gab_extrapolated":
            envelope = co2.get("evaluation_envelope")
            expected_envelope = {
                "temperature_K": [293.15, 393.15],
                "relative_humidity": [0.0, 0.55],
                "co2_partial_pressure_Pa": [0.0, 100000.0],
            }
            if not isinstance(envelope, Mapping):
                raise MofScreeningError(
                    "cold+hot scenario requires co2.evaluation_envelope",
                    field="material.co2.evaluation_envelope",
                )
            for key, expected in expected_envelope.items():
                if envelope.get(key) != expected:
                    raise MofScreeningError(
                        f"cold+hot material.co2.evaluation_envelope.{key} "
                        f"must be {expected}",
                        field=("material.co2.evaluation_envelope." f"{key}"),
                    )
            policy = material.get("water_evaluation_policy")
            if not isinstance(policy, Mapping):
                raise MofScreeningError(
                    "cold+hot scenario requires water_evaluation_policy",
                    field="material.water_evaluation_policy",
                )
            expected_policy = {
                "original_implementation_cap_mol_kg": 4.131251972115188,
                "gab_envelope_max_temperature_K": 302.7029459003439,
                "gab_envelope_max_mol_kg": 4.135780869459574,
                "evaluation_cap_mol_kg": 4.136,
                "evaluation_lower_tolerance_mol_kg": -1e-9,
                "evaluation_upper_tolerance_mol_kg": 1e-9,
            }
            for key, expected in expected_policy.items():
                value = policy.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isclose(
                        float(value),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ):
                    raise MofScreeningError(
                        f"cold+hot material.water_evaluation_policy.{key} "
                        f"must equal {expected!r}",
                        field=("material.water_evaluation_policy." f"{key}"),
                    )
        if family == "young2021_advisory_wadst_gab":
            advisory = _mapping(
                material.get("temperature_advisory"),
                "material.temperature_advisory",
            )
            if advisory.get("mode") != "record_actual_temperatures":
                raise MofScreeningError(
                    "temperature-advisory material must record actual temperatures",
                    field="material.temperature_advisory.mode",
                )
            for key in (
                "requested_temperature_is_advisory",
                "historical_range_is_hard_stop",
                "temporary_temperature_clamping",
                "accepted_state_temperature_clamping",
            ):
                expected = key != "historical_range_is_hard_stop" and (
                    key != "temporary_temperature_clamping"
                    and key != "accepted_state_temperature_clamping"
                )
                if advisory.get(key) is not expected:
                    raise MofScreeningError(
                        f"material.temperature_advisory.{key} must be {expected}",
                        field=f"material.temperature_advisory.{key}",
                    )
            policy = _mapping(
                material.get("water_evaluation_policy"),
                "material.water_evaluation_policy",
            )
            if policy.get("temperature_clamping") is not False:
                raise MofScreeningError(
                    "temperature-advisory water policy cannot clamp temperature",
                    field=("material.water_evaluation_policy.temperature_clamping"),
                )
            if policy.get("upper_bound_known") is not False:
                raise MofScreeningError(
                    "temperature-advisory water policy has no evidenced upper bound",
                    field=("material.water_evaluation_policy.upper_bound_known"),
                )
        _number(water, "heat_J_mol", "material.water", minimum=0.0)
        _string(water, "basis", "material.water")
        n_ads = 2
    else:
        capacities = co2.get("site_capacity_mol_kg")
        prefactors = co2.get("site_affinity_m3_mol")
        energies = co2.get("site_internal_energy_J_mol")
        if not (
            isinstance(capacities, list)
            and isinstance(prefactors, list)
            and isinstance(energies, list)
            and len(capacities) == len(prefactors) == len(energies)
            and len(capacities) == 2
        ):
            raise MofScreeningError(
                "material.co2 requires two equal-length site arrays: "
                "site_capacity_mol_kg, site_affinity_m3_mol, "
                "site_internal_energy_J_mol",
                field="material.co2",
            )
        for index in range(2):
            for key, values, relation in (
                ("site_capacity_mol_kg", capacities, "positive"),
                ("site_affinity_m3_mol", prefactors, "nonnegative"),
                ("site_internal_energy_J_mol", energies, "nonpositive"),
            ):
                item = values[index]
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise MofScreeningError(
                        f"material.co2.{key}[{index}] must be numeric",
                        field=f"material.co2.{key}[{index}]",
                    )
                value = float(item)
                invalid = (
                    not math.isfinite(value)
                    or (relation == "positive" and value <= 0.0)
                    or (relation == "nonnegative" and value < 0.0)
                    or (relation == "nonpositive" and value > 0.0)
                )
                if invalid:
                    raise MofScreeningError(
                        f"material.co2.{key}[{index}] must be finite and {relation}",
                        field=f"material.co2.{key}[{index}]",
                    )
        if water.get("model") != "temperature_activity_pchip":
            raise MofScreeningError(
                "material.water.model must be 'temperature_activity_pchip'",
                field="material.water.model",
            )
        for key in (
            "measurement_temperature_K",
            "sorption_heat_J_mol",
            "reference_temperature_K",
        ):
            _number(water, key, "material.water", minimum=0.0, exclusive_minimum=True)
        points = water.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise MofScreeningError(
                "material.water.points requires at least three [RH, loading] points",
                field="material.water.points",
            )
        rh_values: list[float] = []
        loading_values: list[float] = []
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                raise MofScreeningError(
                    f"material.water.points[{index}] must be [RH, loading_mol_kg]",
                    field=f"material.water.points[{index}]",
                )
            rh, loading = point
            if (
                isinstance(rh, bool)
                or isinstance(loading, bool)
                or not isinstance(rh, (int, float))
                or not isinstance(loading, (int, float))
                or not math.isfinite(float(rh))
                or not math.isfinite(float(loading))
                or not 0.0 <= float(rh) <= 1.0
                or float(loading) < 0.0
            ):
                raise MofScreeningError(
                    f"material.water.points[{index}] is outside RH [0,1] or has a negative loading",
                    field=f"material.water.points[{index}]",
                )
            rh_values.append(float(rh))
            loading_values.append(float(loading))
        if rh_values[0] != 0.0:
            raise MofScreeningError(
                "material.water.points must include the zero-RH anchor",
                field="material.water.points",
            )
        if any(b <= a for a, b in zip(rh_values, rh_values[1:])):
            raise MofScreeningError(
                "material.water.points RH values must be strictly increasing",
                field="material.water.points",
            )
        if any(b < a for a, b in zip(loading_values, loading_values[1:])):
            raise MofScreeningError(
                "material.water.points loading values must be monotone",
                field="material.water.points",
            )
        if max(loading_values) <= 0.0:
            raise MofScreeningError(
                "material.water.points needs a positive loading; an all-zero "
                "curve has no defined adapter storage scale",
                field="material.water.points",
            )
        n_ads = 3

    cp_values = material.get("adsorbed_phase_cp_J_kg_K")
    if not isinstance(cp_values, list) or len(cp_values) != n_ads:
        raise MofScreeningError(
            f"material.adsorbed_phase_cp_J_kg_K requires {n_ads} entries",
            field="material.adsorbed_phase_cp_J_kg_K",
        )
    water_storage_index = (
        2 if family == "split_rate_activity_shift_reference_isotherm" else 1
    )
    for index, value in enumerate(cp_values):
        dry_water_cp = dry_only_water and index == water_storage_index
        allow_dry_only_zero = dry_only_water and index > 1
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (
                float(value) != 0.0
                if dry_water_cp
                else float(value) < 0.0 if allow_dry_only_zero else float(value) <= 0.0
            )
        ):
            raise MofScreeningError(
                f"material.adsorbed_phase_cp_J_kg_K[{index}] must be "
                + (
                    "exactly zero"
                    if dry_water_cp
                    else "nonnegative" if allow_dry_only_zero else "positive"
                ),
                field=f"material.adsorbed_phase_cp_J_kg_K[{index}]",
            )
    if family == "site_resolved_concentration_langmuir":
        gas_constant = 8.31446261815324
        expected_cp = (
            (37.5 - gas_constant) / 0.0440095,
            (37.5 - gas_constant) / 0.0440095,
            34.0 / 0.01801528,
        )
        for index, (actual, expected) in enumerate(zip(cp_values, expected_cp)):
            if not math.isclose(
                float(actual),
                expected,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                raise MofScreeningError(
                    "material.adsorbed_phase_cp_J_kg_K is incompatible "
                    "with the existing two-site concentration/internal-energy "
                    f"law: entry {index} must be {expected:.12g} J/kg "
                    "adsorbed species/K. This is not the dry solid Cp.",
                    field=f"material.adsorbed_phase_cp_J_kg_K[{index}]",
                )
    elif family in (
        "activity_shift_reference_isotherm",
        "split_rate_activity_shift_reference_isotherm",
    ):
        expected_co2_cp = 37.5 / 0.0440095
        count = 2 if family == "split_rate_activity_shift_reference_isotherm" else 1
        for index in range(count):
            if not math.isclose(
                float(cp_values[index]),
                expected_co2_cp,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                raise MofScreeningError(
                    f"material.adsorbed_phase_cp_J_kg_K[{index}] is "
                    "incompatible with the existing constant-isosteric-heat "
                    "reference-curve law: it must be "
                    f"{expected_co2_cp:.12g} J/kg adsorbed species/K so the "
                    "CO2 derivative Cp is constant.",
                    field=f"material.adsorbed_phase_cp_J_kg_K[{index}]",
                )
    elif family == "young2021_wadst_gab":
        expected_cp = (37.5 / 0.0440095, 34.0 / 0.01801528)
        for index, (actual, expected) in enumerate(zip(cp_values, expected_cp)):
            if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
                raise MofScreeningError(
                    "material.adsorbed_phase_cp_J_kg_K is incompatible with "
                    "the native Young2021 CP_GAS energy convention; "
                    f"entry {index} must be {expected:.12g} J/kg/K",
                    field=f"material.adsorbed_phase_cp_J_kg_K[{index}]",
                )

    kinetics = _mapping(material.get("kinetics"), "material.kinetics")
    adsorption = _positive_array(
        kinetics,
        "adsorption_rate_s_inv",
        "material.kinetics",
        length=n_ads,
        allow_zero=True,
    )
    desorption = _positive_array(
        kinetics,
        "desorption_rate_s_inv",
        "material.kinetics",
        length=n_ads,
        allow_zero=True,
    )
    if dry_only_water and (
        adsorption[water_storage_index] != 0.0 or desorption[water_storage_index] != 0.0
    ):
        raise MofScreeningError(
            "dry-only material requires exact zero water adsorption and "
            "desorption rates",
            field=(
                "material.kinetics.adsorption_rate_s_inv" f"[{water_storage_index}]"
                if adsorption[water_storage_index] != 0.0
                else "material.kinetics.desorption_rate_s_inv"
                f"[{water_storage_index}]"
            ),
        )
    _string(kinetics, "evidence_status", "material.kinetics")
    material["kinetics"] = {
        **dict(kinetics),
        "adsorption_rate_s_inv": adsorption,
        "desorption_rate_s_inv": desorption,
    }

    for key in ("provenance", "assumptions", "validity", "uncertainty"):
        if key not in material or material[key] in (None, {}, []):
            raise MofScreeningError(
                f"material.{key} is required and cannot be empty",
                field=f"material.{key}",
            )
    if not isinstance(material["assumptions"], list) or not all(
        isinstance(item, str) and item.strip() for item in material["assumptions"]
    ):
        raise MofScreeningError(
            "material.assumptions must be a nonempty list of strings",
            field="material.assumptions",
        )
    return _mark_builtin_edits(material)


def validate_case(document: Any) -> dict:
    case = deepcopy(dict(_mapping(document, "case")))
    if case.get("schema") != CASE_SCHEMA:
        raise MofScreeningError(
            f"case.schema must be {CASE_SCHEMA!r}", field="case.schema"
        )
    engine = _string(case, "engine", "case")
    if engine not in (SPATIAL, LUMPED):
        raise MofScreeningError(
            "case.engine must be 'spatial' or 'lumped'", field="case.engine"
        )
    _string(case, "label", "case")
    if "runtime_policy" in case:
        case["runtime_policy"] = normalize_runtime_policy(case["runtime_policy"])
    if "temperature_policy" in case:
        case["temperature_policy"] = normalize_temperature_policy(
            case["temperature_policy"]
        )

    weather = _mapping(case.get("weather"), "case.weather")
    mode = _string(weather, "mode", "case.weather")
    if mode not in (FIXED_WEATHER, OFFICIAL_WEATHER):
        raise MofScreeningError(
            "case.weather.mode must be 'fixed' or 'official'",
            field="case.weather.mode",
        )
    maximum_hours = 24 if mode == OFFICIAL_WEATHER else 96
    hours = _integer(
        weather,
        "hours",
        "case.weather",
        minimum=1,
        maximum=maximum_hours,
    )
    if mode == FIXED_WEATHER:
        for key in ("temperature_C", "relative_humidity", "pressure_Pa", "co2_ppm_dry"):
            _number(weather, key, "case.weather")
        if not 255.15 <= 273.15 + float(weather["temperature_C"]) <= 373.0:
            raise MofScreeningError(
                "case.weather.temperature_C is outside the water-property support",
                field="case.weather.temperature_C",
            )
        if not 0.0 <= float(weather["relative_humidity"]) <= 1.0:
            raise MofScreeningError(
                "case.weather.relative_humidity must be in [0,1]",
                field="case.weather.relative_humidity",
            )
        if float(weather["pressure_Pa"]) <= 0.0:
            raise MofScreeningError(
                "case.weather.pressure_Pa must be positive",
                field="case.weather.pressure_Pa",
            )
        if not 0.0 < float(weather["co2_ppm_dry"]) < 1e6:
            raise MofScreeningError(
                "case.weather.co2_ppm_dry must be in (0, 1000000)",
                field="case.weather.co2_ppm_dry",
            )
    else:
        start_utc = _string(weather, "start_utc", "case.weather")
        try:
            datetime.fromisoformat(start_utc.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except ValueError as error:
            raise MofScreeningError(
                "case.weather.start_utc must be an ISO-8601 UTC interval start",
                field="case.weather.start_utc",
            ) from error

    equipment = _mapping(case.get("equipment"), "case.equipment")
    for key in (
        "dry_shaped_mass_kg",
        "bed_thickness_m",
        "bed_face_area_m2",
        "particle_diameter_m",
        "bed_void_fraction",
        "airflow_m3_s",
        "heater_power_W",
        "effective_bed_conductivity_W_m_K",
        "gas_solid_heat_transfer_W_m2_K",
        "wall_heat_capacity_J_K",
        "wall_loss_UA_W_K",
        "wall_cooling_UA_W_K",
    ):
        _number(equipment, key, "case.equipment", minimum=0.0, exclusive_minimum=True)
    if not 0.0 < float(equipment["bed_void_fraction"]) < 1.0:
        raise MofScreeningError(
            "case.equipment.bed_void_fraction must be in (0,1)",
            field="case.equipment.bed_void_fraction",
        )

    for key, floor, ceiling in (
        ("dead_volume_m3", 0.0, None), ("knockout_C", 0.01, 100.0),
        ("cooler_power_W", 1e-6, None), ("leak_Pa_m3_s", 0.0, None),
    ):
        if key in equipment:
            _number(equipment, key, "case.equipment", minimum=floor, maximum=ceiling)
    if engine != SPATIAL and "leak_Pa_m3_s" in equipment and equipment["leak_Pa_m3_s"] != 1e-5:
        raise MofScreeningError("The adjustable leak uses the spatial engine.", field="case.equipment.leak_Pa_m3_s")
    regeneration = _mapping(case.get("regeneration", {"kind": "electrical"}), "case.regeneration")
    kind = _string(regeneration, "kind", "case.regeneration")
    if kind not in ("electrical", "direct_steam", "steam_jacket"):
        raise MofScreeningError("Unknown regeneration method.", field="case.regeneration.kind")
    if kind != "electrical":
        if engine != SPATIAL:
            raise MofScreeningError("Select the spatial engine for steam regeneration.", field="case.engine")
        from ..water_properties import saturation_temperature_K
        pressure = _number(regeneration, "supply_pressure_Pa", "case.regeneration", minimum=620, maximum=1500000)
        feed_C = _number(regeneration, "feedwater_C", "case.regeneration", minimum=0.01, maximum=190)
        if feed_C + 273.15 > saturation_temperature_K(pressure):
            raise MofScreeningError("Feedwater must be at or below the supply boiling point.", field="case.regeneration.feedwater_C")
        _number(regeneration, "generator_efficiency", "case.regeneration", minimum=0.001, maximum=1)
        if regeneration.get("generator_carrier") not in ("electricity", "external_heat"):
            raise MofScreeningError("Select electricity or external heat for steam generation.", field="case.regeneration.generator_carrier")
        if kind == "direct_steam":
            _number(regeneration, "flow_kg_h", "case.regeneration", minimum=0)
            _number(regeneration, "phase_transfer_s", "case.regeneration", minimum=1e-4)
            _number(regeneration, "drain_time_s", "case.regeneration", minimum=1e-4)
            _number(regeneration, "sorption_plateau_activity", "case.regeneration", minimum=0.01, maximum=0.999)
            _boolean(regeneration, "trim_heater", "case.regeneration")
        else:
            _number(regeneration, "jacket_UA_W_K", "case.regeneration", minimum=1e-6)
    cycle = _mapping(case.get("cycle"), "case.cycle")
    for key in (
        "adsorption_s",
        "evacuation_s",
        "desorption_s",
        "cooling_max_s",
        "repressurization_max_s",
        "regeneration_temperature_C",
        "regeneration_pressure_Pa",
        "purge_s",
    ):
        _number(cycle, key, "case.cycle", minimum=0.0)
    for key in (
        "adsorption_s",
        "evacuation_s",
        "desorption_s",
        "cooling_max_s",
        "repressurization_max_s",
        "regeneration_pressure_Pa",
    ):
        _number(cycle, key, "case.cycle", minimum=0.0, exclusive_minimum=True)
    cycle_seconds = sum(
        float(cycle[key])
        for key in (
            "adsorption_s",
            "evacuation_s",
            "desorption_s",
            "cooling_max_s",
            "repressurization_max_s",
        )
    )
    if cycle_seconds > 8.0 * 3600.0:
        raise MofScreeningError(
            "case.cycle full cycle duration cannot exceed eight hours",
            field="case.cycle",
        )

    solver = _mapping(case.get("solver"), "case.solver")
    cells = _integer(solver, "cells", "case.solver", minimum=1, maximum=32)
    if engine == SPATIAL and cells < 2:
        raise MofScreeningError(
            "spatial screening requires at least two through-thickness cells",
            field="case.solver.cells",
        )
    if engine == LUMPED and cells != 1:
        raise MofScreeningError(
            "lumped companion requires exactly one well-mixed cell",
            field="case.solver.cells",
        )
    _number(solver, "rtol", "case.solver", minimum=1e-9, maximum=1e-2)
    _number(solver, "max_step_s", "case.solver", minimum=1.0, maximum=3600.0)
    samples = _integer(
        solver, "samples_per_phase", "case.solver", minimum=3, maximum=121
    )

    initial = _mapping(case.get("initial_state"), "case.initial_state")
    basis = _string(initial, "basis", "case.initial_state")
    if basis not in ("equilibrated_with_first_weather", "clean_dry"):
        raise MofScreeningError(
            "case.initial_state.basis must be 'equilibrated_with_first_weather' "
            "or 'clean_dry'",
            field="case.initial_state.basis",
        )
    _boolean(initial, "cold_start", "case.initial_state")

    collection = _mapping(case.get("collection"), "case.collection")
    if _string(collection, "product_route", "case.collection") not in (
        "all_desorption_and_cooling", "desorption_only"
    ):
        raise MofScreeningError("Unknown product collection route", field="case.collection.product_route")
    condenser_name = "ideal_pump_inlet_configured" if equipment.get("knockout_C", 55.0) != 55.0 else "ideal_pump_inlet_55C"
    if collection.get("knockout_model") != condenser_name:
        raise MofScreeningError(
            f"case.collection.knockout_model must be {condenser_name!r}",
            field="case.collection.knockout_model",
        )
    if engine == LUMPED and (
        float(cycle["purge_s"]) != 0.0
        or collection["product_route"] != "all_desorption_and_cooling"
    ):
        raise MofScreeningError(
            "The accepted lumped port does not support an initial purge or "
            "desorption-only routing; use purge_s=0 and "
            "product_route='all_desorption_and_cooling', or select spatial.",
            field="case.collection.product_route",
        )

    case["weather"]["hours"] = hours
    case["solver"]["cells"] = cells
    case["solver"]["samples_per_phase"] = samples
    case["engine"] = engine
    case["operating_policy"] = deepcopy(SHARED_ACTUATOR_POLICY)
    policy = case["operating_policy"]
    if kind != "electrical":
        policy["regeneration"] = deepcopy(dict(regeneration))
        policy["source"] = "Metered direct steam injection" if kind == "direct_steam" else "Indirect condensing-steam jacket"
        if kind == "direct_steam":
            policy["representation_difference"]["spatial"] = "Axial wet packed bed with distributed wall contact"
            policy["representation_difference"]["pressure_control"] = "Outlet pressure controls pump withdrawal"
            if not regeneration["trim_heater"]:
                policy["phases"]["desorption"]["heater"] = "disabled"
        else:
            for name in ("adsorption", "evacuation", "repressurization"):
                policy["phases"][name]["heater"] = "disabled"
    policy["hardware"]["heater_power_W"] = float(equipment["heater_power_W"])
    if "cooler_power_W" in equipment:
        policy["hardware"]["combined_cooling_power_W"] = equipment["cooler_power_W"]
    if "knockout_C" in equipment:
        policy["hardware"]["conditioner_target_C"] = equipment["knockout_C"]
    return case


def load_defaults() -> dict:
    return read_json_strict(CONFIG_ROOT / "defaults.json")


def load_builtin_material(material_id: str) -> dict:
    if not isinstance(material_id, str) or "/" in material_id or "\\" in material_id:
        raise MofScreeningError("invalid material id", field="material.material_id")
    candidates = {
        "purolite_a110_humid": MATERIAL_ROOT / "purolite_a110_humid.json",
        "hypothetical_mmen_like": MATERIAL_ROOT / "hypothetical_mmen_like.json",
        "calf20_shaped": MATERIAL_ROOT / "calf20_shaped.json",
        "synthetic_independent_demo": MATERIAL_ROOT / "synthetic_independent_demo.json",
        "mmen_mg2_dobpdc_mcdonald2012_conditional": (
            MATERIAL_ROOT / "mmen_mg2_dobpdc_mcdonald2012_conditional.json"
        ),
        "lewatit_vp_oc_1065_low2023_conditional": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_low2023_conditional.json"
        ),
        "mmen_mg2_hob_powder_2024_dry_conditional": (
            MATERIAL_ROOT / "mmen_mg2_hob_powder_2024_dry_conditional.json"
        ),
        "mmen_mg2_hob_pvdf_2024_dry_conditional": (
            MATERIAL_ROOT / "mmen_mg2_hob_pvdf_2024_dry_conditional.json"
        ),
        "mof808_lys_2024_25c_dry_conditional": (
            MATERIAL_ROOT / "mof808_lys_2024_25c_dry_conditional.json"
        ),
        "mof808_tapa_2024_25c_dry_conditional": (
            MATERIAL_ROOT / "mof808_tapa_2024_25c_dry_conditional.json"
        ),
        "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional": (
            MATERIAL_ROOT / "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional.json"
        ),
        "cof1000_2025_25c_split_rate_dry_conditional": (
            MATERIAL_ROOT / "cof1000_2025_25c_split_rate_dry_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_wadst_conditional": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_wadst_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_hot116_conditional": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_hot116_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_cold_hot116_conditional": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_cold_hot116_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_advisory_temperature_conditional": (
            MATERIAL_ROOT
            / "lewatit_vp_oc_1065_young2021_advisory_temperature_conditional.json"
        ),
        "lewatit_vp_oc_1065_young2021_dry_bdf_recovery_v1": (
            MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_dry_bdf_recovery_v1.json"
        ),
        "cof999_2024_25c_dry_conditional": (
            MATERIAL_ROOT / "cof999_2024_25c_dry_conditional.json"
        ),
        "mil101_pei800_30_glycerin_2026_25c_dry_conditional": (
            MATERIAL_ROOT / "mil101_pei800_30_glycerin_2026_25c_dry_conditional.json"
        ),
        "mil101_pei800_30_2026_25c_dry_conditional": (
            MATERIAL_ROOT / "mil101_pei800_30_2026_25c_dry_conditional.json"
        ),
        "cof1000_humid_water_shift_scenario_v1": (
            MATERIAL_ROOT / "cof1000_humid_water_shift_scenario_v1.json"
        ),
        "cof999_humid_water_shift_scenario_v1": (
            MATERIAL_ROOT / "cof999_humid_water_shift_scenario_v1.json"
        ),
        "mmen_independent_unary_humid_diagnostic_v1": (
            MATERIAL_ROOT / "mmen_independent_unary_humid_diagnostic_v1.json"
        ),
        "hob_powder_measured_water_transport_independent_v1": (
            MATERIAL_ROOT / "hob_powder_measured_water_transport_independent_v1.json"
        ),
        "hob_pvdf_same_framework_water_transfer_scenario_v1": (
            MATERIAL_ROOT / "hob_pvdf_same_framework_water_transfer_scenario_v1.json"
        ),
        "tepa50_finite_window_water_proxy_scenario_v1": (
            MATERIAL_ROOT / "tepa50_finite_window_water_proxy_scenario_v1.json"
        ),
        "mof808_lys_humid_one_node_scenario_v1": (
            MATERIAL_ROOT / "mof808_lys_humid_one_node_scenario_v1.json"
        ),
        "mof808_tapa_water_shift_scenario_v1": (
            MATERIAL_ROOT / "mof808_tapa_water_shift_scenario_v1.json"
        ),
        "mil101_pei800_control_measured_water_diagnostic_v1": (
            MATERIAL_ROOT / "mil101_pei800_control_measured_water_diagnostic_v1.json"
        ),
        "mil101_pei800_glycerin_measured_water_diagnostic_v1": (
            MATERIAL_ROOT / "mil101_pei800_glycerin_measured_water_diagnostic_v1.json"
        ),
        "calf20_independent_water_diagnostic_v1": (
            MATERIAL_ROOT / "calf20_independent_water_diagnostic_v1.json"
        ),
    }
    path = candidates.get(material_id)
    if path is None:
        raise MofScreeningError("unknown built-in material", status=404)
    return validate_material(read_json_strict(path))


def build_case(material_document: Any, case_document: Any) -> tuple[dict, dict]:
    material = validate_material(material_document)
    case = validate_case(case_document)
    if (case.get("regeneration", {}).get("kind") == "direct_steam"
            and material["equilibrium_family"] != "young2021_advisory_wadst_gab"):
        raise MofScreeningError("Direct steam currently supports the advisory Young Lewatit wet-material packages. Load the reviewed humid Lewatit material.", field="material.equilibrium_family")
    temperature_policy = effective_temperature_policy(case, material)
    maximum_temperature = temperature_policy["enforcement_limit_C"]
    regeneration = float(case["cycle"]["regeneration_temperature_C"])
    if maximum_temperature is not None and regeneration > float(maximum_temperature):
        raise MofScreeningError(
            "case.cycle.regeneration_temperature_C exceeds the selected "
            "temperature guard policy",
            field="case.cycle.regeneration_temperature_C",
        )
    if material.get("water", {}).get(
        "model"
    ) == "dry_only" and not is_humidity_candidate(material):
        weather = case["weather"]
        if weather["mode"] == FIXED_WEATHER:
            humidity_values = [float(weather["relative_humidity"])]
        else:
            # Imported lazily because the weather module imports this contract.
            from .weather import load_weather_window

            humidity_values = [
                float(row.relative_humidity)
                for row in load_weather_window(weather).rows
            ]
        if any(value != 0.0 for value in humidity_values):
            raise MofScreeningError(
                "This material package contains dry-only water handling; "
                "every selected weather interval must have relative_humidity "
                "exactly zero.",
                field="case.weather.relative_humidity",
            )
    if material["equilibrium_family"] in (
        "young2021_wadst_gab",
        "young2021_hot_wadst_gab_extrapolated",
        "young2021_cold_hot_wadst_gab_extrapolated",
        "young2021_advisory_wadst_gab",
    ):
        weather = case["weather"]
        if weather["mode"] == FIXED_WEATHER:
            rows = [weather]
        else:
            from .weather import load_weather_window

            rows = load_weather_window(weather).rows
        if material["equilibrium_family"] != "young2021_advisory_wadst_gab":
            for row in rows:
                temperature_K = (
                    float(row["temperature_C"])
                    if isinstance(row, Mapping)
                    else float(row.temperature_C)
                ) + 273.15
                relative_humidity = (
                    float(row["relative_humidity"])
                    if isinstance(row, Mapping)
                    else float(row.relative_humidity)
                )
                family = material["equilibrium_family"]
                temperature_floor = (
                    293.15
                    if family == "young2021_cold_hot_wadst_gab_extrapolated"
                    else 298.15
                )
                temperature_limit = (
                    343.15 if family == "young2021_wadst_gab" else 393.15
                )
                if not (
                    temperature_floor - 1e-9
                    <= temperature_K
                    <= temperature_limit + 1e-9
                ):
                    raise MofScreeningError(
                        (
                            "Young2021 weather temperature is outside strict "
                            "298.15-343.15 K support"
                            if temperature_limit == 343.15
                            else (
                                "cold+hot Young2021 weather temperature is outside "
                                "the explicit 293.15-393.15 K evaluation envelope"
                                if temperature_floor == 293.15
                                else "hot Young2021 weather temperature is outside "
                                "the explicit 298.15-393.15 K evaluation envelope"
                            )
                        ),
                        field="case.weather.temperature_C",
                    )
                if not (0.0 <= relative_humidity <= 0.55 + 1e-12):
                    raise MofScreeningError(
                        "Young2021 weather relative humidity is outside strict "
                        "0-0.55 support",
                        field="case.weather.relative_humidity",
                    )
        else:
            if weather["mode"] == FIXED_WEATHER:
                from .weather import (
                    load_weather_window as advisory_load_weather_window,
                )

                rows = advisory_load_weather_window(weather).rows
            from . import young2021_advisory_temperature as advisory_law

            for index, row in enumerate(rows):
                temperature_K = float(row.temperature_C) + 273.15
                relative_humidity = float(row.relative_humidity)
                if not 0.0 <= relative_humidity < 1.0:
                    raise MofScreeningError(
                        "temperature-advisory weather row must have finite "
                        "RH in [0,1)",
                        field="case.weather.relative_humidity",
                        diagnostics={
                            "row_index": index,
                            "relative_humidity": relative_humidity,
                            "domain": "weather_relative_humidity",
                        },
                    )
                co2_partial_Pa = None
                water_partial_Pa = None
                try:
                    water_partial_Pa = row.prepared_buck_water_vapour_partial_Pa
                    co2_partial_Pa = (
                        float(row.pressure_Pa)
                        * (1.0 - float(row.water_mole_fraction))
                        * float(row.co2_ppm_dry)
                        * 1e-6
                    )
                    advisory_law.validate_equation_inputs(
                        temperature_K,
                        co2_partial_Pa,
                        water_partial_Pa,
                        0.0,
                    )
                    local_rh = advisory_law.local_relative_humidity(
                        temperature_K, water_partial_Pa
                    )
                    advisory_law.gab_water_loading(local_rh, temperature_K)
                except Exception as error:
                    raise MofScreeningError(
                        "temperature-advisory weather row is outside a genuine "
                        f"equilibrium branch: {error}",
                        field="case.weather",
                        diagnostics={
                            **(getattr(error, "diagnostics", None) or {}),
                            "row_index": index,
                            "temperature_K": temperature_K,
                            "relative_humidity": relative_humidity,
                            "co2_partial_Pa": co2_partial_Pa,
                            "water_partial_Pa": water_partial_Pa,
                            "discretionary_temperature_range_enforced": False,
                        },
                    ) from error
        if case["engine"] == LUMPED:
            raise MofScreeningError(
                "The Young2021 coupled-humidity material is currently "
                "supported in the spatial engine only. The lumped BDF path "
                "remained CPU-active without accepting a simulated step in "
                "both the default and bounded-Jacobian tests, so it is "
                "withheld rather than presented as usable.",
                field="case.engine",
            )
    return material, case


def preview_payload(material: dict, case: dict, *, source: str) -> dict:
    material_sha = canonical_sha256(material)
    case_sha = canonical_sha256(case)
    runtime_policy = effective_runtime_policy(case)
    temperature_policy = effective_temperature_policy(case, material)
    fixed_phase_seconds = sum(case["cycle"][key] for key in ("adsorption_s", "evacuation_s", "desorption_s"))
    horizon_s = float(case["weather"]["hours"]) * 3600.0
    minimum_cycle = (fixed_phase_seconds if case["engine"] == SPATIAL else
                     case["cycle"]["adsorption_s"] + case["cycle"]["desorption_s"])
    max_cycles = max(1, math.ceil(horizon_s / minimum_cycle))
    maximum_duration = fixed_phase_seconds + case["cycle"]["cooling_max_s"] + case["cycle"]["repressurization_max_s"]
    if case["engine"] == SPATIAL and maximum_duration > horizon_s + 1e-9:
        raise MofScreeningError("Spatial screening needs a horizon covering one configured cycle.",
                                field="case.weather.hours")
    from .materials import build_material_adapter
    from .spatial import state_layout
    adapter = build_material_adapter(material)
    cells = int(case["solver"]["cells"])
    state_count = (state_layout(cells, extended=True, n_storage=adapter.n_storage,
                                wet=case.get("regeneration", {}).get("kind") == "direct_steam").size
                   if case["engine"] == SPATIAL else adapter.n_sites + 45)
    direct_contact = case.get("regeneration", {}).get("kind") == "direct_steam"
    warnings = [
        "Screening diagnostic; it is not hardware qualification or a measured winner.",
        "Requested regeneration, vacuum and cooling targets are advisory outcomes; a target miss alone does not reject or rank down a case.",
        "Humidity is shared gas forcing and is not a validated competitive humid-mixture law unless the selected adapter explicitly provides one.",
        "Electricity covers modeled heater, fan, pump and cooling inside this boundary. Upstream drying and downstream compression are excluded.",
        ("Direct steam uses an axial packed bed with a finite-rate liquid film, distributed drainage and an assumed sorption plateau. Calibrate these inputs on wet beads." if direct_contact else
         "The spatial engine resolves the transverse half-bed with parallel channels; it does not resolve axial breakthrough or manifold redistribution."),
        "Execution runtime controls are operational; stopping or imposing a selected wall-time limit does not change physical operating conditions.",
        "The package temperature scalar is reported through its exact-package guard classification; an unqualified scenario guard is not a measured physical degradation limit.",
    ]
    if material["equilibrium_family"] == "purolite_a110_rh_curves_gab":
        warnings.append(
            "A110 CO2 curves were measured on water-presaturated beads at 25 C and "
            "0/30/55/80% RH. Above 80% the CO2 curve is held fixed while modeled water "
            "uptake continues. Humid feed near 400 ppm is below the first humid "
            "pressure points (52-65 Pa). Other temperatures and hot wet desorption "
            "use screening assumptions."
        )
    if material.get("synthetic"):
        warnings.append(
            "This material package is a synthetic interface fixture, not evidence about a real MOF."
        )
    if material.get("evidence", {}).get("edit_detected"):
        warnings.append(
            "Physical coefficients were edited after loading the built-in "
            "package. The original evidence is preserved in the exported "
            "input, but this run is classified as a user-edited assumption."
        )
    if material.get("water", {}).get(
        "model"
    ) == "dry_only" and not is_humidity_candidate(material):
        warnings.append(
            "This dry-only material has zero physical water storage. "
            "Relative humidity above zero is rejected rather than represented."
        )
    if is_humidity_candidate(material):
        extension = material["humidity_candidate"]
        warnings.append(
            "EXPLORATORY HUMIDITY CANDIDATE: "
            f"evidence tier {extension['evidence_tier']}. Reciprocally closed "
            "thermodynamics and a validated humid cycle are not claimed."
        )
        warnings.append(
            "Water saturation uses retained Antoine through 373 K and IAPWS "
            "above 373 K below the critical point. Water activity outside "
            "[0,1] and temperatures outside the water-support domain are "
            "rejected."
        )
        coupling_kind = material["humidity_candidate"]["co2_coupling"]["kind"]
        if coupling_kind != "unsupported":
            warnings.append(
                "CO2 coupling is one-way and uses actual stored water. The "
                "gas-derived water target controls only water-rate relaxation; "
                "no reciprocal humidity thermodynamics or interaction enthalpy "
                "is represented."
            )
    if material.get("material_id") == ("mmen_mg2_dobpdc_mcdonald2012_conditional"):
        warnings.append(
            "MMEN INDEPENDENT-COMPONENT HUMID DIAGNOSTIC: separate unary "
            "water storage contributes sorption heat and moisture-dependent "
            "gas, pumping and conditioning burdens. Direct humidity "
            "enhancement or competition of CO2 is absent. The water Sips "
            "curve is from a separate Bose specimen; water and hot "
            "desorption rates remain assumed."
        )
    if material.get("equilibrium_family") == "young2021_wadst_gab":
        warnings.append(
            "Young2021 water loading is evaluated from local p_H2O and "
            "Buck p_sat(T_bed), then feeds the CO2 target. Humid-cycle "
            "kinetics and the 70/46 kJ/mol native energy closure are "
            "conditional assumptions, not measured validation. RH between "
            "the dry, 30% and 55% source conditions is model interpolation."
        )
    if material.get("equilibrium_family") == ("young2021_hot_wadst_gab_extrapolated"):
        warnings.append(
            "HOT CONDITIONAL EXTRAPOLATION: the measured Young coupled "
            "support ends at 70 C and RH 0.55. This 25-120 C scenario uses "
            "the unchanged Young coefficients with IAPWS saturation. Hot "
            "binary coupling, LDF rates and energy transfer are unvalidated; "
            "no measured 116 C accuracy is claimed."
        )
        adsorption_rates = tuple(
            float(value) for value in material["kinetics"]["adsorption_rate_s_inv"]
        )
        desorption_rates = tuple(
            float(value) for value in material["kinetics"]["desorption_rate_s_inv"]
        )
        process_heats = (
            float(material["co2"]["heat_J_mol"]),
            float(material["water"]["heat_J_mol"]),
        )
        if (
            adsorption_rates != tuple(APPROVED_LDF_RATES_S_INV)
            or desorption_rates != tuple(APPROVED_LDF_RATES_S_INV)
            or process_heats != tuple(APPROVED_PROCESS_HEATS_J_MOL)
        ):
            warnings.append(
                "The hot package rates or process heats differ from the "
                "approved unchanged Young values; source-reproduction claims "
                "are withdrawn for this edited calculation."
            )
    if material.get("equilibrium_family") == (
        "young2021_cold_hot_wadst_gab_extrapolated"
    ):
        warnings.append(
            "COLD-TRANSIENT + HOT CONDITIONAL EXTRAPOLATION: measured Young "
            "coupled support remains 25-70 C. This scenario continues the "
            "unchanged published correlations down to 20 C and up to 120 C. "
            "The 20 C side is an assumed response to the observed physical "
            "evacuation cooling, not cold-data validation or a Low-to-Young "
            "sample transfer. Water normalization is evaluation-only and "
            "does not alter stored inventories or energy states."
        )
        adsorption_rates = tuple(
            float(value) for value in material["kinetics"]["adsorption_rate_s_inv"]
        )
        desorption_rates = tuple(
            float(value) for value in material["kinetics"]["desorption_rate_s_inv"]
        )
        process_heats = (
            float(material["co2"]["heat_J_mol"]),
            float(material["water"]["heat_J_mol"]),
        )
        if (
            adsorption_rates != tuple(APPROVED_LDF_RATES_S_INV)
            or desorption_rates != tuple(APPROVED_LDF_RATES_S_INV)
            or process_heats != tuple(APPROVED_PROCESS_HEATS_J_MOL)
        ):
            warnings.append(
                "The cold+hot package rates or process heats differ from "
                "the approved unchanged Young values; source-reproduction "
                "claims are withdrawn for this edited calculation."
            )
    if material.get("equilibrium_family") == ("young2021_advisory_wadst_gab"):
        advisory = material.get("temperature_advisory") or {}
        warnings.append(
            "TEMPERATURE-ADVISORY CONDITIONAL EXTRAPOLATION: measured Young "
            "coupled support is 25-70 C. The requested regeneration target "
            "and the former 20-120 C evaluation range are advisory; actual "
            "accepted temperatures and solver query temperatures are "
            "recorded separately. No temperature clamp or discretionary "
            "temperature stop is applied. Equation-domain and nonfinite "
            "failures remain hard rejections, and no measured hot humid "
            "accuracy is claimed."
        )
        if advisory.get("historical_range_is_hard_stop") is not False:
            warnings.append(
                "The temperature-advisory package does not explicitly record "
                "that its historical range is non-enforcing."
            )
        adsorption_rates = tuple(
            float(value) for value in material["kinetics"]["adsorption_rate_s_inv"]
        )
        desorption_rates = tuple(
            float(value) for value in material["kinetics"]["desorption_rate_s_inv"]
        )
        process_heats = (
            float(material["co2"]["heat_J_mol"]),
            float(material["water"]["heat_J_mol"]),
        )
        if (
            adsorption_rates != tuple(APPROVED_LDF_RATES_S_INV)
            or desorption_rates != tuple(APPROVED_LDF_RATES_S_INV)
            or process_heats != tuple(APPROVED_PROCESS_HEATS_J_MOL)
        ):
            warnings.append(
                "The temperature-advisory package rates or process heats "
                "differ from the approved unchanged Young values; "
                "source-reproduction claims are withdrawn."
            )
    return {
        "schema": PREVIEW_SCHEMA,
        "source": source,
        "material_id": material["material_id"],
        "material_name": material["name"],
        "material_sha256": material_sha,
        "case_sha256": case_sha,
        "engine": case["engine"],
        "weather": deepcopy(case["weather"]),
        "cycle": deepcopy(case["cycle"]),
        "solver": deepcopy(case["solver"]),
        "horizon_s": horizon_s,
        "maximum_cycle_starts": max_cycles,
        "state_dimension_estimate": state_count,
        "allocation": {
            "single_process_integration": True,
            "automatic_retries": 0,
            "maximum_wall_s": (
                None
                if runtime_policy["mode"] == MANUAL_STOP
                else runtime_policy["time_limit_s"]
            ),
            "automatic_wall_termination": (runtime_policy["mode"] == TIME_LIMIT),
            "manual_stop_enabled": True,
            "runtime_policy": runtime_policy,
            "soft_runtime_notice_s": 300,
            "reserved_process_starts": 1,
        },
        "assumptions": {
            "material": deepcopy(material.get("assumptions", [])),
            "evidence": deepcopy(material.get("evidence", {})),
            "initial_state": deepcopy(case.get("initial_state", {})),
            "collection": deepcopy(case.get("collection", {})),
            "operating_policy": deepcopy(
                case.get("operating_policy", SHARED_ACTUATOR_POLICY)
            ),
            "runtime_control": runtime_policy,
            "temperature_advisory": deepcopy(material.get("temperature_advisory", {})),
        },
        "warnings": warnings,
        "temperature_policy": temperature_policy,
        "material": material,
        "case": case,
    }
