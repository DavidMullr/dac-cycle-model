"""Build native screening material packages from a guided form payload."""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
import math
import re
from typing import Any, Mapping

from .contract import (
    MATERIAL_SCHEMA,
    SUPPORTED_FAMILIES,
    MofScreeningError,
    canonical_sha256,
    semantic_json_sha256,
    validate_material,
)


GUIDED_RESULT_SCHEMA = "mof-screening-guided-material-result/1"
MATERIAL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def _issue(issues: list[dict], field: str, message: str) -> None:
    issues.append({"field": field, "message": message})


def _text(
    draft: Mapping,
    key: str,
    field: str,
    issues: list[dict],
) -> str | None:
    value = draft.get(key)
    if not isinstance(value, str) or not value.strip():
        _issue(issues, field, "Required text is blank.")
        return None
    return value.strip()


def _number(
    draft: Mapping,
    key: str,
    field: str,
    issues: list[dict],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float | None:
    raw = draft.get(key)
    if raw is None or raw == "":
        _issue(issues, field, "Required number is blank.")
        return None
    if isinstance(raw, bool):
        _issue(issues, field, "Must be a finite number; boolean is invalid.")
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        _issue(issues, field, "Must be a finite number.")
        return None
    if not math.isfinite(value):
        _issue(issues, field, "Must be finite.")
        return None
    invalid = (
        minimum is not None
        and (value <= minimum if exclusive_minimum else value < minimum)
    )
    if invalid:
        relation = ">" if exclusive_minimum else ">="
        _issue(issues, field, f"Must be {relation} {minimum:g}.")
        return None
    if maximum is not None and value > maximum:
        _issue(issues, field, f"Must be <= {maximum:g}.")
        return None
    return value


def _number_list(
    draft: Mapping,
    key: str,
    field: str,
    issues: list[dict],
    *,
    length: int,
    allow_zero: bool = False,
) -> list[float] | None:
    raw = draft.get(key)
    if not isinstance(raw, str) or not raw.strip():
        _issue(
            issues,
            field,
            f"Enter exactly {length} comma-separated numbers.",
        )
        return None
    try:
        values = [float(item.strip()) for item in raw.split(",")]
    except ValueError:
        _issue(issues, field, "Every comma-separated entry must be numeric.")
        return None
    if len(values) != length:
        _issue(issues, field, f"Requires exactly {length} entries.")
        return None
    if any(
        not math.isfinite(value)
        or value < 0.0
        or (not allow_zero and value == 0.0)
        for value in values
    ):
        condition = "nonnegative" if allow_zero else "positive"
        _issue(issues, field, f"All entries must be finite and {condition}.")
        return None
    return values


def _curve(
    draft: Mapping,
    key: str,
    field: str,
    issues: list[dict],
    *,
    kind: str,
) -> list[list[float]] | None:
    raw = draft.get(key)
    if not isinstance(raw, str) or not raw.strip():
        _issue(issues, field, "Required CSV curve is blank.")
        return None
    rows = list(csv.reader(io.StringIO(raw)))
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    expected = (
        ["pressure_Pa", "loading_mol_kg"]
        if kind == "co2"
        else ["relative_humidity", "loading_mol_kg"]
    )
    if not rows or [cell.strip() for cell in rows[0]] != expected:
        _issue(
            issues,
            field,
            "CSV header must be exactly "
            + ",".join(expected)
            + "; units are not inferred.",
        )
        return None
    values: list[list[float]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            _issue(
                issues,
                field,
                f"CSV row {row_number} must contain exactly two values.",
            )
            continue
        try:
            pair = [float(row[0].strip()), float(row[1].strip())]
        except ValueError:
            _issue(
                issues,
                field,
                f"CSV row {row_number} contains a nonnumeric value.",
            )
            continue
        if not all(math.isfinite(item) for item in pair):
            _issue(
                issues,
                field,
                f"CSV row {row_number} contains a non-finite value.",
            )
            continue
        values.append(pair)
    if len(values) < 3:
        _issue(issues, field, "Curve requires at least three data rows.")
    x_values = [row[0] for row in values]
    y_values = [row[1] for row in values]
    if any(b <= a for a, b in zip(x_values, x_values[1:])):
        _issue(issues, field, "Curve x values must be strictly increasing.")
    if any(b < a for a, b in zip(y_values, y_values[1:])):
        _issue(issues, field, "Curve loadings must be monotone nondecreasing.")
    if kind == "co2":
        if any(value <= 0.0 for value in x_values):
            _issue(issues, field, "CO2 pressures must be positive.")
    else:
        if any(not 0.0 <= value <= 1.0 for value in x_values):
            _issue(issues, field, "Relative humidity must stay in [0, 1].")
        if not x_values or x_values[0] != 0.0:
            _issue(issues, field, "Water curve must include RH=0 as its first row.")
    if any(value < 0.0 for value in y_values):
        _issue(issues, field, "Curve loadings cannot be negative.")
    if not y_values or max(y_values) <= 0.0:
        _issue(
            issues,
            field,
            "Curve needs a positive loading; an all-zero curve has no "
            "defined adapter storage scale.",
        )
    return values if not issues else None


def _preserved_block(
    base: Mapping | None,
    key: str,
    known_keys: set[str],
) -> dict:
    if not isinstance(base, Mapping):
        return {}
    block = base.get(key)
    if not isinstance(block, Mapping):
        return {}
    return {
        name: deepcopy(value)
        for name, value in block.items()
        if name not in known_keys
    }


def _declaration(
    evidence_type: str,
    statement: str,
) -> dict:
    return {
        "class": evidence_type,
        "guided_mode": evidence_type,
        "statement": statement,
    }


def _declaration_mode(entry: Any) -> str:
    if not isinstance(entry, Mapping):
        return "assumption"
    class_name = str(entry.get("class") or entry.get("guided_mode") or "").lower()
    if "synthetic" in class_name:
        return "synthetic"
    if any(
        token in class_name
        for token in ("source", "measured", "published")
    ):
        return "source"
    return "assumption"


def _preserved_declaration(
    base: Mapping | None,
    group: str,
    mode: str,
    statement: str,
) -> dict:
    if not isinstance(base, Mapping):
        return _declaration(mode, statement)
    provenance = base.get("provenance")
    if not isinstance(provenance, Mapping):
        return _declaration(mode, statement)
    aliases = {
        "co2": ("co2", "co2_sites", "co2_anchor"),
        "water": ("water", "water_curve"),
        "thermal": ("thermal",),
        "kinetics": ("kinetics",),
    }[group]
    original = next(
        (
            provenance[name]
            for name in aliases
            if isinstance(provenance.get(name), Mapping)
        ),
        None,
    )
    if not isinstance(original, Mapping):
        return _declaration(mode, statement)
    result = deepcopy(dict(original))
    previous_mode = _declaration_mode(original)
    if previous_mode != mode:
        result["loaded_class"] = original.get("class")
        result["class"] = mode
    result["guided_mode"] = mode
    result["statement"] = statement
    return result


def _first_original_evidence(evidence: Any) -> dict | None:
    if not isinstance(evidence, Mapping):
        return None
    original = deepcopy(dict(evidence))
    while isinstance(original.get("original_evidence"), Mapping):
        original = deepcopy(dict(original["original_evidence"]))
    return original


def _declaration_text(entry: Any) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    if isinstance(entry.get("statement"), str) and entry["statement"].strip():
        return entry["statement"].strip()
    class_name = str(entry.get("class") or "declaration").strip()
    source = (
        entry.get("source")
        or entry.get("doi")
        or entry.get("source_locator")
        or ""
    )
    return f"{class_name}: {source}".strip(": ")


def _guided_group_views(material: Mapping) -> dict[str, Any]:
    return {
        "identity_shaping": {
            "material_id": material.get("material_id"),
            "name": material.get("name"),
            "shaped_material": material.get("shaped_material"),
        },
        "co2": {
            "equilibrium_family": material.get("equilibrium_family"),
            "co2": material.get("co2"),
            "co2_water_penalty_per_mol_kg": material.get(
                "co2_water_penalty_per_mol_kg"
            ),
            "nitrogen": material.get("nitrogen"),
            "oxygen": material.get("oxygen"),
        },
        "water": {"water": material.get("water")},
        "thermal": {
            "reference_temperature_K": material.get("reference_temperature_K"),
            "maximum_bed_temperature_C": material.get(
                "maximum_bed_temperature_C"
            ),
            "solid_cp_J_kg_K": material.get("solid_cp_J_kg_K"),
            "adsorbed_phase_cp_J_kg_K": material.get(
                "adsorbed_phase_cp_J_kg_K"
            ),
        },
        "kinetics": {"kinetics": material.get("kinetics")},
    }


def _guided_group_hashes(material: Mapping) -> dict[str, str]:
    return {
        group: semantic_json_sha256(view)
        for group, view in _guided_group_views(material).items()
    }


def _guided_baseline(base: Mapping | None) -> dict | None:
    if not isinstance(base, Mapping):
        return None
    evidence = base.get("evidence")
    if isinstance(evidence, Mapping):
        baseline = evidence.get("guided_baseline")
        if (
            isinstance(baseline, Mapping)
            and isinstance(baseline.get("original_group_sha256"), Mapping)
        ):
            return deepcopy(dict(baseline))
    provenance = base.get("provenance")
    guided_entry = (
        provenance.get("guided_entry")
        if isinstance(provenance, Mapping)
        else None
    )
    original_package_sha256 = (
        guided_entry.get("base_package_sha256")
        if isinstance(guided_entry, Mapping)
        else None
    ) or canonical_sha256(base)
    declarations = {}
    if isinstance(provenance, Mapping):
        declarations = {
            "co2": _declaration_text(
                provenance.get("co2")
                or provenance.get("co2_sites")
                or provenance.get("co2_anchor")
            ),
            "water": _declaration_text(
                provenance.get("water") or provenance.get("water_curve")
            ),
            "thermal": _declaration_text(provenance.get("thermal")),
            "kinetics": _declaration_text(provenance.get("kinetics")),
        }
    return {
        "schema": "mof-screening-guided-baseline/1",
        "original_package_sha256": original_package_sha256,
        "original_group_sha256": _guided_group_hashes(base),
        "original_declarations": declarations,
    }


def _validity_value(
    raw: str,
    field: str,
    issues: list[dict],
) -> Any:
    stripped = raw.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as error:
            _issue(
                issues,
                field,
                f"Structured JSON is invalid: {error.msg}.",
            )
    return stripped


def build_guided_material(payload: Any) -> dict:
    """Return a native package or a complete field-issue list."""
    issues: list[dict] = []
    if not isinstance(payload, Mapping):
        _issue(issues, "draft", "Guided material payload must be an object.")
        return {
            "schema": GUIDED_RESULT_SCHEMA,
            "valid": False,
            "issues": issues,
            "material": None,
        }
    draft = payload.get("draft")
    if not isinstance(draft, Mapping):
        _issue(issues, "draft", "Guided material draft must be an object.")
        return {
            "schema": GUIDED_RESULT_SCHEMA,
            "valid": False,
            "issues": issues,
            "material": None,
        }
    base = draft.get("base_material")
    if base is not None and not isinstance(base, Mapping):
        _issue(issues, "base_material", "Loaded package must be an object.")
        base = None

    material_id = _text(draft, "material_id", "material_id", issues)
    if material_id and not MATERIAL_ID_PATTERN.fullmatch(material_id):
        _issue(
            issues,
            "material_id",
            "Use 3-64 lowercase letters, digits, underscores or hyphens.",
        )
    name = _text(draft, "name", "name", issues)
    family = _text(
        draft, "equilibrium_family", "equilibrium_family", issues
    )
    dry_only_water = (
        family == "split_rate_activity_shift_reference_isotherm"
        or (
            family == "activity_shift_reference_isotherm"
            and isinstance(base, Mapping)
            and isinstance(base.get("water"), Mapping)
            and base["water"].get("model") == "dry_only"
        )
    )
    if family and family not in SUPPORTED_FAMILIES:
        _issue(
            issues,
            "equilibrium_family",
            f"Unsupported law {family!r}; a new bounded adapter is required.",
        )
    group_modes = {}
    for key, group in (
        ("co2_evidence_type", "co2"),
        ("water_evidence_type", "water"),
        ("thermal_evidence_type", "thermal"),
        ("kinetics_evidence_type", "kinetics"),
    ):
        mode = _text(draft, key, f"provenance.{group}.mode", issues)
        if mode not in {"source", "assumption", "synthetic"}:
            _issue(
                issues,
                f"provenance.{group}.mode",
                "Choose source, assumption, or synthetic evidence.",
            )
        group_modes[group] = mode
    evidence_basis = _text(
        draft, "evidence_basis", "evidence_basis", issues
    )
    evidence_not_for = _text(
        draft, "evidence_not_for", "evidence_not_for", issues
    )
    shaping_basis = _text(
        draft, "shaping_basis", "shaped_material.basis", issues
    )
    binder_support = _text(
        draft,
        "binder_support",
        "shaped_material.binder_support",
        issues,
    )
    particle_geometry = _text(
        draft,
        "particle_geometry",
        "shaped_material.particle_geometry",
        issues,
    )
    reference_temperature = _number(
        draft,
        "reference_temperature_K",
        "reference_temperature_K",
        issues,
        minimum=250.0,
        maximum=450.0,
    )
    maximum_temperature = _number(
        draft,
        "maximum_bed_temperature_C",
        "maximum_bed_temperature_C",
        issues,
        minimum=40.0,
        maximum=500.0,
    )
    solid_cp = _number(
        draft,
        "solid_cp_J_kg_K",
        "solid_cp_J_kg_K",
        issues,
        minimum=0.0,
        exclusive_minimum=True,
    )

    declarations = {}
    for key, field in (
        ("co2_declaration", "provenance.co2"),
        ("water_declaration", "provenance.water"),
        ("thermal_declaration", "provenance.thermal"),
        ("kinetics_declaration", "provenance.kinetics"),
    ):
        declarations[field.split(".", 1)[1]] = _text(
            draft, key, field, issues
        )
    assumptions_text = _text(
        draft, "assumptions_text", "assumptions", issues
    )
    assumptions = (
        [
            item.strip()
            for item in re.split(r"[\r\n;]+", assumptions_text)
            if item.strip()
        ]
        if assumptions_text
        else []
    )
    validity = {}
    for key, field in (
        ("co2_validity", "co2"),
        ("water_validity", "water"),
        ("thermal_validity", "thermal"),
        ("kinetics_validity", "kinetics"),
    ):
        raw_validity = draft.get(key)
        if isinstance(raw_validity, (Mapping, list)):
            validity[field] = deepcopy(raw_validity)
        else:
            text_value = _text(
                draft, key, f"validity.{field}", issues
            )
            if text_value is not None:
                validity[field] = _validity_value(
                    text_value, f"validity.{field}", issues
                )
    uncertainty_class = _text(
        draft,
        "uncertainty_classification",
        "uncertainty.classification",
        issues,
    )
    uncertainty_statement = _text(
        draft,
        "uncertainty_statement",
        "uncertainty.statement",
        issues,
    )

    co2_heat = water_heat = None
    water_saturation = water_half_rh = water_sips_exponent = None
    nitrogen_henry = nitrogen_heat = None
    oxygen_henry = oxygen_heat = None
    if family == "independent_langmuir_sips":
        co2_heat = _number(
            draft, "co2_heat_J_mol", "co2.heat_J_mol", issues, minimum=0.0
        )
        co2_saturation = _number(
            draft,
            "co2_saturation_mol_kg",
            "co2.saturation_mol_kg",
            issues,
            minimum=0.0,
            exclusive_minimum=True,
        )
        co2_reference_loading = _number(
            draft,
            "co2_reference_loading_mol_kg",
            "co2.reference_loading_mol_kg",
            issues,
            minimum=0.0,
            exclusive_minimum=True,
        )
        co2_reference_pressure = _number(
            draft,
            "co2_reference_partial_pressure_Pa",
            "co2.reference_partial_pressure_Pa",
            issues,
            minimum=0.0,
            exclusive_minimum=True,
        )
        if (
            co2_saturation is not None
            and co2_reference_loading is not None
            and co2_reference_loading >= co2_saturation
        ):
            _issue(
                issues,
                "co2.reference_loading_mol_kg",
                "Reference loading must be below saturation.",
            )
        water_saturation = _number(
            draft,
            "water_saturation_mol_kg",
            "water.saturation_mol_kg",
            issues,
            minimum=0.0,
            exclusive_minimum=True,
        )
        water_half_rh = _number(
            draft,
            "water_half_loading_relative_humidity_at_reference",
            "water.half_loading_relative_humidity_at_reference",
            issues,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        )
        water_sips_exponent = _number(
            draft,
            "water_sips_exponent",
            "water.sips_exponent",
            issues,
            minimum=0.0,
            exclusive_minimum=True,
        )
        water_heat = _number(
            draft,
            "water_heat_J_mol",
            "water.heat_J_mol",
            issues,
            minimum=0.0,
        )
        co2_block = {
            "saturation_mol_kg": co2_saturation,
            "reference_loading_mol_kg": co2_reference_loading,
            "reference_partial_pressure_Pa": co2_reference_pressure,
            "heat_J_mol": co2_heat,
            "basis": "guided independent dry single-site Langmuir",
        }
        n_ads = 4
    elif family in (
        "activity_shift_reference_isotherm",
        "split_rate_activity_shift_reference_isotherm",
    ):
        co2_heat = _number(
            draft, "co2_heat_J_mol", "co2.heat_J_mol", issues, minimum=0.0
        )
        penalty = _number(
            draft,
            "co2_water_penalty_per_mol_kg",
            "co2_water_penalty_per_mol_kg",
            issues,
            minimum=0.0,
        )
        if dry_only_water and penalty is not None and penalty != 0.0:
            _issue(
                issues,
                "co2_water_penalty_per_mol_kg",
                "Dry-only materials require an exact zero CO2-water penalty.",
            )
        curve_temperature = _number(
            draft,
            "co2_curve_temperature_K",
            "co2.reference_curve.temperature_K",
            issues,
            minimum=250.0,
            maximum=450.0,
        )
        curve = _curve(
            draft,
            "co2_curve_csv",
            "co2.reference_curve",
            issues,
            kind="co2",
        )
        if not dry_only_water:
            water_saturation = _number(
                draft,
                "water_saturation_mol_kg",
                "water.saturation_mol_kg",
                issues,
                minimum=0.0,
                exclusive_minimum=True,
            )
            water_half_rh = _number(
                draft,
                "water_half_loading_relative_humidity_at_reference",
                "water.half_loading_relative_humidity_at_reference",
                issues,
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            )
            water_sips_exponent = _number(
                draft,
                "water_sips_exponent",
                "water.sips_exponent",
                issues,
                minimum=0.0,
                exclusive_minimum=True,
            )
            water_heat = _number(
                draft,
                "water_heat_J_mol",
                "water.heat_J_mol",
                issues,
                minimum=0.0,
            )
        co2_block = {
            "heat_J_mol": co2_heat,
            "reference_curve": {
                "temperature_K": curve_temperature,
                "pressure_Pa": [row[0] for row in curve] if curve else None,
                "loading_mol_kg": [row[1] for row in curve] if curve else None,
            },
            "basis": (
                "guided measured/supplied pressure-loading curve with the "
                "existing constant-isosteric-heat activity shift"
            ),
        }
        if family == "split_rate_activity_shift_reference_isotherm":
            fast_fraction = _number(
                draft,
                "split_fast_fraction",
                "co2.reservoir_fractions[0]",
                issues,
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            )
            slow_fraction = _number(
                draft,
                "split_slow_fraction",
                "co2.reservoir_fractions[1]",
                issues,
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            )
            if (
                fast_fraction is not None
                and slow_fraction is not None
                and not math.isclose(
                    fast_fraction + slow_fraction,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                _issue(
                    issues,
                    "co2.reservoir_fractions",
                    "Fast and slow kinetic reservoir fractions must sum to 1.",
                )
            co2_block["reservoir_order"] = ["fast", "slow"]
            co2_block["reservoir_fractions"] = [
                fast_fraction,
                slow_fraction,
            ]
            n_ads = 5
        else:
            n_ads = 4
    elif family == "site_resolved_concentration_langmuir":
        penalty = None
        site_values = {}
        for site in (1, 2):
            site_values[f"capacity{site}"] = _number(
                draft,
                f"site{site}_capacity_mol_kg",
                f"co2.site_capacity_mol_kg[{site - 1}]",
                issues,
                minimum=0.0,
                exclusive_minimum=True,
            )
            site_values[f"affinity{site}"] = _number(
                draft,
                f"site{site}_affinity_m3_mol",
                f"co2.site_affinity_m3_mol[{site - 1}]",
                issues,
                minimum=0.0,
            )
            site_values[f"energy{site}"] = _number(
                draft,
                f"site{site}_internal_energy_J_mol",
                f"co2.site_internal_energy_J_mol[{site - 1}]",
                issues,
                maximum=0.0,
            )
        co2_block = {
            "site_capacity_mol_kg": [
                site_values["capacity1"],
                site_values["capacity2"],
            ],
            "site_affinity_m3_mol": [
                site_values["affinity1"],
                site_values["affinity2"],
            ],
            "site_internal_energy_J_mol": [
                site_values["energy1"],
                site_values["energy2"],
            ],
            "basis": (
                "two ordered concentration-basis Langmuir sites; affinity is "
                "the prefactor b0 in theta=b0*exp(-U/RT)*c/(1+...)"
            ),
        }
        water_measurement = _number(
            draft,
            "water_measurement_temperature_K",
            "water.measurement_temperature_K",
            issues,
            minimum=0.0,
            exclusive_minimum=True,
        )
        water_reference = _number(
            draft,
            "water_reference_temperature_K",
            "water.reference_temperature_K",
            issues,
            minimum=0.0,
            exclusive_minimum=True,
        )
        water_heat = _number(
            draft,
            "water_sorption_heat_J_mol",
            "water.sorption_heat_J_mol",
            issues,
            minimum=0.0,
        )
        water_points = _curve(
            draft,
            "water_curve_csv",
            "water.points",
            issues,
            kind="water",
        )
        water_block = {
            "model": "temperature_activity_pchip",
            "measurement_temperature_K": water_measurement,
            "reference_temperature_K": water_reference,
            "sorption_heat_J_mol": water_heat,
            "points": water_points,
            "basis": "guided unary water curve with existing PCHIP transfer",
        }
        n_ads = 3
    else:
        co2_block = {}
        water_block = {}
        n_ads = 0

    if family in {
        "independent_langmuir_sips",
        "activity_shift_reference_isotherm",
        "split_rate_activity_shift_reference_isotherm",
    }:
        if dry_only_water:
            water_normalizer = _number(
                draft,
                "water_solver_normalizer_mol_kg",
                "water.solver_normalizer_mol_kg",
                issues,
                minimum=0.0,
                exclusive_minimum=True,
            )
            water_block = {
                "model": "dry_only",
                "physical_storage_mol_kg": 0.0,
                "solver_normalizer_mol_kg": water_normalizer,
                "basis": (
                    "Dry-only structural water state; zero physical water "
                    "storage and nonzero solver normalizer only."
                ),
            }
        else:
            water_block = {
                "saturation_mol_kg": water_saturation,
                "half_loading_relative_humidity_at_reference": water_half_rh,
                "sips_exponent": water_sips_exponent,
                "heat_J_mol": water_heat,
                "basis": "guided independent unary water Sips law",
            }
        nitrogen_henry = _number(
            draft,
            "nitrogen_henry_mol_kg_Pa_at_reference",
            "nitrogen.henry_mol_kg_Pa_at_reference",
            issues,
            minimum=0.0,
        )
        nitrogen_heat = _number(
            draft,
            "nitrogen_heat_J_mol",
            "nitrogen.heat_J_mol",
            issues,
            minimum=0.0,
        )
        oxygen_henry = _number(
            draft,
            "oxygen_henry_mol_kg_Pa_at_reference",
            "oxygen.henry_mol_kg_Pa_at_reference",
            issues,
            minimum=0.0,
        )
        oxygen_heat = _number(
            draft,
            "oxygen_heat_J_mol",
            "oxygen.heat_J_mol",
            issues,
            minimum=0.0,
        )
        nitrogen_block = {
            "henry_mol_kg_Pa_at_reference": nitrogen_henry,
            "heat_J_mol": nitrogen_heat,
        }
        oxygen_block = {
            "henry_mol_kg_Pa_at_reference": oxygen_henry,
            "heat_J_mol": oxygen_heat,
        }
    else:
        nitrogen_block = {}
        oxygen_block = {}

    cp_values = _number_list(
        draft,
        "adsorbed_phase_cp_J_kg_K",
        "adsorbed_phase_cp_J_kg_K",
        issues,
        length=n_ads,
        allow_zero=dry_only_water,
    )
    adsorption_rates = _number_list(
        draft,
        "adsorption_rate_s_inv",
        "kinetics.adsorption_rate_s_inv",
        issues,
        length=n_ads,
        allow_zero=True,
    )
    desorption_rates = _number_list(
        draft,
        "desorption_rate_s_inv",
        "kinetics.desorption_rate_s_inv",
        issues,
        length=n_ads,
        allow_zero=True,
    )
    dry_water_index = (
        2 if family == "split_rate_activity_shift_reference_isotherm" else 1
    )
    if (
        dry_only_water
        and cp_values is not None
        and cp_values[dry_water_index] != 0.0
    ):
        _issue(
            issues,
            f"adsorbed_phase_cp_J_kg_K[{dry_water_index}]",
            "Dry-only water adsorbed Cp must be exactly zero.",
        )
    if (
        dry_only_water
        and adsorption_rates is not None
        and adsorption_rates[dry_water_index] != 0.0
    ):
        _issue(
            issues,
            f"kinetics.adsorption_rate_s_inv[{dry_water_index}]",
            "Dry-only water adsorption rate must be exactly zero.",
        )
    if (
        dry_only_water
        and desorption_rates is not None
        and desorption_rates[dry_water_index] != 0.0
    ):
        _issue(
            issues,
            f"kinetics.desorption_rate_s_inv[{dry_water_index}]",
            "Dry-only water desorption rate must be exactly zero.",
        )

    if issues:
        return {
            "schema": GUIDED_RESULT_SCHEMA,
            "valid": False,
            "issues": issues,
            "material": None,
        }

    curve_metadata = {}
    if isinstance(base, Mapping) and isinstance(base.get("co2"), Mapping):
        base_curve = base["co2"].get("reference_curve")
        if isinstance(base_curve, Mapping):
            curve_metadata = {
                key: deepcopy(value)
                for key, value in base_curve.items()
                if key not in {"pressure_Pa", "loading_mol_kg", "temperature_K"}
            }
    if family in (
        "activity_shift_reference_isotherm",
        "split_rate_activity_shift_reference_isotherm",
    ):
        co2_block["reference_curve"] = {
            **curve_metadata,
            **co2_block["reference_curve"],
        }

    guided_baseline = _guided_baseline(base)
    material = deepcopy(dict(base)) if isinstance(base, Mapping) else {}
    material["schema"] = MATERIAL_SCHEMA
    material["material_id"] = material_id
    material["name"] = name
    material["equilibrium_family"] = family
    material["synthetic"] = "synthetic" in set(group_modes.values())
    material["reference_temperature_K"] = reference_temperature
    material["maximum_bed_temperature_C"] = maximum_temperature
    material["solid_cp_J_kg_K"] = solid_cp
    material["shaped_material"] = {
        **_preserved_block(
            base,
            "shaped_material",
            {"basis", "binder_support", "particle_geometry"},
        ),
        "basis": shaping_basis,
        "binder_support": binder_support,
        "particle_geometry": particle_geometry,
    }
    material["co2"] = {
        **_preserved_block(
            base,
            "co2",
            {
                "saturation_mol_kg",
                "reference_loading_mol_kg",
                "reference_partial_pressure_Pa",
                "heat_J_mol",
                "reference_curve",
                "site_capacity_mol_kg",
                "site_affinity_m3_mol",
                "site_internal_energy_J_mol",
                "basis",
            },
        ),
        **{key: value for key, value in co2_block.items() if value is not None},
    }
    if family in (
        "activity_shift_reference_isotherm",
        "split_rate_activity_shift_reference_isotherm",
    ):
        material["co2_water_penalty_per_mol_kg"] = penalty
    else:
        material.pop("co2_water_penalty_per_mol_kg", None)
    material["water"] = {
        **_preserved_block(
            base,
            "water",
            {
                "saturation_mol_kg",
                "half_loading_relative_humidity_at_reference",
                "sips_exponent",
                "heat_J_mol",
                "model",
                "physical_storage_mol_kg",
                "solver_normalizer_mol_kg",
                "measurement_temperature_K",
                "reference_temperature_K",
                "sorption_heat_J_mol",
                "points",
                "basis",
            },
        ),
        **{key: value for key, value in water_block.items() if value is not None},
    }
    if family == "site_resolved_concentration_langmuir":
        for key in ("nitrogen", "oxygen"):
            material.pop(key, None)
    else:
        material["nitrogen"] = {
            **_preserved_block(
                base,
                "nitrogen",
                {"henry_mol_kg_Pa_at_reference", "heat_J_mol"},
            ),
            **nitrogen_block,
        }
        material["oxygen"] = {
            **_preserved_block(
                base,
                "oxygen",
                {"henry_mol_kg_Pa_at_reference", "heat_J_mol"},
            ),
            **oxygen_block,
        }
    material["adsorbed_phase_cp_J_kg_K"] = cp_values
    material["kinetics"] = {
        **_preserved_block(
            base,
            "kinetics",
            {
                "adsorption_rate_s_inv",
                "desorption_rate_s_inv",
                "evidence_status",
            },
        ),
        "adsorption_rate_s_inv": adsorption_rates,
        "desorption_rate_s_inv": desorption_rates,
        "evidence_status": declarations["kinetics"],
    }
    changed_groups = []
    effective_group_modes = dict(group_modes)
    if guided_baseline is not None:
        current_group_hashes = _guided_group_hashes(material)
        original_group_hashes = guided_baseline["original_group_sha256"]
        changed_groups = sorted(
            group
            for group, digest in current_group_hashes.items()
            if original_group_hashes.get(group) != digest
        )
        inherited_identity_edit = "identity_shaping" in changed_groups
        original_declarations = guided_baseline.get(
            "original_declarations"
        ) or {}
        for group in ("co2", "water", "thermal", "kinetics"):
            inherited_source = effective_group_modes[group] == "source"
            unchanged_declaration = (
                declarations[group] == original_declarations.get(group)
            )
            if inherited_source and unchanged_declaration and (
                group in changed_groups or inherited_identity_edit
            ):
                effective_group_modes[group] = "assumption"
    downgraded_groups = sorted(
        group
        for group in ("co2", "water", "thermal", "kinetics")
        if effective_group_modes[group] != group_modes[group]
    )
    overall_modes = {
        mode for mode in effective_group_modes.values() if mode is not None
    }
    if changed_groups and (
        downgraded_groups or "identity_shaping" in changed_groups
    ):
        status = "USER_EDITED_ASSUMPTION"
    elif overall_modes == {"source"}:
        status = "SUPPLIED_SOURCE_DATA"
    elif overall_modes == {"assumption"}:
        status = "EXPLICIT_ASSUMPTION"
    elif overall_modes == {"synthetic"}:
        status = "SYNTHETIC"
    elif "synthetic" in overall_modes:
        status = "SYNTHETIC_MIXED_DECLARATIONS"
    else:
        status = "CONDITIONAL_MIXED_DECLARATIONS"
    base_evidence = (
        base.get("evidence")
        if isinstance(base, Mapping) and isinstance(base.get("evidence"), Mapping)
        else None
    )
    evidence = {
        key: deepcopy(value)
        for key, value in (base_evidence or {}).items()
        if key
        not in {
            "status",
            "basis",
            "not_for",
            "guided_entry",
            "edit_detected",
            "original_evidence",
            "loaded_package_sha256",
        }
    }
    evidence.update(
        {
            "status": status,
            "basis": evidence_basis,
            "not_for": evidence_not_for,
            "guided_entry": True,
            "edit_detected": bool(changed_groups),
            "changed_groups": changed_groups,
        }
    )
    if guided_baseline is not None:
        evidence["guided_baseline"] = deepcopy(guided_baseline)
    baseline_sha = None
    if isinstance(base, Mapping):
        guided_provenance = (
            (base.get("provenance") or {}).get("guided_entry")
            if isinstance(base.get("provenance"), Mapping)
            else None
        )
        guided_base = bool(
            isinstance(base_evidence, Mapping)
            and base_evidence.get("guided_entry") is True
            and isinstance(guided_provenance, Mapping)
        )
        if guided_base:
            baseline_sha = guided_provenance.get("base_package_sha256")
            if isinstance(base_evidence.get("original_evidence"), Mapping):
                evidence["original_evidence"] = _first_original_evidence(
                    base_evidence["original_evidence"]
                )
        else:
            baseline_sha = canonical_sha256(base)
            evidence["original_evidence"] = _first_original_evidence(
                base_evidence
            )
        if baseline_sha is not None:
            evidence["loaded_package_sha256"] = baseline_sha
    material["evidence"] = evidence
    provenance = deepcopy(material.get("provenance") or {})
    for group, statement in declarations.items():
        provenance[group] = _preserved_declaration(
            base, group, effective_group_modes[group], statement
        )
    provenance["guided_entry"] = {
        "class": "guided_entry",
        "declaration_status": status,
        "base_package_sha256": baseline_sha,
    }
    material["provenance"] = provenance
    existing_assumptions = (
        list(material.get("assumptions") or [])
        if isinstance(material.get("assumptions"), list)
        else []
    )
    material["assumptions"] = existing_assumptions
    for statement in assumptions:
        if statement not in material["assumptions"]:
            material["assumptions"].append(statement)
    material["validity"] = {
        **deepcopy(material.get("validity") or {}),
        **validity,
    }
    material["uncertainty"] = {
        **deepcopy(material.get("uncertainty") or {}),
        "classification": uncertainty_class,
        "statement": uncertainty_statement,
    }
    try:
        validated = validate_material(material)
        if validated is not None:
            validated_evidence = deepcopy(validated.get("evidence") or {})
            validated_status = validated_evidence.get("status")
            validated_evidence.pop("original_evidence", None)
            validated_evidence.update(deepcopy(evidence))
            if validated_status == "USER_EDITED_ASSUMPTION":
                validated_evidence["status"] = "USER_EDITED_ASSUMPTION"
            validated["evidence"] = validated_evidence
    except MofScreeningError as error:
        _issue(issues, error.field or "material", str(error))
        validated = None
    return {
        "schema": GUIDED_RESULT_SCHEMA,
        "valid": not issues,
        "issues": issues,
        "material": validated if not issues else None,
    }
