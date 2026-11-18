import json

import numpy as np
import pytest

from screening.common import CP_GAS, MW
from screening.mof_screening.contract import (
    canonical_sha256,
    load_builtin_material,
    validate_material,
)
from screening.mof_screening.guided_material import build_guided_material
from screening.mof_screening.materials import build_material_adapter


def _common(family):
    return {
        "base_material": None,
        "material_id": f"guided_{family}_sample",
        "name": f"Guided {family} sample",
        "equilibrium_family": family,
        "evidence_basis": "Explicitly declared guided-form fixture",
        "evidence_not_for": "Material performance or hardware qualification",
        "shaping_basis": "Complete dry shaped material basis",
        "binder_support": "Declared 20 wt% binder assumption",
        "particle_geometry": "Declared 1.5 mm particle assumption",
        "reference_temperature_K": 298.15,
        "maximum_bed_temperature_C": 140.0,
        "solid_cp_J_kg_K": 1200.0,
        "co2_evidence_type": "source",
        "water_evidence_type": "assumption",
        "thermal_evidence_type": "assumption",
        "kinetics_evidence_type": "assumption",
        "co2_declaration": "Supplied dry reference point",
        "water_declaration": "Explicit unary water assumption",
        "thermal_declaration": "Declared shaped-material thermal assumption",
        "kinetics_declaration": "Declared LDF assumption",
        "assumptions_text": "The package is for workflow testing only.",
        "co2_validity": "Supplied dry reference conditions only.",
        "water_validity": "Unary water law; no competitive mixture validation.",
        "thermal_validity": "No measured continuous range claimed.",
        "kinetics_validity": "Illustrative LDF rates only.",
        "uncertainty_classification": "UNQUANTIFIED",
        "uncertainty_statement": "No uncertainty distribution is supplied.",
        "adsorption_rate_s_inv": "0.001, 0.0001, 0, 0",
        "desorption_rate_s_inv": "0.002, 0.0002, 0, 0",
    }


def _draft(family):
    draft = _common(family)
    if family == "independent_langmuir_sips":
        draft.update(
            {
                "co2_saturation_mol_kg": "2.5",
                "co2_reference_loading_mol_kg": "1.2",
                "co2_reference_partial_pressure_Pa": "40",
                "co2_heat_J_mol": "0",
                "water_saturation_mol_kg": "18",
                "water_half_loading_relative_humidity_at_reference": "0.5",
                "water_sips_exponent": "5",
                "water_heat_J_mol": "0",
                "nitrogen_henry_mol_kg_Pa_at_reference": "0",
                "nitrogen_heat_J_mol": "0",
                "oxygen_henry_mol_kg_Pa_at_reference": "0",
                "oxygen_heat_J_mol": "0",
                "adsorbed_phase_cp_J_kg_K": "900, 3600, 1000, 900",
            }
        )
    elif family == "activity_shift_reference_isotherm":
        draft.update(
            {
                "co2_heat_J_mol": "70000",
                "co2_water_penalty_per_mol_kg": "0",
                "co2_curve_temperature_K": "298.15",
                "co2_curve_csv": (
                    "pressure_Pa,loading_mol_kg\n"
                    "1,0.01\n39,1.2\n1000,2.0"
                ),
                "water_saturation_mol_kg": "18",
                "water_half_loading_relative_humidity_at_reference": "0.5",
                "water_sips_exponent": "5",
                "water_heat_J_mol": "42000",
                "nitrogen_henry_mol_kg_Pa_at_reference": "0",
                "nitrogen_heat_J_mol": "0",
                "oxygen_henry_mol_kg_Pa_at_reference": "0",
                "oxygen_heat_J_mol": "0",
                "adsorbed_phase_cp_J_kg_K": (
                    f"{CP_GAS[0] / MW[0]}, 3600, 1000, 900"
                ),
            }
        )
    else:
        draft.update(
            {
                "site1_capacity_mol_kg": "2.4",
                "site1_affinity_m3_mol": "5e-7",
                "site1_internal_energy_J_mol": "-35000",
                "site2_capacity_mol_kg": "3.2",
                "site2_affinity_m3_mol": "0",
                "site2_internal_energy_J_mol": "0",
                "water_measurement_temperature_K": "295.15",
                "water_reference_temperature_K": "298.15",
                "water_sorption_heat_J_mol": "52000",
                "water_curve_csv": (
                    "relative_humidity,loading_mol_kg\n"
                    "0,0\n0.2,1.5\n0.5,4\n1,8"
                ),
                "adsorbed_phase_cp_J_kg_K": (
                    "663.1644845317488, 663.1644845317488, "
                    "1887.2867921009274"
                ),
                "adsorption_rate_s_inv": "0.001, 0, 0.0001",
                "desorption_rate_s_inv": "0.002, 0, 0.0002",
            }
        )
    return draft


@pytest.mark.parametrize(
    "family",
    (
        "independent_langmuir_sips",
        "activity_shift_reference_isotherm",
        "site_resolved_concentration_langmuir",
    ),
)
def test_guided_builder_maps_every_supported_family(family):
    result = build_guided_material({"draft": _draft(family)})

    assert result["valid"] is True
    assert result["issues"] == []
    material = result["material"]
    assert material["equilibrium_family"] == family
    adapter = build_material_adapter(material)
    assert adapter.equilibrium_family == family
    assert adapter.n_storage in {3, 4}
    assert material["evidence"]["status"] == (
        "CONDITIONAL_MIXED_DECLARATIONS"
    )
    assert material["provenance"]["co2"]["class"] == "source"
    assert material["provenance"]["water"]["class"] == "assumption"


def test_guided_builder_lists_blank_and_unsupported_fields():
    blank = build_guided_material({"draft": {}})
    unsupported = build_guided_material(
        {"draft": {**_common("not_supported"), "equilibrium_family": "not_supported"}}
    )

    assert blank["valid"] is False
    assert blank["material"] is None
    assert {
        issue["field"] for issue in blank["issues"]
    }.issuperset(
        {
            "material_id",
            "name",
            "equilibrium_family",
            "evidence_basis",
            "provenance.co2.mode",
        }
    )
    assert unsupported["valid"] is False
    assert any(
        "bounded adapter" in issue["message"]
        for issue in unsupported["issues"]
    )


def test_curve_order_and_degenerate_scale_fail_without_silent_fill():
    bad_curve = _draft("activity_shift_reference_isotherm")
    bad_curve["co2_curve_csv"] = (
        "pressure_Pa,loading_mol_kg\n100,2\n39,1.2\n1,0.01"
    )
    zero_co2 = _draft("independent_langmuir_sips")
    zero_co2["co2_saturation_mol_kg"] = "0"

    curve_result = build_guided_material({"draft": bad_curve})
    zero_result = build_guided_material({"draft": zero_co2})

    assert curve_result["valid"] is False
    assert any(
        "strictly increasing" in issue["message"]
        for issue in curve_result["issues"]
    )
    assert zero_result["valid"] is False
    assert any(
        issue["field"] == "co2.saturation_mol_kg"
        and "> 0" in issue["message"]
        for issue in zero_result["issues"]
    )


def _dry_guided_draft():
    base = load_builtin_material(
        "mmen_mg2_hob_powder_2024_dry_conditional"
    )
    draft = _draft("activity_shift_reference_isotherm")
    draft.update(
        {
            "base_material": base,
            "material_id": "guided_hob_dry_copy",
            "name": "Guided HOB dry copy",
            "water_solver_normalizer_mol_kg": base["water"][
                "solver_normalizer_mol_kg"
            ],
            "adsorbed_phase_cp_J_kg_K": (
                f"{CP_GAS[0] / MW[0]}, 0, 0, 0"
            ),
            "adsorption_rate_s_inv": "0.0000554, 0, 0, 0",
            "desorption_rate_s_inv": "0.0000554, 0, 0, 0",
        }
    )
    for key in (
        "water_saturation_mol_kg",
        "water_half_loading_relative_humidity_at_reference",
        "water_sips_exponent",
        "water_heat_J_mol",
    ):
        draft.pop(key, None)
    return draft


def _roundtrip_draft(material):
    curve = material["co2"]["reference_curve"]
    provenance = material["provenance"]
    curve_rows = zip(curve["pressure_Pa"], curve["loading_mol_kg"])
    declaration_mode = lambda entry: (
        "source"
        if any(
            token in str(entry.get("class", "")).lower()
            for token in ("source", "measured", "published")
        )
        else "assumption"
    )
    declarations = {
        group: provenance[group].get("statement")
        or provenance[group].get("class")
        or "declaration"
        for group in ("co2", "water", "thermal", "kinetics")
    }
    return {
        "base_material": material,
        "material_id": material["material_id"],
        "name": material["name"],
        "equilibrium_family": material["equilibrium_family"],
        "evidence_basis": material["evidence"]["basis"],
        "evidence_not_for": material["evidence"]["not_for"],
        "shaping_basis": material["shaped_material"]["basis"],
        "binder_support": material["shaped_material"]["binder_support"],
        "particle_geometry": material["shaped_material"]["particle_geometry"],
        "reference_temperature_K": material["reference_temperature_K"],
        "maximum_bed_temperature_C": material["maximum_bed_temperature_C"],
        "solid_cp_J_kg_K": material["solid_cp_J_kg_K"],
        "co2_evidence_type": declaration_mode(provenance["co2"]),
        "water_evidence_type": declaration_mode(provenance["water"]),
        "thermal_evidence_type": declaration_mode(provenance["thermal"]),
        "kinetics_evidence_type": declaration_mode(provenance["kinetics"]),
        "co2_declaration": declarations["co2"],
        "water_declaration": declarations["water"],
        "thermal_declaration": declarations["thermal"],
        "kinetics_declaration": declarations["kinetics"],
        "assumptions_text": "\n".join(material["assumptions"]),
        "co2_validity": material["validity"]["co2"],
        "water_validity": material["validity"]["water"],
        "thermal_validity": material["validity"]["thermal"],
        "kinetics_validity": material["validity"]["kinetics"],
        "uncertainty_classification": material["uncertainty"]["classification"],
        "uncertainty_statement": material["uncertainty"]["statement"],
        "adsorbed_phase_cp_J_kg_K": ", ".join(
            str(value) for value in material["adsorbed_phase_cp_J_kg_K"]
        ),
        "adsorption_rate_s_inv": ", ".join(
            str(value) for value in material["kinetics"]["adsorption_rate_s_inv"]
        ),
        "desorption_rate_s_inv": ", ".join(
            str(value) for value in material["kinetics"]["desorption_rate_s_inv"]
        ),
        "co2_heat_J_mol": str(material["co2"]["heat_J_mol"]),
        "co2_water_penalty_per_mol_kg": str(
            material["co2_water_penalty_per_mol_kg"]
        ),
        "co2_curve_temperature_K": str(curve["temperature_K"]),
        "co2_curve_csv": (
            "pressure_Pa,loading_mol_kg\n"
            + "\n".join(f"{pressure},{loading}" for pressure, loading in curve_rows)
        ),
        "water_solver_normalizer_mol_kg": str(
            material["water"]["solver_normalizer_mol_kg"]
        ),
        "nitrogen_henry_mol_kg_Pa_at_reference": str(
            material["nitrogen"]["henry_mol_kg_Pa_at_reference"]
        ),
        "nitrogen_heat_J_mol": str(material["nitrogen"]["heat_J_mol"]),
        "oxygen_henry_mol_kg_Pa_at_reference": str(
            material["oxygen"]["henry_mol_kg_Pa_at_reference"]
        ),
        "oxygen_heat_J_mol": str(material["oxygen"]["heat_J_mol"]),
    }


def test_guided_builder_preserves_activity_dry_only_mode():
    draft = _dry_guided_draft()

    result = build_guided_material({"draft": draft})

    assert result["valid"] is True, result["issues"]
    material = result["material"]
    assert material["water"] == {
        "model": "dry_only",
        "physical_storage_mol_kg": 0.0,
        "solver_normalizer_mol_kg": 0.05,
        "basis": (
            "Dry-only structural water state; zero physical water storage "
            "and nonzero solver normalizer only."
        ),
    }
    adapter = build_material_adapter(material)
    equilibrium = adapter.equilibrium(
        298.15, np.array([40.0, 1.0, 79000.0, 21000.0])
    )
    assert equilibrium[1] == 0.0


@pytest.mark.parametrize(
    "material_id",
    (
        "mof808_lys_2024_25c_dry_conditional",
        "mof808_tapa_2024_25c_dry_conditional",
        "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional",
        "cof999_2024_25c_dry_conditional",
        "mil101_pei800_30_glycerin_2026_25c_dry_conditional",
        "mil101_pei800_30_2026_25c_dry_conditional",
    ),
)
def test_new_conditional_material_guided_roundtrip_preserves_source_separation(
    material_id,
):
    source = load_builtin_material(material_id)

    result = build_guided_material({"draft": _roundtrip_draft(source)})

    assert result["valid"] is True, result["issues"]
    roundtripped = validate_material(json.loads(json.dumps(result["material"])))
    assert roundtripped["material_id"] == material_id
    assert roundtripped["equilibrium_family"] == (
        "activity_shift_reference_isotherm"
    )
    assert roundtripped["water"]["model"] == "dry_only"
    assert roundtripped["co2"]["reference_curve"]["pressure_Pa"] == (
        source["co2"]["reference_curve"]["pressure_Pa"]
    )
    assert roundtripped["co2"]["reference_curve"]["loading_mol_kg"] == (
        source["co2"]["reference_curve"]["loading_mol_kg"]
    )
    assert roundtripped["provenance"]["co2_reference_curve"][
        "source_asset_sha256"
    ] == source["provenance"]["co2_reference_curve"]["source_asset_sha256"]
    assert roundtripped["evidence"]["status"] == "USER_EDITED_ASSUMPTION"
    assert roundtripped["evidence"]["original_evidence"]["status"] == (
        source["evidence"]["status"]
    )
    assert roundtripped["missing_evidence"] == source["missing_evidence"]
    adapter = build_material_adapter(roundtripped)
    assert adapter.source_report()["water_mode"] == "dry_only"


@pytest.mark.parametrize(
    ("field", "value", "issue_field"),
    [
        ("adsorbed_phase_cp_J_kg_K", f"{CP_GAS[0] / MW[0]}, 1000, 0, 0", "adsorbed_phase_cp_J_kg_K[1]"),
        ("adsorption_rate_s_inv", "0.001, 0.1, 0, 0", "kinetics.adsorption_rate_s_inv[1]"),
        ("desorption_rate_s_inv", "0.002, 0.1, 0, 0", "kinetics.desorption_rate_s_inv[1]"),
        ("co2_water_penalty_per_mol_kg", "10", "co2_water_penalty_per_mol_kg"),
    ],
)
def test_guided_builder_rejects_conflicting_dry_only_water_semantics(
    field, value, issue_field
):
    draft = _dry_guided_draft()
    draft[field] = value

    result = build_guided_material({"draft": draft})

    assert result["valid"] is False
    assert issue_field in {issue["field"] for issue in result["issues"]}


def test_supported_zero_rates_inert_uptake_and_site_terms_validate():
    independent = build_guided_material(
        {"draft": _draft("independent_langmuir_sips")}
    )
    site = build_guided_material(
        {"draft": _draft("site_resolved_concentration_langmuir")}
    )

    assert independent["valid"] is True
    assert independent["material"]["nitrogen"][
        "henry_mol_kg_Pa_at_reference"
    ] == 0.0
    assert independent["material"]["kinetics"][
        "adsorption_rate_s_inv"
    ][-1] == 0.0
    assert site["valid"] is True
    assert site["material"]["co2"]["site_affinity_m3_mol"][1] == 0.0
    assert site["material"]["co2"][
        "site_internal_energy_J_mol"
    ][1] == 0.0
    assert site["material"]["kinetics"]["adsorption_rate_s_inv"][1] == 0.0


def test_two_site_caloric_compatibility_fails_before_preview():
    draft = _draft("site_resolved_concentration_langmuir")
    draft["adsorbed_phase_cp_J_kg_K"] = "700, 700, 1900"

    result = build_guided_material({"draft": draft})

    assert result["valid"] is False
    assert any(
        issue["field"] == "material.adsorbed_phase_cp_J_kg_K[0]"
        and "concentration/internal-energy law" in issue["message"]
        for issue in result["issues"]
    )


def test_nested_metadata_structured_validity_and_roundtrip_are_stable():
    draft = _draft("activity_shift_reference_isotherm")
    draft["co2_validity"] = json.dumps(
        {
            "temperature_K": [273.15, 373.15],
            "pressure_Pa": [1.0, 1000.0],
        }
    )
    first = build_guided_material({"draft": draft})
    assert first["valid"] is True
    imported = json.loads(json.dumps(first["material"]))
    imported["custom_top_metadata"] = {"instrument": "fixture"}
    imported["co2"]["custom_co2_metadata"] = {"sample": "A"}
    imported["co2"]["reference_curve"]["instrument"] = "source curve"
    imported["co2"]["reference_curve"]["custommeta"] = {"run": 7}
    imported["evidence"].pop("guided_entry", None)
    imported["provenance"].pop("guided_entry", None)
    imported["provenance"]["co2"] = {
        "class": "published measured curve",
        "statement": "Supplied curve",
        "doi": "10.0000/example",
    }

    first_guided = build_guided_material(
        {"draft": {**draft, "base_material": imported}}
    )
    assert first_guided["valid"] is True
    first_material = first_guided["material"]
    assert first_material["custom_top_metadata"] == {
        "instrument": "fixture"
    }
    assert first_material["co2"]["custom_co2_metadata"] == {
        "sample": "A"
    }
    assert first_material["co2"]["reference_curve"]["instrument"] == (
        "source curve"
    )
    assert first_material["co2"]["reference_curve"]["custommeta"] == {
        "run": 7
    }
    assert isinstance(first_material["validity"]["co2"], dict)
    assert "original_evidence" in first_material["evidence"]
    assert "original_evidence" not in first_material["evidence"][
        "original_evidence"
    ]

    second_guided = build_guided_material(
        {"draft": {**draft, "base_material": first_material}}
    )
    assert second_guided["valid"] is True
    assert canonical_sha256(second_guided["material"]) == canonical_sha256(
        first_material
    )
    assert "original_evidence" not in second_guided["material"]["evidence"][
        "original_evidence"
    ]


def test_builtin_edit_keeps_first_evidence_without_measured_promotion():
    base = load_builtin_material("hypothetical_mmen_like")
    draft = _draft("independent_langmuir_sips")
    draft["base_material"] = base
    for group in ("co2", "water", "thermal", "kinetics"):
        draft[f"{group}_evidence_type"] = "assumption"
    first = build_guided_material({"draft": draft})
    assert first["valid"] is True
    assert first["material"]["evidence"]["status"] == (
        "USER_EDITED_ASSUMPTION"
    )
    assert first["material"]["evidence"]["original_evidence"]["status"] == (
        "HYPOTHETICAL"
    )

    second = build_guided_material(
        {"draft": {**draft, "base_material": first["material"]}}
    )
    assert second["valid"] is True
    assert canonical_sha256(second["material"]) == canonical_sha256(
        first["material"]
    )
    assert "original_evidence" not in second["material"]["evidence"][
        "original_evidence"
    ]


def test_custom_sourced_physical_edit_downgrades_and_stays_stable():
    draft = _draft("independent_langmuir_sips")
    for group in ("co2", "water", "thermal", "kinetics"):
        draft[f"{group}_evidence_type"] = "source"
    initial = build_guided_material({"draft": draft})["material"]
    initial["evidence"].pop("guided_entry", None)
    initial["provenance"].pop("guided_entry", None)
    initial["evidence"]["status"] = "SUPPLIED_SOURCE_DATA"
    initial["custom_top_metadata"] = {"instrument": "external"}

    edited_draft = {
        **draft,
        "base_material": initial,
        "co2_reference_loading_mol_kg": "1.25",
    }
    edited = build_guided_material({"draft": edited_draft})

    assert edited["valid"] is True
    material = edited["material"]
    assert material["evidence"]["status"] == "USER_EDITED_ASSUMPTION"
    assert material["evidence"]["changed_groups"] == ["co2"]
    assert material["provenance"]["co2"]["class"] == "assumption"
    assert material["provenance"]["co2"]["loaded_class"] == "source"
    assert material["evidence"]["original_evidence"]["status"] == (
        "SUPPLIED_SOURCE_DATA"
    )
    assert material["custom_top_metadata"] == {"instrument": "external"}

    repeated = build_guided_material(
        {"draft": {**edited_draft, "base_material": material}}
    )
    assert repeated["valid"] is True
    assert canonical_sha256(repeated["material"]) == canonical_sha256(material)
    assert repeated["material"]["evidence"]["status"] == (
        "USER_EDITED_ASSUMPTION"
    )
    assert "original_evidence" not in repeated["material"]["evidence"][
        "original_evidence"
    ]


def test_equivalent_numeric_format_does_not_flag_physical_edit():
    draft = _draft("independent_langmuir_sips")
    for group in ("co2", "water", "thermal", "kinetics"):
        draft[f"{group}_evidence_type"] = "source"
    draft["reference_temperature_K"] = "298"
    initial = build_guided_material({"draft": draft})["material"]
    initial["evidence"].pop("guided_entry", None)
    initial["provenance"].pop("guided_entry", None)

    unchanged = build_guided_material(
        {
            "draft": {
                **draft,
                "base_material": initial,
                "reference_temperature_K": "298.0",
            }
        }
    )

    assert unchanged["valid"] is True
    assert unchanged["material"]["evidence"]["changed_groups"] == []
    assert unchanged["material"]["evidence"]["edit_detected"] is False
    assert unchanged["material"]["evidence"]["status"] == (
        "SUPPLIED_SOURCE_DATA"
    )
