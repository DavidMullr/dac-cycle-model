"""Checks for the explicit dry-air comparison preset."""
import json
from pathlib import Path

import pytest

from screening.mof_screening.contract import (
    canonical_sha256,
    load_builtin_material,
    load_defaults,
    validate_material,
    validate_case,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/mof_screening/matched_supported_case.json"
COMMON_CASE = (
    "37397c393358d68dc593294564619328a36b76915430037fce5919bbd06bc4ca"
)


def test_dry_air_preset_matches_reviewed_case_exactly():
    defaults = load_defaults()
    preset = next(
        item for item in defaults["case_presets"]
        if item["id"] == "dry_air_comparison_baseline_48h"
    )
    reviewed = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert preset["case"] == reviewed
    assert preset["label"] == "Dry-air comparison baseline (48 h)"
    assert preset["case_sha256"] == COMMON_CASE
    assert canonical_sha256(validate_case(preset["case"])) == COMMON_CASE
    assert preset["case"]["runtime_policy"]["mode"] == "manual_stop"
    assert preset["case"]["weather"]["relative_humidity"] == 0.0
    assert preset["case"]["weather"]["hours"] == 48
    assert "material" not in preset["case"]


def test_default_case_remains_the_short_spatial_case():
    defaults = load_defaults()
    assert defaults["case"]["weather"]["hours"] == 4
    assert defaults["case"]["solver"]["cells"] == 4
    assert defaults["case"]["runtime_policy"]["mode"] == "manual_stop"


@pytest.mark.parametrize("condition", ["dry", "humid"])
def test_reviewed_lewatit_preset_matches_complete_export(condition):
    saved = ROOT.parent / "docs" / "data" / "submission" / "results" / f"lewatit_{condition}.json"
    result = json.loads(saved.read_text(encoding="utf-8"))
    preset = next(item for item in load_defaults()["case_presets"]
                  if item["id"] == f"reviewed_lewatit_{condition}_48h")
    assert preset["case"] == result["case"]
    assert preset["case_sha256"] == result["input_identity"]["case_sha256"]
    material = validate_material(load_builtin_material(preset["material_id"]))
    assert material == result["material_package"]
    assert canonical_sha256(material) == result["input_identity"]["material_sha256"]
