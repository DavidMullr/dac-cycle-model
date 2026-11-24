from copy import deepcopy
import json
from pathlib import Path

import pytest

from screening.mof_screening.contract import (
    MofScreeningError,
    build_case,
    load_builtin_material,
    load_defaults,
    material_temperature_guard,
    preview_payload,
    read_json_strict,
    temperature_guard_assessment,
    validate_material,
)
from screening.mof_screening.evaluation import hard_limits


def test_registry_has_all_exact_normalized_package_records():
    from screening.mof_screening.contract import MATERIAL_ROOT

    model = Path(__file__).resolve().parents[1]
    registry = json.loads((model / 'config/mof_screening/temperature_registry.json').read_text(encoding='utf-8'))
    assert registry['active'] is True
    checked = set()
    for record in registry['records'].values():
        name = record['package_relative_path'].replace('\\', '/').rsplit('/', 1)[-1]
        package = MATERIAL_ROOT / name
        if not package.is_file():
            package = MATERIAL_ROOT / 'source_evidence' / name
        if not package.is_file():
            with pytest.raises(MofScreeningError, match='unknown built-in material'):
                load_builtin_material(Path(name).stem)
            continue
        material = validate_material(read_json_strict(package))
        guard = material_temperature_guard(material)
        assert guard['matched'] is True
        assert guard['declared_guard_C'] == record['declared_guard_C']
        checked.add(material['material_id'])
    assert "purolite_a110_humid" in checked
    assert len(checked) == 29


def test_shared_case_hash_is_identical_across_material_guard_classes():
    case = deepcopy(load_defaults()["case"])
    case["weather"]["relative_humidity"] = 0.0
    case["temperature_policy"]["mode"] = "record_scenario_excursions"
    first = preview_payload(
        load_builtin_material("hypothetical_mmen_like"),
        build_case(
            load_builtin_material("hypothetical_mmen_like"), case
        )[1],
        source="test",
    )
    second = preview_payload(
        load_builtin_material("mof808_lys_2024_25c_dry_conditional"),
        build_case(
            load_builtin_material("mof808_lys_2024_25c_dry_conditional"),
            case,
        )[1],
        source="test",
    )
    assert first["case_sha256"] == second["case_sha256"]
    assert first["temperature_policy"]["package_canonical_sha256"] != (
        second["temperature_policy"]["package_canonical_sha256"]
    )
    assert "package_canonical_sha256" not in first["case"]["temperature_policy"]


def test_record_excursion_requires_exact_registry_match():
    case = deepcopy(load_defaults()["case"])
    case["weather"]["relative_humidity"] = 0.0
    case["temperature_policy"]["mode"] = "record_scenario_excursions"
    case["cycle"]["regeneration_temperature_C"] = 115.0
    material = load_builtin_material(
        "mil101_pei800_30_2026_25c_dry_conditional"
    )
    _, recorded_case = build_case(material, case)
    policy = preview_payload(material, recorded_case, source="test")[
        "temperature_policy"
    ]
    assert policy["scenario_excursion_permitted"] is True
    assert policy["physical_limit_known"] is False

    stop_case = deepcopy(case)
    stop_case["temperature_policy"]["mode"] = "stop_at_scenario_guard"
    with pytest.raises(MofScreeningError, match="guard policy"):
        build_case(material, stop_case)

    edited = deepcopy(material)
    edited["solid_cp_J_kg_K"] *= 1.001
    with pytest.raises(MofScreeningError, match="exact reviewed package"):
        build_case(edited, case)


def test_recorded_excursion_is_not_reported_as_physical_hard_limit_pass():
    case = deepcopy(load_defaults()["case"])
    case["weather"]["relative_humidity"] = 0.0
    case["temperature_policy"]["mode"] = "record_scenario_excursions"
    material = load_builtin_material(
        "lewatit_vp_oc_1065_low2023_conditional"
    )
    assessment = temperature_guard_assessment(
        material, case, peak_bed_temperature_K=393.3809
    )
    assert assessment["status"] == "scenario_excursion_recorded"
    assert assessment["scenario_guard_excursion_C"] == pytest.approx(0.23, abs=0.02)
    assert assessment["physical_degradation_limit_pass"] is None

    limits = hard_limits(
        {
            "temperature_guard_assessment": assessment,
            "checks": {"temperature_limit_pass": False},
        }
    )
    assert all(
        check["id"] != "material_temperature_limit"
        for check in limits["checks"]
    )
    assert any(
        "physical degradation limit" in message
        for message in limits["unassessed_limits"]
    )


def test_fixed_horizon_accepts_96h_but_official_remains_24h():
    material = load_builtin_material("hypothetical_mmen_like")
    fixed = deepcopy(load_defaults()["case"])
    fixed["weather"]["hours"] = 96
    assert build_case(material, fixed)[1]["weather"]["hours"] == 96

    official = deepcopy(load_defaults()["case"])
    official["weather"]["mode"] = "official"
    official["weather"]["hours"] = 25
    official["weather"]["start_utc"] = "2023-01-15T00:00:00Z"
    with pytest.raises(MofScreeningError, match="between 1 and 24"):
        build_case(material, official)
