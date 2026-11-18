from copy import deepcopy

import pytest

from screening.mof_screening.contract import (
    MATERIAL_ROOT,
    ROOT,
    MofScreeningError,
    build_case,
    canonical_sha256,
    default_runtime_policy,
    effective_runtime_policy,
    load_builtin_material,
    load_defaults,
    preview_payload,
    read_json_strict,
    runtime_wall_limit_s,
    validate_material,
)
from screening.mof_screening.preflight import (
    execution_source_identity,
    dependency_status,
)
from screening.mof_screening.weather import load_weather_window


def _js_style_numeric_roundtrip(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_js_style_numeric_roundtrip(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _js_style_numeric_roundtrip(item)
            for key, item in value.items()
        }
    return value


def test_builtin_materials_and_shared_case_validate():
    defaults = load_defaults()
    for item in defaults["built_in_materials"]:
        material = load_builtin_material(item["id"])
        case_document = defaults["case"]
        if material.get("water", {}).get("model") == "dry_only":
            case_document = deepcopy(defaults["case"])
            case_document["weather"]["relative_humidity"] = 0.0
        elif material.get("equilibrium_family") == "young2021_wadst_gab":
            case_document = next(
                preset["case"]
                for preset in defaults["case_presets"]
                if preset["material_id"] == material["material_id"]
            )
        normalized, case = build_case(material, case_document)
        preview = preview_payload(normalized, case, source="test")
        assert preview["material_id"] == material["material_id"]
        expected_starts = (
            2
            if material.get("equilibrium_family") != "young2021_wadst_gab"
            else 3
        )
        assert preview["maximum_cycle_starts"] == expected_starts
        assert preview["state_dimension_estimate"] > 0


def test_app_runtime_policy_defaults_to_manual_stop_without_wall_deadline():
    defaults = load_defaults()
    case = defaults["case"]
    policy = effective_runtime_policy(case)
    assert policy == default_runtime_policy()
    assert policy["mode"] == "manual_stop"
    assert policy["physical_operating_condition"] is False
    assert runtime_wall_limit_s(case) is None

    material = load_builtin_material("hypothetical_mmen_like")
    preview = preview_payload(material, case, source="test")
    assert preview["allocation"]["maximum_wall_s"] is None
    assert preview["allocation"]["automatic_wall_termination"] is False
    assert preview["allocation"]["manual_stop_enabled"] is True


def test_explicit_runtime_time_limit_is_validated_and_operational_only():
    defaults = load_defaults()
    case = deepcopy(defaults["case"])
    case["runtime_policy"] = {
        **default_runtime_policy(),
        "mode": "time_limit",
        "time_limit_s": 42.0,
    }
    material = load_builtin_material("hypothetical_mmen_like")
    normalized = build_case(material, case)[1]
    preview = preview_payload(material, normalized, source="test")
    assert runtime_wall_limit_s(normalized) == 42.0
    assert preview["allocation"]["maximum_wall_s"] == 42.0
    assert preview["allocation"]["automatic_wall_termination"] is True
    assert (
        preview["assumptions"]["runtime_control"][
            "physical_operating_condition"
        ]
        is False
    )

    case["runtime_policy"]["time_limit_s"] = 0.0
    with pytest.raises(MofScreeningError, match=r"> 0"):
        build_case(material, case)


def test_legacy_case_without_policy_keeps_direct_caller_default():
    case = deepcopy(load_defaults()["case"])
    del case["runtime_policy"]
    assert runtime_wall_limit_s(case, default_wall_limit_s=300.0) == 300.0


def test_edited_builtin_does_not_retain_source_status():
    defaults = load_defaults()
    material = load_builtin_material("calf20_shaped")
    material["co2"]["site_capacity_mol_kg"][0] *= 1.01
    normalized, _ = build_case(material, defaults["case"])
    assert normalized["evidence"]["status"] == "USER_EDITED_ASSUMPTION"
    assert normalized["evidence"]["edit_detected"] is True
    assert normalized["evidence"]["original_evidence"]["status"] == (
        "CONDITIONAL_SOURCE_TRANSFER"
    )
    assert normalized["provenance"]["input_edit"]["class"] == (
        "user-edited assumption"
    )


def test_native_file_through_js_numeric_json_remains_unedited():
    raw = read_json_strict(MATERIAL_ROOT / "hypothetical_mmen_like.json")
    js_roundtripped = _js_style_numeric_roundtrip(raw)

    validated = validate_material(js_roundtripped)

    assert validated["evidence"]["status"] == "HYPOTHETICAL"
    assert validated["evidence"].get("edit_detected") is not True
    assert validated["provenance"]["loaded_builtin"]["material_id"] == (
        "hypothetical_mmen_like"
    )
    assert "input_edit" not in validated["provenance"]
    assert validate_material(validated) == validated


def test_calf_identity_and_shaping_edit_annotation_is_idempotent():
    raw = read_json_strict(MATERIAL_ROOT / "calf20_shaped.json")
    loaded = validate_material(raw)
    loaded["material_id"] = "calf-renamed-in-app"
    loaded["name"] = "Renamed CALF candidate"
    loaded["shaped_material"]["binder_support"] = (
        "User-declared experimental support"
    )

    edited = validate_material(loaded)
    validated_again = validate_material(edited)
    edit_assumption = (
        "The material identity, sample/shaping declaration or one or more "
        "physical coefficients were edited after loading a built-in package; "
        "results no longer inherit that package's source evidence status."
    )

    assert edited == validated_again
    assert edited["evidence"]["status"] == "USER_EDITED_ASSUMPTION"
    assert edited["evidence"]["edit_detected"] is True
    assert edited["evidence"]["original_evidence"]["status"] == (
        "CONDITIONAL_SOURCE_TRANSFER"
    )
    assert "original_evidence" not in edited["evidence"]["original_evidence"]
    assert edited["provenance"]["loaded_builtin"]["material_id"] == (
        "calf20_nguyen2024_80_20_shaped"
    )
    assert edited["provenance"]["input_edit"][
        "loaded_baseline_material_id"
    ] == "calf20_nguyen2024_80_20_shaped"
    assert edited["assumptions"].count(edit_assumption) == 1
    assert canonical_sha256(edited) == canonical_sha256(validated_again)


def test_missing_required_property_has_field_error():
    defaults = load_defaults()
    material = load_builtin_material("hypothetical_mmen_like")
    del material["co2"]["heat_J_mol"]
    with pytest.raises(MofScreeningError) as captured:
        build_case(material, defaults["case"])
    assert captured.value.field == "material.co2.heat_J_mol"


def test_unsupported_family_is_not_coerced():
    defaults = load_defaults()
    material = load_builtin_material("hypothetical_mmen_like")
    material["equilibrium_family"] = "unknown_material_law"
    with pytest.raises(MofScreeningError) as captured:
        build_case(material, defaults["case"])
    assert "unsupported" in str(captured.value)


def test_activity_shift_reference_family_is_supported():
    import numpy as np

    from screening.common import CP_GAS, MW
    from screening.geometry import build_bare_geometry
    from screening.mof_screening.lumped_port.engine import run_cycle
    from screening.mof_screening.materials import build_material_adapter
    from screening.mof_screening.runtime import (
        build_lumped_config,
        build_spatial_config,
    )
    from screening.mof_screening.spatial import (
        build_model,
        cycle_rhs,
        initial_state,
    )
    from screening.mof_screening.weather import load_weather_window

    defaults = load_defaults()
    material = read_json_strict(
        MATERIAL_ROOT / "synthetic_independent_demo.json"
    )
    material["material_id"] = "synthetic_activity_reference_demo"
    material["name"] = "Synthetic activity-shift reference fixture"
    material["equilibrium_family"] = "activity_shift_reference_isotherm"
    material["co2_water_penalty_per_mol_kg"] = 0.0
    material["co2"] = {
        "heat_J_mol": 71000.0,
        "reference_curve": {
            "pressure_Pa": [1.0, 39.0, 1000.0],
            "loading_mol_kg": [0.1, 2.0, 3.0],
            "temperature_K": 298.15,
        },
    }
    material["adsorbed_phase_cp_J_kg_K"][0] = float(CP_GAS[0] / MW[0])
    normalized, _ = build_case(material, defaults["case"])
    adapter = build_material_adapter(normalized)
    assert adapter.equilibrium_family == "activity_shift_reference_isotherm"
    assert adapter.source_report()["co2_reference_curve_preserved"] is True
    assert adapter.model.data["co2_water_penalty_per_mol_kg"] == 0.0
    partials = np.array([39.0, 1000.0, 79000.0, 21000.0])
    equilibrium = adapter.equilibrium(298.15, partials)
    rates, species_rates = adapter.rates(
        298.15, partials, np.zeros(adapter.n_storage)
    )
    assert np.all(np.isfinite(equilibrium))
    assert equilibrium[0] > 0.0
    assert rates[0] > 0.0
    assert np.all(np.isfinite(species_rates))

    weather = load_weather_window(defaults["case"]["weather"])
    spatial_config = build_spatial_config(
        normalized, defaults["case"], weather.rows[0]
    )
    geometry = build_bare_geometry(spatial_config)
    model = build_model(spatial_config, adapter, geometry)
    state = initial_state(
        spatial_config, adapter, geometry, model
    )
    spatial_derivative, _ = cycle_rhs(
        spatial_config,
        adapter,
        geometry,
        "adsorption",
        model=model,
    )(0.0, state)
    assert np.all(np.isfinite(spatial_derivative))

    lumped_config = build_lumped_config(
        normalized, defaults["case"], weather.rows[0]
    )
    bundle = run_cycle(
        adapter,
        lumped_config,
        relative_humidity=weather.rows[0].relative_humidity,
        rhs_only=True,
    )
    lumped_state = bundle["initial_state"]()
    for phase, rhs in bundle["rhs_by_phase"].items():
        derivative = np.asarray(rhs(0.0, lumped_state), dtype=float)
        assert derivative.shape == lumped_state.shape, phase
        assert np.all(np.isfinite(derivative)), phase

    missing_penalty = deepcopy(material)
    del missing_penalty["co2_water_penalty_per_mol_kg"]
    with pytest.raises(MofScreeningError) as captured:
        validate_material(missing_penalty)
    assert captured.value.field == (
        "material.co2_water_penalty_per_mol_kg"
    )


def test_official_weather_uses_interval_ends():
    defaults = load_defaults()
    spec = deepcopy(defaults["case"]["weather"])
    spec.update(
        {
            "mode": "official",
            "hours": 2,
            "start_utc": "2023-01-15T00:00:00Z",
        }
    )
    window = load_weather_window(spec)
    assert window.rows[0].interval_end_utc == "2023-01-15T01:00:00Z"
    assert window.rows[-1].interval_end_utc == "2023-01-15T02:00:00Z"
    assert window.interval_semantics == "start < utc_timestamp <= end"


def test_source_identity_binds_verified_current_dependencies():
    closure = dependency_status()
    identity = execution_source_identity()

    shared_files = list((ROOT / "screening").glob("*.py")) + list((ROOT / "equipment").glob("*.*"))
    expected = {path.relative_to(ROOT).as_posix() for path in shared_files}
    recorded = read_json_strict(ROOT / closure["path"])["files"]
    assert {row["path"] for row in recorded} == expected
    assert closure["file_count"] == len(expected)
    assert closure["path"] == (
        "config/mof_screening/dependencies.json"
    )
    assert closure["valid"] is True
    assert len(closure["sha256"]) == 64
    assert identity["frozen_closure"] == closure
    assert identity["source_hashes"][closure["path"]] == closure["sha256"]
    assert len(identity["release_sha256"]) == 64


def test_lumped_rejects_unimplemented_purge_routing():
    defaults = load_defaults()
    case = deepcopy(defaults["case"])
    case["engine"] = "lumped"
    case["solver"]["cells"] = 1
    case["cycle"]["purge_s"] = 30.0
    material = load_builtin_material("hypothetical_mmen_like")
    with pytest.raises(MofScreeningError) as captured:
        build_case(material, case)
    assert captured.value.field == "case.collection.product_route"


def test_unknown_collection_route_is_rejected():
    case = load_defaults()["case"]
    case["collection"]["product_route"] = "cooling_only"
    with pytest.raises(MofScreeningError, match="collection route"):
        build_case(load_builtin_material("hypothetical_mmen_like"), case)
