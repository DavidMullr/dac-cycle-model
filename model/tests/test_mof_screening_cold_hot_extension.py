from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from screening.geometry import build_bare_geometry
from screening.mof_screening import (
    young2021,
    young2021_cold_hot,
    young2021_hot,
)
from screening.mof_screening.contract import (
    build_case,
    load_builtin_material,
    load_defaults,
    material_temperature_guard,
    preview_payload,
)
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.runtime import build_spatial_config
from screening.mof_screening.spatial_jacobian import (
    SpatialJacobianAudit,
    bounded_spatial_jacobian,
)
from screening.mof_screening.spatial import (
    build_model,
    cycle_rhs,
    initial_state,
    reset_current_cycle_state_meters,
)
from screening.mof_screening.weather import load_weather_window


MODEL = Path(__file__).resolve().parents[1]
JOB_ROOT = MODEL / "tests/fixtures/mof_screening/jobs"
NEW_ID = "lewatit_vp_oc_1065_young2021_cold_hot116_conditional"
OLD_HOT_ID = "lewatit_vp_oc_1065_young2021_hot116_conditional"
STRICT_ID = "lewatit_vp_oc_1065_young2021_wadst_conditional"
LOW_ID = "lewatit_vp_oc_1065_low2023_conditional"


def _case(relative_humidity: float = 0.30):
    case = deepcopy(load_defaults()["case"])
    case["weather"]["relative_humidity"] = relative_humidity
    case["weather"]["hours"] = 48
    case["cycle"]["regeneration_temperature_C"] = 116.0
    case["cycle"]["regeneration_pressure_Pa"] = 16_000.0
    case["solver"]["cells"] = 2
    return case


def _model(relative_humidity: float = 0.30):
    material, case = build_case(
        load_builtin_material(NEW_ID), _case(relative_humidity)
    )
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    config = build_spatial_config(material, case, weather.rows[0])
    config["_runtime_control"] = {
        "weather_provider": weather.at,
        "cycle_start_s": 0.0,
        "phase_offset_s": 0.0,
        "cycle_number": 1,
        "temperature_guard_limit_K": None,
    }
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = initial_state(config, adapter, geometry, model)
    reset_current_cycle_state_meters(state, model.layout)
    return material, case, adapter, config, geometry, model, state


def test_new_scenario_preserves_source_metadata_and_old_packages():
    new = load_builtin_material(NEW_ID)
    strict = load_builtin_material(STRICT_ID)
    old_hot = load_builtin_material(OLD_HOT_ID)
    low = load_builtin_material(LOW_ID)
    assert new["material_id"] != old_hot["material_id"]
    assert new["equilibrium_family"] == (
        "young2021_cold_hot_wadst_gab_extrapolated"
    )
    assert new["co2"]["support"]["temperature_K"] == [298.15, 343.15]
    assert new["co2"]["evaluation_envelope"]["temperature_K"] == [
        293.15,
        393.15,
    ]
    assert new["water"]["model"] == (
        "young2021_gab_iapws_cold_hot_extrapolated"
    )
    assert new["kinetics"]["adsorption_rate_s_inv"] == old_hot[
        "kinetics"
    ]["adsorption_rate_s_inv"]
    assert new["kinetics"]["desorption_rate_s_inv"] == old_hot[
        "kinetics"
    ]["desorption_rate_s_inv"]
    assert new["co2"]["heat_J_mol"] == strict["co2"]["heat_J_mol"]
    assert new["water"]["heat_J_mol"] == strict["water"]["heat_J_mol"]
    assert young2021_cold_hot.young2021.YOUNG2021_DRY == (
        young2021.YOUNG2021_DRY
    )
    assert young2021_cold_hot.young2021.YOUNG2021_GAB == (
        young2021.YOUNG2021_GAB
    )
    assert low["material_id"] == LOW_ID
    assert material_temperature_guard(new)["matched"] is True


def test_cold_hot_envelope_corner_grid_and_original_cap_distinction():
    for temperature_K in (293.15, 298.15, 393.15):
        for relative_humidity in (0.0, 0.30, 0.55):
            water_partial = (
                relative_humidity
                * young2021_cold_hot.iapws_saturation_pressure_Pa(
                    temperature_K
                )
            )
            co2, water, details = young2021_cold_hot.equilibrium(
                temperature_K,
                40.0,
                water_partial,
            )
            assert np.isfinite([co2, water]).all()
            assert 0.0 <= water <= (
                young2021_cold_hot.WATER_EVALUATION_CAP_MOL_KG
            )
            assert details["measured_cold_validation"] is False

    temperature_K = young2021_cold_hot.GAB_ENVELOPE_MAX_TEMPERATURE_K
    water_partial = (
        0.55
        * young2021_cold_hot.iapws_saturation_pressure_Pa(
            temperature_K
        )
    )
    _co2, water, _details = young2021_cold_hot.equilibrium(
        temperature_K,
        40.0,
        water_partial,
    )
    assert water > (
        young2021_cold_hot.ORIGINAL_IMPLEMENTATION_WATER_CAP_MOL_KG
    )
    assert water <= young2021_cold_hot.WATER_EVALUATION_CAP_MOL_KG

    with pytest.raises(ValueError, match="20-120 C"):
        young2021_cold_hot.validate_evaluation_envelope(
            293.15 - 2e-9, 40.0, 0.30
        )
    with pytest.raises(ValueError, match="0-0.55"):
        young2021_cold_hot.validate_evaluation_envelope(
            298.15, 40.0, 0.55 + 2e-12
        )
    with pytest.raises(ValueError, match="water_loading"):
        young2021_hot.equilibrium(temperature_K, 40.0, water_partial)


def test_water_evaluation_normalization_is_not_inventory_mutation():
    adapter = build_material_adapter(load_builtin_material(NEW_ID))
    partials = np.array(
        [
            40.0,
            0.30 * young2021_cold_hot.iapws_saturation_pressure_Pa(298.15),
            0.0,
            0.0,
        ]
    )
    raw_storage = np.array([0.5, -5e-10])
    before = raw_storage.copy()
    rates, _species = adapter.rates(298.15, partials, raw_storage)
    np.testing.assert_array_equal(raw_storage, before)
    details = adapter.source_report()["water_evaluation_observations"]
    assert details["lower_normalization_count"] >= 1
    assert details["stored_inventory_mutated"] is False
    assert adapter._last_rate_details[
        "evaluation_water_loading_mol_kg"
    ] == 0.0
    assert adapter._last_rate_details["raw_water_loading_mol_kg"] == -5e-10
    assert adapter._last_rate_details["raw_water_delta_mol_kg"] > 0.0
    assert np.all(np.isfinite(rates))
    with pytest.raises(ValueError, match="below cold\\+hot"):
        adapter.rates(
            298.15,
            partials,
            np.array([0.5, -2e-9]),
        )


def test_water_upper_probe_tolerance_and_hard_rejection():
    adapter = build_material_adapter(load_builtin_material(NEW_ID))
    partials = np.array(
        [
            40.0,
            0.30 * young2021_cold_hot.iapws_saturation_pressure_Pa(298.15),
            0.0,
            0.0,
        ]
    )
    cap = young2021_cold_hot.WATER_EVALUATION_CAP_MOL_KG
    adapter.rates(298.15, partials, np.array([0.5, cap + 0.5e-9]))
    assert adapter._last_rate_details[
        "evaluation_water_loading_mol_kg"
    ] == cap
    assert adapter._last_rate_details["upper_normalized"] is True
    with pytest.raises(ValueError, match="above cold\\+hot"):
        adapter.rates(298.15, partials, np.array([0.5, cap + 2e-9]))


def test_spatial_jacobian_handles_loading_bounds_and_records_full_failure():
    material, case, adapter, config, geometry, model, state = _model()
    water_columns = model.layout.q.start + np.arange(
        0, model.layout.q.stop - model.layout.q.start, 2
    ) + 1
    state[water_columns] = (
        young2021_cold_hot.WATER_EVALUATION_CAP_MOL_KG
    )
    rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    audit = SpatialJacobianAudit(
        physical_columns=model.layout.inlet.start,
        ledger_columns=model.layout.size - model.layout.inlet.start,
    )
    jacobian = bounded_spatial_jacobian(
        lambda time, values: rhs(time, values)[0],
        0.0,
        state,
        layout=model.layout,
        material=adapter,
        audit=audit,
    )
    assert np.all(np.isfinite(jacobian))
    assert audit.fallback_steps == 0

    failing_state = state.copy()
    failing_state[water_columns[0]] = -2e-9
    failing_rhs = cycle_rhs(
        config, adapter, geometry, "adsorption", model=model
    )
    with pytest.raises(ValueError, match="below cold\\+hot"):
        failing_rhs(0.0, failing_state)
    report = adapter.failure_report()
    assert report is not None
    assert len(report["rejected_state_vector"]) == model.layout.size
    assert report["raw_storage_mol_kg"][1] == -2e-9
    assert report["cell"] == 0
    assert report["accepted_checkpoint_not_updated"] is True
    assert len(report["state_json_sha256"]) == 64


@pytest.mark.parametrize(
    ("job_id", "phase"),
    (
        ("a7c559cc00c54159a0738b79186c3d87", "adsorption"),
        ("3362a1a7bfa746a1a9eda3e21b6bbbb5", "evacuation"),
    ),
)
def test_actual_failed_checkpoint_states_pass_new_rhs_and_jacobian(
    job_id, phase,
):
    job = JOB_ROOT / job_id
    input_case = json.loads(
        (job / "input_case.json").read_text(encoding="utf-8")
    )
    material, case = build_case(
        load_builtin_material(NEW_ID), input_case
    )
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    config = build_spatial_config(material, case, weather.rows[0])
    config["_runtime_control"] = {
        "weather_provider": weather.at,
        "cycle_start_s": 0.0,
        "phase_offset_s": 0.0,
        "cycle_number": 1,
        "temperature_guard_limit_K": None,
    }
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = np.asarray(
        json.loads(
            (job / "SPATIAL_CHECKPOINT.json").read_text(encoding="utf-8")
        )["state"],
        dtype=float,
    )
    rhs = cycle_rhs(config, adapter, geometry, phase, model=model)
    derivative, _observation = rhs(0.0, state)
    assert np.all(np.isfinite(derivative))
    jacobian = bounded_spatial_jacobian(
        lambda time, values: rhs(time, values)[0],
        0.0,
        state,
        layout=model.layout,
        material=adapter,
    )
    assert np.all(np.isfinite(jacobian))
    assert adapter.failure_report() is None


def test_preview_and_postprocess_identity_are_explicit():
    material, case = build_case(
        load_builtin_material(NEW_ID), _case(0.30)
    )
    preview = preview_payload(material, case, source="cold_hot_test")
    warning = " ".join(preview["warnings"])
    assert "COLD-TRANSIENT + HOT CONDITIONAL EXTRAPOLATION" in warning
    assert "not cold-data validation" in warning
    assert "does not alter stored inventories" in warning
    assert preview["material_id"] == NEW_ID
    assert preview["case"]["weather"]["relative_humidity"] == 0.30
    assert preview["case"]["weather"]["hours"] == 48.0
    assert preview["case"]["cycle"]["regeneration_temperature_C"] == 116.0
    assert preview["case"]["cycle"]["regeneration_pressure_Pa"] == 16_000.0
