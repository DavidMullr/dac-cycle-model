from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from screening.geometry import build_bare_geometry
from screening.mof_screening import (
    young2021,
    young2021_advisory_temperature,
)
from screening.mof_screening.contract import (
    MofScreeningError,
    build_case,
    canonical_sha256,
    load_builtin_material,
    load_defaults,
    material_temperature_guard,
    preview_payload,
    temperature_guard_assessment,
)
from screening.mof_screening.evaluation import hard_limits
from screening.mof_screening.materials import build_material_adapter
from screening.mof_screening.manager import MofScreeningManager
from screening.mof_screening.preflight import execution_source_identity
from screening.mof_screening.runtime import (
    build_spatial_config,
    temperature_reporting,
)
from screening.mof_screening.spatial_jacobian import (
    SpatialJacobianAudit,
    bounded_spatial_jacobian,
)
from screening.mof_screening.spatial import (
    build_model,
    cycle_rhs,
    energy_inventory,
)
from screening.mof_screening.weather import load_weather_window


MODEL = Path(__file__).resolve().parents[1]
JOB = (
    MODEL
    / "tests/fixtures/mof_screening/jobs/330d27a0aa6b4efb8fa4a17823c7b823"
)
MATERIAL_ID = (
    "lewatit_vp_oc_1065_young2021_advisory_temperature_conditional"
)
DRY_PRESET = "young2021_advisory_temperature_48h_rh0"
HUMID_PRESET = "young2021_advisory_temperature_48h_rh030"


def _preset(preset_id: str) -> dict:
    return next(
        row
        for row in load_defaults()["case_presets"]
        if row["id"] == preset_id
    )


def test_advisory_package_and_presets_keep_evidence_and_stops_distinct():
    material = load_builtin_material(MATERIAL_ID)
    dry = _preset(DRY_PRESET)
    humid = _preset(HUMID_PRESET)
    _, normalized_dry = build_case(material, dry["case"])
    _, normalized_humid = build_case(material, humid["case"])
    preview = preview_payload(material, normalized_humid, source="test")

    assert dry["case_sha256"] == (
        "c246e39357249046e9308bd53ef1478bb85aadda055ba0b806c820180e6bbe90"
    )
    assert dry["case"] == json.loads(
        (
            MODEL / "tests/fixtures/mof_screening/advisory_case.json"
        ).read_text(encoding="utf-8")
    )
    assert normalized_dry["cycle"]["regeneration_temperature_C"] == 116.0
    assert normalized_humid["weather"]["relative_humidity"] == 0.30
    assert material["temperature_advisory"][
        "historical_range_is_hard_stop"
    ] is False
    assert material["temperature_advisory"][
        "accepted_state_temperature_clamping"
    ] is False
    assert preview["temperature_policy"]["enforcement_limit_C"] is None
    assert preview["temperature_policy"]["scenario_excursion_permitted"] is True
    warning = " ".join(preview["warnings"])
    assert "No temperature clamp or discretionary temperature stop" in warning
    assert "measured Young coupled support is 25-70 C" in warning
    assert material_temperature_guard(material)["matched"] is True

    comparable = deepcopy(dry["case"])
    comparable["weather"]["relative_humidity"] = 0.30
    comparable["label"] = humid["case"]["label"]
    assert comparable == humid["case"]
    assert canonical_sha256(normalized_dry) != canonical_sha256(
        normalized_humid
    )


@pytest.mark.parametrize(
    "temperature_K",
    (
        292.9,
        393.1531748676392,
        500.0,
    ),
)
def test_advisory_provider_accepts_below_and_above_old_discretionary_limits(
    temperature_K,
):
    adapter = build_material_adapter(load_builtin_material(MATERIAL_ID))
    water_partial = (
        0.05
        * young2021_advisory_temperature.iapws_saturation_pressure_Pa(
            temperature_K
        )
    )
    partials = np.array([400.0, water_partial, 0.0, 0.0])
    storage = np.array([0.4, 0.2])
    before = storage.copy()
    rates, _species = adapter.rates(
        temperature_K, partials, storage
    )
    equilibrium = adapter.equilibrium(temperature_K, partials)
    np.testing.assert_array_equal(storage, before)
    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(equilibrium))
    report = adapter.source_report()
    assert report["historical_range_enforced"] is False
    assert report["reference_temperature_enforced"] is False
    assert report["advisory_evaluation_observations"][
        "stored_inventory_mutated"
    ] is False


def test_implicit_equilibrium_uses_gab_water_for_both_output_components():
    temperature_K = 298.15
    partials = np.array(
        [
            42.0,
            0.30
            * young2021_advisory_temperature.iapws_saturation_pressure_Pa(
                temperature_K
            ),
            0.0,
            0.0,
        ]
    )
    implicit_co2, implicit_water, implicit_details = (
        young2021_advisory_temperature.equilibrium(
            temperature_K,
            partials[0],
            partials[1],
        )
    )
    explicit_co2, explicit_water, explicit_details = (
        young2021_advisory_temperature.equilibrium(
            temperature_K,
            partials[0],
            partials[1],
            water_loading_mol_kg=implicit_water,
        )
    )
    assert implicit_co2 == pytest.approx(1.226318321, rel=0, abs=2e-9)
    assert implicit_water == pytest.approx(1.962024676, rel=0, abs=2e-9)
    assert implicit_co2 == pytest.approx(explicit_co2, rel=0, abs=1e-14)
    assert implicit_water == pytest.approx(explicit_water, rel=0, abs=1e-14)
    assert implicit_details["wet_site_weight"] == pytest.approx(
        explicit_details["wet_site_weight"], rel=0, abs=1e-14
    )
    assert implicit_details["water_loading_normalization"][
        "raw_water_loading_mol_kg"
    ] == pytest.approx(implicit_water)


def test_adapter_rejects_negative_co2_and_preserves_tiny_tolerance_raw_delta():
    adapter = build_material_adapter(load_builtin_material(MATERIAL_ID))
    partials = np.array([42.0, 1_000.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="storage components"):
        adapter.rates(350.0, partials, np.array([-2e-9, 0.2]))

    storage = np.array([-0.5e-9, 0.2])
    before = storage.copy()
    rates, _species = adapter.rates(350.0, partials, storage)
    np.testing.assert_array_equal(storage, before)
    target = storage[0] + rates[0] / adapter._ad_rate[0]
    assert adapter._last_rate_details["raw_co2_delta_mol_kg"] == pytest.approx(
        target - storage[0]
    )
    assert adapter._last_rate_details["raw_water_loading_mol_kg"] == 0.2


def test_true_equation_branches_still_reject_with_precise_reasons():
    adapter = build_material_adapter(load_builtin_material(MATERIAL_ID))
    with pytest.raises(ValueError, match="IAPWS saturation equation domain"):
        adapter.rates(
            273.15,
            np.array([400.0, 100.0, 0.0, 0.0]),
            np.array([0.4, 0.2]),
        )
    assert adapter.failure_report() is None
    with pytest.raises(ValueError, match="storage components"):
        adapter.rates(
            350.0,
            np.array([400.0, 1_000.0, 0.0, 0.0]),
            np.array([0.4, -2e-9]),
        )
    assert adapter.source_report()[
        "material_query_temperature_observations"
    ]["failed_query_count"] == 2


def test_guard_assessment_records_excursion_without_physical_pass_claim():
    material, case = build_case(
        load_builtin_material(MATERIAL_ID), _preset(HUMID_PRESET)["case"]
    )
    assessment = temperature_guard_assessment(
        material,
        case,
        peak_bed_temperature_K=393.1531748676392,
    )
    assert assessment["status"] == "scenario_excursion_recorded"
    assert assessment["policy"]["enforcement_limit_C"] is None
    assert assessment["physical_degradation_limit_pass"] is None
    limits = hard_limits(
        {
            "temperature_guard_assessment": assessment,
            "checks": {"temperature_limit_pass": None},
        }
    )
    assert all(
        check["id"] != "material_temperature_limit"
        for check in limits["checks"]
    )


def test_accepted_temperature_extrema_are_separate_from_query_extrema():
    material = load_builtin_material(MATERIAL_ID)
    case = deepcopy(_preset(HUMID_PRESET)["case"])
    case["cycle"]["regeneration_temperature_C"] = 117.5
    accepted, queries, advisory = temperature_reporting(
        material,
        case,
        [
            {
                "bed_temperature_min_K": 389.0,
                "bed_temperature_max_K": 393.1531748676392,
            },
            {
                "bed_temperature_min_K": 390.0,
                "bed_temperature_max_K": 393.0,
            },
        ],
        {
            "material_query_temperature_observations": {
                "basis": "test query observations",
                "minimum_finite_temperature_K": 388.999,
                "maximum_finite_temperature_K": 393.154,
            }
        },
    )
    assert advisory is True
    assert accepted["minimum_bed_temperature_K"] == 389.0
    assert accepted["maximum_bed_temperature_K"] == pytest.approx(
        393.1531748676392
    )
    assert queries["minimum_finite_temperature_K"] == 388.999
    assert queries["maximum_finite_temperature_K"] == 393.154
    assert accepted["requested_regeneration_temperature_C"] == 117.5
    assert accepted["maximum_bed_temperature_K"] != (
        queries["maximum_finite_temperature_K"]
    )


def test_legacy_temperature_reporting_does_not_invent_advisory_semantics():
    case = deepcopy(_preset(HUMID_PRESET)["case"])
    case["cycle"]["regeneration_temperature_C"] = 108.0
    accepted, _queries, advisory = temperature_reporting(
        load_builtin_material("lewatit_vp_oc_1065_young2021_wadst_conditional"),
        case,
        [
            {
                "bed_temperature_min_K": 330.0,
                "bed_temperature_max_K": 350.0,
            }
        ],
        {"material_query_temperature_observations": None},
    )
    assert advisory is False
    assert accepted["requested_regeneration_temperature_C"] == 108.0
    assert accepted["historical_evaluation_range_K"] is None
    assert accepted["historical_range_hard_stop"] is None
    assert accepted["temperature_clamped"] is None


def test_preview_validates_actual_weather_equations_without_advisory_range(
    monkeypatch,
):
    material = load_builtin_material(MATERIAL_ID)
    below_old = deepcopy(_preset(HUMID_PRESET)["case"])
    below_old["weather"]["temperature_C"] = 19.75
    _, accepted_below = build_case(material, below_old)
    assert accepted_below["weather"]["temperature_C"] == 19.75

    invalid = deepcopy(below_old)
    invalid["weather"]["temperature_C"] = 0.0
    with pytest.raises(
        MofScreeningError, match="outside a genuine equilibrium branch"
    ):
        build_case(material, invalid)

    invalid_rh = deepcopy(below_old)
    invalid_rh["weather"]["relative_humidity"] = 1.0
    with pytest.raises(MofScreeningError, match=r"RH in \[0,1\)"):
        build_case(material, invalid_rh)

    official = deepcopy(below_old)
    official["weather"] = {
        "mode": "official",
        "hours": 1,
        "start_utc": "2023-01-15T00:00:00Z",
    }
    row = SimpleNamespace(
        interval_start_utc="2023-01-15T00:00:00+00:00",
        interval_end_utc="2023-01-15T01:00:00+00:00",
        temperature_C=130.0,
        relative_humidity=0.30,
        pressure_Pa=100_000.0,
        co2_ppm_dry=420.0,
        water_mole_fraction=0.003,
        prepared_buck_water_vapour_partial_Pa=30_000.0,
    )
    monkeypatch.setattr(
        "screening.mof_screening.weather.load_weather_window",
        lambda _weather: SimpleNamespace(rows=[row]),
    )
    _, accepted_hot = build_case(material, official)
    assert accepted_hot["weather"]["mode"] == "official"

    row.temperature_C = -0.01
    with pytest.raises(
        MofScreeningError, match="outside a genuine equilibrium branch"
    ):
        build_case(material, official)

    row.temperature_C = 130.0
    row.relative_humidity = 1.0
    with pytest.raises(MofScreeningError, match=r"RH in \[0,1\)"):
        build_case(material, official)


def test_latest_captured_state_passes_native_rhs_and_jacobian_without_integration():
    failure = json.loads(
        (JOB / "MATERIAL_DOMAIN_FAILURE.json").read_text(encoding="utf-8")
    )
    state_document = json.loads(
        (JOB / "STATE.json").read_text(encoding="utf-8")
    )
    input_case = json.loads(
        (JOB / "input_case.json").read_text(encoding="utf-8")
    )
    material, case = build_case(
        load_builtin_material(MATERIAL_ID), input_case
    )
    adapter = build_material_adapter(material)
    weather = load_weather_window(case["weather"])
    config = build_spatial_config(material, case, weather.rows[0])
    phase_offset_s = float(state_document["simulated_s"]) - float(
        state_document["phase_elapsed_s"]
    )
    config["_runtime_control"] = {
        "weather_provider": weather.at,
        "cycle_start_s": 0.0,
        "phase_offset_s": phase_offset_s,
        "cycle_number": 1,
        "temperature_guard_limit_K": None,
    }
    geometry = build_bare_geometry(config)
    model = build_model(config, adapter, geometry)
    state = np.asarray(failure["rejected_state_vector"], dtype=float)
    before = state.copy()
    inventory_before = energy_inventory(
        config, adapter, geometry, state, model
    )
    rhs = cycle_rhs(
        config, adapter, geometry, failure["phase"], model=model
    )
    local_time = float(state_document["phase_elapsed_s"])
    derivative, _observation = rhs(local_time, state)
    jacobian = bounded_spatial_jacobian(
        lambda time, values: rhs(time, values)[0],
        local_time,
        state,
        layout=model.layout,
        material=adapter,
        audit=SpatialJacobianAudit(
            physical_columns=model.layout.inlet.start,
            ledger_columns=model.layout.size - model.layout.inlet.start,
        ),
    )
    inventory_after = energy_inventory(
        config, adapter, geometry, state, model
    )
    assert np.array_equal(before, state)
    assert np.all(np.isfinite(derivative))
    assert np.all(np.isfinite(jacobian))
    assert inventory_before == inventory_after
    assert failure["temperature_K"] == pytest.approx(393.1531748676392)
    assert state_document["simulated_s"] == pytest.approx(11974.062369633008)
    assert local_time == pytest.approx(4474.062369633007)
    assert adapter.failure_report() is None


def test_csv_export_retains_temperature_and_water_diagnostics(tmp_path):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    job_id = "1" * 32
    root = manager.job_root / job_id
    root.mkdir(parents=True)
    (root / "STATE.json").write_text(
        json.dumps({"state": "completed", "job_id": job_id}),
        encoding="utf-8",
    )
    result = {
        "schema": "mof-screening-result/1",
        "job_id": job_id,
        "temperature_policy": {"mode": "record_scenario_excursions"},
        "temperature_guard_assessment": {
            "status": "scenario_excursion_recorded"
        },
        "accepted_state_temperature_observations": {
            "maximum_bed_temperature_K": 393.1531748676392
        },
        "material_query_temperature_observations": {
            "minimum_finite_temperature_K": 389.0,
            "maximum_finite_temperature_K": 394.0,
        },
        "water_burden": {
            "initial_loading_mol_kg": 0.5,
            "final_loading_mol_kg": 0.1,
        },
        "metrics": {
            "water_condensed_mol": 2.0,
            "water_vented_mol": 3.0,
            "water_in_product_mol": 4.0,
        },
        "checks": {
            "mass_closure_pass": True,
            "energy_closure_pass": True,
            "energy_relative_residual": 1e-8,
        },
        "traces": [
            {
                "elapsed_s": 1.0,
                "cycle": 1,
                "phase": "desorption",
                "bed_temperature_max_K": 393.1531748676392,
            }
        ],
    }
    (root / "RESULT.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    text, _filename, _content_type = manager.export(job_id, "csv")
    assert "accepted_peak_bed_temperature_K" in text
    assert "material_query_max_temperature_K" in text
    assert "water_loading_final_mol_kg" in text
    assert "energy_relative_residual" in text
    assert "scenario_excursion_recorded" in text
    manager.close()


def test_source_identity_binds_new_provider_package_and_configs():
    identity = execution_source_identity()
    hashes = identity["source_hashes"]
    for relative in (
        "screening/mof_screening/young2021_advisory_temperature.py",
        "materials/screening/"
        "lewatit_vp_oc_1065_young2021_advisory_temperature_conditional.json",
        "config/mof_screening/defaults.json",
        "config/mof_screening/temperature_registry.json",
    ):
        assert len(hashes[relative]) == 64
    assert identity["frozen_closure"]["sha256"] == hashes["config/mof_screening/dependencies.json"]
    assert young2021.YOUNG2021_DRY.q_inf_0_mol_kg == 4.86
