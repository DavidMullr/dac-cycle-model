import json
from pathlib import Path
import time

import pytest

from screening.mof_screening import manager as manager_module
from screening.mof_screening.contract import (
    RESULT_SCHEMA,
    canonical_sha256,
    load_builtin_material,
    load_defaults,
    read_json_strict,
)
from screening.mof_screening.evaluation import evaluate_existing_result
from screening.mof_screening.manager import MofScreeningManager
from screening.mof_screening.preflight import execution_source_identity
from screening.mof_screening import runtime as runtime_module


def _result(*, target_miss=True, hard_failure=False):
    endpoint = {
        "all_pass": not target_miss,
        "regeneration_temperature_K": 354.0,
        "regeneration_temperature_target_K": 373.0,
        "regeneration_temperature_pass": not target_miss,
        "desorption_pressure_Pa": 18300.0,
        "desorption_pressure_target_Pa": 20000.0,
        "desorption_pressure_pass": True,
        "cooling_temperature_K": 298.65,
        "cooling_temperature_target_K": 298.65,
        "cooling_pass": True,
        "repressurization_pressure_Pa": 99500.0,
    }
    return {
        "schema": RESULT_SCHEMA,
        "endpoint": endpoint,
        "checks": {
            "mass_closure_pass": True,
            "energy_closure_pass": True,
            "mass_relative_residual_by_species": [1e-12],
            "energy_relative_residual": 1e-9,
            "temperature_limit_pass": not hard_failure,
            "finite_observations": True,
        },
        "numerical_qualification": {
            "status": "not_established_for_this_run",
            "statement": "Single-run diagnostic.",
        },
        "cycles": [],
    }


def test_target_miss_is_advisory_when_hard_limits_and_conservation_pass():
    evaluation = evaluate_existing_result(_result(target_miss=True))

    targets = evaluation["target_attainment"]
    assert targets["status"] == "advisory_miss"
    assert targets["advisory"] is True
    assert targets["miss_alone_excludes_case"] is False
    assert targets["miss_alone_ranks_case_down"] is False
    assert targets["checks"][0]["achieved"] == 354.0
    assert targets["checks"][0]["target"] == 373.0
    assert evaluation["hard_limits"]["status"] == "partial"
    assert evaluation["conservation"]["status"] == "pass"
    assert (
        evaluation["numerical_reliability"]["status"]
        == "not_established_for_this_run"
    )


def test_hard_limit_failure_stays_separate_from_target_attainment():
    evaluation = evaluate_existing_result(
        _result(target_miss=False, hard_failure=True)
    )

    assert evaluation["target_attainment"]["status"] == "satisfied"
    assert evaluation["hard_limits"]["status"] == "fail"
    assert evaluation["conservation"]["status"] == "pass"


def test_unknown_targets_and_limits_are_not_reported_as_pass():
    result = _result(target_miss=False)
    result["endpoint"] = {"all_pass": False}
    result["checks"].pop("temperature_limit_pass")

    evaluation = evaluate_existing_result(result)

    assert evaluation["target_attainment"]["status"] == "not_available"
    assert evaluation["hard_limits"]["status"] == "not_assessed"
    assert all(
        check["passed"] is None
        for check in evaluation["target_attainment"]["checks"]
    )


def test_partial_spatial_tail_does_not_override_completed_cycle_targets():
    result = _result(target_miss=False)
    result["endpoint"].update(
        {
            # This mirrors the historical wrapper's last-observed trace.
            "regeneration_temperature_K": 354.154794435,
            "regeneration_temperature_pass": False,
            "desorption_pressure_Pa": 18331.329412495,
            "cooling_temperature_K": 298.65,
            "cooling_pass": True,
        }
    )
    result["horizon"] = {
        "cycles_completed": 1,
        "cycles_started": 2,
        "partial_final_cycle": True,
    }
    result["cycles"] = [
        {
            "cycle": 1,
            "complete": True,
            "windows": [
                {"phase": "adsorption", "start_s": 0.0, "end_s": 1800.0},
                {"phase": "evacuation", "start_s": 1800.0, "end_s": 3000.0},
                {"phase": "desorption", "start_s": 3000.0, "end_s": 8400.0},
                {"phase": "cooling", "start_s": 8400.0, "end_s": 10186.0},
                {
                    "phase": "repressurization",
                    "start_s": 10186.0,
                    "end_s": 10196.0,
                },
            ],
            "phase_completion": {
                phase: {"horizon_truncated": False}
                for phase in (
                    "adsorption",
                    "evacuation",
                    "desorption",
                    "cooling",
                    "repressurization",
                )
            },
            "phase_evidence": [
                {"phase": phase}
                for phase in (
                    "adsorption",
                    "evacuation",
                    "desorption",
                    "cooling",
                    "repressurization",
                )
            ],
        },
        {
            "cycle": 2,
            "complete": False,
            "windows": [
                {"phase": "adsorption", "start_s": 0.0, "end_s": 1800.0},
                {"phase": "evacuation", "start_s": 1800.0, "end_s": 3000.0},
                {"phase": "desorption", "start_s": 3000.0, "end_s": 4203.57},
            ],
            "phase_completion": {
                "adsorption": {"horizon_truncated": False},
                "evacuation": {"horizon_truncated": False},
                "desorption": {"horizon_truncated": True},
            },
            "phase_evidence": [
                {"phase": "adsorption"},
                {"phase": "evacuation"},
                {"phase": "desorption"},
            ],
        },
    ]
    result["traces"] = [
        {
            "cycle": 1,
            "phase": "desorption",
            "elapsed_s": 8400.0,
            "bed_temperature_min_K": 373.1209234901842,
            "min_pressure_Pa": 19428.514819183947,
        },
        {
            "cycle": 1,
            "phase": "cooling",
            "elapsed_s": 10186.0,
            "bed_temperature_max_K": 298.65,
            "min_pressure_Pa": 15953.471563963145,
        },
        {
            "cycle": 1,
            "phase": "repressurization",
            "elapsed_s": 10196.0,
            "bed_temperature_max_K": 298.65,
            "min_pressure_Pa": 99500.0,
        },
        {
            "cycle": 2,
            "phase": "desorption",
            "elapsed_s": 14400.0,
            "bed_temperature_min_K": 354.1547944350525,
            "min_pressure_Pa": 18331.329412495477,
        },
    ]

    targets = evaluate_existing_result(result)["target_attainment"]
    temperature = next(
        check
        for check in targets["checks"]
        if check["id"] == "regeneration_temperature"
    )
    partial = [
        row
        for row in targets["partial_stage_observations"]
        if row["phase"] == "desorption"
    ][0]

    assert targets["status"] == "satisfied"
    assert targets["latest_completed_cycle"] == 1
    assert temperature["cycle"] == 1
    assert temperature["duration_s"] == 5400.0
    assert temperature["achieved"] == 373.1209234901842
    assert temperature["passed"] is True
    assert partial["cycle"] == 2
    assert partial["duration_s"] == pytest.approx(1203.57)
    assert partial["achieved"] == 354.1547944350525
    assert partial["counts_toward_completed_targets"] is False
    assert targets["partial_stages_counted_as_completed_misses"] is False


def test_truncated_repressurization_with_all_five_phases_stays_partial():
    result = _result(target_miss=False)
    result["horizon"] = {
        "cycles_completed": 0,
        "cycles_started": 1,
        "partial_final_cycle": True,
    }
    result["cycles"] = [
        {
            "cycle": 1,
            "complete": False,
            "windows": [
                {"phase": phase, "start_s": index, "end_s": index + 1}
                for index, phase in enumerate(
                    (
                        "adsorption",
                        "evacuation",
                        "desorption",
                        "cooling",
                        "repressurization",
                    )
                )
            ],
            "phase_completion": {
                **{
                    phase: {"horizon_truncated": False}
                    for phase in (
                        "adsorption",
                        "evacuation",
                        "desorption",
                        "cooling",
                    )
                },
                "repressurization": {"horizon_truncated": True},
            },
            "phase_evidence": [
                {"phase": phase}
                for phase in (
                    "adsorption",
                    "evacuation",
                    "desorption",
                    "cooling",
                    "repressurization",
                )
            ],
        }
    ]
    result["traces"] = [
        {
            "cycle": 1,
            "phase": "repressurization",
            "elapsed_s": 5.0,
            "min_pressure_Pa": 90000.0,
        }
    ]

    targets = evaluate_existing_result(result)["target_attainment"]

    assert targets["latest_completed_cycle"] is None
    assert targets["status"] == "not_available"
    assert len(targets["checks"]) == 3
    assert all(not check["known"] for check in targets["checks"])
    assert targets["partial_stage_observations"][0]["phase"] == (
        "repressurization"
    )


def test_explicit_zero_completed_cycles_is_not_treated_as_missing():
    result = _result(target_miss=False)
    result["horizon"] = {
        "cycles_completed": 0,
        "cycles_started": 1,
        "partial_final_cycle": True,
    }
    result["cycles"] = [
        {
            "cycle": 1,
            "phase": phase,
            "start_elapsed_s": index,
            "end_elapsed_s": index + 1,
            "duration_s": 1.0,
            "endpoint": {
                "endpoint_ok": True,
                "temperature_end_K": 373.0,
                "temperature_target_K": 373.0,
                "temperature_ok": True,
                "pressure_end_Pa": 20000.0,
                "pressure_target_Pa": 20000.0,
                "pressure_ok": True,
            },
        }
        for index, phase in enumerate(
            (
                "adsorption",
                "evacuation",
                "desorption",
                "cooling",
                "repressurization",
            )
        )
    ]

    targets = evaluate_existing_result(result)["target_attainment"]

    assert targets["latest_completed_cycle"] is None
    assert targets["status"] == "not_available"
    assert len(targets["partial_stage_observations"]) == 1
    assert targets["partial_stage_observations"][0]["phase"] == (
        "repressurization"
    )


def test_legacy_official_weather_target_uses_phase_end_interval():
    saved_path = (
        Path(__file__).resolve().parent / "fixtures/mof_screening/lumped_result_9.json"
    )
    result = read_json_strict(saved_path)

    targets = evaluate_existing_result(result)["target_attainment"]
    cooling = next(
        check
        for check in targets["checks"]
        if check["id"] == "cooling_temperature"
    )

    assert targets["status"] == "satisfied"
    assert cooling["achieved"] == pytest.approx(281.65)
    assert cooling["target"] == pytest.approx(281.65)
    assert cooling["passed"] is True


def test_missing_target_checks_remain_visible_as_unknown():
    result = _result(target_miss=False)
    result["endpoint"] = {
        "regeneration_temperature_target_K": 373.0
    }
    result["horizon"] = {
        "cycles_completed": 1,
        "cycles_started": 1,
        "partial_final_cycle": False,
    }
    result["cycles"] = [
        {
            "cycle": 1,
            "complete": True,
            "windows": [
                {"phase": phase, "start_s": index, "end_s": index + 1}
                for index, phase in enumerate(
                    (
                        "adsorption",
                        "evacuation",
                        "desorption",
                        "cooling",
                        "repressurization",
                    )
                )
            ],
            "phase_completion": {
                phase: {"horizon_truncated": False}
                for phase in (
                    "adsorption",
                    "evacuation",
                    "desorption",
                    "cooling",
                    "repressurization",
                )
            },
            "phase_evidence": [
                {"phase": phase}
                for phase in (
                    "adsorption",
                    "evacuation",
                    "desorption",
                    "cooling",
                    "repressurization",
                )
            ],
        }
    ]
    result["traces"] = [
        {
            "cycle": 1,
            "phase": "desorption",
            "elapsed_s": 4.0,
            "bed_temperature_min_K": 373.0,
            "min_pressure_Pa": None,
        }
    ]

    targets = evaluate_existing_result(result)["target_attainment"]
    known = [check["known"] for check in targets["checks"]]

    assert len(targets["checks"]) == 3
    assert known == [True, False, False]
    assert targets["status"] == "incomplete"


def _comparison_result(material, case, source_sha="a" * 64):
    return {
        "schema": RESULT_SCHEMA,
        "job_id": "",
        "case_label": case["label"],
        "engine": case["engine"],
        "engine_identity": "mocked",
        "source_identity": {"release_sha256": source_sha},
        "input_identity": {
            "material_sha256": canonical_sha256(material),
            "case_sha256": "c" * 64,
            "combined_sha256": "d" * 64,
        },
        "material": {
            "material_id": material["material_id"],
            "name": material["name"],
            "synthetic": material["synthetic"],
        },
        "case": json.loads(json.dumps(case)),
        "weather": {"mode": case["weather"]["mode"]},
        "operating_policy": case.get("operating_policy", {}),
        "geometry": {"mass_kg": case["equipment"]["dry_shaped_mass_kg"]},
        "state_layout": {
            "model": "mock",
            "material_sites": (
                2
                if material["material_id"]
                == "calf20_nguyen2024_80_20_shaped"
                else 1
            ),
        },
        "horizon": {
            "complete": True,
            "requested_s": 14400.0,
            "simulated_s": 14400.0,
            "cycles_completed": (
                2 if case["cycle"]["adsorption_s"] <= 1200.0 else 1
            ),
            "cycles_started": (
                2 if case["cycle"]["adsorption_s"] <= 1200.0 else 2
            ),
            "partial_final_cycle": (
                case["cycle"]["adsorption_s"] > 1200.0
            ),
            "initial_state": case["initial_state"]["basis"],
        },
        "metrics": {
            "collected_CO2_kg_day": 1.0,
            "electricity_kWh_day": 10.0,
            "electricity_kWh_kg_CO2": 10.0,
            "dry_product_CO2_fraction": 0.9,
            "wet_product_CO2_fraction": 0.5,
        },
        "endpoint": {
            "all_pass": False,
            "regeneration_temperature_K": 354.0,
            "regeneration_temperature_target_K": 373.0,
            "regeneration_temperature_pass": False,
            "desorption_pressure_Pa": 18300.0,
            "desorption_pressure_target_Pa": 20000.0,
            "desorption_pressure_pass": True,
            "cooling_temperature_K": 298.65,
            "cooling_temperature_target_K": 298.65,
            "cooling_pass": True,
        },
        "checks": {
            "mass_closure_pass": True,
            "energy_closure_pass": True,
            "temperature_limit_pass": True,
            "finite_observations": True,
        },
        "numerical_qualification": {
            "status": "not_established_for_this_run"
        },
        "assumptions": {"engine": "mock"},
        "cycles": [],
        "traces": [],
    }


def _wait_for(manager, job_id, states, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = manager.inspect(job_id)
        if state["state"] in states:
            return state
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {states}")


def test_deliberate_operating_change_is_a_controlled_comparison(
    monkeypatch, tmp_path
):
    defaults = load_defaults()
    material = load_builtin_material("hypothetical_mmen_like")

    def mocked_run(material_document, case_document, **_kwargs):
        return _comparison_result(material_document, case_document)

    monkeypatch.setattr(manager_module, "run_spatial", mocked_run)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    ids = []
    for adsorption_s, label in ((1800.0, "Baseline"), (1200.0, "Shorter cycle")):
        case = json.loads(json.dumps(defaults["case"]))
        case["label"] = label
        case["cycle"]["adsorption_s"] = adsorption_s
        preview = manager.preview(material, case)
        started = manager.start(preview["preview_id"])
        assert _wait_for(manager, started["job_id"], {"completed"})[
            "state"
        ] == "completed"
        ids.append(started["job_id"])

    comparison = manager.compare(ids)
    assert comparison["same_configuration"] is False
    assert comparison["exact_configuration"] is False
    assert comparison["compatible_comparison"] is True
    assert comparison["comparison_mode"] == "controlled_comparison"
    assert "cycle" in comparison["controlled_change_fields"]
    assert all(
        row["target_attainment_status"] == "advisory_miss"
        for row in comparison["rows"]
    )
    manager.close()


def test_same_resolution_mixed_material_layouts_are_compatible(
    monkeypatch, tmp_path
):
    defaults = load_defaults()

    def mocked_run(material_document, case_document, **_kwargs):
        return _comparison_result(material_document, case_document)

    monkeypatch.setattr(manager_module, "run_spatial", mocked_run)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    ids = []
    for material_id in ("hypothetical_mmen_like", "calf20_shaped"):
        case = json.loads(json.dumps(defaults["case"]))
        case["label"] = material_id
        preview = manager.preview(
            load_builtin_material(material_id), case
        )
        started = manager.start(preview["preview_id"])
        _wait_for(manager, started["job_id"], {"completed"})
        ids.append(started["job_id"])

    comparison = manager.compare(ids)
    assert comparison["material_changed"] is True
    assert comparison["same_resolution"] is True
    assert comparison["family_layout_changed"] is True
    assert comparison["compatible_comparison"] is True
    assert "state_layout" not in comparison["incompatible_fields"]
    assert comparison["comparison_mode"] == "exact_configuration"
    assert "controlled comparisons" in comparison["ranking_note"]
    assert "mismatched" not in comparison["ranking_note"]
    manager.close()


def test_edited_material_with_same_id_is_detected_by_package_hash(
    monkeypatch, tmp_path
):
    defaults = load_defaults()

    def mocked_run(material_document, case_document, **_kwargs):
        return _comparison_result(material_document, case_document)

    monkeypatch.setattr(manager_module, "run_spatial", mocked_run)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    baseline = load_builtin_material("hypothetical_mmen_like")
    edited = json.loads(json.dumps(baseline))
    edited["co2"]["heat_J_mol"] *= 1.01
    ids = []
    for material in (baseline, edited):
        preview = manager.preview(material, defaults["case"])
        started = manager.start(preview["preview_id"])
        _wait_for(manager, started["job_id"], {"completed"})
        ids.append(started["job_id"])

    comparison = manager.compare(ids)
    assert comparison["rows"][0]["material_id"] == comparison["rows"][1][
        "material_id"
    ]
    assert comparison["material_changed"] is True
    manager.close()


def test_legacy_saved_result_gets_read_only_derived_evaluation(tmp_path):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    job_id = "a" * 32
    job_root = manager.job_root / job_id
    job_root.mkdir()
    material = load_builtin_material("hypothetical_mmen_like")
    case = load_defaults()["case"]
    result = _comparison_result(material, case)
    result["job_id"] = job_id
    (job_root / "STATE.json").write_text(
        json.dumps(
            {
                "schema": "mof-screening-job-state/1",
                "job_id": job_id,
                "state": "completed",
                "progress": 1.0,
                "message": "Saved legacy result",
                "updated_utc": "2026-11-02T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    result_path = job_root / "RESULT.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )
    before = result_path.read_bytes()

    inspected = manager.inspect(job_id)

    assert inspected["result"]["derived_evaluation"]["target_attainment"][
        "status"
    ] == "advisory_miss"
    assert inspected["result"]["derived_evaluation"]["hard_limits"][
        "status"
    ] == "partial"
    assert result_path.read_bytes() == before
    manager.close()


def test_lumped_postprocess_completes_with_mocked_saved_session(
    monkeypatch, tmp_path
):
    saved_path = (
        Path(__file__).resolve().parent / "fixtures/mof_screening/lumped_result_8.json"
    )
    saved = read_json_strict(saved_path)
    summary = saved["raw_model_evidence"]["summary"]

    def mocked_session(*_args, **_kwargs):
        return {
            "summary": summary,
            "checkpoint_path": str(tmp_path / "checkpoint.json"),
            "checkpoint_sha256": "e" * 64,
        }

    monkeypatch.setattr(
        runtime_module, "run_weather_session", mocked_session
    )
    result = runtime_module.run_lumped(
        saved["material_package"],
        saved["case"],
        checkpoint_path=tmp_path / "checkpoint.json",
        source_identity=saved["source_identity"],
    )

    desorption = next(row for row in summary["phase_history"] if row["phase"] == "desorption")
    assert result["endpoint"]["desorption_pressure_Pa"] == desorption["pressure_end_Pa"]
    assert result["metrics"]["peak_bed_temperature_K"] == max(
        row["diagnostics"]["maximum_temperature_K"] for row in summary["chunk_records"]
    )
    assert result["engine"] == "lumped"
    assert result["endpoint"]["regeneration_temperature_K"] == pytest.approx(
        372.9700724152217
    )
    assert result["endpoint"]["all_pass"] is True
    assert result["target_attainment"]["status"] == "satisfied"
    assert isinstance(
        result["checks"]["mass_relative_residual_by_species"], list
    )
    assert result["checks"]["mass_relative_residual_by_species"] == [
        pytest.approx(summary["mass_closure_relative_error"])
    ]
    assert result["checks"]["mass_relative_residual_basis"] == (
        "maximum_relative_residual_across_species"
    )


def test_source_identity_binds_active_static_workflow_assets():
    source_hashes = execution_source_identity()["source_hashes"]

    assert "app/mof_screening/static/index.html" in source_hashes
    assert "app/mof_screening/static/app.js" in source_hashes
    assert "app/mof_screening/static/style.css" in source_hashes
    assert "app/mof_screening/static/material_builder.js" in source_hashes
    assert "app/mof_screening/static/material_state.js" in source_hashes
    assert "app/mof_screening/static/comparison_view.js" in source_hashes
    assert "app/mof_screening/static/result_view.js" in source_hashes
    assert "screening/mof_screening/evaluation.py" in source_hashes
    assert "screening/mof_screening/guided_material.py" in source_hashes
