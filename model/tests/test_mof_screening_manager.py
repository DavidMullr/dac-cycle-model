import csv
import io
import json
import os
import threading
import time

import pytest

from screening.mof_screening import manager as manager_module
from screening.mof_screening.contract import (
    MofScreeningError,
    RESULT_SCHEMA,
    build_case,
    canonical_sha256,
    load_builtin_material,
    load_defaults,
    preview_payload,
    read_json_strict,
    validate_material,
)
from screening.mof_screening.manager import MofScreeningManager


def _windows_share_error(winerror):
    error = PermissionError(
        13, f"simulated Windows sharing violation {winerror}"
    )
    error.winerror = winerror
    return error


@pytest.mark.parametrize("winerror", (5, 32, 33))
def test_atomic_json_retries_transient_windows_share_errors(
    monkeypatch, tmp_path, winerror
):
    target = tmp_path / "PROGRESS.json"
    original = b'{"state":"original"}\n'
    target.write_bytes(original)
    replacement = {"state": "updated", "progress": 0.42}
    real_replace = manager_module.os.replace
    attempts = []
    sleeps = []

    def flaky_replace(source, destination):
        attempts.append((source, destination))
        assert destination.read_bytes() == original
        if len(attempts) == 1:
            raise _windows_share_error(winerror)
        real_replace(source, destination)

    monkeypatch.setattr(manager_module.os, "replace", flaky_replace)
    monkeypatch.setattr(
        manager_module, "WINDOWS_REPLACE_RETRY_DELAYS_S", (0.0, 0.0)
    )
    monkeypatch.setattr(manager_module.time, "sleep", sleeps.append)

    manager_module._atomic_json(target, replacement)

    assert json.loads(target.read_text(encoding="utf-8")) == replacement
    assert len(attempts) == 2
    assert sleeps == [0.0]
    assert not list(tmp_path.glob("PROGRESS.json.*.tmp"))


def test_atomic_json_retry_exhaustion_preserves_state_and_temp(
    monkeypatch, tmp_path
):
    target = tmp_path / "STATE.json"
    original = b'{"state":"running","progress":0.75}\n'
    target.write_bytes(original)
    replacement = {"state": "running", "progress": 0.9}
    attempts = []

    def blocked_replace(_source, _destination):
        attempts.append(1)
        assert target.read_bytes() == original
        raise _windows_share_error(32)

    monkeypatch.setattr(manager_module.os, "replace", blocked_replace)
    monkeypatch.setattr(
        manager_module, "WINDOWS_REPLACE_RETRY_DELAYS_S", (0.0, 0.0)
    )
    monkeypatch.setattr(manager_module.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError) as captured:
        manager_module._atomic_json(target, replacement)

    assert captured.value.winerror == 32
    assert len(attempts) == 3
    assert target.read_bytes() == original
    temporary_files = list(tmp_path.glob("STATE.json.*.tmp"))
    assert len(temporary_files) == 1
    assert json.loads(
        temporary_files[0].read_text(encoding="utf-8")
    ) == replacement


@pytest.mark.parametrize(
    "platform_name,winerror",
    (("nt", 87), ("posix", 5)),
)
def test_atomic_json_does_not_retry_permanent_or_non_windows_error(
    monkeypatch, tmp_path, platform_name, winerror
):
    target = tmp_path / "STATE.json"
    original = b'{"state":"original"}\n'
    target.write_bytes(original)
    attempts = []

    def blocked_replace(_source, _destination):
        attempts.append(1)
        raise _windows_share_error(winerror)

    monkeypatch.setattr(manager_module.os, "name", platform_name)
    monkeypatch.setattr(manager_module.os, "replace", blocked_replace)
    monkeypatch.setattr(
        manager_module, "WINDOWS_REPLACE_RETRY_DELAYS_S", (0.0, 0.0)
    )
    monkeypatch.setattr(manager_module.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError):
        manager_module._atomic_json(target, {"state": "updated"})

    assert attempts == [1]
    assert target.read_bytes() == original
    assert len(list(tmp_path.glob("STATE.json.*.tmp"))) == 1


def test_history_reader_waits_for_state_writer_and_history_stays_untouched(
    monkeypatch, tmp_path
):
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    current_id = "a" * 32
    historical_id = "b" * 32
    current_root = manager.job_root / current_id
    historical_root = manager.job_root / historical_id
    manager_module._atomic_json(
        current_root / "STATE.json",
        {
            "schema": "mof-screening-job-state/1",
            "job_id": current_id,
            "state": "running",
            "progress": 0.1,
            "payload": {"sequence": 1},
        },
    )
    manager_module._atomic_json(
        historical_root / "STATE.json",
        {
            "schema": "mof-screening-job-state/1",
            "job_id": historical_id,
            "state": "completed",
            "progress": 1.0,
            "updated_utc": "2026-11-05T00:00:00+00:00",
        },
    )
    manager_module._atomic_json(
        historical_root / "RESULT.json",
        {
            "job_id": historical_id,
            "case_label": "Preserved history",
            "engine": "spatial",
            "material": {"name": "Historical material", "synthetic": False},
            "metrics": {"collected_CO2_kg_day": 1.0},
            "input_identity": {"material_sha256": "1" * 64},
        },
    )
    historical_state_bytes = (historical_root / "STATE.json").read_bytes()
    historical_result_bytes = (historical_root / "RESULT.json").read_bytes()

    writer_holds_lock = threading.Event()
    allow_write = threading.Event()
    reader_ready = threading.Event()
    reader_done = threading.Event()
    foreign_state_read = threading.Event()
    reader_errors = []
    reader_rows = []
    real_read_json = manager_module.read_json_strict

    def observed_read(path):
        if (
            path.name == "STATE.json"
            and threading.current_thread().name == "history-reader"
        ):
            foreign_state_read.set()
        return real_read_json(path)

    monkeypatch.setattr(manager_module, "read_json_strict", observed_read)

    def writer():
        with manager.state_lock:
            writer_holds_lock.set()
            assert allow_write.wait(timeout=2.0)
            manager._write_state(
                current_id,
                state="running",
                progress=0.55,
                message="Concurrent state update",
                payload={"sequence": 2},
            )

    def reader():
        reader_ready.set()
        try:
            reader_rows.extend(manager.history())
        except Exception as error:  # pragma: no cover - asserted below
            reader_errors.append(error)
        finally:
            reader_done.set()

    writer_thread = threading.Thread(target=writer, name="state-writer")
    reader_thread = threading.Thread(target=reader, name="history-reader")
    writer_thread.start()
    assert writer_holds_lock.wait(timeout=2.0)
    reader_thread.start()
    assert reader_ready.wait(timeout=2.0)
    assert not foreign_state_read.wait(timeout=0.05)
    assert not reader_done.is_set()

    allow_write.set()
    writer_thread.join(timeout=2.0)
    reader_thread.join(timeout=2.0)

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert reader_errors == []
    current = next(row for row in reader_rows if row["job_id"] == current_id)
    historical = next(
        row for row in reader_rows if row["job_id"] == historical_id
    )
    assert current["state"] == "running"
    assert current["progress"] == pytest.approx(0.55)
    assert current["message"] == "Concurrent state update"
    assert historical["case_label"] == "Preserved history"
    assert historical["state"] == "completed"
    saved_current = read_json_strict(current_root / "STATE.json")
    assert saved_current["payload"] == {"sequence": 2}
    assert saved_current["progress"] == pytest.approx(0.55)
    assert (historical_root / "STATE.json").read_bytes() == historical_state_bytes
    assert (historical_root / "RESULT.json").read_bytes() == historical_result_bytes
    manager.close()


def _fake_result(material, case, **kwargs):
    assert kwargs["source_identity"]["release_sha256"]
    progress = kwargs.get("progress")
    if progress:
        progress(1.0, "mocked complete")
    return {
        "schema": RESULT_SCHEMA,
        "case_label": case["label"],
        "engine": "spatial",
        "engine_identity": "mocked-spatial",
        "recorded_utc": "2026-11-02T00:00:00+00:00",
        "source_identity": {"release_sha256": "d" * 64},
        "input_identity": {
            "material_sha256": "a" * 64,
            "case_sha256": "b" * 64,
            "combined_sha256": "c" * 64,
        },
        "material": {
            "material_id": material["material_id"],
            "name": "Mocked material",
            "evidence": {"status": "MOCK"},
            "synthetic": True,
            "assumptions": ["mock"],
            "validity": {},
            "uncertainty": {},
        },
        "metrics": {
            "collected_CO2_kg_day": 1.25,
            "electricity_kWh_day": 20.0,
            "electricity_kWh_kg_CO2": 16.0,
            "dry_product_CO2_fraction": 0.9,
            "wet_product_CO2_fraction": 0.5,
        },
        "horizon": {
            "complete": True,
            "cycles_completed": 1,
            "cycles_started": 1,
        },
        "endpoint": {"all_pass": True},
        "checks": {
            "mass_closure_pass": True,
            "energy_closure_pass": True,
        },
        "numerical_qualification": {"status": "not_established_for_this_run"},
        "traces": [
            {
                "elapsed_s": 10.0,
                "phase_elapsed_s": 0.0,
                "cycle": 1,
                "phase": "adsorption",
            },
            {
                "elapsed_s": 20.0,
                "phase_elapsed_s": 10.0,
                "cycle": 1,
                "phase": "adsorption",
            },
            {
                "elapsed_s": 30.0,
                "phase_elapsed_s": 0.0,
                "cycle": 2,
                "phase": "adsorption",
            },
        ],
    }


def _wait_for(manager, job_id, states, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = manager.inspect(job_id)
        if current["state"] in states:
            return current
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {states}")


def test_preview_start_history_compare_export_with_mock(monkeypatch, tmp_path):
    defaults = load_defaults()
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    monkeypatch.setattr(manager_module, "run_spatial", _fake_result)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    material = load_builtin_material("hypothetical_mmen_like")
    ids = []
    for index in range(2):
        case = json.loads(json.dumps(defaults["case"]))
        case["label"] = f"Mock case {index + 1}"
        preview = manager.preview(material, case)
        started = manager.start(preview["preview_id"])
        state = _wait_for(
            manager, started["job_id"], {"completed", "failed"}
        )
        assert state["state"] == "completed"
        ids.append(started["job_id"])
        assert not manager.execution_lock_path.exists()
    comparison = manager.compare(ids)
    assert len(comparison["rows"]) == 2
    assert manager.export(ids[0], "json")[1].endswith(".json")
    csv_text = manager.export(ids[0], "csv")[0]
    csv_rows = list(csv.reader(io.StringIO(csv_text)))
    assert csv_rows[0][:3] == ["time_s", "cycle", "phase"]
    assert csv_rows[0][16:21] == [
        "minimum_storage_mol_kg",
        "co2_loading_mean_mol_kg",
        "water_loading_mean_mol_kg",
        "local_rh_min",
        "local_rh_mean",
    ]
    assert "accepted_peak_bed_temperature_K" in csv_rows[0]
    assert "material_query_max_temperature_K" in csv_rows[0]
    assert "energy_relative_residual" in csv_rows[0]
    assert csv_rows[0][21] == "local_rh_max"
    assert csv_rows[0][22] == "phase_elapsed_s"
    assert len(csv_rows) == 4
    assert all(len(row) == len(csv_rows[0]) for row in csv_rows)
    assert all(
        float(row[0]) == pytest.approx(value)
        for row, value in zip(csv_rows[1:], [10, 20, 30])
    )
    assert [row[1] for row in csv_rows[1:]] == ["1", "1", "2"]
    assert len(manager.history()) == 2
    manager.close()


def test_immutable_preview_starts_with_mock_engine(monkeypatch, tmp_path):
    defaults = load_defaults()
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    monkeypatch.setattr(manager_module, "run_spatial", _fake_result)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    preview = manager.preview(
        load_builtin_material("hypothetical_mmen_like"),
        defaults["case"],
    )
    stored = read_json_strict(
        manager.preview_root / f"{preview['preview_id']}.json"
    )
    normalized = validate_material(stored["material"])

    assert canonical_sha256(normalized) == stored["material_sha256"]
    started = manager.start(preview["preview_id"])
    state = _wait_for(
        manager, started["job_id"], {"completed", "failed"}
    )
    assert state["state"] == "completed"
    manager.close()


def test_legacy_preview_without_runtime_policy_remains_startable(
    monkeypatch, tmp_path
):
    defaults = load_defaults()
    case = json.loads(json.dumps(defaults["case"]))
    del case["runtime_policy"]
    material, normalized_case = build_case(
        load_builtin_material("hypothetical_mmen_like"), case
    )
    payload = preview_payload(material, normalized_case, source="legacy")
    preview_id = "a" * 32
    payload.update(
        {
            "preview_id": preview_id,
            "created_utc": "2026-11-02T00:00:00+00:00",
        }
    )
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    monkeypatch.setattr(manager_module, "run_spatial", _fake_result)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    (manager.preview_root / f"{preview_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    started = manager.start(preview_id)
    state = _wait_for(
        manager, started["job_id"], {"completed", "failed"}
    )

    assert state["state"] == "completed"
    saved_case = read_json_strict(
        manager.job_root / started["job_id"] / "input_case.json"
    )
    assert "runtime_policy" not in saved_case
    assert state["runtime_policy"]["mode"] == "manual_stop"
    manager.close()


def test_start_refuses_disk_identity_changed_after_manager_startup(
    monkeypatch, tmp_path
):
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    preview = manager.preview(
        load_builtin_material("hypothetical_mmen_like"),
        load_defaults()["case"],
    )
    monkeypatch.setattr(
        manager_module,
        "execution_source_identity",
        lambda: {"release_sha256": "changed"},
    )
    with pytest.raises(Exception, match="changed after server startup"):
        manager.start(preview["preview_id"])
    assert not manager.execution_lock_path.exists()
    manager.close()


def test_comparison_flags_changed_runtime_source(monkeypatch, tmp_path):
    defaults = load_defaults()
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    source_ids = iter(["a" * 64, "b" * 64])

    def changed_source_result(material, case, **kwargs):
        payload = _fake_result(material, case, **kwargs)
        payload["source_identity"]["release_sha256"] = next(source_ids)
        return payload

    monkeypatch.setattr(
        manager_module, "run_spatial", changed_source_result
    )
    material = load_builtin_material("hypothetical_mmen_like")
    ids = []
    for index in range(2):
        case = json.loads(json.dumps(defaults["case"]))
        case["label"] = f"Source case {index + 1}"
        preview = manager.preview(material, case)
        started = manager.start(preview["preview_id"])
        state = _wait_for(
            manager, started["job_id"], {"completed", "failed"}
        )
        assert state["state"] == "completed"
        ids.append(started["job_id"])

    comparison = manager.compare(ids)
    assert comparison["same_configuration"] is True
    assert comparison["same_source_identity"] is False
    assert comparison["same_execution_basis"] is False
    assert comparison["basis_fields"][-1]["field"] == "source_identity"
    assert comparison["basis_fields"][-1]["matching"] is False
    manager.close()


def test_comparison_accepts_32_jobs_and_rejects_33(monkeypatch, tmp_path):
    defaults = load_defaults()
    material = load_builtin_material("hypothetical_mmen_like")
    case = defaults["case"]
    base = _fake_result(
        material,
        case,
        source_identity={"release_sha256": "d" * 64},
    )
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    job_ids = [f"{index:032x}" for index in range(33)]

    def inspect(job_id):
        result = json.loads(json.dumps(base))
        result["job_id"] = job_id
        return {"state": "completed", "result": result}

    monkeypatch.setattr(manager, "inspect", inspect)
    comparison = manager.compare(job_ids[:32])
    assert len(comparison["rows"]) == 32
    assert comparison["rows"][0]["material_evidence_status"] == "MOCK"

    with pytest.raises(Exception, match="2-32"):
        manager.compare(job_ids)
    manager.close()


def test_cancel_callback_is_retained_as_cancelled(monkeypatch, tmp_path):
    defaults = load_defaults()

    def blocking_job(
        material, case, *, cancel_check, progress, source_identity
    ):
        assert source_identity
        for _ in range(500):
            cancel_check()
            time.sleep(0.01)
        raise AssertionError("cancel was not observed")

    monkeypatch.setattr(manager_module, "run_spatial", blocking_job)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    preview = manager.preview(
        load_builtin_material("hypothetical_mmen_like"),
        defaults["case"],
    )
    started = manager.start(preview["preview_id"])
    manager.cancel(started["job_id"])
    state = _wait_for(manager, started["job_id"], {"cancelled"})
    assert state["failure"]["state"] == "cancelled"
    assert not manager.execution_lock_path.exists()
    manager.close()


def test_manager_records_runtime_policy_progress_and_partial_cancel(
    monkeypatch, tmp_path
):
    defaults = load_defaults()

    def blocking_job(material, case, *, cancel_check, progress, source_identity):
        assert case["runtime_policy"]["mode"] == "manual_stop"
        progress(
            0.25,
            "completed first phase",
            {
                "event": "phase_completed",
                "engine": "spatial",
                "cycle": 1,
                "phase": "evacuation",
                "elapsed_s": 7200.0,
                "checkpoint_state": [300.0, 301.0],
            },
        )
        progress(
            0.30,
            "completed first cycle",
            {
                "event": "cycle_completed",
                "engine": "spatial",
                "cycle": 1,
                "elapsed_s": 18000.0,
                "checkpoint_state": [301.0, 302.0],
            },
        )
        progress(
            0.37,
            "spatial cycle 1 desorption at 1200 s",
            {
                "event": "phase_completed",
                "engine": "spatial",
                "cycle": 1,
                "phase": "desorption",
                "elapsed_s": 1200.0,
                "phase_elapsed_s": 120.0,
                "phase_duration_s": 5400.0,
                "checkpoint_state": [300.0, 301.0],
            },
        )
        for _ in range(500):
            cancel_check()
            time.sleep(0.01)
        raise AssertionError("cancel was not observed")

    monkeypatch.setattr(manager_module, "run_spatial", blocking_job)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    preview = manager.preview(
        load_builtin_material("hypothetical_mmen_like"),
        defaults["case"],
    )
    assert preview["case"]["runtime_policy"]["mode"] == "manual_stop"
    started = manager.start(preview["preview_id"])
    manager.cancel(started["job_id"])
    state = _wait_for(manager, started["job_id"], {"cancelled"})

    assert state["progress"] == pytest.approx(0.37)
    assert state["current_phase"] == "desorption"
    assert state["elapsed_wall_s"] >= 0.0
    partial = state["partial"]
    assert partial["status"] == "cancelled"
    assert partial["complete"] is False
    assert partial["last_progress"]["simulated_s"] == 1200.0
    assert partial["last_checkpoint"]["state_sha256"]
    assert partial["last_checkpoint"]["resumable"] is False
    assert partial["accepted_progress_summaries"]["count"] == 3
    summary_lines = (
        manager.job_root
        / started["job_id"]
        / "PROGRESS_SUMMARIES.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(summary_lines) == 3
    assert state["failure"]["state"] == "cancelled"
    assert not manager.execution_lock_path.exists()
    manager.close()


def test_manager_explicit_deadline_is_timed_out_and_incomplete(
    monkeypatch, tmp_path
):
    defaults = load_defaults()
    case = json.loads(json.dumps(defaults["case"]))
    case["runtime_policy"] = {
        **case["runtime_policy"],
        "mode": "time_limit",
        "time_limit_s": 2.0,
    }

    def deadline_job(material, case, *, cancel_check, progress, source_identity):
        assert case["runtime_policy"]["time_limit_s"] == 2.0
        progress(
            0.44,
            "spatial cycle 1 cooling at 8000 s",
            {
                "engine": "spatial",
                "cycle": 1,
                "phase": "cooling",
                "elapsed_s": 8000.0,
                "event": "integration_step",
            },
        )
        raise MofScreeningError(
            "screening reached the user-selected execution time limit",
            status=504,
        )

    monkeypatch.setattr(manager_module, "run_spatial", deadline_job)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=tmp_path / "LOCK.json"
    )
    preview = manager.preview(
        load_builtin_material("hypothetical_mmen_like"), case
    )
    started = manager.start(preview["preview_id"])
    state = _wait_for(manager, started["job_id"], {"timed_out"})

    assert state["progress"] == pytest.approx(0.44)
    assert state["partial"]["status"] == "timed_out"
    assert state["partial"]["complete"] is False
    assert state["partial"]["last_progress"]["phase"] == "cooling"
    assert state["failure"]["state"] == "timed_out"
    manager.close()


def test_shared_execution_lock_blocks_new_job(monkeypatch, tmp_path):
    lock = tmp_path / "LOCK.json"
    lock.write_text(
        json.dumps(
            {
                "owner_token": "other",
                "pid": os.getpid(),
                "study_id": "old-study",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "job", execution_lock_path=lock
    )
    preview = manager.preview(
        load_builtin_material("hypothetical_mmen_like"),
        load_defaults()["case"],
    )
    with pytest.raises(Exception, match="shared execution lock"):
        manager.start(preview["preview_id"])
    manager.close()
