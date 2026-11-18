"""Real child-process lifecycle checks with a zero-ODE latch worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from screening.mof_screening import parallel_manager as parallel
from screening.mof_screening import manager as legacy
from screening.mof_screening.contract import load_defaults, load_builtin_material, MofScreeningError
from test_mof_screening_manager import _fake_result


LATCH_WORKER = r'''
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1])
(root/'READY').write_text(str(os.getpid()))
while True:
 if (root/'CRASH').exists():os._exit(7)
 cancel=(root/'CANCEL_REQUEST.json').exists()
 if cancel or (root/'RELEASE').exists():
  if not cancel:(root/'RESULT.json').write_bytes((root/'FIXTURE_RESULT.json').read_bytes())
  p=root/'STATE.json';state=json.loads(p.read_text());state['state']='cancelled' if cancel else 'completed'
  temp=root/'fixture-state.tmp';temp.write_text(json.dumps(state))
  for attempt in range(10):
   try:temp.replace(p);break
   except PermissionError:time.sleep(.01)
  break
 time.sleep(.02)
'''


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.025)
    raise AssertionError("worker lifecycle did not reach expected state")


@pytest.fixture
def manager(monkeypatch, tmp_path):
    script = tmp_path / "latch_worker.py"
    script.write_text(LATCH_WORKER)
    monkeypatch.setattr(parallel, "require_ready", lambda: {"start_enabled": True})

    def launch(root, owner_pid):
        preview = json.loads((root / "PREVIEW.json").read_text())
        result = _fake_result(preview["material"], preview["case"], source_identity={"release_sha256": "fixture"})
        result.update(job_id=root.name, source_identity=json.loads((root / "WORKER_INPUT.json").read_text())["source_identity"])
        (root / "FIXTURE_RESULT.json").write_text(json.dumps(result))
        return subprocess.Popen([sys.executable, str(script), str(root)], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    instance = parallel.ConcurrentMofScreeningManager(tmp_path / "output", tmp_path / "LOCK.json", launcher=launch)
    yield instance
    instance.close()


def start(manager):
    preview = manager.preview(load_builtin_material("hypothetical_mmen_like"), load_defaults()["case"])
    job = manager.start(preview["preview_id"])["job_id"]
    wait_until(lambda: (manager.job_root / job / "READY").exists())
    return job


def test_three_processes_cancel_one_and_retain_shared_exclusion(manager, monkeypatch, tmp_path):
    manager.set_concurrency(3)
    ids = [start(manager) for _ in range(3)]
    assert len({manager._jobs[job]["pid"] for job in ids}) == 3
    assert len(manager.status()["active_jobs"]) == 3
    assert manager.thread.is_alive() and manager.execution_lock_path.exists()
    with pytest.raises(MofScreeningError, match="maximum simultaneous"):
        start(manager)
    monkeypatch.setattr(legacy, "require_ready", lambda: {})
    other = legacy.MofScreeningManager(tmp_path / "other", manager.execution_lock_path)
    preview = other.preview(load_builtin_material("hypothetical_mmen_like"), load_defaults()["case"])
    with pytest.raises(MofScreeningError, match="another study"):
        other.start(preview["preview_id"])
    manager.cancel(ids[0])
    wait_until(lambda: manager.status()["concurrency"]["active_count"] == 2)
    assert manager.inspect(ids[0])["state"] == "cancelled"
    assert manager.execution_lock_path.exists()
    assert {row["job_id"] for row in manager.status()["active_jobs"]} == set(ids[1:])
    (manager.job_root / ids[1] / "RELEASE").touch()
    wait_until(lambda: manager.status()["concurrency"]["active_count"] == 1)
    assert manager.execution_lock_path.exists()
    (manager.job_root / ids[2] / "RELEASE").touch()
    wait_until(lambda: not manager.thread.is_alive())
    assert not manager.execution_lock_path.exists()
    assert manager.export(ids[1], "json")[1].endswith(".json")
    assert len(manager.history()) == 3
    other.close()


def test_zero_means_unlimited_and_lower_limit_does_not_cancel(manager):
    assert manager.concurrency()["unlimited"]
    ids = [start(manager) for _ in range(5)]
    assert manager.concurrency()["active_count"] == 5
    manager.set_concurrency(1)
    assert manager.concurrency()["active_count"] == 5
    assert not any((manager.job_root / job / "CANCEL_REQUEST.json").exists() for job in ids)
    with pytest.raises(MofScreeningError, match="maximum simultaneous"):
        start(manager)
    manager.set_concurrency(0)
    ids.append(start(manager))
    assert manager.concurrency()["active_count"] == 6


def test_crash_does_not_release_sibling_lock_and_is_durable(manager):
    first, second = start(manager), start(manager)
    (manager.job_root / first / "CRASH").touch()
    wait_until(lambda: manager.status()["concurrency"]["active_count"] == 1)
    assert manager.inspect(first)["state"] == "failed"
    assert manager.inspect(first)["partial"]["complete"] is False
    assert manager.inspect(second)["state"] in parallel.ACTIVE
    assert manager.execution_lock_path.exists()


@pytest.mark.parametrize("limit", [-1, True, 1.5, None, "3"])
def test_invalid_operational_limits_rejected(manager, limit):
    with pytest.raises(MofScreeningError):
        manager.set_concurrency(limit)


def test_launch_failure_releases_aggregate_lock(manager):
    def fail(*args):
        raise OSError("cannot create process")
    manager._launcher = fail
    with pytest.raises(MofScreeningError, match="cannot create process"):
        start(manager)
    assert not manager.execution_lock_path.exists()
    assert not manager.thread.is_alive()


def test_real_worker_uses_sealed_input_and_existing_runtime(monkeypatch, tmp_path):
    from screening.mof_screening import process_worker
    monkeypatch.setattr(parallel, "require_ready", lambda: {})
    captured = {}
    def mocked(material, case, **kwargs):
        captured.update(material=material, case=case, binding=kwargs["source_identity"])
        return _fake_result(material, case, **kwargs)
    monkeypatch.setattr(legacy, "run_spatial", mocked)
    manager = parallel.ConcurrentMofScreeningManager(tmp_path / "output", tmp_path / "LOCK.json")
    # A minimal sealed job is prepared without starting another process or ODE.
    job = "a" * 32
    root = manager.job_root / job
    root.mkdir()
    preview = manager.preview(load_builtin_material("hypothetical_mmen_like"), load_defaults()["case"])
    (root / "PREVIEW.json").write_text(json.dumps(preview))
    (root / "WORKER_INPUT.json").write_text(json.dumps({"source_identity": manager.startup_source_identity}))
    process_worker.execute_job(root, os.getpid())
    result = manager.inspect(job)
    assert result["state"] == "completed"
    assert captured["material"] == preview["material"] and captured["case"] == preview["case"]
    assert captured["binding"] == manager.startup_source_identity
    manager.close()


def test_input_write_failure_releases_lock_without_starting_worker(manager, monkeypatch):
    real_atomic = parallel._atomic_json
    def fail_case(path, payload):
        if path.name == "input_case.json":
            raise OSError("simulated disk write failure")
        return real_atomic(path, payload)
    monkeypatch.setattr(parallel, "_atomic_json", fail_case)
    with pytest.raises(MofScreeningError, match="simulated disk write failure"):
        start(manager)
    assert not manager.execution_lock_path.exists()
    assert not manager.thread.is_alive()


def test_malformed_history_does_not_block_startup_or_get_rewritten(tmp_path):
    root = tmp_path / "output/jobs" / ("b" * 32)
    root.mkdir(parents=True)
    bad = b'{"incomplete":'
    (root / "STATE.json").write_bytes(bad)
    instance = parallel.ConcurrentMofScreeningManager(tmp_path / "output", tmp_path / "LOCK.json")
    assert instance.history()[0]["state"] == "invalid_record"
    assert (root / "STATE.json").read_bytes() == bad
    instance.close()
