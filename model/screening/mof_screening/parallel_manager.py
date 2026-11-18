"""Process-isolated jobs; concurrency is an operational setting, not physics."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .contract import MofScreeningError, effective_runtime_policy, read_json_strict
from .manager import MofScreeningManager as BaseManager, _atomic_json, _utc_now
from .preflight import execution_source_identity, require_ready

ACTIVE = {"queued", "running", "cancelling"}


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def launch_worker(job_root: Path, owner_pid: int):
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    model_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(model_root) + os.pathsep + environment.get("PYTHONPATH", "")
    with (job_root / "WORKER.log").open("ab") as log:
        return subprocess.Popen(
            [sys.executable, "-u", "-m", "screening.mof_screening.process_worker",
             "--job-root", str(job_root), "--owner-pid", str(owner_pid)],
            cwd=model_root, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


class WorkerGroup:
    """reference cross-study callers ask whether manager.thread is alive."""
    def __init__(self, manager):
        self.manager = manager

    def is_alive(self):
        with self.manager.lock:
            self.manager._reap()
            return bool(self.manager._jobs)

    def join(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.is_alive() and (deadline is None or time.monotonic() < deadline):
            time.sleep(0.05)


class ConcurrentMofScreeningManager(BaseManager):
    def __init__(self, output_root=None, execution_lock_path=None, *,
                 max_concurrent_jobs=None, launcher=None):
        self._recovered = {}
        super().__init__(output_root, execution_lock_path)
        self.lock = threading.RLock()
        self._jobs = dict(self._recovered)
        self._launcher = launcher or launch_worker
        self._closing = False
        self._settings_path = self.output_root / "CONCURRENCY.json"
        stored = read_json_strict(self._settings_path) if self._settings_path.exists() else {}
        self._max_jobs = self._validate_limit(
            stored.get("max_concurrent_jobs", 0) if max_concurrent_jobs is None else max_concurrent_jobs
        )
        self.thread = WorkerGroup(self)
        self._stop_reaper = threading.Event()
        if self._jobs:
            self._acquire_execution_lock(next(iter(self._jobs)))
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="mof-worker-watch")
        self._reaper.start()

    @staticmethod
    def _validate_limit(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MofScreeningError("maximum simultaneous runs must be an integer; 0 means unlimited", field="max_concurrent_jobs")
        return value

    def _mark_restart_interrupted(self):
        for path in self.job_root.glob("*/STATE.json"):
            try:
                state = read_json_strict(path)
            except (OSError, ValueError, MofScreeningError):
                continue  # Preserve malformed history; history() reports that individual row.
            if state.get("state") not in ACTIVE:
                continue
            worker_path = path.parent / "WORKER.json"
            try:
                worker = read_json_strict(worker_path) if worker_path.exists() else {}
            except (OSError, ValueError, MofScreeningError):
                continue  # An uncertain worker record must not trigger a destructive rewrite.
            if worker.get("job_id") == path.parent.name and pid_alive(worker.get("pid")):
                self._recovered[path.parent.name] = {"process": None, "pid": worker["pid"]}
                continue
            self._write_state(path.parent.name, state="interrupted_on_restart",
                              message="The server stopped before this job completed; partial evidence retained.")

    def _acquire_execution_lock(self, job_id):
        if self.execution_lock_path.exists():
            current = read_json_strict(self.execution_lock_path)
            if current.get("owner_token") == self.owner_token:
                return
        super()._acquire_execution_lock(job_id)

    def _reap_loop(self):
        while not self._stop_reaper.wait(0.25):
            with self.lock:
                self._reap()

    def _reap(self):
        for job_id, worker in list(self._jobs.items()):
            process = worker["process"]
            exit_code = process.poll() if process is not None else (None if pid_alive(worker["pid"]) else -1)
            if exit_code is None:
                continue
            state = self._state(job_id)
            if state.get("state") in ACTIVE:
                elapsed = float(state.get("elapsed_wall_s") or 0)
                reason = f"Simulation worker exited unexpectedly (exit {exit_code}); incomplete evidence retained."
                partial = self._record_partial_evidence(job_id, status="failed", reason=reason, elapsed_wall_s=elapsed)
                self._write_state(job_id, state="failed", message=reason,
                                  error={"message": reason, "status": 500, "field": None},
                                  partial_diagnostic="PARTIAL.json", complete=False)
            del self._jobs[job_id]
        if not self._jobs:
            self.active_event.clear()
            super()._release_execution_lock()

    def concurrency(self):
        with self.lock:
            self._reap()
            return {"max_concurrent_jobs": self._max_jobs, "unlimited": self._max_jobs == 0,
                    "active_count": len(self._jobs), "worker_type": "separate_process",
                    "math_threads_per_worker": 1}

    def set_concurrency(self, limit):
        limit = self._validate_limit(limit)
        with self.lock:
            _atomic_json(self._settings_path, {"max_concurrent_jobs": limit})
            self._max_jobs = limit
            return self.concurrency()

    def start(self, preview_id):
        with self.lock:
            self._reap()
            if self._closing:
                raise MofScreeningError("server is shutting down", status=409)
            require_ready()
            if execution_source_identity() != self.startup_source_identity:
                raise MofScreeningError("runtime source changed after startup; restart before starting another job", status=409)
            if self._max_jobs and len(self._jobs) >= self._max_jobs:
                raise MofScreeningError("maximum simultaneous runs reached; raise the limit or wait", status=409)
            preview = self._load_preview(preview_id)
            consumed = self.preview_root / f"{preview_id}.CONSUMED.json"
            if consumed.exists():
                raise MofScreeningError("preview has already been consumed", status=409)
            job_id = uuid.uuid4().hex
            self._acquire_execution_lock(job_id)
            root = self.job_root / job_id
            try:
                root.mkdir(exist_ok=False)
                _atomic_json(root / "PREVIEW.json", preview)
                _atomic_json(root / "input_material.json", preview["material"])
                _atomic_json(root / "input_case.json", preview["case"])
                _atomic_json(root / "WORKER_INPUT.json", {"source_identity": self.startup_source_identity})
                _atomic_json(root / "STATE.json", {
                    "schema": "mof-screening-job-state/1", "job_id": job_id,
                    "preview_id": preview_id, "state": "queued", "progress": 0.0,
                    "message": "Starting independent simulation process", "created_utc": _utc_now(),
                    "updated_utc": _utc_now(), "cancel_requested": False,
                    "runtime_policy": effective_runtime_policy(preview["case"]),
                    "temperature_policy": preview.get("temperature_policy"),
                    "material_id": preview["material_id"], "material_name": preview["material"]["name"],
                    "elapsed_wall_s": 0.0, "simulated_s": None,
                })
                _atomic_json(consumed, {"preview_id": preview_id, "job_id": job_id, "consumed_utc": _utc_now()})
                process = self._launcher(root, os.getpid())
                self._jobs[job_id] = {"process": process, "pid": process.pid}
                _atomic_json(root / "WORKER.json", {"job_id": job_id, "pid": process.pid, "owner_pid": os.getpid()})
            except Exception as error:
                if job_id in self._jobs:
                    self._jobs[job_id]["process"].terminate()
                self._write_state(job_id, state="failed", message=f"Worker could not start: {error}")
                self._reap()
                raise MofScreeningError(f"simulation process could not start: {error}", status=500) from error
            self.current_job_id = job_id
            self.active_event.set()
            return {"job_id": job_id, "state": "queued"}

    def _state(self, job_id, *, include_result=False):
        state = super()._state(job_id, include_result=include_result)
        if state.get("state") in ACTIVE and (self.job_root / job_id / "CANCEL_REQUEST.json").exists():
            state = {**state, "state": "cancelling", "cancel_requested": True}
        return state

    def status(self):
        with self.lock:
            self._reap()
            active = [self._state(job_id) for job_id in self._jobs]
            current = active[-1] if active else (self._state(self.current_job_id) if self.current_job_id else None)
            return {"state": (current or {}).get("state"), "active": bool(active),
                    "current": current, "active_jobs": active, "concurrency": self.concurrency()}

    def cancel(self, job_id):
        with self.lock:
            self._reap()
            state = self._state(job_id)
            if job_id not in self._jobs:
                return {"job_id": job_id, "state": state.get("state"), "cancel_requested": False}
            _atomic_json(self.job_root / job_id / "CANCEL_REQUEST.json", {"job_id": job_id, "requested_utc": _utc_now()})
            return {"job_id": job_id, "state": "cancelling", "cancel_requested": True}

    def close(self):
        with self.lock:
            self._closing = True
            for job_id in list(self._jobs):
                self.cancel(job_id)
        self.thread.join(timeout=30)
        if not self.thread.is_alive():
            self._stop_reaper.set()
            self._reaper.join(timeout=1)
