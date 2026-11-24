"""Durable preview, asynchronous run, history, comparison, and export manager."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback
import uuid

from .contract import (
    MofScreeningError,
    PREVIEW_SCHEMA,
    RESULT_SCHEMA,
    ROOT,
    build_case,
    canonical_sha256,
    default_runtime_policy,
    default_temperature_policy,
    effective_runtime_policy,
    preview_payload,
    read_json_strict,
)
from .evaluation import evaluate_existing_result
from .runtime import run_lumped, run_spatial
from .preflight import execution_source_identity, require_ready
from . import windows_io


ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
MAX_COMPARISON_JOBS = 32
RUNTIME_PROGRESS_WRITE_S = 0.75
FIRST_LONG_RUNTIME_NOTICE_S = 300.0
LONG_RUNTIME_NOTICE_INTERVAL_S = 900.0
WINDOWS_REPLACE_RETRY_DELAYS_S = windows_io.WINDOWS_IO_RETRY_DELAYS_S
WINDOWS_TRANSIENT_REPLACE_WINERRORS = frozenset({5, 32, 33})


def _is_transient_windows_replace_error(error: OSError) -> bool:
    return windows_io.is_transient_windows_sharing_error(error)


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    windows_io.replace_with_retry(
        temporary,
        destination,
        retry_delays=WINDOWS_REPLACE_RETRY_DELAYS_S,
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _append_json_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MofScreeningManager:
    def __init__(
        self,
        output_root: Path | None = None,
        execution_lock_path: Path | None = None,
    ):
        self.output_root = Path(
            output_root or ROOT / "outputs" / "mof_screening"
        ).resolve()
        self.preview_root = self.output_root / "previews"
        self.job_root = self.output_root / "jobs"
        self.preview_root.mkdir(parents=True, exist_ok=True)
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.active_event = threading.Event()
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.current_job_id: str | None = None
        self.owner_token = uuid.uuid4().hex
        self.startup_source_identity = execution_source_identity()
        self.execution_lock_path = Path(
            execution_lock_path or ROOT / "outputs" / "STUDY_EXECUTION_LOCK.json"
        )
        self._mark_restart_interrupted()

    def _acquire_execution_lock(self, job_id: str) -> None:
        path = self.execution_lock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                current = read_json_strict(path)
                pid = current.get("pid")
                if isinstance(pid, int) and pid > 0:
                    from .parallel_manager import pid_alive

                    if pid_alive(pid):
                        raise MofScreeningError(
                            "another study or screening job owns the shared execution lock",
                            status=409,
                        )
            except MofScreeningError:
                raise
            except Exception as error:
                raise MofScreeningError(
                    "shared execution lock is uncertain; inspect it before starting",
                    status=409,
                ) from error
            path.unlink()
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "owner_token": self.owner_token,
                        "pid": os.getpid(),
                        "study_id": job_id,
                        "owner": "mof_screening",
                        "created_utc": _utc_now(),
                    },
                    stream,
                )
        except FileExistsError as error:
            raise MofScreeningError(
                "another study or screening job owns the shared execution lock",
                status=409,
            ) from error

    def _release_execution_lock(self) -> None:
        try:
            current = read_json_strict(self.execution_lock_path)
            if current.get("owner_token") == self.owner_token:
                self.execution_lock_path.unlink()
        except Exception:
            pass

    def _mark_restart_interrupted(self) -> None:
        for state_path in self.job_root.glob("*/STATE.json"):
            try:
                state = read_json_strict(state_path)
            except Exception:
                continue
            if state.get("state") in {"queued", "running", "cancelling"}:
                state.update(
                    {
                        "state": "interrupted_on_restart",
                        "message": (
                            "The local server stopped before this job reached "
                            "a terminal state."
                        ),
                        "updated_utc": _utc_now(),
                    }
                )
                _atomic_json(state_path, state)

    def preview(
        self,
        material_document,
        case_document,
        *,
        source: str = "built_in_or_imported",
    ) -> dict:
        if isinstance(case_document, dict) and "runtime_policy" not in case_document:
            case_document = {
                **case_document,
                "runtime_policy": default_runtime_policy(),
            }
        if (
            isinstance(case_document, dict)
            and "temperature_policy" not in case_document
        ):
            case_document = {
                **case_document,
                "temperature_policy": default_temperature_policy(),
            }
        material, case = build_case(material_document, case_document)
        payload = preview_payload(material, case, source=source)
        preview_id = uuid.uuid4().hex
        payload["preview_id"] = preview_id
        payload["created_utc"] = _utc_now()
        _atomic_json(self.preview_root / f"{preview_id}.json", payload)
        return payload

    def _load_preview(self, preview_id: str) -> dict:
        if not isinstance(preview_id, str) or not ID_PATTERN.fullmatch(preview_id):
            raise MofScreeningError("preview id is invalid", field="preview_id")
        path = self.preview_root / f"{preview_id}.json"
        if not path.is_file():
            raise MofScreeningError("preview does not exist", status=404)
        preview = read_json_strict(path)
        if (
            preview.get("schema") != PREVIEW_SCHEMA
            or preview.get("preview_id") != preview_id
        ):
            raise MofScreeningError("preview identity is invalid", status=409)
        normalized_material, normalized_case = build_case(
            preview.get("material"), preview.get("case")
        )
        if canonical_sha256(normalized_material) != preview.get("material_sha256"):
            raise MofScreeningError("preview material changed", status=409)
        if canonical_sha256(normalized_case) != preview.get("case_sha256"):
            raise MofScreeningError("preview case changed", status=409)
        preview["material"] = normalized_material
        preview["case"] = normalized_case
        return preview

    def start(self, preview_id: str) -> dict:
        with self.lock:
            require_ready()
            current_source_identity = execution_source_identity()
            if current_source_identity.get(
                "release_sha256"
            ) != self.startup_source_identity.get("release_sha256"):
                raise MofScreeningError(
                    "runtime source files changed after server startup; "
                    "restart the server before starting another job",
                    status=409,
                )
            if self.thread is not None and self.thread.is_alive():
                raise MofScreeningError("another screening job is running", status=409)
            preview = self._load_preview(preview_id)
            consumed = self.preview_root / f"{preview_id}.CONSUMED.json"
            if consumed.exists():
                raise MofScreeningError("preview has already been consumed", status=409)
            job_id = uuid.uuid4().hex
            self._acquire_execution_lock(job_id)
            job_root = self.job_root / job_id
            job_root.mkdir(parents=True, exist_ok=False)
            _atomic_json(job_root / "PREVIEW.json", preview)
            _atomic_json(job_root / "input_material.json", preview["material"])
            _atomic_json(job_root / "input_case.json", preview["case"])
            state = {
                "schema": "mof-screening-job-state/1",
                "job_id": job_id,
                "preview_id": preview_id,
                "state": "queued",
                "progress": 0.0,
                "message": "Queued",
                "created_utc": _utc_now(),
                "updated_utc": _utc_now(),
                "cancel_requested": False,
                "runtime_policy": effective_runtime_policy(preview["case"]),
                "temperature_policy": preview.get("temperature_policy"),
                "started_utc": None,
                "elapsed_wall_s": 0.0,
                "simulated_s": None,
                "current_cycle": None,
                "current_phase": None,
                "phase_elapsed_s": None,
                "phase_duration_s": None,
                "last_checkpoint": None,
            }
            _atomic_json(job_root / "STATE.json", state)
            _atomic_json(
                consumed,
                {
                    "preview_id": preview_id,
                    "job_id": job_id,
                    "consumed_utc": _utc_now(),
                },
            )
            self.current_job_id = job_id
            self.cancel_event = threading.Event()
            self.active_event.set()
            self.thread = threading.Thread(
                target=self._run_job,
                args=(job_id, preview),
                daemon=True,
                name=f"mof-screening-{job_id[:8]}",
            )
            self.thread.start()
            return {"job_id": job_id, "state": "queued"}

    def _write_state(self, job_id: str, **changes) -> None:
        with self.state_lock:
            path = self.job_root / job_id / "STATE.json"
            try:
                state = read_json_strict(path)
            except Exception:
                state = {
                    "schema": "mof-screening-job-state/1",
                    "job_id": job_id,
                }
            state.update(changes)
            state["updated_utc"] = _utc_now()
            _atomic_json(path, state)

    def _run_job(self, job_id: str, preview: dict) -> None:
        last_write = 0.0
        next_runtime_notice_s = FIRST_LONG_RUNTIME_NOTICE_S
        last_notice = None
        last_checkpoint_record = None
        started_monotonic = time.monotonic()
        runtime_policy = effective_runtime_policy(preview["case"])
        self._write_state(
            job_id,
            state="running",
            message="Initializing",
            progress=0.01,
            started_utc=_utc_now(),
            elapsed_wall_s=0.0,
            runtime_policy=runtime_policy,
        )

        def progress(fraction: float, message: str, details=None) -> None:
            nonlocal last_checkpoint_record, last_notice
            nonlocal last_write, next_runtime_notice_s
            now = time.monotonic()
            details = dict(details or {})
            force = bool(
                fraction >= 1.0
                or details.get("event") in {"phase_completed", "cycle_completed"}
            )
            if not force and now - last_write < RUNTIME_PROGRESS_WRITE_S:
                return
            elapsed_wall_s = now - started_monotonic
            notice = None
            if elapsed_wall_s >= next_runtime_notice_s:
                notice = {
                    "soft_notice": True,
                    "message": (
                        "Long calculation continues; no automatic wall-time "
                        "termination applies in manual-stop mode."
                        if runtime_policy["mode"] == "manual_stop"
                        else "Long calculation continues toward the selected "
                        "execution time limit."
                    ),
                    "elapsed_wall_s": elapsed_wall_s,
                }
                while next_runtime_notice_s <= elapsed_wall_s:
                    next_runtime_notice_s += LONG_RUNTIME_NOTICE_INTERVAL_S
                last_notice = notice
            checkpoint_state = details.pop("checkpoint_state", None)
            last_checkpoint = last_checkpoint_record
            if checkpoint_state is not None:
                checkpoint_path = self.job_root / job_id / "SPATIAL_CHECKPOINT.json"
                checkpoint = {
                    "schema": "mof-screening-spatial-progress-checkpoint/1",
                    "job_id": job_id,
                    "phase": details.get("phase"),
                    "cycle": details.get("cycle"),
                    "event": details.get("event"),
                    "simulated_s": details.get("simulated_s", details.get("elapsed_s")),
                    "state": checkpoint_state,
                    "state_sha256": canonical_sha256(checkpoint_state),
                    "resumable": False,
                    "resume_note": (
                        "Diagnostic phase-boundary state only; no safe resume "
                        "protocol is implemented."
                    ),
                    "recorded_utc": _utc_now(),
                }
                _atomic_json(checkpoint_path, checkpoint)
                last_checkpoint = {
                    "kind": "spatial_phase_or_cycle",
                    "path": str(checkpoint_path),
                    "state_sha256": checkpoint["state_sha256"],
                    "event": checkpoint["event"],
                    "simulated_s": checkpoint["simulated_s"],
                    "resumable": False,
                }
                last_checkpoint_record = last_checkpoint
            progress_record = {
                "schema": "mof-screening-job-progress/1",
                "job_id": job_id,
                "utc": _utc_now(),
                "fraction": max(0.0, min(float(fraction), 0.99)),
                "message": message,
                "elapsed_wall_s": elapsed_wall_s,
                "engine": details.get("engine", preview["case"]["engine"]),
                "simulated_s": details.get("simulated_s", details.get("elapsed_s")),
                "cycle": details.get("cycle"),
                "phase": details.get("phase"),
                "phase_elapsed_s": details.get("phase_elapsed_s"),
                "phase_duration_s": details.get("phase_duration_s"),
                "event": details.get("event"),
                "runtime_policy": runtime_policy,
                "last_checkpoint": last_checkpoint,
                "runtime_notice": last_notice,
            }
            if progress_record["event"] in {
                "phase_completed",
                "cycle_completed",
            }:
                _append_json_line(
                    self.job_root / job_id / "PROGRESS_SUMMARIES.jsonl",
                    {
                        "schema": "mof-screening-accepted-progress-summary/1",
                        "job_id": job_id,
                        "recorded_utc": progress_record["utc"],
                        "engine": progress_record["engine"],
                        "event": progress_record["event"],
                        "cycle": progress_record["cycle"],
                        "phase": progress_record["phase"],
                        "simulated_s": progress_record["simulated_s"],
                        "phase_elapsed_s": progress_record["phase_elapsed_s"],
                        "phase_duration_s": progress_record["phase_duration_s"],
                        "state_sha256": (
                            checkpoint["state_sha256"]
                            if checkpoint_state is not None
                            else details.get("state_sha256")
                        ),
                    },
                )
            _atomic_json(
                self.job_root / job_id / "PROGRESS.json",
                progress_record,
            )
            notice_text = f" {last_notice['message']}" if last_notice else ""
            self._write_state(
                job_id,
                state="running",
                message=f"{message}.{notice_text}".strip(),
                progress=progress_record["fraction"],
                cancel_requested=self.cancel_event.is_set(),
                elapsed_wall_s=elapsed_wall_s,
                simulated_s=progress_record["simulated_s"],
                current_cycle=progress_record["cycle"],
                current_phase=progress_record["phase"],
                phase_elapsed_s=progress_record["phase_elapsed_s"],
                phase_duration_s=progress_record["phase_duration_s"],
                last_checkpoint=last_checkpoint,
                runtime_notice=last_notice,
            )
            last_write = now

        def cancel_check() -> None:
            if self.cancel_event.is_set():
                raise MofScreeningError("job cancelled by user", status=409)

        try:
            if preview["case"]["engine"] == "spatial":
                result = run_spatial(
                    preview["material"],
                    preview["case"],
                    cancel_check=cancel_check,
                    progress=progress,
                    source_identity=self.startup_source_identity,
                )
            else:
                result = run_lumped(
                    preview["material"],
                    preview["case"],
                    cancel_check=cancel_check,
                    progress=progress,
                    checkpoint_path=self.job_root / job_id / "lumped_checkpoint.json",
                    source_identity=self.startup_source_identity,
                )
            if result.get("schema") != RESULT_SCHEMA:
                raise MofScreeningError("engine returned an invalid result schema")
            result["job_id"] = job_id
            result["preview_id"] = preview["preview_id"]
            result["preview_material_sha256"] = preview["material_sha256"]
            result["preview_case_sha256"] = preview["case_sha256"]
            result["elapsed_utc"] = _utc_now()
            _atomic_json(self.job_root / job_id / "RESULT.json", result)
            self._release_execution_lock()
            self._write_state(
                job_id,
                state="completed",
                progress=1.0,
                message="Screening completed",
                cancel_requested=self.cancel_event.is_set(),
                elapsed_wall_s=float(time.monotonic() - started_monotonic),
            )
        except MofScreeningError as error:
            interrupted = self.cancel_event.is_set()
            terminal_state = (
                "cancelled"
                if interrupted
                else ("timed_out" if error.status == 504 else "failed")
            )
            partial = self._record_partial_evidence(
                job_id,
                status=terminal_state,
                reason=str(error),
                elapsed_wall_s=float(time.monotonic() - started_monotonic),
            )
            self._release_execution_lock()
            diagnostics_path = None
            if isinstance(error.diagnostics, dict):
                diagnostic_file = (
                    self.job_root / job_id / "MATERIAL_DOMAIN_FAILURE.json"
                )
                _atomic_json(diagnostic_file, error.diagnostics)
                diagnostics_path = diagnostic_file.name
            _atomic_json(
                self.job_root / job_id / "FAILURE.json",
                {
                    "schema": "mof-screening-failure/1",
                    "job_id": job_id,
                    "state": terminal_state,
                    "message": str(error),
                    "field": error.field,
                    "material_domain_diagnostic": diagnostics_path,
                    "progress": partial.get("last_progress", {}).get("fraction", 0.0),
                    "partial_evidence": partial,
                    "recorded_utc": _utc_now(),
                },
            )
            self._write_state(
                job_id,
                state=terminal_state,
                progress=partial.get("last_progress", {}).get("fraction", 0.0),
                message=str(error),
                error={
                    "message": str(error),
                    "field": error.field,
                    "status": error.status,
                },
                cancel_requested=interrupted,
                elapsed_wall_s=partial.get("elapsed_wall_s"),
                partial_diagnostic="PARTIAL.json",
                material_domain_diagnostic=diagnostics_path,
            )
        except Exception as error:  # noqa: BLE001 - failure must remain durable
            partial = self._record_partial_evidence(
                job_id,
                status="failed",
                reason=f"Internal screening failure: {error}",
                elapsed_wall_s=float(time.monotonic() - started_monotonic),
            )
            self._release_execution_lock()
            _atomic_json(
                self.job_root / job_id / "FAILURE.json",
                {
                    "schema": "mof-screening-failure/1",
                    "job_id": job_id,
                    "state": "failed",
                    "message": str(error),
                    "traceback": traceback.format_exc(limit=8),
                    "partial_evidence": partial,
                    "recorded_utc": _utc_now(),
                },
            )
            self._write_state(
                job_id,
                state="failed",
                progress=partial.get("last_progress", {}).get("fraction", 0.0),
                message=f"Internal screening failure: {error}",
                error={"message": str(error), "field": None, "status": 500},
                elapsed_wall_s=partial.get("elapsed_wall_s"),
                partial_diagnostic="PARTIAL.json",
            )
        finally:
            self._release_execution_lock()
            self.active_event.clear()

    def _record_partial_evidence(
        self,
        job_id: str,
        *,
        status: str,
        reason: str,
        elapsed_wall_s: float,
    ) -> dict:
        root = self.job_root / job_id
        try:
            last_progress = read_json_strict(root / "PROGRESS.json")
        except Exception:
            last_progress = {}
        checkpoint_summary = None
        spatial_path = root / "SPATIAL_CHECKPOINT.json"
        lumped_path = root / "lumped_checkpoint.json"
        try:
            if spatial_path.is_file():
                checkpoint = read_json_strict(spatial_path)
                checkpoint_summary = {
                    "kind": "spatial_phase_or_cycle",
                    "path": str(spatial_path),
                    "phase": checkpoint.get("phase"),
                    "cycle": checkpoint.get("cycle"),
                    "event": checkpoint.get("event"),
                    "simulated_s": checkpoint.get("simulated_s"),
                    "state_sha256": checkpoint.get("state_sha256"),
                    "resumable": False,
                }
            elif lumped_path.is_file():
                checkpoint = read_json_strict(lumped_path)
                state = checkpoint.get("state")
                checkpoint_summary = {
                    "kind": "lumped_active_checkpoint",
                    "path": str(lumped_path),
                    "status": checkpoint.get("checkpoint_status"),
                    "sequence": checkpoint.get("sequence"),
                    "cursor": checkpoint.get("cursor"),
                    "state_sha256": (
                        None if state is None else canonical_sha256(state)
                    ),
                    "resumable": False,
                }
        except Exception as error:
            checkpoint_summary = {
                "kind": "checkpoint_unreadable",
                "path": str(spatial_path if spatial_path.is_file() else lumped_path),
                "error": str(error),
                "resumable": False,
            }
        partial = {
            "schema": "mof-screening-partial-evidence/1",
            "job_id": job_id,
            "status": status,
            "complete": False,
            "reason": reason,
            "elapsed_wall_s": elapsed_wall_s,
            "last_progress": last_progress,
            "last_checkpoint": (
                checkpoint_summary or last_progress.get("last_checkpoint")
            ),
            "accepted_progress_summaries": {
                "path": str(root / "PROGRESS_SUMMARIES.jsonl"),
                "append_only": True,
                "count": (
                    sum(
                        1
                        for _line in (root / "PROGRESS_SUMMARIES.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if _line.strip()
                    )
                    if (root / "PROGRESS_SUMMARIES.jsonl").is_file()
                    else 0
                ),
            },
            "resume_implemented": False,
            "resume_note": (
                "Partial evidence is retained for diagnosis. It is not a "
                "completed result and no safe resume is claimed."
            ),
            "recorded_utc": _utc_now(),
        }
        _atomic_json(root / "PARTIAL.json", partial)
        return partial

    def status(self) -> dict:
        current = None
        if self.current_job_id:
            current = self._state(self.current_job_id)
        return {
            "state": None if current is None else current.get("state"),
            "active": bool(self.thread and self.thread.is_alive()),
            "current": current,
            "history": self.history(),
        }

    def _state(self, job_id: str, *, include_result: bool = False) -> dict:
        if not ID_PATTERN.fullmatch(job_id):
            raise MofScreeningError("job id is invalid", field="job_id")
        root = self.job_root / job_id
        state_path = root / "STATE.json"
        if not state_path.is_file():
            raise MofScreeningError("job does not exist", status=404)
        with self.state_lock:
            state = read_json_strict(state_path)
        if include_result and (root / "RESULT.json").is_file():
            state["result"] = read_json_strict(root / "RESULT.json")
        if (root / "FAILURE.json").is_file():
            state["failure"] = read_json_strict(root / "FAILURE.json")
        if (root / "PARTIAL.json").is_file():
            state["partial"] = read_json_strict(root / "PARTIAL.json")
        return state

    def inspect(self, job_id: str) -> dict:
        state = self._state(job_id, include_result=True)
        root = self.job_root / job_id
        if (root / "RESULT.json").is_file():
            document = read_json_strict(root / "RESULT.json")
            if document.get("job_id") != job_id:
                raise MofScreeningError("saved result identity differs", status=409)
            # Interpret reference records on read without rewriting their bytes.
            document["derived_evaluation"] = evaluate_existing_result(document)
            state["result"] = document
        return state

    def cancel(self, job_id: str) -> dict:
        if job_id != self.current_job_id:
            state = self._state(job_id)
            return {
                "job_id": job_id,
                "state": state.get("state"),
                "cancel_requested": False,
            }
        self.cancel_event.set()
        self._write_state(
            job_id,
            state="cancelling",
            message="Cancellation requested",
            cancel_requested=True,
        )
        return {
            "job_id": job_id,
            "state": "cancelling",
            "cancel_requested": True,
        }

    def history(self) -> list[dict]:
        rows = []
        for state_path in sorted(
            self.job_root.glob("*/STATE.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            try:
                with self.state_lock:
                    state = read_json_strict(state_path)
                root = state_path.parent
                result = (
                    read_json_strict(root / "RESULT.json")
                    if (root / "RESULT.json").is_file()
                    else None
                )
                rows.append(
                    {
                        "job_id": state.get("job_id"),
                        "state": state.get("state"),
                        "progress": state.get("progress"),
                        "message": state.get("message"),
                        "updated_utc": state.get("updated_utc"),
                        "elapsed_wall_s": state.get("elapsed_wall_s"),
                        "simulated_s": state.get("simulated_s"),
                        "current_cycle": state.get("current_cycle"),
                        "current_phase": state.get("current_phase"),
                        "runtime_policy": state.get("runtime_policy"),
                        "temperature_policy": state.get("temperature_policy"),
                        "partial": state.get("partial_diagnostic") is not None,
                        "case_label": (
                            None if result is None else result.get("case_label")
                        ),
                        "engine": (None if result is None else result.get("engine")),
                        "material": (
                            None
                            if result is None
                            else result.get("material", {}).get("name")
                        ),
                        "metrics": None if result is None else result.get("metrics"),
                        "input_identity": (
                            None if result is None else result.get("input_identity")
                        ),
                        "synthetic": (
                            False
                            if result is None
                            else bool(result.get("material", {}).get("synthetic"))
                        ),
                        "target_attainment_status": (
                            None
                            if result is None
                            else evaluate_existing_result(result)["target_attainment"][
                                "status"
                            ]
                        ),
                        "hard_limits_status": (
                            None
                            if result is None
                            else evaluate_existing_result(result)["hard_limits"][
                                "status"
                            ]
                        ),
                        "conservation_status": (
                            None
                            if result is None
                            else evaluate_existing_result(result)["conservation"][
                                "status"
                            ]
                        ),
                        "valid": bool(
                            result is not None and result.get("schema") == RESULT_SCHEMA
                        ),
                    }
                )
            except Exception as error:  # one corrupt record must not hide others
                rows.append(
                    {
                        "job_id": state_path.parent.name,
                        "state": "invalid_record",
                        "message": str(error),
                        "valid": False,
                    }
                )
        return rows

    def compare(self, job_ids: list[str]) -> dict:
        if (
            not isinstance(job_ids, list)
            or not 2 <= len(job_ids) <= MAX_COMPARISON_JOBS
        ):
            raise MofScreeningError(
                f"comparison requires 2-{MAX_COMPARISON_JOBS} job ids",
                field="job_ids",
            )
        if len(job_ids) != len(set(job_ids)):
            raise MofScreeningError(
                "comparison job ids must be unique", field="job_ids"
            )
        results = []
        for job_id in job_ids:
            state = self.inspect(job_id)
            if state.get("state") != "completed" or not state.get("result"):
                raise MofScreeningError(
                    f"job {job_id} is not a completed result", field="job_ids"
                )
            results.append(state["result"])

        def field_hashes(key: str, getter=None) -> list[str | None]:
            getter = getter or (lambda result: result.get(key))
            values = [getter(result) for result in results]
            return [
                None if value is None else canonical_sha256(value) for value in values
            ]

        field_hashes_by_name = {
            "weather": field_hashes("weather"),
            "geometry": field_hashes("geometry"),
            "state_layout": field_hashes("state_layout"),
            "time_basis": field_hashes(
                "time_basis",
                lambda result: {
                    "requested_s": (result.get("horizon") or {}).get("requested_s"),
                    "simulated_s": (result.get("horizon") or {}).get("simulated_s"),
                    "weather_mode": (
                        (result.get("case") or {}).get("weather") or {}
                    ).get("mode"),
                    "weather_hours": (
                        (result.get("case") or {}).get("weather") or {}
                    ).get("hours"),
                    "weather_start_utc": (
                        (result.get("case") or {}).get("weather") or {}
                    ).get("start_utc"),
                },
            ),
            "assumptions": field_hashes("assumptions"),
            "mass_basis": field_hashes(
                "mass_basis",
                lambda result: (result.get("case") or {})
                .get("equipment", {})
                .get("dry_shaped_mass_kg"),
            ),
            "equipment": field_hashes(
                "equipment",
                lambda result: (result.get("case") or {}).get("equipment"),
            ),
            "cycle": field_hashes(
                "cycle",
                lambda result: (result.get("case") or {}).get("cycle"),
            ),
            "operating_policy": field_hashes(
                "operating_policy",
                lambda result: (result.get("case") or {}).get("operating_policy"),
            ),
            "initial_state": field_hashes(
                "initial_state",
                lambda result: (result.get("case") or {}).get("initial_state"),
            ),
            "collection": field_hashes(
                "collection",
                lambda result: (result.get("case") or {}).get("collection"),
            ),
            "engine": field_hashes(
                "engine",
                lambda result: (result.get("case") or {}).get("engine"),
            ),
            "solver": field_hashes(
                "solver",
                lambda result: (result.get("case") or {}).get("solver"),
            ),
            "resolution": field_hashes(
                "resolution",
                lambda result: {
                    "engine": (result.get("case") or {}).get("engine"),
                    "cells": ((result.get("case") or {}).get("solver") or {}).get(
                        "cells"
                    ),
                    "representation": (result.get("geometry") or {}).get(
                        "representation"
                    ),
                },
            ),
        }
        basis_fields = []
        for key in (
            "weather",
            "geometry",
            "state_layout",
            "time_basis",
            "assumptions",
            "mass_basis",
            "equipment",
            "cycle",
            "operating_policy",
            "initial_state",
            "collection",
            "engine",
            "solver",
            "resolution",
        ):
            values = field_hashes_by_name[key]
            basis_fields.append(
                {
                    "field": key,
                    "matching": bool(all(values) and len(set(values)) == 1),
                    "hashes": values,
                }
            )
        source_identities = [result.get("source_identity") for result in results]
        source_hashes = [
            None if source_identity is None else canonical_sha256(source_identity)
            for source_identity in source_identities
        ]
        basis_fields.append(
            {
                "field": "source_identity",
                "matching": bool(all(source_hashes) and len(set(source_hashes)) == 1),
                "hashes": source_hashes,
            }
        )
        same_source_identity = bool(all(source_hashes) and len(set(source_hashes)) == 1)
        runtime_policies = [
            effective_runtime_policy(result.get("case") or {}) for result in results
        ]
        runtime_policy_hashes = [
            canonical_sha256(policy) for policy in runtime_policies
        ]
        same_runtime_policy = len(set(runtime_policy_hashes)) == 1
        configuration_hashes = []
        for result in results:
            case = result.get("case") or {}
            configuration_hashes.append(
                canonical_sha256(
                    {
                        "weather": case.get("weather"),
                        "equipment": case.get("equipment"),
                        "cycle": case.get("cycle"),
                        "initial_state": case.get("initial_state"),
                        "collection": case.get("collection"),
                        "operating_policy": case.get("operating_policy"),
                        "engine": case.get("engine"),
                    }
                )
            )
        exact_configuration_hashes = [
            canonical_sha256(
                {
                    "configuration": configuration_hash,
                    "solver": ((result.get("case") or {}).get("solver")),
                }
            )
            for result, configuration_hash in zip(results, configuration_hashes)
        ]
        rows = []
        for result, configuration_hash, runtime_policy in zip(
            results, configuration_hashes, runtime_policies
        ):
            metrics = result.get("metrics") or {}
            evaluation = evaluate_existing_result(result)
            rows.append(
                {
                    "job_id": result["job_id"],
                    "case_label": result["case_label"],
                    "material_id": result["material"]["material_id"],
                    "material_name": result["material"]["name"],
                    "material_evidence_status": (
                        (result.get("material") or {}).get("evidence", {}).get("status")
                    ),
                    "engine": result["engine"],
                    "configuration_sha256": configuration_hash,
                    "regeneration_method": result.get("case", {}).get("regeneration", {}).get("kind", "electrical"),
                    "steam_generator_input_kWh": metrics.get("steam_generator_input_kWh"),
                    "steam_generator_carrier": metrics.get("steam_generator_carrier"),
                    "steam_generator_kWh_kg_CO2": metrics.get("steam_generator_kWh_kg_CO2"),
                    "steam_kg_per_kg_CO2": metrics.get("steam_kg_per_kg_CO2"),
                    "last_complete_cycle_CO2_kg_day": metrics.get("last_complete_cycle_CO2_kg_day"),
                    "cycle_repeatability_established": (
                        (result.get("horizon", {}).get("cycle_repeatability") or {}).get("established")
                    ),
                    "collected_CO2_kg_day": metrics.get("collected_CO2_kg_day"),
                    "electricity_kWh_day": metrics.get("electricity_kWh_day"),
                    "electricity_kWh_kg_CO2": metrics.get("electricity_kWh_kg_CO2"),
                    "dry_product_CO2_fraction": metrics.get("dry_product_CO2_fraction"),
                    "wet_product_CO2_fraction": metrics.get("wet_product_CO2_fraction"),
                    "cycles_completed": result.get("horizon", {}).get(
                        "cycles_completed"
                    ),
                    "cycles_started": result.get("horizon", {}).get("cycles_started"),
                    "basis_complete": result.get("horizon", {}).get("complete"),
                    "endpoint_all_pass": result.get("endpoint", {}).get("all_pass"),
                    "numerical_status": result.get("numerical_qualification", {}).get(
                        "status"
                    ),
                    "partial_final_cycle": result.get("horizon", {}).get(
                        "partial_final_cycle"
                    ),
                    "initial_state": result.get("horizon", {}).get("initial_state"),
                    "dry_shaped_mass_kg": (
                        (result.get("case") or {})
                        .get("equipment", {})
                        .get("dry_shaped_mass_kg")
                    ),
                    "source_release_sha256": (
                        None
                        if result.get("source_identity") is None
                        else result["source_identity"].get("release_sha256")
                    ),
                    "closure_pass": bool(
                        result.get("checks", {}).get("mass_closure_pass")
                        and result.get("checks", {}).get("energy_closure_pass")
                    ),
                    "conservation_status": evaluation["conservation"]["status"],
                    "hard_limits_status": evaluation["hard_limits"]["status"],
                    "target_attainment_status": evaluation["target_attainment"][
                        "status"
                    ],
                    "target_attainment_advisory": evaluation["target_attainment"][
                        "advisory"
                    ],
                    "target_checks": evaluation["target_attainment"]["checks"],
                    "hard_limit_checks": evaluation["hard_limits"]["checks"],
                    "hard_limit_gaps": evaluation["hard_limits"]["unassessed_limits"],
                    "numerical_reliability": evaluation["numerical_reliability"],
                    "synthetic": bool(result["material"].get("synthetic")),
                    "runtime_policy": runtime_policy,
                    "temperature_guard_status": (
                        (result.get("temperature_guard_assessment") or {}).get("status")
                    ),
                    "temperature_policy": result.get("temperature_policy"),
                }
            )
        matching_by_name = {field["field"]: field["matching"] for field in basis_fields}
        required_match_fields = (
            "weather",
            "mass_basis",
            "time_basis",
            "assumptions",
            "engine",
            "initial_state",
        )
        incompatible_fields = [
            field for field in required_match_fields if not matching_by_name[field]
        ]
        if not same_source_identity:
            incompatible_fields.append("source_identity")
        controlled_change_fields = [
            field
            for field in (
                "geometry",
                "equipment",
                "cycle",
                "operating_policy",
                "collection",
                "solver",
                "resolution",
            )
            if not matching_by_name[field]
        ]
        material_hashes = [
            (
                (result.get("input_identity") or {}).get("material_sha256")
                or canonical_sha256(
                    result.get("material_package") or result["material"]
                )
            )
            for result in results
        ]
        material_changed = len(set(material_hashes)) > 1
        exact_configuration = bool(
            len(set(exact_configuration_hashes)) == 1 and same_source_identity
        )
        controlled_comparison = bool(
            not incompatible_fields and (not exact_configuration or material_changed)
        )
        comparison_mode = (
            "exact_configuration"
            if exact_configuration
            else (
                "controlled_comparison"
                if controlled_comparison
                else "incompatible_basis"
            )
        )
        warnings = [
            (
                "Executable source identities match."
                if same_source_identity
                else "Executable source identities differ or are missing; "
                "do not present this comparison as one kernel basis."
            ),
            (
                "Requested target misses are advisory and do not by "
                "themselves exclude or rank down a case."
            ),
            (
                "Conservation, known hard limits and numerical reliability "
                "are reported separately from target attainment."
            ),
            "No row earns a hardware, measured-material, annual or CSS qualification badge.",
        ]
        if incompatible_fields:
            warnings.append(
                "Incompatible comparison basis: " + ", ".join(incompatible_fields) + "."
            )
        if controlled_change_fields:
            warnings.append(
                "Deliberate operating/design inputs differ: "
                + ", ".join(controlled_change_fields)
                + ". Review these changes as the comparison variables."
            )
        warnings.append(
            "Execution runtime policies are operational metadata and are "
            "excluded from the physical comparison basis."
        )
        if not incompatible_fields:
            rows.sort(key=lambda row: (
                not row["closure_pass"] or row["hard_limits_status"] == "fail",
                row["collected_CO2_kg_day"] is None,
                -(row["collected_CO2_kg_day"] or 0.0),
            ))
        return {
            "schema": "mof-screening-comparison/1",
            "rows": rows,
            "basis_fields": basis_fields,
            "same_configuration": len(set(configuration_hashes)) == 1,
            "exact_configuration": exact_configuration,
            "comparison_mode": comparison_mode,
            "compatible_comparison": not incompatible_fields,
            "incompatible_fields": incompatible_fields,
            "controlled_change_fields": controlled_change_fields,
            "material_changed": material_changed,
            "same_resolution": matching_by_name["resolution"],
            "family_layout_changed": bool(
                material_changed and not matching_by_name["state_layout"]
            ),
            "same_engine": len({row["engine"] for row in rows}) == 1,
            "same_source_identity": same_source_identity,
            "runtime_policies": runtime_policies,
            "runtime_policy_matches": same_runtime_policy,
            "same_execution_basis": bool(
                len(set(configuration_hashes)) == 1 and same_source_identity
            ),
            "ranking_note": (
                "Collected CO2/day is shown as a screening metric. Rankings "
                "are presented for exact configurations or compatible "
                "controlled comparisons. Failed conservation or known hard-limit violations sort last. Missing requested operating targets "
                "never removes a case by itself; weather, mass basis, engine, "
                "initial state and source incompatibilities remain warnings."
            ),
            "warnings": warnings,
        }

    def export(self, job_id: str, fmt: str) -> tuple[str, str, str]:
        state = self.inspect(job_id)
        result = state.get("result")
        if state.get("state") != "completed" or not result:
            raise MofScreeningError(
                "only completed results can be exported", status=409
            )
        base = f"mof_screening_{job_id[:12]}"
        if fmt == "json":
            return (
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                base + ".json",
                "application/json; charset=utf-8",
            )
        if fmt != "csv":
            raise MofScreeningError("export format must be json or csv")
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        accepted_temperature = (
            result.get("accepted_state_temperature_observations") or {}
        )
        material_query_temperature = (
            result.get("material_query_temperature_observations") or {}
        )
        water_burden = result.get("water_burden") or {}
        metrics = result.get("metrics") or {}
        checks = result.get("checks") or {}
        constant_columns = {
            "accepted_peak_bed_temperature_K": (
                accepted_temperature.get("maximum_bed_temperature_K")
            ),
            "material_query_min_temperature_K": (
                material_query_temperature.get("minimum_finite_temperature_K")
            ),
            "material_query_max_temperature_K": (
                material_query_temperature.get("maximum_finite_temperature_K")
            ),
            "temperature_policy_mode": (
                (result.get("temperature_policy") or {}).get("mode")
            ),
            "temperature_boundary_status": (
                (result.get("temperature_guard_assessment") or {}).get("status")
            ),
            "water_loading_initial_mol_kg": water_burden.get("initial_loading_mol_kg"),
            "water_loading_final_mol_kg": water_burden.get("final_loading_mol_kg"),
            "water_condensed_mol": metrics.get("water_condensed_mol"),
            "water_vented_mol": metrics.get("water_vented_mol"),
            "water_in_product_mol": metrics.get("water_in_product_mol"),
            "mass_closure_pass": checks.get("mass_closure_pass"),
            "energy_closure_pass": checks.get("energy_closure_pass"),
            "energy_relative_residual": checks.get("energy_relative_residual"),
        }
        wet_fields = []
        if "steam_generator_input_kWh" in metrics:
            for key in ("steam_generator_input_kWh", "steam_generator_carrier", "steam_in_kg",
                        "bed_condensed_kg", "bed_evaporated_kg", "bed_drain_kg", "bed_liquid_final_kg",
                        "jacket_steam_kg", "jacket_delivered_heat_kWh"):
                if key in metrics:
                    constant_columns[key] = metrics[key]
            if "steam_in_kg" in metrics:
                wet_fields = ["outlet_water_mole_fraction", "steam_in_mol_cycle", "bed_drain_mol_cycle",
                              "axial_solid_temperature_K", "axial_pressure_Pa", "axial_water_partial_Pa",
                              "axial_adsorbed_water_mol_kg", "bed_liquid_mol_by_cell"]
        headers = [
            "time_s",
            "cycle",
            "phase",
            "pressure_Pa",
            "min_pressure_Pa",
            "bed_temperature_min_K",
            "bed_temperature_mean_K",
            "bed_temperature_max_K",
            "gas_temperature_mean_K",
            "wall_temperature_K",
            "heater_W",
            "cooling_thermal_W",
            "fan_electric_W",
            "pump_electric_W",
            "pressure_spread_Pa",
            "minimum_gas_inventory_mol",
            "minimum_storage_mol_kg",
            "co2_loading_mean_mol_kg",
            "water_loading_mean_mol_kg",
            "local_rh_min",
            "local_rh_mean",
            "local_rh_max",
            "phase_elapsed_s",
            *wet_fields,
            *constant_columns,
        ]
        writer.writerow(headers)
        for row in result.get("traces", []):
            writer.writerow(
                [
                    (
                        row.get("elapsed_s")
                        if header == "time_s"
                        else (
                            constant_columns[header]
                            if header in constant_columns
                            else row.get(header)
                        )
                    )
                    for header in headers
                ]
            )
        return buffer.getvalue(), base + ".csv", "text/csv; charset=utf-8"

    def close(self) -> None:
        self.cancel_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
