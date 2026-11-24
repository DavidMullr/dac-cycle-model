"""Routes for the general MOF screening workspace."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from screening.mof_screening.contract import (
    MofScreeningError,
    SUPPORTED_FAMILIES,
    WEATHER_PATH,
    file_sha256,
    load_builtin_material,
    load_defaults,
)
from screening.mof_screening.guided_material import build_guided_material
from screening.mof_screening.manager import MAX_COMPARISON_JOBS
from screening.mof_screening.preflight import readiness


STATIC = Path(__file__).resolve().parent / "static"
API_PREFIX = "/api/mof-screening/"
MAX_JSON_BODY = 800_000
JOB_ID = re.compile(r"^[a-f0-9]{32}$")


def _manager(handler):
    manager = getattr(handler, "mof_screening_manager", None)
    if manager is None:
        raise MofScreeningError("MOF screening manager is not configured", status=500)
    return manager


def _body(handler) -> dict:
    lengths = handler.headers.get_all("Content-Length", [])
    if (
        len(lengths) != 1
        or not lengths[0].isdigit()
        or not 0 < int(lengths[0]) <= MAX_JSON_BODY
    ):
        raise MofScreeningError(
            "a JSON body of at most 800 kB is required", status=413
        )
    content_type = (
        handler.headers.get("Content-Type") or ""
    ).split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise MofScreeningError(
            "Content-Type must be application/json", status=415
        )
    raw = handler.rfile.read(int(lengths[0]))

    def reject(token):
        raise MofScreeningError(
            f"non-finite JSON value is not allowed: {token}", status=400
        )

    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MofScreeningError("request must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise MofScreeningError("request JSON must be an object")
    return value


def get(handler, path: str, query: dict) -> bool:
    if path in {"/screening", "/screening/"}:
        handler.file(STATIC / "index.html")
        return True
    if path == "/screening/style.css":
        handler.file(STATIC / "style.css")
        return True
    if path == "/screening/app.js":
        handler.file(STATIC / "app.js")
        return True
    if path == "/screening/material_state.js":
        handler.file(STATIC / "material_state.js")
        return True
    if path == "/screening/case_presets.js":
        handler.file(STATIC / "case_presets.js")
        return True
    if path == "/screening/progress_clock.js":
        handler.file(STATIC / "progress_clock.js")
        return True
    if path == "/screening/result_view.js":
        handler.file(STATIC / "result_view.js")
        return True
    if path == "/screening/comparison_view.js":
        handler.file(STATIC / "comparison_view.js")
        return True
    if path == "/screening/job_reattach.js":
        handler.file(STATIC / "job_reattach.js")
        return True
    if path == "/screening/material_builder.js":
        handler.file(STATIC / "material_builder.js")
        return True
    if path != "/api/mof-screening" and not path.startswith(API_PREFIX):
        return False
    if not handler._allowed_host():
        handler.json(403, {"error": "host rejected"})
        return True
    try:
        manager = _manager(handler)
        if path in {"/api/mof-screening", API_PREFIX + "scope"}:
            defaults = load_defaults()
            handler.json(
                200,
                {
                    "schema": "mof-screening-public-scope/1",
                    "status": "SCREENING_DIAGNOSTIC",
                    "status_meaning": (
                        "Calculations are runnable screening diagnostics, not "
                        "hardware qualification or measured-material winners."
                    ),
                    "defaults": defaults,
                    "comparison_max_jobs": MAX_COMPARISON_JOBS,
                    "concurrency": manager.concurrency() if hasattr(manager, "concurrency") else {"max_concurrent_jobs": 1, "active_count": 0},
                    "weather": {
                        "official_dataset": WEATHER_PATH.name,
                        "official_dataset_sha256": file_sha256(WEATHER_PATH),
                        "pico": "MeteoSwiss Pully normalized hourly interval ends",
                        "interval_semantics": "start < utc_timestamp <= end",
                        "maximum_hours": {
                            "fixed": 96,
                            "official": 24,
                        },
                    },
                    "engines": {
                        "spatial": (
                            "Transverse symmetric half-bed for wall heating. Direct injection uses axial "
                            "cells with gas, solid and free-liquid inventories."
                        ),
                        "lumped": (
                            "Portable accepted Matched_material_weather "
                            "finite-cycle engine using the same material, "
                            "weather, equipment and output contract."
                        ),
                    },
                    "supported_families": list(SUPPORTED_FAMILIES),
                    "compatibility_gaps": {},
                    "limitations": [
                        "Requested regeneration, vacuum and cooling targets are advisory outcomes; target misses do not by themselves invalidate a case.",
                        "Humidity is shared gas forcing; competitive humid-mixture laws require an explicit material adapter.",
                        "Electricity excludes upstream drying, downstream compression and storage.",
                        "Fixed-weather spatial runs report numerical cycle repeatability. Annual, lifetime and prototype performance require additional evidence.",
                        "A 0.2% mesh-family gate is reported separately and is not inferred from a single run.",
                    ],
                    "history": manager.history(),
                    "gate": readiness(),
                },
            )
        elif path.startswith(API_PREFIX + "materials/"):
            material_id = path[len(API_PREFIX + "materials/") :]
            if "/" in material_id or not material_id:
                raise MofScreeningError(
                    "unknown MOF screening endpoint", status=404
                )
            handler.json(200, load_builtin_material(material_id))
        elif path == API_PREFIX + "status":
            handler.json(200, manager.status())
        elif path == API_PREFIX + "history":
            handler.json(
                200,
                {
                    "schema": "mof-screening-history/1",
                    "jobs": manager.history(),
                },
            )
        elif path.startswith(API_PREFIX + "jobs/"):
            value = path[len(API_PREFIX + "jobs/") :]
            if value.endswith("/export"):
                job_id = value[: -len("/export")]
                if not JOB_ID.fullmatch(job_id):
                    raise MofScreeningError("job id is invalid", field="job_id")
                fmt = (query.get("format") or ["json"])[0].lower()
                text, filename, content_type = manager.export(job_id, fmt)
                handler._send_text(
                    200, text, content_type, filename=filename
                )
            else:
                if "/" in value or not JOB_ID.fullmatch(value):
                    raise MofScreeningError(
                        "unknown MOF screening endpoint", status=404
                    )
                handler.json(200, manager.inspect(value))
        else:
            raise MofScreeningError("unknown MOF screening endpoint", status=404)
        return True
    except MofScreeningError as error:
        handler.json(error.status, {"error": str(error), "field": error.field})
        return True
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        handler.json(
            400,
            {
                "error": f"MOF screening request could not be derived safely: {error}",
                "field": None,
            },
        )
        return True


def post(handler, path: str) -> bool:
    if not path.startswith(API_PREFIX):
        return False
    known = {
        API_PREFIX + "preview",
        API_PREFIX + "material/guided",
        API_PREFIX + "start",
        API_PREFIX + "cancel",
        API_PREFIX + "compare",
        API_PREFIX + "concurrency",
    }
    if path not in known:
        handler.json(404, {"error": "unknown MOF screening endpoint"})
        return True
    if not handler._allowed_host() or not handler._allowed_origin():
        length = handler.headers.get("Content-Length")
        if length and length.isdigit() and int(length) <= MAX_JSON_BODY:
            handler.rfile.read(int(length))
        handler.json(403, {"error": "host/origin rejected"})
        return True
    try:
        body = _body(handler)
        manager = _manager(handler)
        if path == API_PREFIX + "material/guided":
            if set(body) != {"draft"}:
                raise MofScreeningError(
                    "guided material body must contain only draft",
                    field="body",
                )
            handler.json(200, build_guided_material(body))
        elif path == API_PREFIX + "preview":
            if set(body) != {"material", "case"}:
                raise MofScreeningError(
                    "preview body must contain exactly material and case",
                    field="body",
                )
            material_value = body["material"]
            source = (
                material_value.get("material_id", "imported")
                if isinstance(material_value, dict)
                else "invalid_import"
            )
            handler.json(
                200,
                manager.preview(
                    body["material"], body["case"], source=str(source)
                ),
            )
        elif path == API_PREFIX + "concurrency":
            if set(body) != {"max_concurrent_jobs"}:
                raise MofScreeningError("concurrency body must contain only max_concurrent_jobs", field="body")
            if not hasattr(manager, "set_concurrency"):
                raise MofScreeningError("parallel execution is unavailable in this manager", status=409)
            handler.json(200, manager.set_concurrency(body["max_concurrent_jobs"]))
        elif path == API_PREFIX + "start":
            if set(body) != {"preview_id"}:
                raise MofScreeningError(
                    "start body must contain only preview_id", field="body"
                )
            with handler.execution_gate:
                handler.json(202, manager.start(body["preview_id"]))
        elif path == API_PREFIX + "cancel":
            if set(body) != {"job_id"}:
                raise MofScreeningError(
                    "cancel body must contain only job_id", field="body"
                )
            handler.json(200, manager.cancel(body["job_id"]))
        else:
            if set(body) != {"job_ids"}:
                raise MofScreeningError(
                    "compare body must contain only job_ids", field="body"
                )
            handler.json(200, manager.compare(body["job_ids"]))
        return True
    except MofScreeningError as error:
        handler.json(error.status, {"error": str(error), "field": error.field})
        return True
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        handler.json(
            400,
            {
                "error": f"MOF screening request could not be derived safely: {error}",
                "field": None,
            },
        )
        return True
