import io
import json
import threading
import time

import pytest

from app.mof_screening import routes
from screening.mof_screening.manager import MofScreeningManager


class Headers(dict):
    def get_all(self, key, default=None):
        value = self.get(key)
        if value is not None:
            return [value]
        return [] if default is None else list(default)


class Handler:
    def __init__(self, payload, manager):
        body = json.dumps(payload).encode("utf-8")
        self.headers = Headers(
            {
                "Content-Length": str(len(body)),
                "Content-Type": "application/json",
            }
        )
        self.rfile = io.BytesIO(body)
        self.mof_screening_manager = manager
        self.execution_gate = threading.Lock()
        self.response = None

    def _allowed_host(self):
        return True

    def _allowed_origin(self):
        return True

    def json(self, status, payload):
        self.response = (status, payload)

    def file(self, path):
        self.response = (200, path)

    def _send_text(self, status, text, content_type, filename=None):
        self.response = (
            status,
            {
                "text": text,
                "content_type": content_type,
                "filename": filename,
            },
        )


def test_screening_page_boot_assets_include_progress_clock():
    handler = Handler({}, None)
    assert routes.get(handler, "/screening/", {}) is True
    assert handler.response[0] == 200
    assert handler.response[1].name == "index.html"

    handler = Handler({}, None)
    assert routes.get(handler, "/screening/progress_clock.js", {}) is True
    assert handler.response[0] == 200
    assert handler.response[1].name == "progress_clock.js"
    html = routes.STATIC.joinpath("index.html").read_text(encoding="utf-8")
    app = routes.STATIC.joinpath("app.js").read_text(encoding="utf-8")
    assert html.index("/screening/progress_clock.js") < html.index("/screening/app.js")
    assert "createElapsedClock" in app


def test_native_preview_start_state_and_export_flow_with_mock(
    monkeypatch, tmp_path
):
    from screening.mof_screening import manager as manager_module
    from screening.mof_screening.contract import (
        RESULT_SCHEMA,
        load_builtin_material,
        load_defaults,
    )

    def mocked_run(material, case, **_kwargs):
        return {
            "schema": RESULT_SCHEMA,
            "case_label": case["label"],
            "engine": "spatial",
            "engine_identity": "mock",
            "source_identity": {"release_sha256": "a" * 64},
            "input_identity": {
                "material_sha256": "b" * 64,
                "case_sha256": "c" * 64,
                "combined_sha256": "d" * 64,
            },
            "material": {
                "material_id": material["material_id"],
                "name": material["name"],
                "evidence": material["evidence"],
                "synthetic": material["synthetic"],
                "assumptions": material["assumptions"],
                "validity": material["validity"],
                "uncertainty": material["uncertainty"],
            },
            "metrics": {},
            "horizon": {},
            "endpoint": {},
            "checks": {},
            "numerical_qualification": {},
            "traces": [],
            "execution_runtime_policy": case["runtime_policy"],
        }

    monkeypatch.setattr(manager_module, "run_spatial", mocked_run)
    monkeypatch.setattr(
        manager_module, "require_ready", lambda: {"status": "ready"}
    )
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    handler = Handler(
        {
            "material": load_builtin_material("hypothetical_mmen_like"),
            "case": load_defaults()["case"],
        },
        manager,
    )
    assert routes.post(handler, "/api/mof-screening/preview") is True
    assert handler.response[0] == 200
    preview = handler.response[1]
    assert preview["case"]["runtime_policy"]["mode"] == "manual_stop"
    assert preview["allocation"]["maximum_wall_s"] is None

    handler = Handler(
        {"preview_id": preview["preview_id"]}, manager
    )
    assert routes.post(handler, "/api/mof-screening/start") is True
    assert handler.response[0] == 202
    started = handler.response[1]
    deadline = time.monotonic() + 5.0
    state = None
    while time.monotonic() < deadline:
        state = manager.inspect(started["job_id"])
        if state["state"] in {"completed", "failed", "timed_out"}:
            break
        time.sleep(0.02)
    assert state["state"] == "completed"
    assert state["runtime_policy"]["mode"] == "manual_stop"

    handler = Handler({}, manager)
    assert routes.get(
        handler,
        f"/api/mof-screening/jobs/{started['job_id']}/export",
        {"format": ["json"]},
    ) is True
    assert handler.response[0] == 200
    assert handler.response[1]["content_type"].startswith("application/json")
    manager.close()


def test_comparison_view_static_asset_is_served(tmp_path):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    handler = Handler({}, manager)

    assert routes.get(handler, "/screening/comparison_view.js", {}) is True
    assert handler.response == (
        200,
        routes.STATIC / "comparison_view.js",
    )
    manager.close()


def test_case_presets_static_asset_is_served(tmp_path):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    handler = Handler({}, manager)

    assert routes.get(handler, "/screening/case_presets.js", {}) is True
    assert handler.response == (200, routes.STATIC / "case_presets.js")
    manager.close()


def test_job_reattach_static_asset_is_served(tmp_path):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    handler = Handler({}, manager)
    assert routes.get(handler, "/screening/job_reattach.js", {}) is True
    assert handler.response == (200, routes.STATIC / "job_reattach.js")
    manager.close()


def test_preview_route_returns_field_error_for_array_material(tmp_path):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    handler = Handler({"material": [], "case": {}}, manager)
    assert routes.post(handler, "/api/mof-screening/preview") is True
    assert handler.response[0] == 400
    assert "material" in handler.response[1]["error"]
    manager.close()


def test_start_route_is_locked_when_preflight_missing(monkeypatch, tmp_path):
    from screening.mof_screening.contract import (
        load_builtin_material,
        load_defaults,
    )

    from screening.mof_screening import manager as manager_module
    from screening.mof_screening.contract import MofScreeningError

    def blocked():
        raise MofScreeningError(
            "route is temporarily unavailable: test preflight", status=423
        )

    monkeypatch.setattr(manager_module, "require_ready", blocked)
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    preview = manager.preview(
        load_builtin_material("hypothetical_mmen_like"),
        load_defaults()["case"],
    )
    handler = Handler({"preview_id": preview["preview_id"]}, manager)
    assert routes.post(handler, "/api/mof-screening/start") is True
    assert handler.response[0] == 423
    assert "temporarily unavailable" in handler.response[1]["error"]
    manager.close()


def test_guided_material_route_returns_all_blank_field_issues(tmp_path):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    handler = Handler({"draft": {}}, manager)

    assert routes.post(
        handler, "/api/mof-screening/material/guided"
    ) is True
    assert handler.response[0] == 200
    payload = handler.response[1]
    assert payload["valid"] is False
    assert payload["material"] is None
    assert len(payload["issues"]) >= 10
    manager.close()


@pytest.mark.parametrize(
    ("material_id", "family"),
    (
        (
            "mof808_lys_2024_25c_dry_conditional",
            "activity_shift_reference_isotherm",
        ),
        (
            "mof808_tapa_2024_25c_dry_conditional",
            "activity_shift_reference_isotherm",
        ),
        (
            "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional",
            "activity_shift_reference_isotherm",
        ),
        (
            "cof1000_2025_25c_split_rate_dry_conditional",
            "split_rate_activity_shift_reference_isotherm",
        ),
        (
            "cof999_2024_25c_dry_conditional",
            "activity_shift_reference_isotherm",
        ),
        (
            "mil101_pei800_30_glycerin_2026_25c_dry_conditional",
            "activity_shift_reference_isotherm",
        ),
        (
            "mil101_pei800_30_2026_25c_dry_conditional",
            "activity_shift_reference_isotherm",
        ),
    ),
)
def test_conditional_material_is_selectable_through_native_route(
    material_id, family, tmp_path
):
    manager = MofScreeningManager(
        tmp_path / "jobs", execution_lock_path=tmp_path / "LOCK.json"
    )
    handler = Handler({}, manager)

    assert routes.get(
        handler, f"/api/mof-screening/materials/{material_id}", {}
    ) is True

    assert handler.response[0] == 200
    assert handler.response[1]["material_id"] == material_id
    assert handler.response[1]["water"]["model"] == "dry_only"
    assert handler.response[1]["equilibrium_family"] == family
    manager.close()
