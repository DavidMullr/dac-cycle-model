"""Source-bound readiness gate for MOF screening execution."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..source_closure import (
    CLOSURE_PATH as DEPENDENCY_PATH,
    SourceClosureError,
    validate_source_closure,
)
from .contract import CONFIG_ROOT, MATERIAL_ROOT, ROOT, MofScreeningError


PREFLIGHT_PATH = CONFIG_ROOT / "PREFLIGHT.json"
SOURCE_PATHS = (
    ROOT / "screening/mof_screening/__init__.py",
    ROOT / "screening/mof_screening/parallel_manager.py",
    ROOT / "screening/mof_screening/process_worker.py",
    ROOT / "screening/mof_screening/contract.py",
    ROOT / "screening/mof_screening/windows_io.py",
    ROOT / "screening/mof_screening/purolite.py",
    ROOT / "screening/mof_screening/materials.py",
    ROOT / "screening/mof_screening/stored_water_shift.py",
    ROOT / "screening/mof_screening/young2021.py",
    ROOT / "screening/mof_screening/young2021_hot.py",
    ROOT / "screening/mof_screening/young2021_cold_hot.py",
    ROOT / "screening/mof_screening/young2021_advisory_temperature.py",
    ROOT / "screening/mof_screening/weather.py",
    ROOT / "screening/mof_screening/runtime.py",
    ROOT / "screening/mof_screening/evaluation.py",
    ROOT / "screening/mof_screening/guided_material.py",
    ROOT / "screening/mof_screening/energy_balance.py",
    ROOT / "screening/mof_screening/spatial.py",
    ROOT / "screening/mof_screening/direct_steam.py",
    ROOT / "screening/mof_screening/spatial_jacobian.py",
    ROOT / "screening/mof_screening/bdf_trial_recovery.py",
    ROOT / "screening/mof_screening/lumped_port/engine.py",
    ROOT / "screening/mof_screening/lumped_port/pressure_control.py",
    ROOT / "screening/mof_screening/lumped_port/pump_conditioner.py",
    ROOT / "screening/mof_screening/lumped_port/water_properties.py",
    ROOT / "screening/mof_screening/lumped_port/bounded_jacobian.py",
    ROOT / "screening/mof_screening/lumped_port/solver_start.py",
    ROOT / "screening/mof_screening/lumped_port/weather_driver.py",
    ROOT / "screening/mof_screening/manager.py",
    ROOT / "screening/mof_screening/preflight.py",
    ROOT / "app/mof_screening/routes.py",
    ROOT / "app/mof_screening/static/index.html",
    ROOT / "app/mof_screening/static/app.js",
    ROOT / "app/mof_screening/static/style.css",
    ROOT / "app/mof_screening/static/material_builder.js",
    ROOT / "app/mof_screening/static/material_state.js",
    ROOT / "app/mof_screening/static/case_presets.js",
    ROOT / "app/mof_screening/static/progress_clock.js",
    ROOT / "app/mof_screening/static/comparison_view.js",
    ROOT / "app/mof_screening/static/job_reattach.js",
    ROOT / "app/mof_screening/static/result_view.js",
    ROOT / "app/server.py",
    ROOT / "app/engineering.py",
    DEPENDENCY_PATH,
    CONFIG_ROOT / "defaults.json",
    CONFIG_ROOT / "temperature_registry.json",
    CONFIG_ROOT / "spatial_profile.json",
    MATERIAL_ROOT / "purolite_a110_humid.json",
    MATERIAL_ROOT / "hypothetical_mmen_like.json",
    MATERIAL_ROOT / "calf20_shaped.json",
    MATERIAL_ROOT / "synthetic_independent_demo.json",
    MATERIAL_ROOT / "mmen_mg2_dobpdc_mcdonald2012_conditional.json",
    MATERIAL_ROOT / "lewatit_vp_oc_1065_low2023_conditional.json",
    MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_wadst_conditional.json",
    MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_hot116_conditional.json",
    MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_cold_hot116_conditional.json",
    MATERIAL_ROOT
    / "lewatit_vp_oc_1065_young2021_advisory_temperature_conditional.json",
    MATERIAL_ROOT / "lewatit_vp_oc_1065_young2021_dry_bdf_recovery_v1.json",
    MATERIAL_ROOT / "mmen_mg2_hob_powder_2024_dry_conditional.json",
    MATERIAL_ROOT / "mmen_mg2_hob_pvdf_2024_dry_conditional.json",
    MATERIAL_ROOT / "mof808_lys_2024_25c_dry_conditional.json",
    MATERIAL_ROOT / "mof808_tapa_2024_25c_dry_conditional.json",
    MATERIAL_ROOT / "tepa50_mg2_dobpdc_2025_25c_apparent_dry_conditional.json",
    MATERIAL_ROOT / "cof1000_2025_25c_split_rate_dry_conditional.json",
    MATERIAL_ROOT / "cof999_2024_25c_dry_conditional.json",
    MATERIAL_ROOT / "mil101_pei800_30_glycerin_2026_25c_dry_conditional.json",
    MATERIAL_ROOT / "mil101_pei800_30_2026_25c_dry_conditional.json",
    MATERIAL_ROOT / "cof1000_humid_water_shift_scenario_v1.json",
    MATERIAL_ROOT / "cof999_humid_water_shift_scenario_v1.json",
    MATERIAL_ROOT / "mmen_independent_unary_humid_diagnostic_v1.json",
    MATERIAL_ROOT / "hob_powder_measured_water_transport_independent_v1.json",
    MATERIAL_ROOT / "hob_pvdf_same_framework_water_transfer_scenario_v1.json",
    MATERIAL_ROOT / "tepa50_finite_window_water_proxy_scenario_v1.json",
    MATERIAL_ROOT / "mof808_lys_humid_one_node_scenario_v1.json",
    MATERIAL_ROOT / "mof808_tapa_water_shift_scenario_v1.json",
    MATERIAL_ROOT / "mil101_pei800_control_measured_water_diagnostic_v1.json",
    MATERIAL_ROOT / "mil101_pei800_glycerin_measured_water_diagnostic_v1.json",
    MATERIAL_ROOT / "calf20_independent_water_diagnostic_v1.json",
    MATERIAL_ROOT / "pully_hourly_2023_2025_normalized.csv",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): _sha256_file(path) for path in SOURCE_PATHS
    }


def dependency_status() -> dict:
    """Verify and summarize the current numerical dependency record."""
    closure = validate_source_closure(DEPENDENCY_PATH)
    if closure["file_count"] < 1:
        raise SourceClosureError(
            "current numerical dependency record must not be empty"
        )
    return {
        "path": closure["path"],
        "sha256": closure["sha256"],
        "file_count": closure["file_count"],
        "valid": True,
    }


def execution_source_identity() -> dict:
    """Return a portable, verifiable identity for executed screening results."""
    hashes = source_hashes()
    return {
        "schema": "mof-screening-source-identity/1",
        "release_sha256": _canonical_hash(hashes),
        "source_hashes": hashes,
        "frozen_closure": dependency_status(),
    }


def _canonical_hash(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def readiness() -> dict:
    try:
        record = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "status": "not_ready",
            "start_enabled": False,
            "reason": "zero-integration preflight record is missing",
        }
    expected = source_hashes()
    recorded = record.get("source_hashes")
    mismatches = [
        path
        for path, digest in expected.items()
        if not isinstance(recorded, dict) or recorded.get(path) != digest
    ]
    try:
        closure = dependency_status()
        closure_error = None
    except (OSError, SourceClosureError) as error:
        closure = None
        closure_error = str(error)
    recorded_closure = record.get("frozen_closure") or {}
    closure_matches = bool(
        closure is not None
        and recorded_closure.get("path") == closure["path"]
        and recorded_closure.get("sha256") == closure["sha256"]
        and recorded_closure.get("file_count") == closure["file_count"]
    )
    ready = (
        record.get("schema") == "mof-screening-preflight/1"
        and record.get("status") == "zero_integration_ready"
        and record.get("checks_passed") is True
        and not mismatches
        and closure_matches
    )
    return {
        "status": "ready" if ready else "stale_or_incomplete",
        "start_enabled": ready,
        "reason": (
            None
            if ready
            else (
                "kernel preflight failed, source files changed after validation, "
                "or current numerical dependency record is invalid"
            )
        ),
        "source_mismatches": mismatches,
        "frozen_closure": closure,
        "frozen_closure_error": closure_error,
        "record": record,
    }


def require_ready() -> dict:
    state = readiness()
    if not state["start_enabled"]:
        raise MofScreeningError(
            f"route is temporarily unavailable: {state['reason']}",
            status=423,
        )
    return state


def write_preflight(*, checks: list[dict], tests: dict) -> dict:
    if any(not check.get("passed") for check in checks):
        raise MofScreeningError("cannot create a passing preflight with failed checks")
    closure = dependency_status()
    record = {
        "schema": "mof-screening-preflight/1",
        "status": "zero_integration_ready",
        "checks_passed": True,
        "process_integration_starts": 0,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes(),
        "frozen_closure": closure,
        "checks": checks,
        "tests": tests,
        "scope": (
            "Zero-integration parser, material-fixture, EOS/species/energy "
            "derivative, runtime-guard, lock, and mocked orchestration checks only."
        ),
    }
    temporary = PREFLIGHT_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PREFLIGHT_PATH)
    return record
