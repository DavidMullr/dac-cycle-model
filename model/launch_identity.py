"""Loopback launcher identity and conflict checks for DAC."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT_ID = "carbon-team-dac"
DEFAULT_PORT = 8772
EXPECTED_MODULE = "app.server"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent


def canonical_root(path: str | Path) -> str:
    return str(Path(path).resolve())


def read_health(port: int = DEFAULT_PORT, timeout: float = 1.5) -> dict | None:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def port_open(port: int = DEFAULT_PORT, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def inspect_port(
    port: int = DEFAULT_PORT,
    *,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    expected_nonce: str | None = None,
) -> dict:
    health = read_health(port)
    expected_root = canonical_root(project_root)
    if health:
        root_matches = (
            isinstance(health.get("project_root"), str)
            and canonical_root(health["project_root"]) == expected_root
        )
        nonce_matches = expected_nonce is None or health.get("launch_nonce") == expected_nonce
        if (
            health.get("project_id") == PROJECT_ID
            and health.get("port") == port
            and root_matches
            and nonce_matches
        ):
            return {"state": "matching", "health": health}
    if port_open(port):
        if health:
            reason = (
                "health project/root/session identity does not match this checkout"
            )
        else:
            reason = "unrelated or unhealthy listener"
        return {
            "state": "port_conflict",
            "error": (
                f"127.0.0.1:{port} is used by a {reason}; refusing to terminate it."
            ),
        }
    return {"state": "stopped"}


def sha256_commandline(commandline: str) -> str:
    if not isinstance(commandline, str) or not commandline.strip():
        raise ValueError("empty command line")
    return hashlib.sha256(" ".join(commandline.split()).encode("utf-8")).hexdigest()


def build_identity(
    *,
    pid: int,
    port: int,
    executable: str,
    commandline: str,
    process_start_time: str,
    project_root: Path,
    health_started_at: str,
    launch_nonce: str,
    launch_pid: int,
) -> dict:
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid PID")
    if port != DEFAULT_PORT and port <= 0:
        raise ValueError("invalid port")
    if not isinstance(launch_nonce, str) or len(launch_nonce) < 16:
        raise ValueError("invalid launch nonce")
    if type(launch_pid) is not int or launch_pid <= 0:
        raise ValueError("invalid launch PID")
    return {
        "schema": "dac-launcher-identity/2",
        "project_id": PROJECT_ID,
        "pid": pid,
        "port": port,
        "executable": executable,
        "commandline": commandline,
        "commandline_sha256": sha256_commandline(commandline),
        "process_start_time": process_start_time,
        "project_root": canonical_root(project_root),
        "health_started_at": health_started_at,
        "launch_nonce": launch_nonce,
        "launch_pid": launch_pid,
    }


def validate_stop_identity(
    identity: dict,
    *,
    health: dict,
    current_pid: int,
    current_executable: str,
    current_commandline: str,
    current_process_start_time: str,
    expected_project_root: str | Path = DEFAULT_PROJECT_ROOT,
) -> tuple[bool, str]:
    if identity.get("schema") != "dac-launcher-identity/2":
        return False, "identity schema mismatch"
    if identity.get("project_id") != PROJECT_ID:
        return False, "identity project mismatch"
    if int(identity.get("pid", -1)) != int(current_pid):
        return False, "stored PID differs from current process"
    if int(identity.get("port", -1)) != int(health.get("port", -2)):
        return False, "identity/health port mismatch"
    if health.get("project_id") != PROJECT_ID:
        return False, "health project mismatch"
    if int(health.get("pid", -1)) != int(current_pid):
        return False, "health PID differs from current process"
    expected_root = canonical_root(expected_project_root)
    if identity.get("project_root") != expected_root:
        return False, "identity project root differs from this checkout"
    if health.get("project_root") != expected_root:
        return False, "health project root differs from this checkout"
    nonce = identity.get("launch_nonce")
    if not isinstance(nonce, str) or len(nonce) < 16:
        return False, "launch nonce is missing"
    if health.get("launch_nonce") != nonce:
        return False, "health launch nonce differs"
    if identity.get("executable") != current_executable:
        return False, "executable differs"
    if identity.get("process_start_time") != current_process_start_time:
        return False, "process creation time differs; possible PID reuse"
    if identity.get("commandline_sha256") != sha256_commandline(current_commandline):
        return False, "process command line differs"
    if EXPECTED_MODULE not in current_commandline:
        return False, "process command line is not the DAC server"
    if identity.get("health_started_at") != health.get("started_at"):
        return False, "health session identity differs"
    return True, "identity verified"


def validate_start_identity(
    identity: dict,
    *,
    health: dict,
    launched_pid: int,
    current_pid: int,
    current_executable: str,
    current_commandline: str,
    current_process_start_time: str,
    parent_pids: set[int],
    expected_project_root: str | Path,
) -> tuple[bool, str]:
    if identity.get("launch_pid") != launched_pid:
        return False, "identity launch PID differs from the process this launcher started"
    if launched_pid not in parent_pids and current_pid != launched_pid:
        return False, "health PID is not the launched process or its descendant"
    return validate_stop_identity(
        identity,
        health=health,
        current_pid=current_pid,
        current_executable=current_executable,
        current_commandline=current_commandline,
        current_process_start_time=current_process_start_time,
        expected_project_root=expected_project_root,
    )
