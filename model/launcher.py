"""Normal Windows launcher path used by the DAC .cmd entry points."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import struct
import subprocess
import sys
import time
import webbrowser

from launch_identity import (
    DEFAULT_PORT,
    PROJECT_ID,
    canonical_root,
    inspect_port,
    sha256_commandline,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME = ROOT / "outputs/app_runtime"
IDENTITY_SCHEMA = "dac-launcher-identity/2"


class LauncherError(RuntimeError):
    pass


def _same_instant(left: str, right: str) -> bool:
    try:
        left_value = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
        right_value = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
    except ValueError:
        return False
    return abs((left_value - right_value).total_seconds()) <= 0.002


def _runtime(runtime_root: str | Path | None) -> Path:
    return Path(runtime_root).resolve() if runtime_root else DEFAULT_RUNTIME


def _powershell_json(command: str):
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / (
        "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    if not powershell.is_file():
        raise LauncherError("Windows PowerShell 5.1 is unavailable for process inspection")
    encoded_command = (
        "$OutputEncoding=[Console]::OutputEncoding="
        "[Text.UTF8Encoding]::new($false); " + command
    )
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", encoded_command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        raise LauncherError(
            "cannot inspect the launched Windows process: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    text = (result.stdout or "").strip()
    if not text:
        return None
    return json.loads(text)


def process_info(pid: int) -> dict | None:
    pid = int(pid)
    command = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' "
        "-ErrorAction Stop; if($null -ne $p){"
        "[pscustomobject]@{ProcessId=[int]$p.ProcessId;"
        "ParentProcessId=[int]$p.ParentProcessId;"
        "ExecutablePath=$p.ExecutablePath;CommandLine=$p.CommandLine;"
        "CreationTime=$p.CreationDate.ToUniversalTime().ToString('o')}"
        "|ConvertTo-Json -Compress}"
    )
    return _powershell_json(command)


def process_snapshot() -> list[dict]:
    command = (
        "@(Get-CimInstance Win32_Process -ErrorAction Stop | ForEach-Object {"
        "[pscustomobject]@{ProcessId=[int]$_.ProcessId;"
        "ParentProcessId=[int]$_.ParentProcessId;"
        "ExecutablePath=$_.ExecutablePath;CommandLine=$_.CommandLine;"
        "CreationTime=$_.CreationDate.ToUniversalTime().ToString('o')}"
        "})|ConvertTo-Json -Compress"
    )
    value = _powershell_json(command)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def ancestor_ids(pid: int, snapshot: list[dict]) -> set[int]:
    by_pid = {int(row["ProcessId"]): row for row in snapshot}
    found = set()
    current = int(pid)
    for _ in range(12):
        row = by_pid.get(current)
        if row is None:
            break
        parent = int(row.get("ParentProcessId") or 0)
        if parent <= 0 or parent in found:
            break
        found.add(parent)
        current = parent
    return found


def descendant_ids(pid: int, snapshot: list[dict]) -> set[int]:
    children = {}
    for row in snapshot:
        children.setdefault(int(row.get("ParentProcessId") or 0), []).append(
            int(row["ProcessId"])
        )
    found = set()
    queue = [int(pid)]
    while queue:
        parent = queue.pop()
        for child in children.get(parent, []):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


def _terminate(pid: int) -> bool:
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError:
        return process_info(int(pid)) is None
    for _ in range(20):
        if process_info(int(pid)) is None:
            return True
        time.sleep(0.25)
    return process_info(int(pid)) is None


def _same_process_record(expected: dict, current: dict) -> bool:
    return (
        int(expected["ProcessId"]) == int(current["ProcessId"])
        and str(expected.get("ExecutablePath") or "")
        == str(current.get("ExecutablePath") or "")
        and str(expected.get("CommandLine") or "")
        == str(current.get("CommandLine") or "")
        and _same_instant(
            str(expected.get("CreationTime") or ""),
            str(current.get("CreationTime") or ""),
        )
    )


def _stop_owned_tree(launch_record: dict, *, root: Path, nonce: str) -> None:
    launch_pid = int(launch_record["ProcessId"])
    current_launch = process_info(launch_pid)
    if current_launch is None:
        raise LauncherError(
            "launch process already exited; descendants cannot be safely attributed "
            "to the current process session"
        )
    if not _same_process_record(launch_record, current_launch):
        raise LauncherError(
            "launch PID no longer matches the original process creation identity; "
            "possible PID reuse, no termination performed"
        )
    snapshot = process_snapshot()
    ids = descendant_ids(launch_pid, snapshot)
    records = {int(row["ProcessId"]): row for row in snapshot}
    root_text = canonical_root(root).lower()
    for candidate in sorted(ids, reverse=True):
        expected = records.get(candidate)
        if expected is None:
            continue
        current = process_info(candidate)
        if current is None:
            continue
        command = str(current.get("CommandLine") or "")
        if not _same_process_record(expected, current):
            raise LauncherError(
                f"descendant PID {candidate} changed identity; no further termination"
            )
        if (
            "app.server" not in command
            or nonce not in command
            or root_text not in command.lower()
        ):
            raise LauncherError(
                f"descendant PID {candidate} is not the verified DAC launch tree"
            )
        if not _terminate(candidate):
            raise LauncherError(f"verified descendant PID {candidate} did not exit")
    if not _terminate(launch_pid):
        raise LauncherError(f"verified launch PID {launch_pid} did not exit")


def _health(port: int, *, root: Path, nonce: str | None = None) -> dict | None:
    result = inspect_port(port, project_root=root, expected_nonce=nonce)
    return result.get("health") if result["state"] == "matching" else None


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def _identity_path(runtime: Path) -> Path:
    return runtime / "server_identity.json"


def _validate_process_identity(identity: dict, health: dict, *, expected_root: Path) -> dict:
    root = canonical_root(expected_root)
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise LauncherError("launcher identity schema mismatch")
    if identity.get("project_id") != PROJECT_ID:
        raise LauncherError("launcher project identity mismatch")
    if canonical_root(identity.get("project_root", "")) != root:
        raise LauncherError("launcher identity belongs to another checkout")
    if canonical_root(health.get("project_root", "")) != root:
        raise LauncherError("health response belongs to another checkout")
    nonce = str(identity.get("launch_nonce") or "")
    if len(nonce) < 16 or health.get("launch_nonce") != nonce:
        raise LauncherError("launch-session nonce differs")
    pid = int(identity.get("pid", -1))
    if pid <= 0 or int(health.get("pid", -2)) != pid:
        raise LauncherError("health PID differs from launcher identity")
    if int(identity.get("port", -1)) != int(health.get("port", -2)):
        raise LauncherError("health port differs from launcher identity")
    info = process_info(pid)
    if info is None:
        raise LauncherError("identified DAC process is no longer running")
    if str(info.get("ExecutablePath") or "") != str(identity.get("executable") or ""):
        raise LauncherError("identified DAC executable differs")
    command = str(info.get("CommandLine") or "")
    if (
        "app.server" not in command
        or nonce not in command
        or canonical_root(root).lower() not in command.lower()
        or sha256_commandline(command) != identity.get("commandline_sha256")
    ):
        raise LauncherError("identified DAC command line differs")
    if not _same_instant(
        str(info.get("CreationTime") or ""),
        str(identity.get("process_start_time") or ""),
    ):
        raise LauncherError("identified process creation time differs")
    if not _same_instant(
        str(health.get("started_at") or ""),
        str(identity.get("health_started_at") or ""),
    ):
        raise LauncherError("health session start time differs")
    return info


def _validate_launch_record(identity: dict) -> dict | None:
    launch_pid = int(identity.get("launch_pid", 0))
    if launch_pid <= 0:
        raise LauncherError("launcher identity lacks the owned launch PID")
    current = process_info(launch_pid)
    if current is None:
        return None
    command = str(current.get("CommandLine") or "")
    nonce = str(identity.get("launch_nonce") or "")
    root = canonical_root(identity.get("project_root", ""))
    if (
        not identity.get("launch_process_start_time")
        or not identity.get("launch_commandline_sha256")
        or str(identity.get("launch_executable") or "")
        != str(current.get("ExecutablePath") or "")
        or not _same_instant(
            str(identity["launch_process_start_time"]),
            str(current.get("CreationTime") or ""),
        )
        or sha256_commandline(command)
        != identity.get("launch_commandline_sha256")
        or "app.server" not in command
        or nonce not in command
        or root.lower() not in command.lower()
    ):
        raise LauncherError(
            "recorded launch PID no longer matches its creation/executable/command identity"
        )
    return current


def _existing_verified(port: int, root: Path, runtime: Path) -> dict | None:
    path = _identity_path(runtime)
    if not path.is_file():
        return None
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("schema") != IDENTITY_SCHEMA or int(identity.get("port", -1)) != port:
        return None
    if canonical_root(identity.get("project_root", "")) != canonical_root(root):
        return None
    health = _health(port, root=root, nonce=identity.get("launch_nonce"))
    if health is None:
        return None
    _validate_process_identity(identity, health, expected_root=root)
    return {"health": health, "identity": identity}


def start(
    *,
    port: int = DEFAULT_PORT,
    runtime_root: str | Path | None = None,
    no_browser: bool = False,
) -> dict:
    root = ROOT
    runtime = _runtime(runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    matching = _existing_verified(port, root, runtime)
    if matching:
        if not no_browser:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        return {
            "status": "already_running",
            "pid": matching["health"]["pid"],
            "url": f"http://127.0.0.1:{port}/",
        }
    probe = inspect_port(port, project_root=root)
    if probe["state"] == "port_conflict":
        raise LauncherError(probe["error"])
    if probe["state"] == "matching":
        raise LauncherError(
            "a DAC health response exists for this checkout, but no verified "
            "launcher identity was found; refusing to reuse or terminate it"
        )

    env = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "env-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if env.returncode != 0:
        raise LauncherError(f"DAC environment check failed: {env.stdout.strip()}")

    try:
        if process_info(os.getpid()) is None:
            raise LauncherError("current process could not be inspected")
    except LauncherError as error:
        raise LauncherError(
            "Windows process inspection is unavailable; no server was launched. "
            "Check Windows permissions before trying Start again. " + str(error)
        ) from error

    nonce = secrets.token_hex(16)
    stdout_path = runtime / "server.stdout.log"
    stderr_path = runtime / "server.stderr.log"
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        launch = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "app.server",
                "--port",
                str(port),
                "--launch-nonce",
                nonce,
                "--project-root",
                str(ROOT),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
            ),
        )
    launch_record = process_info(launch.pid)
    if launch_record is None:
        raise LauncherError(
            "cannot establish the launched process creation identity; "
            "no cleanup termination was attempted"
        )
    try:
        health = None
        for _ in range(60):
            time.sleep(0.5)
            health = _health(port, root=root, nonce=nonce)
            if health is not None:
                break
        if health is None:
            raise LauncherError(
                "DAC did not become healthy with the launched root/session identity"
            )
        server_pid = int(health["pid"])
        snapshot = process_snapshot()
        lineage = ancestor_ids(server_pid, snapshot)
        if server_pid != launch.pid and launch.pid not in lineage:
            raise LauncherError(
                "health PID is not the launched process or its proven descendant"
            )
        server = process_info(server_pid)
        if server is None:
            raise LauncherError("health process disappeared during identity validation")
        base_text = subprocess.run(
            [sys.executable, "-c", "import sys; print(sys._base_executable or sys.executable)"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        allowed = {
            canonical_root(sys.executable),
            canonical_root(base_text.stdout.strip()),
        }
        executable = str(server.get("ExecutablePath") or "")
        if canonical_root(executable) not in allowed:
            raise LauncherError(
                "health process executable does not match the selected Python"
            )
        command = str(server.get("CommandLine") or "")
        if "app.server" not in command or nonce not in command:
            raise LauncherError("health process command line does not prove this launch")
        identity = {
            "schema": IDENTITY_SCHEMA,
            "project_id": PROJECT_ID,
            "pid": server_pid,
            "port": port,
            "executable": executable,
            "commandline": command,
            "commandline_sha256": sha256_commandline(command),
            "process_start_time": server["CreationTime"],
            "project_root": canonical_root(root),
            "health_started_at": health["started_at"],
            "launch_nonce": nonce,
            "launch_pid": launch.pid,
            "launch_process_start_time": launch_record["CreationTime"],
            "launch_executable": launch_record.get("ExecutablePath"),
            "launch_commandline_sha256": sha256_commandline(
                str(launch_record.get("CommandLine") or "")
            ),
        }
        path = _identity_path(runtime)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        temporary.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        _stop_owned_tree(launch_record, root=root, nonce=nonce)
        raise
    if not no_browser:
        webbrowser.open(f"http://127.0.0.1:{port}/")
    return {
        "status": "started",
        "pid": server_pid,
        "url": f"http://127.0.0.1:{port}/",
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def stop(
    *,
    port: int = DEFAULT_PORT,
    runtime_root: str | Path | None = None,
) -> dict:
    root = ROOT
    runtime = _runtime(runtime_root)
    path = _identity_path(runtime)
    if not path.is_file():
        probe = inspect_port(port, project_root=root)
        if probe["state"] == "stopped":
            return {"status": "already_stopped", "port": port}
        if probe["state"] == "port_conflict":
            raise LauncherError(probe["error"])
        raise LauncherError("refusing to stop: launcher identity is missing")
    identity = json.loads(path.read_text(encoding="utf-8"))
    if canonical_root(identity.get("project_root", "")) != canonical_root(root):
        raise LauncherError("refusing to stop: launcher belongs to another checkout")
    health = _health(port, root=root, nonce=identity.get("launch_nonce"))
    if health is None:
        probe = inspect_port(port, project_root=root)
        if probe["state"] == "stopped":
            path.unlink()
            return {"status": "already_stopped", "port": port}
        raise LauncherError("refusing to stop: health identity does not match")
    _validate_process_identity(identity, health, expected_root=root)
    launch_record = _validate_launch_record(identity)
    server_pid = int(identity["pid"])
    if not _terminate(server_pid):
        raise LauncherError(
            f"identified DAC PID {server_pid} did not exit; launcher identity retained"
        )
    launch_pid = int(identity.get("launch_pid", 0))
    if launch_record is not None and launch_pid != server_pid:
        if not _terminate(launch_pid):
            raise LauncherError(
                f"owned launch PID {launch_pid} did not exit; launcher identity retained"
            )
    path.unlink()
    return {"status": "stopped", "pid": server_pid, "port": port}


def validate_installation_runtime(version, implementation, pointer_bits, os_name):
    if tuple(version[:2]) != (3, 12) or implementation != "cpython" \
            or pointer_bits != 64 or os_name != "nt":
        raise LauncherError(
            "Setup requires CPython 3.12, 64-bit Windows. "
            "The bundled packages are built for that interpreter."
        )


def install(*, wheelhouse: str | Path | None = None) -> dict:
    validate_installation_runtime(
        sys.version_info, sys.implementation.name, struct.calcsize("P") * 8, os.name)
    venv = ROOT / ".venv"
    venv_python = venv / "Scripts/python.exe"
    requirements = ROOT / "requirements-lock.txt"
    wheelhouse_path = Path(wheelhouse).resolve() if wheelhouse else ROOT / "wheelhouse"
    if not wheelhouse_path.is_dir():
        raise LauncherError(f"offline wheelhouse not found: {wheelhouse_path}")
    if not venv_python.is_file():
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            cwd=ROOT,
            timeout=120,
            check=False,
        )
        if created.returncode != 0:
            raise LauncherError("project-local virtual environment creation failed")
    installed = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse_path),
            "-r",
            str(requirements),
        ],
        cwd=ROOT,
        timeout=600,
        check=False,
    )
    if installed.returncode != 0:
        raise LauncherError(
            "offline dependency installation failed; no network fallback was attempted"
        )
    checked = subprocess.run(
        [str(venv_python), str(ROOT / "run.py"), "env-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if checked.returncode != 0:
        raise LauncherError(f"installed environment validation failed: {checked.stdout}")
    return {"status": "installed", "venv": str(venv), "no_network": True}


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("--port", "-Port", type=int, default=DEFAULT_PORT)
    start_parser.add_argument("--runtime-root", "-RuntimeRoot")
    start_parser.add_argument("--no-browser", "-NoBrowser", action="store_true")
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("--port", "-Port", type=int, default=DEFAULT_PORT)
    stop_parser.add_argument("--runtime-root", "-RuntimeRoot")
    install_parser = sub.add_parser("install")
    install_parser.add_argument("--wheelhouse", "-Wheelhouse")
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            result = start(
                port=args.port,
                runtime_root=args.runtime_root,
                no_browser=args.no_browser,
            )
            if result["status"] == "started":
                print(
                    f"DAC portal started at {result['url']} (PID {result['pid']})."
                )
                print(f"Logs: {result['stdout']} ; {result['stderr']}")
            else:
                print(
                    f"DAC portal already running at {result['url']} "
                    f"(PID {result['pid']})."
                )
        elif args.command == "stop":
            result = stop(port=args.port, runtime_root=args.runtime_root)
            if result["status"] == "stopped":
                print(
                    f"DAC portal stopped (PID {result['pid']}). "
                    "Existing numerical campaign results were not modified."
                )
            else:
                print(f"DAC portal is not running on 127.0.0.1:{args.port}.")
        else:
            result = install(wheelhouse=args.wheelhouse)
            print(
                f"DAC environment installed at {result['venv']}. "
                "No simulation or global package was changed."
            )
    except (LauncherError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"DAC launcher error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
