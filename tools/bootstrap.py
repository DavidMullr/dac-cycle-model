"""Create and validate the pinned local runtime without starting the model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
DEFAULT_VENV = MODEL / ".venv"
LOCK_FILE = MODEL / "requirements-lock.txt"
EXPECTED_LOCK_COUNT = 17
LOCK_LINE = re.compile(r"^[A-Za-z0-9_.-]+==[^\s;]+$")


class BootstrapError(RuntimeError):
    pass


def read_lock(path: Path = LOCK_FILE) -> list[str]:
    requirements = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(requirements) != EXPECTED_LOCK_COUNT:
        raise BootstrapError(
            f"expected {EXPECTED_LOCK_COUNT} exact lock entries, found "
            f"{len(requirements)}"
        )
    invalid = [line for line in requirements if not LOCK_LINE.fullmatch(line)]
    if invalid:
        raise BootstrapError("lock contains an unpinned or malformed requirement")
    names = [line.split("==", 1)[0].casefold() for line in requirements]
    if len(set(names)) != len(names):
        raise BootstrapError("lock contains duplicate package names")
    return requirements


def validate_interpreter() -> dict[str, object]:
    version = sys.version_info
    implementation = platform.python_implementation()
    if implementation != "CPython" or sys.implementation.name != "cpython":
        raise BootstrapError("the exact CPython implementation is required")
    if version < (3, 12):
        raise BootstrapError("CPython 3.12 or newer is required")
    if struct.calcsize("P") * 8 != 64:
        raise BootstrapError("a 64-bit Python interpreter is required")
    return {
        "executable": sys.executable,
        "implementation": implementation,
        "version": platform.python_version(),
        "architecture_bits": struct.calcsize("P") * 8,
    }


def venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def build_plan(
    *, offline_wheelhouse: Path | None, environment: Path = DEFAULT_VENV
) -> list[dict[str, object]]:
    requirements = read_lock()
    if offline_wheelhouse is not None:
        wheelhouse = offline_wheelhouse.resolve(strict=True)
        if not wheelhouse.is_dir():
            raise BootstrapError("offline wheelhouse is not a directory")
        wheels = sorted(wheelhouse.glob("*.whl"))
        if len(wheels) != EXPECTED_LOCK_COUNT:
            raise BootstrapError(
                f"offline wheelhouse must contain {EXPECTED_LOCK_COUNT} wheels, "
                f"found {len(wheels)}"
            )
    else:
        wheelhouse = None

    python = venv_python(environment)
    plan: list[dict[str, object]] = []
    if not python.is_file():
        plan.append(
            {
                "purpose": "create exact local virtual environment",
                "command": [sys.executable, "-m", "venv", str(environment)],
            }
        )
    install = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--requirement",
        str(LOCK_FILE),
    ]
    if wheelhouse is not None:
        install.extend(["--no-index", "--find-links", str(wheelhouse)])
    plan.append(
        {
            "purpose": "install the 17 exact lock entries",
            "command": install,
        }
    )
    plan.append(
        {
            "purpose": "verify installed dependency consistency",
            "command": [str(python), "-m", "pip", "check"],
        }
    )
    plan.append(
        {
            "purpose": "verify exact runtime packages without starting the model",
            "command": [str(python), "-B", "-m", "run", "env-check"],
            "cwd": str(MODEL),
        }
    )
    return plan


def run_plan(plan: list[dict[str, object]]) -> None:
    for step in plan:
        command = [str(value) for value in step["command"]]
        completed = subprocess.run(
            command,
            cwd=str(step.get("cwd", ROOT)),
            check=False,
        )
        if completed.returncode:
            raise BootstrapError(
                f"setup step failed with exit code {completed.returncode}: "
                f"{step['purpose']}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-wheelhouse",
        type=Path,
        help="Use the 17-wheel local set and disable package-index access",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the setup plan without modifying the environment",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate an existing environment without installing or downloading",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    try:
        interpreter = validate_interpreter()
        plan = build_plan(
            offline_wheelhouse=args.offline_wheelhouse,
            environment=args.venv.resolve(),
        )
        if args.check and not venv_python(args.venv.resolve()).is_file():
            raise BootstrapError("the requested local environment does not exist")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "mode": "offline"
                        if args.offline_wheelhouse
                        else "standard-index",
                        "interpreter": interpreter,
                        "lock_entries": len(read_lock()),
                        "plan": plan,
                        "network_requested": False,
                        "model_started": False,
                    },
                    indent=2,
                )
            )
            return 0
        if args.check:
            check_plan = [
                step
                for step in plan
                if step["purpose"]
                in {
                    "verify installed dependency consistency",
                    "verify exact runtime packages without starting the model",
                }
            ]
            run_plan(check_plan)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "environment": str(args.venv.resolve()),
                        "model_started": False,
                    }
                )
            )
            return 0
        run_plan(plan)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "environment": str(args.venv.resolve()),
                    "model_started": False,
                    "next": "Run INSTALL.cmd --check, then START.cmd",
                }
            )
        )
        return 0
    except (BootstrapError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
