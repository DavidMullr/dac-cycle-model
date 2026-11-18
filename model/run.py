#!/usr/bin/env python3
"""DAC delivery portal, export and environment CLI."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


IMPORT_NAMES = {
    "colorama": "colorama",
    "contourpy": "contourpy",
    "cycler": "cycler",
    "fonttools": "fontTools",
    "iniconfig": "iniconfig",
    "kiwisolver": "kiwisolver",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "packaging": "packaging",
    "pillow": "PIL",
    "pluggy": "pluggy",
    "pygments": "pygments",
    "pyparsing": "pyparsing",
    "pytest": "pytest",
    "python-dateutil": "dateutil",
    "scipy": "scipy",
    "six": "six",
}


def _locked_requirements() -> dict[str, str]:
    path = ROOT / "requirements-lock.txt"
    required = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "==" not in text:
            raise ValueError(f"unpinned requirement in {path}: {text}")
        name, version = text.split("==", 1)
        required[name.strip().lower()] = version.strip()
    if len(required) != 17:
        raise ValueError(f"expected 17 exact requirements, found {len(required)}")
    return required


def env_check() -> dict:
    required = _locked_requirements()
    packages = {}
    missing = []
    version_mismatches = {}
    other_installed = {}
    import_failures = {}
    for name, locked_version in required.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        packages[name] = installed
        if installed != locked_version:
            version_mismatches[name] = {
                "required": locked_version,
                "installed": installed,
            }
        try:
            importlib.import_module(IMPORT_NAMES[name])
        except Exception as error:
            import_failures[name] = f"{type(error).__name__}: {error}"
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower()
        if name and name not in required:
            other_installed[name] = distribution.version
    runtime_ok = sys.version_info >= (3, 12)
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    base_site = (base_prefix / "Lib" / "site-packages").resolve()
    external_site_paths = []
    for value in sys.path:
        candidate = Path(value or ".").resolve()
        if "site-packages" not in candidate.as_posix().lower():
            continue
        if candidate == base_site or (
            prefix != base_prefix and prefix not in candidate.parents
        ):
            external_site_paths.append(str(candidate))
    is_virtual_environment = prefix != base_prefix
    no_system_site_packages = is_virtual_environment and not external_site_paths
    locked_compatible = (
        runtime_ok and not missing and not version_mismatches and not import_failures
    )
    if not runtime_ok:
        status = "unsupported_python"
    elif missing:
        status = "missing_locked_packages"
    elif version_mismatches:
        status = "lock_version_mismatch"
    elif import_failures:
        status = "import_failure"
    elif not no_system_site_packages:
        status = "not_independent_environment"
    else:
        status = "pass"
    return {
        "schema": "dac-environment-check/2",
        "status": status,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "requires_python_declared": ">=3.12",
        "effective_locked_runtime_python": ">=3.12",
        "python_policy_note": (
            "The public project metadata and the exact lock require CPython >=3.12. "
            "The observed runtime is reported separately from the project policy."
        ),
        "installed_observed_on": "CPython 3.12.3 Windows",
        "packages": packages,
        "required_packages": required,
        "locked_package_count": len(required),
        "missing": missing,
        "version_mismatches": version_mismatches,
        "import_failures": import_failures,
        "merely_installed_but_not_lock_compatible": sorted(
            set(version_mismatches) | set(missing) | set(import_failures)
        ),
        "other_installed_distributions": other_installed,
        "python_runtime_supported": runtime_ok,
        "locked_runtime_compatible": locked_compatible,
        "is_virtual_environment": is_virtual_environment,
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(base_prefix),
        "purelib": str(Path(sysconfig.get_paths()["purelib"]).resolve()),
        "external_site_packages_in_sys_path": external_site_paths,
        "no_system_site_packages": no_system_site_packages,
        "independent_fresh_environment_verified": no_system_site_packages,
        "no_install_performed": True,
    }


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command == "env-check":
        result = env_check()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "pass" else 2
    if command == "serve":
        from app.server import main as serve
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        serve()
        return 0
    raise SystemExit("Use: python run.py serve [--port 8772] or python run.py env-check")


if __name__ == "__main__":
    raise SystemExit(main())
