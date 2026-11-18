import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from screening.mof_screening import windows_io


def _share_error(winerror):
    error = PermissionError(13, f"Windows sharing violation {winerror}")
    error.winerror = winerror
    return error


@pytest.mark.skipif(os.name != "nt", reason="requires Windows sharing semantics")
def test_real_windows_replace_contention_exceeds_old_budget(tmp_path):
    target = tmp_path / "STATE.json"
    temporary = tmp_path / "STATE.next.tmp"
    target.write_text('{"state":"old"}\n', encoding="utf-8")
    temporary.write_text('{"state":"new"}\n', encoding="utf-8")
    holder = tmp_path / "hold.py"
    holder.write_text(
        "import ctypes,sys,time\n"
        "from ctypes import wintypes\n"
        "k=ctypes.WinDLL('kernel32',use_last_error=True)\n"
        "k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]\n"
        "k.CreateFileW.restype=wintypes.HANDLE\n"
        "k.CloseHandle.argtypes=[wintypes.HANDLE]\n"
        "k.CloseHandle.restype=wintypes.BOOL\n"
        "handle=k.CreateFileW(sys.argv[1],0x80000000,0,None,3,0x80,None)\n"
        "invalid=wintypes.HANDLE(-1).value\n"
        "assert handle is not None and handle != invalid\n"
        "print('locked',flush=True)\n"
        "try: time.sleep(float(sys.argv[2]))\n"
        "finally: assert k.CloseHandle(handle)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(holder), str(target), "0.35"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None and process.stdout.readline().strip() == "locked"
    started = time.monotonic()
    windows_io.replace_with_retry(
        temporary, target, retry_delays=(0.02, 0.05, 0.1, 0.2, 0.3)
    )
    elapsed = time.monotonic() - started
    assert process.wait(timeout=5) == 0
    assert elapsed > 0.185
    assert json.loads(target.read_text(encoding="utf-8"))["state"] == "new"


@pytest.mark.skipif(os.name != "nt", reason="requires Windows sharing semantics")
def test_real_windows_read_contention_retries(tmp_path):
    target = tmp_path / "STATE.json"
    target.write_text('{"state":"readable"}\n', encoding="utf-8")
    holder = tmp_path / "hold_read.py"
    holder.write_text(
        "import ctypes,sys,time\n"
        "from ctypes import wintypes\n"
        "k=ctypes.WinDLL('kernel32',use_last_error=True)\n"
        "k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]\n"
        "k.CreateFileW.restype=wintypes.HANDLE\n"
        "k.CloseHandle.argtypes=[wintypes.HANDLE]\n"
        "k.CloseHandle.restype=wintypes.BOOL\n"
        "handle=k.CreateFileW(sys.argv[1],0x80000000,0,None,3,0x80,None)\n"
        "invalid=wintypes.HANDLE(-1).value\n"
        "assert handle is not None and handle != invalid\n"
        "print('locked',flush=True)\n"
        "try: time.sleep(0.12)\n"
        "finally: assert k.CloseHandle(handle)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(holder), str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None and process.stdout.readline().strip() == "locked"
    text = windows_io.read_text_with_retry(
        target, retry_delays=(0.01, 0.025, 0.05, 0.1)
    )
    assert process.wait(timeout=5) == 0
    assert json.loads(text)["state"] == "readable"


def test_permanent_error_is_not_retried(monkeypatch, tmp_path):
    attempts = []
    monkeypatch.setattr(windows_io.os, "name", "nt")

    def fail(*_):
        attempts.append(1)
        raise _share_error(87)

    monkeypatch.setattr(windows_io.os, "replace", fail)
    with pytest.raises(PermissionError):
        windows_io.replace_with_retry(
            tmp_path / "a", tmp_path / "b", retry_delays=(0, 0, 0)
        )
    assert attempts == [1]


def test_transient_exhaustion_raises(monkeypatch, tmp_path):
    attempts = []
    sleeps = []
    monkeypatch.setattr(windows_io.os, "name", "nt")

    def fail(*_):
        attempts.append(1)
        raise _share_error(32)

    monkeypatch.setattr(windows_io.os, "replace", fail)
    monkeypatch.setattr(windows_io.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        windows_io.replace_with_retry(
            tmp_path / "a", tmp_path / "b", retry_delays=(0.0, 0.0)
        )
    assert len(attempts) == 3
    assert sleeps == [0.0, 0.0]


@pytest.mark.skipif(os.name != "nt", reason="requires Windows file replacement")
def test_concurrent_producer_reader_stress_in_bounded_subprocess(tmp_path):
    script = tmp_path / "stress.py"
    target = tmp_path / "STATE.json"
    script.write_text(
        "import json,threading,sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0,sys.argv[2])\n"
        "from screening.mof_screening.windows_io import replace_with_retry,read_text_with_retry\n"
        "target=Path(sys.argv[1]); errors=[]; stop=False\n"
        "def writer():\n"
        "  global stop\n"
        "  for i in range(120):\n"
        "    temp=target.with_suffix(f'.{i}.tmp'); temp.write_text(json.dumps({'i':i})+'\\n'); replace_with_retry(temp,target)\n"
        "  stop=True\n"
        "def reader():\n"
        "  while not stop:\n"
        "    try: json.loads(read_text_with_retry(target))\n"
        "    except FileNotFoundError: pass\n"
        "    except Exception as e: errors.append(repr(e))\n"
        "r=threading.Thread(target=reader); r.start(); writer(); r.join()\n"
        "raise SystemExit(1 if errors else 0)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(target), str(Path(__file__).resolve().parents[1])],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(target.read_text())["i"] == 119
