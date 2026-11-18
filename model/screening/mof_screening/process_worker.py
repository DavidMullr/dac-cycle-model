"""Execute one sealed job in a fresh interpreter using the existing engine."""
import argparse
import os
from pathlib import Path
import threading
import time

from .contract import read_json_strict
from .manager import MofScreeningManager, _atomic_json, _utc_now, ID_PATTERN
from .parallel_manager import pid_alive
from .preflight import execution_source_identity


class CancelFlag:
    def __init__(self, path, owner_pid):
        self.path = path
        self.owner_pid = owner_pid
        self.next_owner_check = 0.0
        self.owner_lost = False

    def is_set(self):
        now = time.monotonic()
        if now >= self.next_owner_check:
            self.owner_lost = not pid_alive(self.owner_pid)
            self.next_owner_check = now + 0.5
        return self.owner_lost or self.path.exists()


def execute_job(root, owner_pid):
    root = Path(root).resolve()
    assert ID_PATTERN.fullmatch(root.name) and root.parent.name == "jobs"
    preview = read_json_strict(root / "PREVIEW.json")
    binding = read_json_strict(root / "WORKER_INPUT.json")["source_identity"]
    assert binding == execution_source_identity(), "worker source differs from sealed launch"
    import numpy
    import scipy
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    _atomic_json(root / "WORKER_RUNTIME.json", {"pid": os.getpid(),
                 "python": os.sys.version, "numpy": numpy.__version__, "scipy": scipy.__version__,
                 "math_thread_environment": {key: os.environ.get(key) for key in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                 "threadpools": pools})
    manager = MofScreeningManager.__new__(MofScreeningManager)
    manager.output_root = root.parent.parent
    manager.job_root = root.parent
    manager.state_lock = threading.RLock()
    manager.cancel_event = CancelFlag(root / "CANCEL_REQUEST.json", owner_pid)
    manager.active_event = threading.Event()
    manager.active_event.set()
    manager.startup_source_identity = binding
    manager._release_execution_lock = lambda: None  # Parent owns the aggregate study lock.
    manager._run_job(root.name, preview)
    if manager.cancel_event.owner_lost:
        state = read_json_strict(root / "STATE.json")
        if state.get("state") == "cancelled":
            manager._write_state(root.name, state="interrupted_on_restart",
                                 message="Owning server stopped; partial diagnostic retained.")
    _atomic_json(root / "WORKER_EXIT.json", {"pid": os.getpid(), "utc": _utc_now(),
                 "owner_lost": manager.cancel_event.owner_lost})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    args = parser.parse_args()
    execute_job(args.job_root, args.owner_pid)
