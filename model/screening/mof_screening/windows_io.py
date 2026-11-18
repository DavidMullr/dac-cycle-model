"""Bounded Windows file-sharing retries for native JSON persistence."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import time

WINDOWS_IO_RETRY_DELAYS_S = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
TRANSIENT_WINDOWS_WINERRORS = frozenset({5, 32, 33})
TRANSIENT_ERRNOS = frozenset({errno.EACCES, errno.EPERM})


def is_transient_windows_sharing_error(error: OSError) -> bool:
    if os.name != "nt":
        return False
    winerror = getattr(error, "winerror", None)
    if winerror is not None:
        return winerror in TRANSIENT_WINDOWS_WINERRORS
    return isinstance(error, PermissionError) and error.errno in TRANSIENT_ERRNOS


def _retry(callable_, *, retry_delays):
    delays = tuple(retry_delays)
    for attempt in range(len(delays) + 1):
        try:
            return callable_()
        except OSError as error:
            if not is_transient_windows_sharing_error(error) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def read_text_with_retry(
    path, *, encoding="utf-8", retry_delays=WINDOWS_IO_RETRY_DELAYS_S
):
    path = Path(path)
    return _retry(lambda: path.read_text(encoding=encoding), retry_delays=retry_delays)


def replace_with_retry(
    temporary, destination, *, retry_delays=WINDOWS_IO_RETRY_DELAYS_S
):
    _retry(lambda: os.replace(temporary, destination), retry_delays=retry_delays)
