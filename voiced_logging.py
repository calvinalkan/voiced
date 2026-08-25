"""Thread-safe daemon logging with native systemd journal priorities."""

from __future__ import annotations

import os
import sys
import threading
import traceback
from typing import Literal

LogLevel = Literal["debug", "info", "warn", "error", "critical"]

_JOURNAL_PRIORITY: dict[LogLevel, int] = {
    "debug": 7,
    "info": 6,
    "warn": 4,
    "error": 3,
    "critical": 2,
}

_debug_enabled = False
_write_lock = threading.Lock()


def stdout_is_journal_stream() -> bool:
    journal_stream = os.environ.get("JOURNAL_STREAM")
    if journal_stream is None:
        return False
    try:
        expected_device, expected_inode = map(int, journal_stream.split(":"))
        stdout_stat = os.fstat(sys.stdout.fileno())
    except (OSError, ValueError):
        return False
    return stdout_stat.st_dev == expected_device and stdout_stat.st_ino == expected_inode


def configure_logging(*, debug: bool) -> None:
    """Configure process-wide debug filtering before worker threads start."""
    global _debug_enabled
    _debug_enabled = debug


def log(component: str, level: LogLevel, message: str) -> None:
    """Write one ordered log event and preserve its severity in journald."""
    if level == "debug" and not _debug_enabled:
        return

    priority_prefix = f"<{_JOURNAL_PRIORITY[level]}>" if stdout_is_journal_stream() else ""
    lines = message.splitlines() or [""]
    with _write_lock:
        for line in lines:
            print(
                f"{priority_prefix}[{component}] [{level}] {line}",
                file=sys.stdout,
                flush=True,
            )


def log_exception(component: str, message: str) -> None:
    """Log an unexpected exception with each traceback line at error priority."""
    log(component, "error", message)
    for line in traceback.format_exc().rstrip().splitlines():
        log(component, "error", line)
