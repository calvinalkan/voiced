"""Unix-socket control protocol shared by the daemon and CLI."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import cast

MAX_CONTROL_MESSAGE_BYTES = 64 * 1024


class ControlSocketError(RuntimeError):
    """The CLI could not exchange one complete request with the daemon."""


def control_runtime_directory() -> Path:
    instance = os.environ.get("VOICED_INSTANCE", "")
    directory_name = f"voiced-{instance}" if instance else "voiced"
    runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    return runtime_root / directory_name


def control_socket_path() -> Path:
    return control_runtime_directory() / "control.sock"


def send_control_request(
    request: dict[str, object], *, timeout_seconds: float = 2.0
) -> dict[str, object]:
    """Send one request and return the daemon's one-line JSON response."""
    payload = json.dumps(request, separators=(",", ":")).encode() + b"\n"
    response = bytearray()

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_seconds)
            client.connect(str(control_socket_path()))
            client.sendall(payload)
            while b"\n" not in response:
                chunk = client.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > MAX_CONTROL_MESSAGE_BYTES:
                    raise ControlSocketError("daemon response exceeded 64 KiB")
    except OSError as error:
        raise ControlSocketError(str(error)) from error

    response_line = bytes(response).partition(b"\n")[0]
    if not response_line:
        raise ControlSocketError("daemon closed the control socket without a response")
    try:
        decoded: object = json.loads(response_line)  # pyright: ignore[reportAny]
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ControlSocketError(f"daemon returned invalid JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ControlSocketError("daemon response must be a JSON object")
    return cast(dict[str, object], decoded)
