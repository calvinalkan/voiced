#!/usr/bin/env python3
"""Command-line interface for the voiced daemon."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO, cast

from control_socket import ControlSocketError, send_control_request
from typer import detect_keyboard_layout, find_typer_backend

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
STATE_DIRECTORY = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "voiced"
)
HISTORY_FILE = STATE_DIRECTORY / "history.json"


def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]
    arguments = sys.argv[2:]
    if command == "serve":
        serve(arguments)
    elif command in ("listen", "record"):
        send_recording_command(command, arguments)
    elif command == "stop":
        require_no_arguments(command, arguments)
        _ = send_command({"cmd": "stop"})
    elif command == "status":
        require_no_arguments(command, arguments)
        show_status()
    elif command == "kill":
        require_no_arguments(command, arguments)
        stop_daemon()
    elif command == "history":
        show_history(arguments)
    elif command in ("-h", "--help"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage(file=sys.stderr)
        raise SystemExit(1)


def serve(arguments: list[str]) -> None:
    venv_python = SCRIPT_DIRECTORY / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        print("Virtual environment missing. Run: uv sync", file=sys.stderr)
        raise SystemExit(os.EX_CONFIG)
    os.execv(
        venv_python,
        [str(venv_python), str(SCRIPT_DIRECTORY / "daemon.py"), *arguments],
    )


def send_recording_command(command: str, arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog=f"voiced {command}", add_help=False)
    _ = parser.add_argument("-t", "--toggle", action="store_true")
    try:
        parsed = parser.parse_args(arguments)
    except SystemExit as error:
        raise SystemExit(1) from error

    action = f"{command}-toggle" if cast(bool, parsed.toggle) else command
    request: dict[str, object] = {"cmd": action}
    optional_environment = {
        "VOICED_TEST_INPUT": "test_input",
        "VOICED_TEST_OUTPUT": "test_output",
        "VOICED_SAVE_AUDIO": "save_audio",
    }
    for environment_name, request_name in optional_environment.items():
        if value := os.environ.get(environment_name):
            request[request_name] = value
    if fault := os.environ.get("VOICED_TEST_CAPTURE_FAULT"):
        request["test_capture_fault"] = fault

    _ = send_command(request)


def send_command(request: dict[str, object]) -> dict[str, object]:
    try:
        response = send_control_request(request)
    except ControlSocketError as error:
        print(f"Daemon not running or unavailable: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if response.get("ok") is not True:
        print(str(response.get("error", "daemon rejected command")), file=sys.stderr)
        raise SystemExit(1)
    return response


def show_status() -> None:
    try:
        response = send_control_request({"cmd": "status"}, timeout_seconds=0.5)
    except ControlSocketError:
        print("Daemon not running")
        return

    phase = response.get("phase", "unknown")
    pid = response.get("pid", "unknown")
    print(f"Daemon running (PID: {pid}, phase: {phase})")


def stop_daemon() -> None:
    try:
        response = send_control_request({"cmd": "shutdown"})
    except ControlSocketError:
        print("Daemon not running")
        return
    if response.get("ok") is not True:
        print(str(response.get("error", "daemon rejected shutdown")), file=sys.stderr)
        raise SystemExit(1)
    print(f"Shutdown requested (PID: {response.get('pid', 'unknown')})")


def show_history(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="voiced history", add_help=False)
    _ = parser.add_argument("number", nargs="?", type=int)
    _ = parser.add_argument("-c", "--copy", action="store_true")
    try:
        parsed = parser.parse_args(arguments)
    except SystemExit as error:
        raise SystemExit(1) from error

    if not HISTORY_FILE.exists():
        print("No history yet")
        return
    try:
        with HISTORY_FILE.open() as file:
            raw_history: object = json.load(file)  # pyright: ignore[reportAny]
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read history: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if not isinstance(raw_history, list):
        print("History file is invalid", file=sys.stderr)
        raise SystemExit(1)
    history = cast(list[object], raw_history)

    number = cast(int | None, parsed.number)
    if number is None:
        print("Recent transcriptions:\n")
        for index, raw_entry in enumerate(history, start=1):
            if not isinstance(raw_entry, dict):
                continue
            entry = cast(dict[object, object], raw_entry)
            timestamp = str(entry.get("time", ""))
            text = str(entry.get("text", ""))
            display = f"{text[:60]}..." if len(text) > 60 else text
            clock = timestamp.split(" ", maxsplit=1)[-1]
            print(f"  {index:2}. [{clock}] {display}")
        print("\nUsage: voiced history N       to type entry N")
        print("       voiced history N -c    to copy to clipboard")
        return

    if number < 1 or number > len(history):
        print(f"Invalid entry number. Valid range: 1-{len(history)}", file=sys.stderr)
        raise SystemExit(1)
    raw_entry = history[number - 1]
    if not isinstance(raw_entry, dict):
        print(f"History entry {number} is invalid", file=sys.stderr)
        raise SystemExit(1)
    entry = cast(dict[object, object], raw_entry)
    text_value = entry.get("text")
    if not isinstance(text_value, str):
        print(f"History entry {number} is invalid", file=sys.stderr)
        raise SystemExit(1)
    text = text_value
    display = f"{text[:60]}..." if len(text) > 60 else text

    if cast(bool, parsed.copy):
        try:
            copy_result = subprocess.run(
                ["wl-copy", "--", text], timeout=2, check=False
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            print(f"Clipboard command failed: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        if copy_result.returncode != 0:
            print("Could not copy the history entry", file=sys.stderr)
            raise SystemExit(1)
        print(f"Copied to clipboard: {display}")
        return

    backend_name, backend_path = find_typer_backend()
    if backend_name is None or backend_path is None:
        print("No typing backend found", file=sys.stderr)
        raise SystemExit(1)

    try:
        old_clipboard = subprocess.run(
            ["wl-paste", "-n"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
        copy_result = subprocess.run(
            ["wl-copy", "--", text], timeout=2, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(f"Clipboard command failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if copy_result.returncode != 0:
        print("Could not copy the history entry", file=sys.stderr)
        raise SystemExit(1)

    time.sleep(0.05)
    environment = os.environ.copy()
    try:
        if backend_name == "dotool":
            environment["DOTOOL_XKB_LAYOUT"] = detect_keyboard_layout()
            paste_result = subprocess.run(
                [backend_path],
                input="key ctrl+shift+v\n",
                text=True,
                timeout=2,
                env=environment,
                check=False,
            )
        else:
            for socket_path in (
                "/tmp/.ydotool_socket",
                f"/run/user/{os.getuid()}/.ydotool_socket",
                "/run/ydotoold/socket",
            ):
                if Path(socket_path).exists():
                    environment["YDOTOOL_SOCKET"] = socket_path
                    break
            paste_result = subprocess.run(
                [
                    backend_path,
                    "key",
                    "29:1",
                    "42:1",
                    "47:1",
                    "47:0",
                    "42:0",
                    "29:0",
                ],
                timeout=2,
                env=environment,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Typing backend failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        time.sleep(0.05)
        if old_clipboard:
            try:
                _ = subprocess.run(
                    ["wl-copy", "--", old_clipboard], timeout=2, check=False
                )
            except (OSError, subprocess.TimeoutExpired):
                pass

    if paste_result.returncode != 0:
        print(f"{backend_name} failed to paste the history entry", file=sys.stderr)
        raise SystemExit(1)
    print(f"Typed: {display}")


def require_no_arguments(command: str, arguments: list[str]) -> None:
    if arguments:
        print(f"voiced {command} takes no arguments", file=sys.stderr)
        raise SystemExit(1)


def print_usage(*, file: TextIO = sys.stdout) -> None:
    print(
        """voiced - Voice dictation

Commands:
  serve [opts]        Start daemon (for systemd/foreground)
  listen              Listen and transcribe (auto-stop on silence)
  record              Record until 'stop' command
  record -t, --toggle Toggle recording on/off (for keyboard shortcuts)
  stop                Stop current recording
  status              Show daemon status
  kill                Stop the daemon
  history             Show recent transcriptions
  history N           Re-type transcription N
  history N -c        Copy transcription N to clipboard

Options for 'serve':
  --config FILE       Config file path
  --model MODEL       tiny, base, small, medium, large-v3
  --silence-threshold FLOAT
  --silence-duration FLOAT
  --auto-enter        Press Enter after typing
  -d, --debug         Debug output

Config: ~/.config/voiced/config.json
History: ~/.local/state/voiced/history.json""",
        file=file,
    )


if __name__ == "__main__":
    main()
