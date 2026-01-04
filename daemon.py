#!/usr/bin/env python3
"""
voiced daemon - Voice dictation backend

Handles recording, transcription, typing, and history.
Controlled via JSON commands from the 'voiced' bash script.

Protocol:
  - Commands appended as JSON lines to /tmp/voiced-command
  - SIGUSR1 signal triggers reading last line
  - Format: {"cmd": "listen"} or {"cmd": "record"} or {"cmd": "stop"}
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from audio import Audio
from transcriber import Transcriber
from typer import Typer


class Config(TypedDict):
    model: str
    device: str
    silence_threshold: float
    silence_duration: float
    speech_start_duration: float
    auto_enter: bool
    debug: bool
    history_size: int
    keyboard_layout: str | None
    typer_backend: str
    clipboard_copy: bool

# File paths - support multiple instances via VOICED_INSTANCE env var
_INSTANCE = os.environ.get("VOICED_INSTANCE", "")
_SUFFIX = f"-{_INSTANCE}" if _INSTANCE else ""
PID_FILE = Path(f"/tmp/voiced{_SUFFIX}.pid")
COMMAND_FILE = Path(f"/tmp/voiced{_SUFFIX}-command")
READY_FILE = Path(f"/tmp/voiced{_SUFFIX}-ready")
CONFIG_DIR = Path.home() / ".config" / "voiced"
CONFIG_FILE = CONFIG_DIR / "config.json"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "voiced"
HISTORY_FILE = STATE_DIR / "history.json"

DEFAULT_CONFIG: Config = {
    "model": "base",
    "device": "cpu",
    "silence_threshold": 0.02,
    "silence_duration": 0.8,
    "speech_start_duration": 0.2,
    "auto_enter": False,
    "debug": False,
    "history_size": 20,
    "keyboard_layout": None,
    "typer_backend": "auto",
    "clipboard_copy": True,
}


def log(component: str, level: str, message: str) -> None:
    """Unified logging: [component] [level] message"""
    print(f"[{component}] [{level}] {message}", flush=True)


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard using wl-copy (Wayland). Fire and forget."""
    try:
        proc = subprocess.Popen(
            ["wl-copy", "--"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.stdin:
            _ = proc.stdin.write(text.encode())
            proc.stdin.close()
        log("clipboard", "info", f"started: {len(text)} chars")
    except FileNotFoundError:
        log("clipboard", "warn", "wl-copy not found, skipping clipboard")
    except Exception as e:
        log("clipboard", "warn", f"clipboard copy failed: {e}")


def notify(message: str, urgency: str = "normal", timeout: int = 2000) -> int | None:
    """Send desktop notification (non-blocking). Returns notification ID."""
    # Skip notifications in test mode
    if _INSTANCE:
        return None
    try:
        result = subprocess.run(
            [
                "notify-send",
                "-a", "voiced",
                "-i", "audio-input-microphone",
                "-u", urgency,
                "-t", str(timeout),
                "-p",  # Print notification ID
                message,
            ],
            capture_output=True,
            text=True,
            timeout=1,
        )
        return int(result.stdout.strip()) if result.stdout.strip() else None
    except Exception as e:
        log("notify", "warn", f"notify-send failed: {e}")
        return None


def notify_close(notify_id: int | None) -> None:
    """Close a notification by ID."""
    if notify_id is None:
        return
    try:
        _ = subprocess.run(
            ["notify-send", "-r", str(notify_id), "-t", "1", ""],
            capture_output=True,
            timeout=1,
        )
    except Exception:
        pass


class ConfigError(Exception):
    """Raised when configuration is invalid."""


def parse_config(data: object) -> Config:
    """Parse and validate config data, returning a Config or raising ConfigError."""
    if not isinstance(data, dict):
        raise ConfigError("config must be a JSON object")

    # Cast to typed dict after validation
    d: dict[str, object] = cast(dict[str, object], data)
    config: Config = DEFAULT_CONFIG.copy()

    # Helper to get and validate a field
    def get_str(key: str) -> str | None:
        if key not in d:
            return None
        val = d[key]
        if not isinstance(val, str):
            raise ConfigError(f"{key} must be a string, got {type(val).__name__}")
        return val

    def get_float(key: str) -> float | None:
        if key not in d:
            return None
        val = d[key]
        if not isinstance(val, (int, float)):
            raise ConfigError(f"{key} must be a number, got {type(val).__name__}")
        return float(val)

    def get_int(key: str) -> int | None:
        if key not in d:
            return None
        val = d[key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise ConfigError(f"{key} must be an integer, got {type(val).__name__}")
        return val

    def get_bool(key: str) -> bool | None:
        if key not in d:
            return None
        val = d[key]
        if not isinstance(val, bool):
            raise ConfigError(f"{key} must be a boolean, got {type(val).__name__}")
        return val

    def get_str_or_none(key: str) -> str | None:
        if key not in d:
            return None
        val = d[key]
        if val is None:
            return None
        if not isinstance(val, str):
            raise ConfigError(f"{key} must be a string or null, got {type(val).__name__}")
        return val

    if (v := get_str("model")) is not None:
        config["model"] = v
    if (v := get_str("device")) is not None:
        config["device"] = v
    if (v := get_float("silence_threshold")) is not None:
        config["silence_threshold"] = v
    if (v := get_float("silence_duration")) is not None:
        config["silence_duration"] = v
    if (v := get_float("speech_start_duration")) is not None:
        config["speech_start_duration"] = v
    if (v := get_bool("auto_enter")) is not None:
        config["auto_enter"] = v
    if (v := get_bool("debug")) is not None:
        config["debug"] = v
    if (v := get_int("history_size")) is not None:
        config["history_size"] = v
    if "keyboard_layout" in d:
        config["keyboard_layout"] = get_str_or_none("keyboard_layout")
    if (v := get_str("typer_backend")) is not None:
        config["typer_backend"] = v
    if (v := get_bool("clipboard_copy")) is not None:
        config["clipboard_copy"] = v

    return config


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from file, with defaults. Exits on invalid config."""
    path: Path | None = None
    if config_path:
        path = Path(config_path)
    elif CONFIG_FILE.exists():
        path = CONFIG_FILE

    if path is None or not path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with open(path) as f:
            data: object = json.load(f)  # pyright: ignore[reportAny]
        return parse_config(data)
    except json.JSONDecodeError as e:
        log("config", "error", f"invalid JSON in {path}: {e}")
        sys.exit(1)
    except ConfigError as e:
        log("config", "error", f"invalid config in {path}: {e}")
        sys.exit(1)


def load_history() -> list[dict[str, str]]:
    """Load history from file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE) as f:
            data: object = json.load(f)  # pyright: ignore[reportAny]
        if not isinstance(data, list):
            return []
        # Validate list contents
        typed_list = cast(list[object], data)
        history: list[dict[str, str]] = []
        for item in typed_list:
            if isinstance(item, dict):
                # Validate it has the expected string keys
                typed_item = cast(dict[object, object], item)
                entry: dict[str, str] = {}
                for k, v in typed_item.items():
                    if isinstance(k, str) and isinstance(v, str):
                        entry[k] = v
                if entry:
                    history.append(entry)
        return history
    except Exception as e:
        log("history", "warn", f"failed to load: {e}")
        return []


def save_history(history: list[dict[str, str]]) -> None:
    """Save history to file."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except OSError as e:
        log("history", "warn", f"failed to save: {e}")


def add_to_history(text: str, max_size: int) -> None:
    """Add a transcription to history."""
    history = load_history()
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
        },
    )
    save_history(history[:max_size])


def write_pid() -> None:
    _ = PID_FILE.write_text(str(os.getpid()))


def remove_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


def read_last_command() -> dict[str, str] | None:
    """Read the last JSON line from command file."""
    if not COMMAND_FILE.exists():
        return None
    try:
        lines = COMMAND_FILE.read_text().strip().split("\n")
        if not lines:
            return None
        data: object = json.loads(lines[-1])  # pyright: ignore[reportAny]
        if not isinstance(data, dict):
            return None
        # Validate and extract string values
        typed_data = cast(dict[object, object], data)
        result: dict[str, str] = {}
        for k, v in typed_data.items():
            if isinstance(k, str) and isinstance(v, str):
                result[k] = v
        return result if result else None
    except Exception as e:
        log("command", "warn", f"failed to parse: {e}")
    return None


class VoiceDaemon:
    config: Config
    debug: bool
    is_tty: bool
    running: bool
    recording: bool
    stop_event: threading.Event
    audio: Audio
    transcriber: Transcriber
    typer: Typer

    def __init__(self, config: Config) -> None:
        self.config = config
        self.debug = config["debug"]
        self.is_tty = sys.stdout.isatty()
        self.running = True
        self.recording = False
        self.stop_event = threading.Event()

        self.audio = Audio(
            silence_threshold=config["silence_threshold"],
            silence_duration=config["silence_duration"],
            speech_start_duration=config["speech_start_duration"],
            debug=self.debug,
            is_tty=self.is_tty,
        )
        self.transcriber = Transcriber(
            model_size=config["model"],
            device=config["device"],
            debug=self.debug,
        )
        self.typer = Typer(
            backend=config["typer_backend"],
            keyboard_layout=config["keyboard_layout"],
            auto_enter=config["auto_enter"],
            debug=self.debug,
        )

    def _start_recording_thread(
        self,
        auto_stop: bool,
        test_input: str | None,
        test_output: str | None,
        save_audio: str | None,
    ) -> None:
        """Start a recording thread with the given parameters."""
        threading.Thread(
            target=lambda: self.do_record(
                auto_stop=auto_stop,
                test_input=test_input,
                test_output=test_output,
                save_audio=save_audio,
            ),
            daemon=True,
        ).start()

    def handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        if signum == signal.SIGUSR1:
            cmd = read_last_command()
            if not cmd:
                log("daemon", "warn", "signal received but no command in file")
                return

            action = cmd.get("cmd")
            test_input = cmd.get("test_input")
            test_output = cmd.get("test_output")
            save_audio = cmd.get("save_audio")
            log("daemon", "info", f"got command: {action}")

            if action == "listen":
                if not self.recording:
                    self._start_recording_thread(True, test_input, test_output, save_audio)
                else:
                    log("daemon", "warn", "already recording, ignoring")

            elif action == "listen-toggle":
                if not self.recording:
                    self._start_recording_thread(True, test_input, test_output, save_audio)
                else:
                    log("daemon", "info", "stopping (listen toggle)")
                    self.stop_event.set()

            elif action == "record":
                if not self.recording:
                    self._start_recording_thread(False, test_input, test_output, save_audio)
                else:
                    log("daemon", "warn", "already recording, ignoring")

            elif action == "record-toggle":
                if not self.recording:
                    self._start_recording_thread(False, test_input, test_output, save_audio)
                else:
                    log("daemon", "info", "toggle: stopping recording")
                    self.stop_event.set()

            elif action == "stop":
                if self.recording:
                    self.stop_event.set()
                else:
                    log("daemon", "warn", "not recording, ignoring")

            else:
                log("daemon", "warn", f"unknown command: {action}")

        elif signum in (signal.SIGTERM, signal.SIGINT):
            log("daemon", "info", "shutting down...")
            self.stop_event.set()
            self.running = False

    def do_record(self, auto_stop: bool = True, test_input: str | None = None, test_output: str | None = None, save_audio: str | None = None) -> None:
        if self.recording:
            return

        self.recording = True
        self.stop_event.clear()

        if test_input:
            log("audio", "info", f"using test input: {test_input}")
        else:
            log("audio", "info", "waiting for speech...")

        try:
            audio = self.audio.record(
                stop_event=self.stop_event,
                auto_stop_on_silence=auto_stop,
                test_input=test_input,
                save_audio=save_audio,
            )
            self.process_audio(audio, test_output=test_output)
        finally:
            self.recording = False
            self.stop_event.clear()

    def process_audio(
        self, audio: NDArray[np.float32] | None, test_output: str | None = None
    ) -> None:
        if audio is None or len(audio) < 8000:
            log("audio", "info", "no speech detected")
            _ = notify("No speech detected")
            return

        audio_duration = len(audio) / 16000
        log("audio", "info", f"done ({audio_duration:.1f}s audio)")

        notify_id = notify("Transcribing...", timeout=30000)
        log("transcriber", "info", "processing...")

        start_time = time.time()
        text = self.transcriber.transcribe(audio)
        elapsed_ms = int((time.time() - start_time) * 1000)

        if not text:
            log("transcriber", "info", f"done ({elapsed_ms}ms): no speech detected")
            notify_close(notify_id)
            _ = notify("No speech detected")
            return

        # Truncate for display
        display_text = text[:80] + "..." if len(text) > 80 else text
        log("transcriber", "info", f'done ({elapsed_ms}ms, {len(text)} chars): "{display_text}"')

        # Save to history (skip in test mode)
        if test_output:
            log("history", "debug", "test mode: skipping history save")
        else:
            add_to_history(text, self.config.get("history_size", 20))

        # Output: write to file (test mode) or type it
        if test_output:
            # Test mode: write to file instead of typing
            notify_close(notify_id)
            log("typer", "info", f"test mode: skipping typing (output: {test_output})")
            log("typer", "info", "test mode: skipping clipboard copy")
            try:
                with open(test_output, "w") as f:
                    _ = f.write(text)
                log("typer", "info", f"test output written to: {test_output}")
            except OSError as e:
                log("typer", "error", f"failed to write test output: {e}")
            return

        start_time = time.time()
        success, total, failed = self.typer.type_text(text)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Copy to clipboard (async, won't block)
        if self.config.get("clipboard_copy", True):
            copy_to_clipboard(text)

        # Close "Transcribing..." notification now that we're done
        notify_close(notify_id)

        if success and failed == 0:
            log("typer", "info", f"done ({elapsed_ms}ms): {total} chars")
        elif success and failed > 0:
            log(
                "typer",
                "warn",
                f"done ({elapsed_ms}ms): {total - failed}/{total} chars, {failed} failed",
            )
            _ = notify(f"Typing incomplete: {failed} chars failed - check history")
        else:
            log("typer", "error", "failed (saved to history)")
            _ = notify("Failed to type - use 'voiced history' to recover")

    def run(self) -> None:
        print("=" * 50, flush=True)
        print("voiced daemon", flush=True)
        print("=" * 50, flush=True)
        print(f"  Model:             faster-whisper ({self.config['model']})", flush=True)
        print(f"  Device:            {self.config['device']}", flush=True)
        print(f"  Silence threshold: {self.config['silence_threshold']} (amplitude 0.0-1.0)", flush=True)
        print(f"  Silence duration:  {self.config['silence_duration']}s", flush=True)
        print(f"  Speech start:      {self.config['speech_start_duration']}s", flush=True)
        print(f"  Auto-enter:        {self.config['auto_enter']}", flush=True)
        print(f"  Clipboard copy:    {self.config['clipboard_copy']}", flush=True)
        print(f"  Typer backend:     {self.typer.backend}", flush=True)
        print(f"  Keyboard layout:   {self.typer.keyboard_layout}", flush=True)
        print(f"  Debug:             {self.debug}", flush=True)
        print(f"  TTY:               {self.is_tty}", flush=True)
        print("-" * 50, flush=True)

        log("daemon", "info", "loading model...")
        self.transcriber.load(on_progress=lambda msg: print(f"    {msg}", flush=True))

        _ = signal.signal(signal.SIGUSR1, self.handle_signal)
        _ = signal.signal(signal.SIGTERM, self.handle_signal)
        _ = signal.signal(signal.SIGINT, self.handle_signal)

        READY_FILE.touch()

        print("-" * 50, flush=True)
        log("daemon", "info", f"ready (pid: {os.getpid()})")
        print("=" * 50, flush=True)

        while self.running:
            time.sleep(0.1)

        # Clean up
        self.typer.shutdown()
        log("daemon", "info", "shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="voiced daemon")
    _ = parser.add_argument("--config", help="Config file path")
    _ = parser.add_argument("--model", "-m", choices=["tiny", "base", "small", "medium", "large-v3"])
    _ = parser.add_argument("--silence-threshold", type=float)
    _ = parser.add_argument("--silence-duration", type=float)
    _ = parser.add_argument(
        "--speech-start-duration",
        type=float,
        help="Sustained speech needed to start (default: 0.2s)",
    )
    _ = parser.add_argument("--auto-enter", action="store_true")
    _ = parser.add_argument(
        "--keyboard-layout", help="Keyboard layout (e.g., de, us). Auto-detected if not set."
    )
    _ = parser.add_argument(
        "--typer-backend", choices=["auto", "dotool", "ydotool"], help="Typing backend"
    )
    _ = parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    # Extract typed values from args (argparse returns Any, so we cast)
    arg_config = cast(str | None, args.config)
    arg_model = cast(str | None, args.model)
    arg_silence_threshold = cast(float | None, args.silence_threshold)
    arg_silence_duration = cast(float | None, args.silence_duration)
    arg_speech_start_duration = cast(float | None, args.speech_start_duration)
    arg_auto_enter = cast(bool, args.auto_enter)
    arg_keyboard_layout = cast(str | None, args.keyboard_layout)
    arg_typer_backend = cast(str | None, args.typer_backend)
    arg_debug = cast(bool, args.debug)

    config = load_config(arg_config)

    if arg_model:
        config["model"] = arg_model
    if arg_silence_threshold:
        config["silence_threshold"] = arg_silence_threshold
    if arg_silence_duration:
        config["silence_duration"] = arg_silence_duration
    if arg_speech_start_duration:
        config["speech_start_duration"] = arg_speech_start_duration
    if arg_auto_enter:
        config["auto_enter"] = True
    if arg_keyboard_layout:
        config["keyboard_layout"] = arg_keyboard_layout
    if arg_typer_backend:
        config["typer_backend"] = arg_typer_backend
    if arg_debug:
        config["debug"] = True

    write_pid()
    try:
        daemon = VoiceDaemon(config)
        daemon.run()
    finally:
        remove_pid()
        READY_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
