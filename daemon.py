#!/usr/bin/env python3
"""Persistent voice-capture, transcription, and text-insertion daemon."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import IO, NoReturn, SupportsFloat, TypedDict, cast, final

import numpy as np
from numpy.typing import NDArray

from audio import Audio, CaptureMonitor, InputDeviceError
from control_socket import (
    MAX_CONTROL_MESSAGE_BYTES,
    control_runtime_directory,
    control_socket_path,
)
from transcriber import (
    MOONSHINE_MODELS,
    TRANSCRIBER_ENGINES,
    WHISPER_MODELS,
    TranscriberProtocol,
    TranscriptionTimeoutError,
    make_transcriber,
)
from typer import Typer
from voiced_logging import configure_logging, log, log_exception

_INSTANCE = os.environ.get("VOICED_INSTANCE", "")
CONFIG_FILE = Path.home() / ".config" / "voiced" / "config.json"
STATE_DIRECTORY = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "voiced"
)
HISTORY_FILE = STATE_DIRECTORY / "history.json"

CAPTURE_START_TIMEOUT_SECONDS = 3.0
CALLBACK_STALL_SECONDS = 2.0
CAPTURE_STOP_TIMEOUT_SECONDS = 2.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
TRANSCRIPTION_TIMEOUT_SECONDS = 120.0
TOGGLE_DEBOUNCE_SECONDS = 0.3


class Config(TypedDict):
    model: str
    device: str
    input_device: str | None
    silence_threshold: float
    silence_duration: float
    speech_start_duration: float
    auto_enter: bool
    debug: bool
    history_size: int
    keyboard_layout: str | None
    typer_backend: str
    clipboard_copy: bool
    insertion_method: str
    paste_keybind: str
    transcriber_engine: str
    streaming: bool
    whisper_vad_filter: bool


DEFAULT_CONFIG: Config = {
    "model": "base",
    "device": "cpu",
    "input_device": None,
    "silence_threshold": 0.02,
    "silence_duration": 0.8,
    "speech_start_duration": 0.2,
    "auto_enter": False,
    "debug": False,
    "history_size": 20,
    "keyboard_layout": None,
    "typer_backend": "auto",
    "clipboard_copy": True,
    "insertion_method": "paste",
    "paste_keybind": "ctrl+shift+v",
    "transcriber_engine": "whisper",
    "streaming": False,
    "whisper_vad_filter": False,
}


class ConfigError(Exception):
    """The supplied configuration cannot define a valid daemon."""


def load_config(config_path: str | None = None) -> Config:
    path = Path(config_path) if config_path else CONFIG_FILE
    if not path.exists():
        if config_path is not None:
            log("config", "error", f"config file does not exist: {path}")
            raise SystemExit(os.EX_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        with path.open() as file:
            data: object = json.load(file)  # pyright: ignore[reportAny]
        return parse_config(data)
    except json.JSONDecodeError as error:
        log("config", "error", f"invalid JSON in {path}: {error}")
        raise SystemExit(os.EX_CONFIG) from error
    except ConfigError as error:
        log("config", "error", f"invalid config in {path}: {error}")
        raise SystemExit(os.EX_CONFIG) from error


def parse_config(data: object) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("config must be a JSON object")

    values = cast(dict[str, object], data)
    unknown_keys = values.keys() - DEFAULT_CONFIG.keys()
    if unknown_keys:
        raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown_keys))}")
    config = DEFAULT_CONFIG.copy()

    def string_value(key: str) -> str | None:
        if key not in values:
            return None
        value = values[key]
        if not isinstance(value, str):
            raise ConfigError(f"{key} must be a string, got {type(value).__name__}")
        return value

    def optional_string_value(key: str) -> str | None:
        value = values.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"{key} must be a string or null, got {type(value).__name__}")
        return value

    def float_value(key: str) -> float | None:
        if key not in values:
            return None
        value = values[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{key} must be a number, got {type(value).__name__}")
        return float(value)

    def integer_value(key: str) -> int | None:
        if key not in values:
            return None
        value = values[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{key} must be an integer, got {type(value).__name__}")
        return value

    def boolean_value(key: str) -> bool | None:
        if key not in values:
            return None
        value = values[key]
        if not isinstance(value, bool):
            raise ConfigError(f"{key} must be a boolean, got {type(value).__name__}")
        return value

    if (value := string_value("model")) is not None:
        config["model"] = value
    if (value := string_value("device")) is not None:
        config["device"] = value
    if "input_device" in values:
        input_device = optional_string_value("input_device")
        if input_device == "":
            raise ConfigError("input_device must not be empty")
        config["input_device"] = input_device
    if (value := float_value("silence_threshold")) is not None:
        config["silence_threshold"] = value
    if (value := float_value("silence_duration")) is not None:
        config["silence_duration"] = value
    if (value := float_value("speech_start_duration")) is not None:
        config["speech_start_duration"] = value
    if (value := boolean_value("auto_enter")) is not None:
        config["auto_enter"] = value
    if (value := boolean_value("debug")) is not None:
        config["debug"] = value
    if (value := integer_value("history_size")) is not None:
        config["history_size"] = value
    if "keyboard_layout" in values:
        config["keyboard_layout"] = optional_string_value("keyboard_layout")
    if (value := string_value("typer_backend")) is not None:
        config["typer_backend"] = value
    if (value := boolean_value("clipboard_copy")) is not None:
        config["clipboard_copy"] = value
    if (value := string_value("insertion_method")) is not None:
        config["insertion_method"] = value
    if (value := string_value("paste_keybind")) is not None:
        config["paste_keybind"] = value
    if (value := string_value("transcriber_engine")) is not None:
        config["transcriber_engine"] = value
    if (value := boolean_value("streaming")) is not None:
        config["streaming"] = value
    if (value := boolean_value("whisper_vad_filter")) is not None:
        config["whisper_vad_filter"] = value

    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    engine = config["transcriber_engine"]
    model = config["model"]
    if engine == "whisper":
        if model not in WHISPER_MODELS:
            raise ConfigError(f"{model!r} is not a Whisper model")
        if config["streaming"]:
            raise ConfigError(
                "streaming=true requires transcriber_engine=moonshine; Whisper has no streaming inference API"
            )
    elif engine == "moonshine":
        if model not in MOONSHINE_MODELS:
            raise ConfigError(f"{model!r} is not a Moonshine model")
        if config["device"] != "cpu":
            raise ConfigError("Moonshine does not support the device setting; use device='cpu'")
        if config["whisper_vad_filter"]:
            raise ConfigError("whisper_vad_filter applies only to the Whisper engine")
    else:
        raise ConfigError(f"transcriber_engine must be one of {TRANSCRIBER_ENGINES}")

    if config["device"] not in ("cpu", "cuda"):
        raise ConfigError("device must be 'cpu' or 'cuda'")
    if config["typer_backend"] not in ("auto", "dotool", "ydotool"):
        raise ConfigError("typer_backend must be 'auto', 'dotool', or 'ydotool'")
    if config["insertion_method"] not in ("paste", "type"):
        raise ConfigError("insertion_method must be 'paste' or 'type'")
    if not math.isfinite(config["silence_threshold"]) or not (
        0 <= config["silence_threshold"] <= 1
    ):
        raise ConfigError("silence_threshold must be finite and between 0 and 1")
    if not math.isfinite(config["silence_duration"]) or config["silence_duration"] <= 0:
        raise ConfigError("silence_duration must be finite and greater than zero")
    if (
        not math.isfinite(config["speech_start_duration"])
        or config["speech_start_duration"] <= 0
    ):
        raise ConfigError("speech_start_duration must be finite and greater than zero")
    if config["history_size"] < 0:
        raise ConfigError("history_size must be non-negative")
    if config["keyboard_layout"] == "":
        raise ConfigError("keyboard_layout must not be empty")
    if not config["paste_keybind"]:
        raise ConfigError("paste_keybind must not be empty")


def copy_to_clipboard(text: str) -> bool:
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            ["wl-copy", "--"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _ = process.communicate(input=text.encode(), timeout=2)
        if process.returncode == 0:
            log("clipboard", "info", f"copied: {len(text)} chars")
            return True
        log("clipboard", "warn", f"clipboard copy failed: exit {process.returncode}")
    except subprocess.TimeoutExpired:
        log("clipboard", "warn", "wl-copy timed out after 2s, killing")
        if process:
            process.kill()
            _ = process.wait()
    except FileNotFoundError:
        log("clipboard", "warn", "wl-copy not found, skipping clipboard")
    except Exception as error:
        log("clipboard", "warn", f"clipboard copy failed: {error}")
        if process:
            process.kill()
            _ = process.wait()
    return False


def notify(
    message: str,
    urgency: str = "normal",
    timeout: int = 2000,
    body: str | None = None,
) -> int | None:
    if _INSTANCE:
        return None
    try:
        command = [
            "notify-send",
            "-a",
            "voiced",
            "-i",
            "audio-input-microphone",
            "-u",
            urgency,
            "-t",
            str(timeout),
            "-p",
            message,
        ]
        if body is not None:
            command.append(body)
        result = subprocess.run(command, capture_output=True, text=True, timeout=1)
        return int(result.stdout.strip()) if result.stdout.strip() else None
    except Exception as error:
        log("notify", "warn", f"notify-send failed: {error}")
        return None


def notify_close(notification_id: int | None) -> None:
    if notification_id is None:
        return
    try:
        _ = subprocess.run(
            [
                "busctl",
                "--user",
                "call",
                "org.freedesktop.Notifications",
                "/org/freedesktop/Notifications",
                "org.freedesktop.Notifications",
                "CloseNotification",
                "u",
                str(notification_id),
            ],
            capture_output=True,
            timeout=1,
        )
    except Exception:
        pass


def add_to_history(text: str, max_size: int) -> None:
    history: list[dict[str, str]] = []
    if HISTORY_FILE.exists():
        try:
            with HISTORY_FILE.open() as file:
                raw_history: object = json.load(file)  # pyright: ignore[reportAny]
            if isinstance(raw_history, list):
                for raw_entry in cast(list[object], raw_history):
                    if not isinstance(raw_entry, dict):
                        continue
                    entry = {
                        key: value
                        for key, value in cast(dict[object, object], raw_entry).items()
                        if isinstance(key, str) and isinstance(value, str)
                    }
                    if entry:
                        history.append(entry)
        except Exception as error:
            log("history", "warn", f"failed to load: {error}")

    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
        },
    )
    temporary_history_file = HISTORY_FILE.with_suffix(".json.tmp")
    try:
        STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with temporary_history_file.open("w") as file:
            json.dump(history[:max_size], file, indent=2)
        _ = temporary_history_file.replace(HISTORY_FILE)
    except OSError as error:
        temporary_history_file.unlink(missing_ok=True)
        log("history", "warn", f"failed to save: {error}")


def audio_health_summary(
    audio: NDArray[np.float32], sample_rate: int, speech_threshold: float
) -> tuple[str, list[str]]:
    if len(audio) == 0:
        return "empty", ["empty audio buffer"]

    absolute_audio = np.abs(audio)
    mean_square = float(cast(SupportsFloat, np.mean(np.square(audio))))
    rms = math.sqrt(mean_square)
    peak = float(cast(SupportsFloat, np.max(absolute_audio)))
    mean_absolute = float(cast(SupportsFloat, np.mean(absolute_audio)))
    non_finite_count = int(np.count_nonzero(~np.isfinite(audio)))
    clipped_percent = float(cast(SupportsFloat, np.mean(absolute_audio >= 0.98))) * 100
    near_zero_percent = float(cast(SupportsFloat, np.mean(absolute_audio < 1e-4))) * 100

    chunk_samples = max(1, int(sample_rate * 0.1))
    chunk_count = (len(audio) + chunk_samples - 1) // chunk_samples
    speech_chunk_count = 0
    for start_index in range(0, len(audio), chunk_samples):
        chunk = audio[start_index : start_index + chunk_samples]
        chunk_mean_absolute = float(cast(SupportsFloat, np.mean(np.abs(chunk))))
        if chunk_mean_absolute > speech_threshold:
            speech_chunk_count += 1
    speech_percent = (speech_chunk_count / chunk_count) * 100 if chunk_count else 0.0

    summary = (
        f"rms={rms:.4f}, mean={mean_absolute:.4f}, peak={peak:.3f}, "
        f"speech_chunks={speech_percent:.0f}%, clipped={clipped_percent:.1f}%, "
        f"near_zero={near_zero_percent:.1f}%"
    )
    warnings: list[str] = []
    if non_finite_count:
        warnings.append(f"{non_finite_count} non-finite samples")
    if peak < speech_threshold:
        warnings.append("peak stayed below speech threshold")
    elif speech_percent < 15:
        warnings.append("mostly below speech threshold")
    if clipped_percent > 1:
        warnings.append("possible input clipping")
    if near_zero_percent > 80:
        warnings.append("mostly near-zero samples")
    return summary, warnings


class OperationPhase(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    STOPPING = "stopping"
    TRANSCRIBING = "transcribing"


@dataclass(frozen=True)
class RecordingRequest:
    auto_stop: bool
    test_input: str | None
    test_output: str | None
    save_audio: str | None
    test_capture_fault: str | None


@dataclass(frozen=True)
class CaptureFinished:
    session_id: int
    request: RecordingRequest
    audio: NDArray[np.float32] | None
    error: BaseException | None
    traceback_text: str | None


@dataclass(frozen=True)
class TranscriptionFinished:
    session_id: int
    error: BaseException | None
    traceback_text: str | None


WorkerEvent = CaptureFinished | TranscriptionFinished


@final
class VoiceDaemon:
    """Own one daemon lifecycle and serialize every externally visible transition."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.phase = OperationPhase.IDLE
        self.shutdown_requested = False
        self.shutdown_deadline: float | None = None
        self.shutdown_signal = threading.Event()
        self.worker_events: queue.SimpleQueue[WorkerEvent] = queue.SimpleQueue()
        self.session_id = 0
        self.last_toggle_monotonic = 0.0
        self.notified_input_device_error: str | None = None

        self.capture_thread: threading.Thread | None = None
        self.capture_stop_event: threading.Event | None = None
        self.capture_monitor: CaptureMonitor | None = None
        self.capture_started_monotonic: float | None = None
        self.capture_stop_deadline: float | None = None
        self.capture_failure_reason: str | None = None
        self.capture_samples_count = 0
        self.capture_request: RecordingRequest | None = None
        self.recovery_after_transcription_reason: str | None = None
        self.transcription_thread: threading.Thread | None = None

        self.control_server: socket.socket | None = None
        self.control_lock_file: IO[str] | None = None

        self.audio = Audio(
            silence_threshold=config["silence_threshold"],
            silence_duration=config["silence_duration"],
            speech_start_duration=config["speech_start_duration"],
            input_device=config["input_device"],
            debug=config["debug"],
            is_tty=sys.stdout.isatty(),
        )
        self.transcriber: TranscriberProtocol = make_transcriber(
            engine=config["transcriber_engine"],
            model=config["model"],
            device=config["device"],
            debug=config["debug"],
            streaming=config["streaming"],
            whisper_vad_filter=config["whisper_vad_filter"],
        )

        def typer_notify(message: str) -> None:
            _ = notify(message, urgency="low")

        self.typer = Typer(
            backend=config["typer_backend"],
            keyboard_layout=config["keyboard_layout"],
            auto_enter=config["auto_enter"],
            debug=config["debug"],
            insertion_method=config["insertion_method"],
            paste_keybind=config["paste_keybind"],
            copy_to_clipboard=copy_to_clipboard,
            notify=typer_notify,
        )

    def run(self) -> None:
        _ = signal.signal(signal.SIGTERM, self.handle_signal)
        _ = signal.signal(signal.SIGINT, self.handle_signal)
        self.acquire_control_lock()

        startup_message = (
            f"starting engine={self.config['transcriber_engine']} "
            f"model={self.config['model']} streaming={self.config['streaming']} "
            f"device={self.config['device']} "
            f"input_device={self.config['input_device'] or 'system-default'}"
        )
        log("daemon", "info", startup_message)
        log("daemon", "info", "loading model")
        self.transcriber.load(on_progress=lambda message: log("transcriber", "info", message))

        if self.shutdown_signal.is_set():
            self.request_shutdown("signal-during-startup")
        else:
            self.open_control_socket()
            log(
                "daemon",
                "info",
                f"ready pid={os.getpid()} socket={control_socket_path()}",
            )

        try:
            self.run_event_loop()
        finally:
            self.close_control_socket()

        if not self.transcriber.shutdown(timeout_seconds=2):
            log(
                "daemon",
                "critical",
                "event=transcriber_shutdown_timeout action=forced-exit",
            )
            self.typer.shutdown()
            os._exit(0)
        self.typer.shutdown()
        log("daemon", "info", "event=shutdown_complete")
        self.release_control_lock()

    def handle_signal(self, _signum: int, _frame: FrameType | None) -> None:
        self.shutdown_signal.set()

    def acquire_control_lock(self) -> None:
        runtime_directory = control_runtime_directory()
        runtime_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        runtime_directory.chmod(0o700)

        lock_path = runtime_directory / "daemon.lock"
        self.control_lock_file = lock_path.open("w")
        try:
            fcntl.flock(self.control_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            log("daemon", "error", "another daemon owns the control socket")
            raise SystemExit(os.EX_UNAVAILABLE) from error

    def open_control_socket(self) -> None:
        socket_path = control_socket_path()
        socket_path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(socket_path))
            socket_path.chmod(0o600)
            server.listen(16)
            server.settimeout(0.05)
        except Exception:
            server.close()
            socket_path.unlink(missing_ok=True)
            raise
        self.control_server = server

    def close_control_socket(self) -> None:
        if self.control_server is not None:
            self.control_server.close()
            self.control_server = None
        control_socket_path().unlink(missing_ok=True)

    def release_control_lock(self) -> None:
        if self.control_lock_file is not None:
            self.control_lock_file.close()
            self.control_lock_file = None

    def run_event_loop(self) -> None:
        while True:
            if self.shutdown_signal.is_set() and not self.shutdown_requested:
                self.request_shutdown("signal")

            self.drain_worker_events()
            now = time.monotonic()
            self.monitor_capture(now)

            if self.shutdown_requested and self.phase == OperationPhase.IDLE:
                return
            if self.shutdown_deadline is not None and now >= self.shutdown_deadline:
                self.force_shutdown("operation_did_not_finish")

            self.serve_control_request()

    def serve_control_request(self) -> None:
        if self.control_server is None:
            time.sleep(0.05)
            return
        try:
            connection = self.control_server.accept()[0]
        except TimeoutError:
            return
        except OSError as error:
            if self.shutdown_requested:
                return
            raise error

        with connection:
            connection.settimeout(0.5)
            try:
                payload = bytearray()
                while b"\n" not in payload:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    payload.extend(chunk)
                    if len(payload) > MAX_CONTROL_MESSAGE_BYTES:
                        raise ValueError("control request exceeded 64 KiB")
                request_line = bytes(payload).partition(b"\n")[0]
                request: object = json.loads(request_line)  # pyright: ignore[reportAny]
                if not isinstance(request, dict):
                    raise ValueError("control request must be a JSON object")
                response = self.dispatch_control_request(cast(dict[str, object], request))
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                ValueError,
                OSError,
            ) as error:
                log("control", "warn", f"invalid request: {error}")
                response = {"ok": False, "error": str(error)}
            try:
                connection.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
            except OSError:
                log("control", "debug", "client disconnected before response")

    def dispatch_control_request(self, request: dict[str, object]) -> dict[str, object]:
        action = request.get("cmd")
        if not isinstance(action, str):
            return {"ok": False, "error": "cmd must be a string"}

        if action == "status":
            return {
                "ok": True,
                "pid": os.getpid(),
                "phase": self.phase.value,
                "shutdown_requested": self.shutdown_requested,
            }
        if action == "shutdown":
            self.request_shutdown("control-command")
            return {"ok": True, "pid": os.getpid(), "phase": self.phase.value}

        if action not in ("listen", "listen-toggle", "record", "record-toggle", "stop"):
            log("control", "warn", f"unknown command: {action}")
            return {"ok": False, "error": f"unknown command: {action}"}
        if self.shutdown_requested:
            return {"ok": True, "status": "ignored", "reason": "shutting-down"}

        if action == "stop":
            if self.phase == OperationPhase.CAPTURING:
                self.request_capture_stop("stop-command")
                return {"ok": True, "status": "stopping"}
            log("control", "debug", f"ignored stop phase={self.phase.value}")
            return {"ok": True, "status": "ignored", "reason": self.phase.value}

        auto_stop = action.startswith("listen")
        is_toggle = action.endswith("-toggle")
        now = time.monotonic()
        if is_toggle:
            if now - self.last_toggle_monotonic < TOGGLE_DEBOUNCE_SECONDS:
                log("control", "debug", f"ignored repeated toggle phase={self.phase.value}")
                return {"ok": True, "status": "ignored", "reason": "debounced"}
            self.last_toggle_monotonic = now

            if self.phase == OperationPhase.CAPTURING:
                self.request_capture_stop("toggle")
                return {"ok": True, "status": "stopping"}
            if self.phase != OperationPhase.IDLE:
                log("control", "debug", f"ignored toggle phase={self.phase.value}")
                return {"ok": True, "status": "ignored", "reason": self.phase.value}
        elif self.phase != OperationPhase.IDLE:
            log("control", "warn", f"ignored {action}: phase={self.phase.value}")
            return {"ok": True, "status": "ignored", "reason": self.phase.value}

        test_capture_fault = request.get("test_capture_fault")
        if test_capture_fault not in (None, "stall", "hang"):
            return {"ok": False, "error": "test_capture_fault must be 'stall' or 'hang'"}
        if test_capture_fault is not None and not _INSTANCE:
            return {"ok": False, "error": "capture fault injection requires VOICED_INSTANCE"}

        optional_strings: dict[str, str | None] = {}
        for key in ("test_input", "test_output", "save_audio"):
            value = request.get(key)
            if value is not None and not isinstance(value, str):
                return {"ok": False, "error": f"{key} must be a string"}
            optional_strings[key] = value

        self.start_capture(
            RecordingRequest(
                auto_stop=auto_stop,
                test_input=optional_strings["test_input"],
                test_output=optional_strings["test_output"],
                save_audio=optional_strings["save_audio"],
                test_capture_fault=cast(str | None, test_capture_fault),
            )
        )
        return {"ok": True, "status": "capturing", "session": self.session_id}

    def start_capture(self, request: RecordingRequest) -> None:
        self.session_id += 1
        session_id = self.session_id
        stop_event = threading.Event()
        monitor = CaptureMonitor()
        self.capture_stop_event = stop_event
        self.capture_monitor = monitor
        self.capture_started_monotonic = time.monotonic()
        self.capture_stop_deadline = None
        self.capture_failure_reason = None
        self.capture_samples_count = 0
        self.capture_request = request
        self.recovery_after_transcription_reason = None
        self.phase = OperationPhase.CAPTURING

        self.transcriber.start_session()

        def feed_transcriber(audio: NDArray[np.float32]) -> None:
            self.capture_samples_count += len(audio)
            self.transcriber.feed(audio)

        def capture() -> None:
            try:
                audio = self.audio.record(
                    stop_event=stop_event,
                    auto_stop_on_silence=request.auto_stop,
                    on_chunk=feed_transcriber,
                    test_input=request.test_input,
                    save_audio=request.save_audio,
                    monitor=monitor,
                    test_capture_fault=request.test_capture_fault,
                )
                event = CaptureFinished(session_id, request, audio, None, None)
            except BaseException as error:
                event = CaptureFinished(
                    session_id,
                    request,
                    None,
                    error,
                    traceback.format_exc(),
                )
            self.worker_events.put(event)

        self.capture_thread = threading.Thread(
            target=capture,
            name=f"capture-{session_id}",
            daemon=True,
        )
        self.capture_thread.start()
        mode = "listen" if request.auto_stop else "record"
        log("daemon", "info", f"event=capture_started session={session_id} mode={mode}")

    def request_capture_stop(self, reason: str) -> None:
        if self.phase == OperationPhase.STOPPING:
            log("daemon", "debug", f"capture stop already requested session={self.session_id}")
            return
        if self.phase != OperationPhase.CAPTURING or self.capture_stop_event is None:
            return

        self.phase = OperationPhase.STOPPING
        self.capture_stop_deadline = time.monotonic() + CAPTURE_STOP_TIMEOUT_SECONDS
        self.capture_stop_event.set()
        log(
            "daemon",
            "info",
            f"event=capture_stop_requested session={self.session_id} reason={reason}",
        )

    def monitor_capture(self, now: float) -> None:
        if self.phase == OperationPhase.CAPTURING and self.capture_monitor is not None:
            stream_started, last_callback = self.capture_monitor.snapshot()
            if (
                stream_started is None
                and self.capture_started_monotonic is not None
                and now - self.capture_started_monotonic >= CAPTURE_START_TIMEOUT_SECONDS
            ):
                self.capture_failure_reason = "capture_start_timeout"
                log(
                    "audio",
                    "error",
                    f"event=capture_start_timeout session={self.session_id}",
                )
                self.request_capture_stop("capture-start-timeout")
            elif (
                last_callback is not None
                and now - last_callback >= CALLBACK_STALL_SECONDS
            ):
                callback_age_ms = int((now - last_callback) * 1000)
                self.capture_failure_reason = "callback_stall"
                log(
                    "audio",
                    "error",
                    f"event=callback_stall session={self.session_id} callback_age_ms={callback_age_ms}",
                )
                self.request_capture_stop("callback-stall")

        if (
            self.phase == OperationPhase.STOPPING
            and self.capture_stop_deadline is not None
            and now >= self.capture_stop_deadline
        ):
            if self.shutdown_requested:
                self.force_shutdown("capture_stop_timeout")
            capture_request = self.capture_request
            if capture_request is None or self.capture_samples_count == 0:
                self.restart_after_failure("capture_stop_timeout")
            assert capture_request is not None

            # The capture thread may be blocked in PortAudio teardown, but every
            # completed callback has already queued its samples on the model
            # worker. Stop accepting callbacks and finalize that queue before
            # process-level recovery releases the stuck native thread.
            self.capture_stop_deadline = None
            self.capture_monitor = None
            self.capture_started_monotonic = None
            self.recovery_after_transcription_reason = "capture_stop_timeout"
            log(
                "daemon",
                "warn",
                f"event=recovery_transcription session={self.session_id} reason=capture_stop_timeout samples={self.capture_samples_count}",
            )
            self.start_transcription(
                request=capture_request,
                audio=None,
                captured_samples_count=self.capture_samples_count,
                capture_error=None,
            )

    def drain_worker_events(self) -> None:
        while True:
            try:
                event = self.worker_events.get_nowait()
            except queue.Empty:
                return
            if isinstance(event, CaptureFinished):
                self.finish_capture(event)
            else:
                self.finish_transcription(event)

    def finish_capture(self, event: CaptureFinished) -> None:
        if event.session_id != self.session_id:
            log("daemon", "warn", f"ignored stale capture event session={event.session_id}")
            return
        if self.recovery_after_transcription_reason == "capture_stop_timeout":
            if self.capture_thread is not None:
                self.capture_thread.join()
            self.capture_thread = None
            log("daemon", "debug", f"detached capture returned session={event.session_id}")
            return

        if self.capture_thread is not None:
            self.capture_thread.join()
        self.capture_thread = None
        self.capture_stop_event = None
        self.capture_monitor = None
        self.capture_started_monotonic = None
        self.capture_stop_deadline = None
        self.capture_request = None

        if self.shutdown_requested:
            log(
                "daemon",
                "info",
                f"event=capture_discarded session={event.session_id} reason=shutdown",
            )
            self.phase = OperationPhase.IDLE
            return

        recovery_reason = self.capture_failure_reason
        if event.error is not None:
            if isinstance(event.error, InputDeviceError) and recovery_reason is None:
                error_message = str(event.error)
                log("audio", "error", error_message)
                if error_message != self.notified_input_device_error:
                    self.notified_input_device_error = error_message
                    _ = notify(
                        "voiced: microphone error",
                        urgency="critical",
                        timeout=2000,
                        body=error_message,
                    )
            else:
                log("audio", "error", event.traceback_text or str(event.error))
                recovery_reason = recovery_reason or "capture_exception"
        else:
            self.notified_input_device_error = None

        capture_error = event.error
        if recovery_reason is not None:
            capture_error = None
            log(
                "daemon",
                "warn",
                f"event=recovery_transcription session={event.session_id} reason={recovery_reason} samples={self.capture_samples_count}",
            )
            if recovery_reason in ("callback_stall", "capture_start_timeout"):
                _ = notify(
                    "voiced: microphone stream stopped",
                    urgency="critical",
                    timeout=3000,
                    body="Available audio will be transcribed. Reconnect the microphone before the next recording.",
                )
            else:
                self.recovery_after_transcription_reason = recovery_reason

        self.start_transcription(
            request=event.request,
            audio=event.audio,
            captured_samples_count=self.capture_samples_count,
            capture_error=capture_error,
        )

    def start_transcription(
        self,
        *,
        request: RecordingRequest,
        audio: NDArray[np.float32] | None,
        captured_samples_count: int,
        capture_error: BaseException | None,
    ) -> None:
        session_id = self.session_id
        self.phase = OperationPhase.TRANSCRIBING
        self.capture_request = None

        def transcribe() -> None:
            try:
                if capture_error is not None:
                    _ = self.transcriber.finalize(timeout_seconds=10)
                else:
                    self.process_audio(
                        audio,
                        captured_samples_count=captured_samples_count,
                        session_id=session_id,
                        test_output=request.test_output,
                    )
                finished = TranscriptionFinished(session_id, None, None)
            except BaseException as error:
                finished = TranscriptionFinished(
                    session_id,
                    error,
                    traceback.format_exc(),
                )
            self.worker_events.put(finished)

        self.transcription_thread = threading.Thread(
            target=transcribe,
            name=f"transcription-{session_id}",
            daemon=True,
        )
        self.transcription_thread.start()

    def finish_transcription(self, event: TranscriptionFinished) -> None:
        if event.session_id != self.session_id:
            log("daemon", "warn", f"ignored stale transcription event session={event.session_id}")
            return
        if self.transcription_thread is not None:
            self.transcription_thread.join()
        self.transcription_thread = None

        if event.error is not None:
            if self.shutdown_requested:
                log(
                    "transcriber",
                    "error",
                    f"transcription ended during shutdown: {event.error}",
                )
                self.phase = OperationPhase.IDLE
                return
            if isinstance(event.error, TranscriptionTimeoutError):
                log("transcriber", "critical", str(event.error))
                self.restart_after_failure("transcription_timeout")
            log("transcriber", "error", event.traceback_text or str(event.error))
            self.restart_after_failure("transcription_exception")

        if self.recovery_after_transcription_reason is not None:
            recovery_reason = self.recovery_after_transcription_reason
            self.recovery_after_transcription_reason = None
            if self.shutdown_requested:
                self.phase = OperationPhase.IDLE
                return
            self.restart_after_failure(recovery_reason, recording_processed=True)

        self.phase = OperationPhase.IDLE
        if self.capture_failure_reason is not None:
            log(
                "daemon",
                "info",
                f"event=capture_recovered session={event.session_id} reason={self.capture_failure_reason} action=kept-running",
            )
            self.capture_failure_reason = None
        log("daemon", "debug", f"event=session_complete session={event.session_id}")

    def process_audio(
        self,
        audio: NDArray[np.float32] | None,
        *,
        captured_samples_count: int,
        session_id: int,
        test_output: str | None,
    ) -> None:
        audio_samples_count = len(audio) if audio is not None else captured_samples_count
        if audio_samples_count < 8000:
            _ = self.transcriber.finalize(timeout_seconds=TRANSCRIPTION_TIMEOUT_SECONDS)
            log("audio", "info", f"no speech detected session={session_id}")
            _ = notify("No speech detected")
            return

        audio_duration = audio_samples_count / 16000
        if audio is None:
            log(
                "audio",
                "warn",
                f"processing {audio_duration:.1f}s buffered before stuck capture session={session_id}",
            )
        else:
            health, health_warnings = audio_health_summary(
                audio,
                sample_rate=16000,
                speech_threshold=self.config["silence_threshold"],
            )
            log(
                "audio",
                "info",
                f"done session={session_id} ({audio_duration:.1f}s audio, {health})",
            )
            for warning in health_warnings:
                log("audio", "warn", f"session={session_id} {warning}")

        notification_id = notify("Transcribing...", timeout=30000)
        log("transcriber", "info", f"processing session={session_id}")
        start_time = time.time()
        try:
            text = self.transcriber.finalize(timeout_seconds=TRANSCRIPTION_TIMEOUT_SECONDS)
        except BaseException:
            notify_close(notification_id)
            raise
        elapsed_ms = int((time.time() - start_time) * 1000)

        if not text:
            log(
                "transcriber",
                "info",
                f"done session={session_id} ({elapsed_ms}ms): no speech detected",
            )
            notify_close(notification_id)
            _ = notify("No speech detected")
            return

        display_text = f"{text[:80]}..." if len(text) > 80 else text
        log(
            "transcriber",
            "info",
            f'done session={session_id} ({elapsed_ms}ms, {len(text)} chars): "{display_text}"',
        )
        notify_close(notification_id)
        if audio_duration >= 10 and len(text) / audio_duration < 2:
            log(
                "transcriber",
                "warn",
                f"session={session_id} transcript is unusually short for the captured audio; check audio health stats or capture a sample with VOICED_SAVE_AUDIO",
            )

        if test_output:
            log("history", "debug", "test mode: skipping history save")
        else:
            add_to_history(text, self.config["history_size"])

        if test_output:
            log("typer", "info", f"test mode: writing output to {test_output}")
            try:
                with open(test_output, "w") as file:
                    _ = file.write(text)
            except OSError as error:
                log("typer", "error", f"failed to write test output: {error}")
            return

        start_time = time.time()
        success, method = self.typer.insert_text(text)
        elapsed_ms = int((time.time() - start_time) * 1000)
        if method != "paste" and self.config["clipboard_copy"]:
            _ = copy_to_clipboard(text)

        if success:
            log("typer", "info", f"done ({elapsed_ms}ms, {method}): {len(text)} chars")
        else:
            log("typer", "error", "failed (saved to history)")
            _ = notify("Failed to insert text - use 'voiced history' to recover")

    def request_shutdown(self, reason: str) -> None:
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.shutdown_deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
        log(
            "daemon",
            "info",
            f"event=shutdown_requested reason={reason} phase={self.phase.value}",
        )
        if self.phase == OperationPhase.CAPTURING:
            self.request_capture_stop("shutdown")
        elif self.phase == OperationPhase.STOPPING:
            log("daemon", "debug", "shutdown is waiting for capture teardown")
        elif self.phase == OperationPhase.TRANSCRIBING:
            log("daemon", "debug", "shutdown is waiting for transcription")

    def restart_after_failure(
        self, reason: str, *, recording_processed: bool = False
    ) -> NoReturn:
        log(
            "daemon",
            "critical",
            f"event=recovery_restart reason={reason} session={self.session_id} exit_status={os.EX_TEMPFAIL}",
        )
        notification_body = (
            "Available audio was processed before restart. Try the hotkey again."
            if recording_processed
            else "No usable audio was available. Try the hotkey again after the restart."
        )
        _ = notify(
            "voiced is restarting after an audio failure",
            urgency="critical",
            timeout=3000,
            body=notification_body,
        )
        self.close_control_socket()
        os._exit(os.EX_TEMPFAIL)

    def force_shutdown(self, reason: str) -> NoReturn:
        log(
            "daemon",
            "critical",
            f"event=forced_shutdown reason={reason} phase={self.phase.value}",
        )
        self.close_control_socket()
        os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="voiced daemon")
    _ = parser.add_argument("--config", help="Config file path")
    _ = parser.add_argument("--model", "-m", choices=list(WHISPER_MODELS + MOONSHINE_MODELS))
    _ = parser.add_argument("--silence-threshold", type=float)
    _ = parser.add_argument("--silence-duration", type=float)
    _ = parser.add_argument("--speech-start-duration", type=float)
    _ = parser.add_argument("--auto-enter", action="store_true")
    _ = parser.add_argument("--keyboard-layout")
    _ = parser.add_argument("--typer-backend", choices=["auto", "dotool", "ydotool"])
    _ = parser.add_argument("--insertion-method", choices=["paste", "type"])
    _ = parser.add_argument("--transcriber-engine", choices=list(TRANSCRIBER_ENGINES))
    _ = parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=None)
    _ = parser.add_argument(
        "--whisper-vad-filter", action=argparse.BooleanOptionalAction, default=None
    )
    _ = parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    config = load_config(cast(str | None, args.config))
    if (value := cast(str | None, args.model)) is not None:
        config["model"] = value
    if (value := cast(float | None, args.silence_threshold)) is not None:
        config["silence_threshold"] = value
    if (value := cast(float | None, args.silence_duration)) is not None:
        config["silence_duration"] = value
    if (value := cast(float | None, args.speech_start_duration)) is not None:
        config["speech_start_duration"] = value
    if cast(bool, args.auto_enter):
        config["auto_enter"] = True
    if (value := cast(str | None, args.keyboard_layout)) is not None:
        config["keyboard_layout"] = value
    if (value := cast(str | None, args.typer_backend)) is not None:
        config["typer_backend"] = value
    if (value := cast(str | None, args.insertion_method)) is not None:
        config["insertion_method"] = value
    if (value := cast(str | None, args.transcriber_engine)) is not None:
        config["transcriber_engine"] = value
    if (value := cast(bool | None, args.streaming)) is not None:
        config["streaming"] = value
    if (value := cast(bool | None, args.whisper_vad_filter)) is not None:
        config["whisper_vad_filter"] = value
    if cast(bool, args.debug):
        config["debug"] = True

    try:
        validate_config(config)
    except ConfigError as error:
        log("config", "error", str(error))
        raise SystemExit(os.EX_CONFIG) from error

    configure_logging(debug=config["debug"])
    try:
        daemon = VoiceDaemon(config)
    except (RuntimeError, ValueError) as error:
        log("config", "error", str(error))
        raise SystemExit(os.EX_CONFIG) from error

    try:
        daemon.run()
    except SystemExit:
        raise
    except BaseException:
        log_exception("daemon", "unhandled daemon exception")
        raise


if __name__ == "__main__":
    main()
