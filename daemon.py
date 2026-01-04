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

from audio import Audio
from transcriber import Transcriber
from typer import Typer

# File paths
PID_FILE = Path("/tmp/voiced.pid")
COMMAND_FILE = Path("/tmp/voiced-command")
READY_FILE = Path("/tmp/voiced-ready")
CONFIG_DIR = Path.home() / ".config" / "voiced"
CONFIG_FILE = CONFIG_DIR / "config.json"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "voiced"
HISTORY_FILE = STATE_DIR / "history.json"

DEFAULT_CONFIG = {
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


def log(component, level, message):
    """Unified logging: [component] [level] message"""
    print(f"[{component}] [{level}] {message}", flush=True)


def copy_to_clipboard(text):
    """Copy text to clipboard using wl-copy (Wayland). Fire and forget."""
    try:
        proc = subprocess.Popen(
            ["wl-copy", "--"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(text.encode())
        proc.stdin.close()
        log("clipboard", "info", f"started: {len(text)} chars")
    except FileNotFoundError:
        log("clipboard", "warn", "wl-copy not found, skipping clipboard")
    except Exception as e:
        log("clipboard", "warn", f"clipboard copy failed: {e}")


def notify(message, urgency="normal", timeout=2000):
    """Send desktop notification (non-blocking). Returns notification ID."""
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


def notify_close(notify_id):
    """Close a notification by ID."""
    if notify_id is None:
        return
    try:
        subprocess.run(
            ["notify-send", "-r", str(notify_id), "-t", "1", ""],
            capture_output=True,
            timeout=1,
        )
    except Exception:
        pass


def load_config(config_path=None):
    """Load configuration from file, with defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                config.update(json.load(f))
        except Exception as e:
            log("config", "warn", f"failed to load {config_path}: {e}")
    elif CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config.update(json.load(f))
        except Exception as e:
            log("config", "warn", f"failed to load {CONFIG_FILE}: {e}")

    return config


def load_history():
    """Load history from file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception as e:
        log("history", "warn", f"failed to load: {e}")
        return []


def save_history(history):
    """Save history to file."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except OSError as e:
        log("history", "warn", f"failed to save: {e}")


def add_to_history(text, max_size):
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


def write_pid():
    PID_FILE.write_text(str(os.getpid()))


def remove_pid():
    PID_FILE.unlink(missing_ok=True)


def read_last_command():
    """Read the last JSON line from command file."""
    if not COMMAND_FILE.exists():
        return None
    try:
        lines = COMMAND_FILE.read_text().strip().split("\n")
        if lines:
            return json.loads(lines[-1])
    except Exception as e:
        log("command", "warn", f"failed to parse: {e}")
    return None


class VoiceDaemon:
    def __init__(self, config):
        self.config = config
        self.debug = config.get("debug", False)
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

    def handle_signal(self, signum, frame):
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
                    threading.Thread(
                        target=lambda ti=test_input, to=test_output, sa=save_audio: self.do_record(
                            auto_stop=True, test_input=ti, test_output=to, save_audio=sa
                        ),
                        daemon=True,
                    ).start()
                else:
                    log("daemon", "warn", "already recording, ignoring")

            elif action == "record":
                if not self.recording:
                    threading.Thread(
                        target=lambda ti=test_input, to=test_output, sa=save_audio: self.do_record(
                            auto_stop=False, test_input=ti, test_output=to, save_audio=sa
                        ),
                        daemon=True,
                    ).start()
                else:
                    log("daemon", "warn", "already recording, ignoring")

            elif action == "record-toggle":
                if not self.recording:
                    threading.Thread(
                        target=lambda ti=test_input, to=test_output, sa=save_audio: self.do_record(
                            auto_stop=False, test_input=ti, test_output=to, save_audio=sa
                        ),
                        daemon=True,
                    ).start()
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

    def do_record(self, auto_stop=True, test_input=None, test_output=None, save_audio=None):
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

    def process_audio(self, audio, test_output=None):
        if audio is None or len(audio) < 8000:
            log("audio", "info", "no speech detected")
            notify("No speech detected")
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
            notify("No speech detected")
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
                    f.write(text)
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
            notify(f"Typing incomplete: {failed} chars failed - check history")
        else:
            log("typer", "error", "failed (saved to history)")
            notify("Failed to type - use 'voiced history' to recover")

    def run(self):
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

        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

        READY_FILE.touch()

        print("-" * 50, flush=True)
        log("daemon", "info", f"ready (pid: {os.getpid()})")
        print("=" * 50, flush=True)

        while self.running:
            time.sleep(0.1)

        # Clean up
        self.typer.shutdown()
        log("daemon", "info", "shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="voiced daemon")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--model", "-m", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--silence-threshold", type=float)
    parser.add_argument("--silence-duration", type=float)
    parser.add_argument(
        "--speech-start-duration",
        type=float,
        help="Sustained speech needed to start (default: 0.2s)",
    )
    parser.add_argument("--auto-enter", action="store_true")
    parser.add_argument(
        "--keyboard-layout", help="Keyboard layout (e.g., de, us). Auto-detected if not set."
    )
    parser.add_argument(
        "--typer-backend", choices=["auto", "dotool", "ydotool"], help="Typing backend"
    )
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["model"] = args.model
    if args.silence_threshold:
        config["silence_threshold"] = args.silence_threshold
    if args.silence_duration:
        config["silence_duration"] = args.silence_duration
    if args.speech_start_duration:
        config["speech_start_duration"] = args.speech_start_duration
    if args.auto_enter:
        config["auto_enter"] = True
    if args.keyboard_layout:
        config["keyboard_layout"] = args.keyboard_layout
    if args.typer_backend:
        config["typer_backend"] = args.typer_backend
    if args.debug:
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
