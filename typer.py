"""
Text insertion module for voiced.
Uses dotool for layout-aware keyboard input.
"""

import os
import subprocess
import threading
import time
from pathlib import Path


def detect_keyboard_layout():
    """Auto-detect keyboard layout from system settings."""
    # Try gsettings first (GNOME)
    try:
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.input-sources", "sources"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if "xkb" in output:
                import re

                match = re.search(r"'xkb',\s*'([^'+]+)", output)
                if match:
                    return match.group(1)
    except Exception:
        pass

    # Try localectl
    try:
        result = subprocess.run(["localectl", "status"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "X11 Layout" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass

    return "us"


def find_typer_backend():
    """Find available typing backend. Returns (name, path) or (None, None)."""
    script_dir = Path(__file__).parent

    # Check for dotool in script directory first
    local_dotool = script_dir / "dotool"
    if local_dotool.exists() and os.access(local_dotool, os.X_OK):
        return ("dotool", str(local_dotool))

    # Check PATH for dotool
    for path in os.environ.get("PATH", "").split(":"):
        dotool = Path(path) / "dotool"
        if dotool.exists() and os.access(dotool, os.X_OK):
            return ("dotool", str(dotool))

    # Check for ydotool
    for path in os.environ.get("PATH", "").split(":"):
        ydotool = Path(path) / "ydotool"
        if ydotool.exists() and os.access(ydotool, os.X_OK):
            return ("ydotool", str(ydotool))

    return (None, None)


class Typer:
    def __init__(self, backend="auto", keyboard_layout=None, auto_enter=False, debug=False):
        self.auto_enter = auto_enter
        self.debug = debug
        self.keyboard_layout = keyboard_layout or detect_keyboard_layout()
        self._start_time = None

        # Persistent dotool process
        self._dotool_proc = None
        self._dotool_lock = threading.Lock()

        # Find backend
        if backend == "auto":
            name, path = find_typer_backend()
            if name is None:
                raise RuntimeError("No typing backend found. Install dotool or ydotool.")
            self.backend = name
            self.backend_path = path
        else:
            name, path = find_typer_backend()
            if name != backend:
                raise RuntimeError(f"{backend} not found")
            self.backend = name
            self.backend_path = path

        # Dotool process started lazily on first use

    def _log(self, level, msg):
        if level == "debug" and not self.debug:
            return
        print(f"[typer] [{level}] {msg}", flush=True)

    def _start_dotool(self):
        """Start a new dotool process. Returns True on success."""
        try:
            env = os.environ.copy()
            env["DOTOOL_XKB_LAYOUT"] = self.keyboard_layout

            self._dotool_proc = subprocess.Popen(
                [self.backend_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Send initial config
            self._dotool_proc.stdin.write(b"typedelay 0\ntypehold 0\n")
            self._dotool_proc.stdin.flush()

            self._log("debug", "dotool started")
            return True
        except Exception as e:
            self._log("error", f"failed to start dotool: {e}")
            self._dotool_proc = None
            return False

    def _type_dotool(self, text):
        """Type using persistent dotool process. Returns (total, failed)."""
        with self._dotool_lock:
            # Start process on first use
            if self._dotool_proc is None:
                if not self._start_dotool():
                    raise RuntimeError("dotool not available")

            try:
                # Send type command
                cmd = f"type {text}\n"
                if self.auto_enter:
                    cmd += "key Return\n"

                self._dotool_proc.stdin.write(cmd.encode())
                self._dotool_proc.stdin.flush()

                # Note: We can't easily get per-character failure info with persistent process
                # The warnings go to stderr but we can't read them without blocking
                return (len(text), 0)

            except (BrokenPipeError, OSError) as e:
                # Process died, try once to restart
                self._log("warn", f"dotool pipe error: {e}, restarting...")
                self._dotool_proc = None

                if not self._start_dotool():
                    raise RuntimeError("dotool restart failed")

                # Retry the write
                try:
                    cmd = f"type {text}\n"
                    if self.auto_enter:
                        cmd += "key Return\n"
                    self._dotool_proc.stdin.write(cmd.encode())
                    self._dotool_proc.stdin.flush()
                    return (len(text), 0)
                except (BrokenPipeError, OSError):
                    self._dotool_proc = None
                    raise RuntimeError("dotool failed after restart")

    def _type_ydotool(self, text):
        """Type using ydotool (instant, may have layout issues)."""
        env = os.environ.copy()
        for s in ["/tmp/.ydotool_socket", f"/run/user/{os.getuid()}/.ydotool_socket"]:
            if os.path.exists(s):
                env["YDOTOOL_SOCKET"] = s
                break

        cmd = [self.backend_path, "type", "-d", "0", "-H", "0", "--", text]
        result = subprocess.run(cmd, capture_output=True, timeout=10, env=env)

        if result.returncode != 0:
            raise RuntimeError(f"ydotool failed: {result.stderr.decode()}")

        if self.auto_enter:
            subprocess.run(
                [self.backend_path, "key", "28:1", "28:0"],
                capture_output=True,
                timeout=1,
                env=env,
            )

    def type_text(self, text):
        """Insert text at cursor position. Returns (success, total_chars, failed_chars)."""
        self._start_time = time.time()

        if not text:
            return (False, 0, 0)

        try:
            if self.backend == "dotool":
                total, failed = self._type_dotool(text)
                return (True, total, failed)
            else:
                self._type_ydotool(text)
                return (True, len(text), 0)

        except Exception as e:
            self._log("error", f"type failed: {e}")
            return (False, len(text), len(text))

    def shutdown(self):
        """Clean up persistent dotool process."""
        with self._dotool_lock:
            if self._dotool_proc is not None:
                try:
                    self._dotool_proc.stdin.close()
                    self._dotool_proc.wait(timeout=1)
                except Exception:
                    self._dotool_proc.kill()
                self._dotool_proc = None
                self._log("debug", "dotool process stopped")
