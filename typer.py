"""
Text insertion module for voiced.

Inserts dictated text via paste-first (wl-clipboard + Ctrl+Shift+V), falling
back to layout-aware keyboard typing through dotool when paste is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path


def detect_keyboard_layout() -> str:
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


def find_typer_backend() -> tuple[str | None, str | None]:
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
    auto_enter: bool
    debug: bool
    keyboard_layout: str
    backend: str
    backend_path: str
    insertion_method: str
    paste_keybind: str
    _start_time: float | None
    _dotool_proc: subprocess.Popen[bytes] | None
    _dotool_lock: threading.Lock
    _copy_to_clipboard: Callable[[str], bool]
    _notify: Callable[[str], None]

    def __init__(
        self,
        backend: str = "auto",
        keyboard_layout: str | None = None,
        auto_enter: bool = False,
        debug: bool = False,
        insertion_method: str = "paste",
        paste_keybind: str = "ctrl+shift+v",
        copy_to_clipboard: Callable[[str], bool] | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.auto_enter = auto_enter
        self.debug = debug
        self.keyboard_layout = keyboard_layout or detect_keyboard_layout()
        self.paste_keybind = paste_keybind
        self._copy_to_clipboard = copy_to_clipboard or (lambda _t: False)
        self._notify = notify or (lambda _m: None)
        self._start_time = None
        self._dotool_proc = None
        self._dotool_lock = threading.Lock()

        # Find backend
        if backend == "auto":
            name, path = find_typer_backend()
            if name is None or path is None:
                raise RuntimeError("No typing backend found. Install dotool or ydotool.")
            self.backend = name
            self.backend_path = path
        else:
            name, path = find_typer_backend()
            if name != backend:
                raise RuntimeError(f"{backend} not found")
            if name is None or path is None:
                raise RuntimeError(f"{backend} not found")
            self.backend = name
            self.backend_path = path

        # Paste support is dotool-only: ydotool's key API takes raw evdev
        # keycodes which would need a separate keymap translation layer.
        if insertion_method == "paste" and self.backend != "dotool":
            self._log(
                "warn",
                f"paste not implemented for {self.backend}, using type insertion",
            )
            insertion_method = "type"
        self.insertion_method = insertion_method

    def _log(self, level: str, msg: str) -> None:
        if level == "debug" and not self.debug:
            return
        print(f"[typer] [{level}] {msg}", flush=True)

    def _start_dotool(self) -> bool:
        """Start a new dotool process. Returns True on success."""
        try:
            env = os.environ.copy()
            env["DOTOOL_XKB_LAYOUT"] = self.keyboard_layout

            # stdout/stderr to DEVNULL: dotool emits warnings (e.g. unmapped
            # keysyms) to stderr; with PIPE the buffer fills over a long-lived
            # session and dotool blocks on its next stderr write, freezing input.
            self._dotool_proc = subprocess.Popen(
                [self.backend_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )

            self._log("debug", "dotool started")
            return True
        except Exception as e:
            self._log("error", f"failed to start dotool: {e}")
            self._dotool_proc = None
            return False

    def _send_dotool(self, cmd: str) -> bool:
        """Write a command to dotool stdin. Reconnects once on broken pipe."""
        with self._dotool_lock:
            if self._dotool_proc is None:
                if not self._start_dotool():
                    return False

            for attempt in range(2):
                try:
                    if self._dotool_proc and self._dotool_proc.stdin:
                        _ = self._dotool_proc.stdin.write(cmd.encode())
                        self._dotool_proc.stdin.flush()
                    return True
                except (BrokenPipeError, OSError) as e:
                    if attempt == 0:
                        self._log("warn", f"dotool pipe error: {e}, restarting")
                        self._dotool_proc = None
                        if not self._start_dotool():
                            return False
                        continue
                    self._log("error", f"dotool failed after restart: {e}")
                    self._dotool_proc = None
                    return False
            return False

    def _try_paste(self, text: str) -> bool:
        """Copy to clipboard and send paste keybind. All-or-nothing."""
        if not self._copy_to_clipboard(text):
            return False
        # wl-copy forks to background after acquiring the selection, but the
        # compositor's selection handover takes a few ms; a brief settle keeps
        # the paste keystroke from racing the clipboard owner change.
        time.sleep(0.05)
        cmd = f"key {self.paste_keybind}\n"
        if self.auto_enter:
            cmd += "key enter\n"
        return self._send_dotool(cmd)

    def _type_dotool(self, text: str) -> bool:
        """Type via dotool at a paced rate. Used as fallback when paste fails."""
        # 4ms typedelay + 4ms typehold ≈ 125 keys/sec. Faster rates drop
        # characters mid-stream (kernel uinput overflow) or trigger Wayland
        # EPIPE crashes in apps like Zed under sustained synthetic input.
        cmd = f"typedelay 4\ntypehold 4\ntype {text}\n"
        if self.auto_enter:
            cmd += "key enter\n"
        return self._send_dotool(cmd)

    def _type_ydotool(self, text: str) -> bool:
        """Type via ydotool (no layout awareness, same paced rate)."""
        env = os.environ.copy()
        for s in ["/tmp/.ydotool_socket", f"/run/user/{os.getuid()}/.ydotool_socket"]:
            if os.path.exists(s):
                env["YDOTOOL_SOCKET"] = s
                break

        try:
            result = subprocess.run(
                [self.backend_path, "type", "-d", "4", "-H", "4", "--", text],
                capture_output=True,
                timeout=15,
                env=env,
            )
            if result.returncode != 0:
                self._log("error", f"ydotool failed: {result.stderr.decode()}")
                return False
            if self.auto_enter:
                _ = subprocess.run(
                    [self.backend_path, "key", "28:1", "28:0"],
                    capture_output=True,
                    timeout=1,
                    env=env,
                )
            return True
        except subprocess.TimeoutExpired:
            self._log("error", "ydotool timed out after 15s")
            return False

    def insert_text(self, text: str) -> tuple[bool, str]:
        """Insert text at cursor. Returns (success, method) where method is
        'paste' or 'type'. In paste mode, falls back to typing on failure."""
        self._start_time = time.time()
        if not text:
            return (False, "none")

        if self.insertion_method == "paste":
            if self._try_paste(text):
                return (True, "paste")
            self._log("warn", "paste failed, falling back to type")
            self._notify("voiced: paste failed, typing instead")

        if self.backend == "dotool":
            return (self._type_dotool(text), "type")
        return (self._type_ydotool(text), "type")

    def shutdown(self) -> None:
        """Clean up persistent dotool process."""
        with self._dotool_lock:
            if self._dotool_proc is not None:
                try:
                    if self._dotool_proc.stdin:
                        self._dotool_proc.stdin.close()
                    _ = self._dotool_proc.wait(timeout=1)
                except Exception:
                    self._dotool_proc.kill()
                self._dotool_proc = None
                self._log("debug", "dotool process stopped")
