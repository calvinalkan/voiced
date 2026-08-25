"""
Audio recording module for voiced.
Handles microphone input and silence detection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from typing import cast, final

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from voiced_logging import LogLevel, log

# Type alias for audio arrays - float32 audio samples
AudioArray = NDArray[np.float32]


class InputDeviceError(RuntimeError):
    """The configured or default microphone cannot be opened for recording."""


@final
class CaptureMonitor:
    """Expose capture progress to the daemon without transferring stream ownership."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream_started_monotonic: float | None = None
        self._last_callback_monotonic: float | None = None

    def mark_stream_started(self, monotonic_time: float) -> None:
        with self._lock:
            self._stream_started_monotonic = monotonic_time
            self._last_callback_monotonic = monotonic_time

    def mark_callback(self, monotonic_time: float) -> None:
        with self._lock:
            self._last_callback_monotonic = monotonic_time

    def snapshot(self) -> tuple[float | None, float | None]:
        with self._lock:
            return self._stream_started_monotonic, self._last_callback_monotonic


def get_terminal_width() -> int:
    """Get terminal width, default to 80 if unavailable."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


class Audio:
    sample_rate: int
    silence_threshold: float
    silence_duration: float
    speech_start_duration: float
    input_device: str | None
    debug: bool
    is_tty: bool
    pre_buffer_samples: int
    _start_time: float | None
    _waveform_history: list[str]

    def __init__(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.6,
        sample_rate: int = 16000,
        speech_start_duration: float = 0.2,
        input_device: str | None = None,
        debug: bool = False,
        is_tty: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.speech_start_duration = speech_start_duration  # Sustained speech needed to start
        self.input_device = input_device
        self.debug = debug
        self.is_tty = is_tty
        self._start_time = None
        self._waveform_history = []

        # Pre-buffer to catch start of speech (1 second)
        self.pre_buffer_samples = sample_rate

    def _log(self, level: LogLevel, message: str) -> None:
        """Log a message with audio namespace."""
        # Clear any in-progress live display line first
        if self.is_tty:
            width = get_terminal_width() - 1
            _ = sys.stdout.write(f"\r{' ' * width}\r")
            _ = sys.stdout.flush()
        log("audio", level, message)

    def _level_to_bar(self, level: float) -> str:
        """Convert audio level to waveform character, scaled to threshold."""
        bars = "▁▂▃▄▅▆▇█"
        # Scale so threshold = ▅ (index 4), giving headroom for speech above
        scale = self.silence_threshold / 0.6  # threshold at 60% of bar height
        normalized = min(level / scale, 1.0) if scale > 0 else 0
        idx = int(normalized * (len(bars) - 1))
        return bars[idx]

    def _format_live_display(
        self,
        level: float,
        is_speech: bool,
        is_recording: bool,
        speech_samples: int,
        silence_samples: int,
        elapsed: float,
    ) -> str:
        """Format the live-updating display line (TTY only)."""
        # Add to waveform history
        self._waveform_history.append(self._level_to_bar(level))
        if len(self._waveform_history) > 40:
            _ = self._waveform_history.pop(0)

        waveform = "".join(self._waveform_history).ljust(40)

        # Compact level display
        cmp = ">" if is_speech else "<"
        level_info = f"{level:.3f}{cmp}{self.silence_threshold:.3f}"

        if is_recording:
            # Recording state: show silence countdown
            silence_sec = silence_samples / self.sample_rate
            silence_left = max(0, self.silence_duration - silence_sec)
            if silence_sec > 0.05:
                state = f"● REC {elapsed:.1f}s pause:{silence_left:.1f}s"
            else:
                state = f"● REC {elapsed:.1f}s speaking"
        else:
            # Waiting state: show speech progress
            speech_sec = speech_samples / self.sample_rate
            if is_speech:
                # Building up speech - show progress bar
                speech_left = max(0, self.speech_start_duration - speech_sec)
                pct = min(speech_sec / self.speech_start_duration, 1.0)
                filled = int(pct * 5)
                progress = "▓" * filled + "░" * (5 - filled)
                state = f"◆ [{progress}] {speech_left:.2f}s→rec"
            else:
                state = "○ waiting..."

        return f"{waveform} {level_info} {state}"

    def to_file(self, audio: AudioArray, path: str) -> None:
        """
        Save audio to a WAV file.

        Args:
            audio: numpy array of audio samples (float32)
            path: path to save WAV file

        Raises:
            OSError: if file cannot be written
        """
        try:
            # Convert float32 [-1.0, 1.0] to int16. Clamp first so overloaded
            # input gain does not wrap around when saving diagnostics.
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

            with wave.open(path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)  # 16-bit
                f.setframerate(self.sample_rate)
                f.writeframes(audio_int16.tobytes())

            self._log("debug", f"saved audio to: {path}")
        except OSError as e:
            self._log("error", f"failed to save audio to {path}: {e}")
            raise

    def from_file(self, path: str) -> AudioArray:
        """
        Load audio from a WAV file.

        Args:
            path: path to WAV file

        Returns:
            numpy array of audio samples (float32), resampled to self.sample_rate

        Raises:
            FileNotFoundError: if file doesn't exist
            ValueError: if file format is invalid or unsupported
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            with wave.open(path, "rb") as f:
                if f.getnchannels() not in (1, 2):
                    raise ValueError(f"Unsupported channel count: {f.getnchannels()}")
                if f.getsampwidth() not in (1, 2, 4):
                    raise ValueError(f"Unsupported sample width: {f.getsampwidth()}")

                frames = f.readframes(f.getnframes())
                file_rate = f.getframerate()
                sample_width = f.getsampwidth()
                channels = f.getnchannels()
        except wave.Error as e:
            raise ValueError(f"Invalid WAV file: {e}") from e

        # Convert bytes to numpy array based on sample width
        audio: AudioArray
        if sample_width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert stereo to mono by averaging channels
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.float32)  # pyright: ignore[reportAny]

        # Resample if needed (simple linear interpolation)
        if file_rate != self.sample_rate:
            duration = len(audio) / file_rate
            new_length = int(duration * self.sample_rate)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        return audio

    def _select_input_device(self) -> None:
        if self.input_device is None:
            if self.debug:
                self._log("debug", "using system default input device")
            return

        selection_start = time.monotonic()
        try:
            result = subprocess.run(
                ["pactl", "--format=json", "list", "sources"],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            )
            sources = cast(list[dict[str, object]], json.loads(result.stdout))
        except FileNotFoundError as e:
            raise InputDeviceError(
                "cannot select a configured microphone because pactl is unavailable"
            ) from e
        except (subprocess.SubprocessError, json.JSONDecodeError, TypeError) as e:
            raise InputDeviceError(f"could not list PipeWire input devices: {e}") from e

        input_sources: list[tuple[str, str]] = []
        for source in sources:
            name = source.get("name")
            description = source.get("description")
            if isinstance(name, str) and not name.endswith(".monitor"):
                input_sources.append(
                    (name, description if isinstance(description, str) else name)
                )

        if self.debug:
            available = "; ".join(
                f"{description} ({name})" for name, description in input_sources
            ) or "none"
            self._log("debug", f"available input devices: {available}")

        configured_name = self.input_device.casefold()
        matches = [
            (name, description)
            for name, description in input_sources
            if configured_name in name.casefold()
            or configured_name in description.casefold()
        ]
        if not matches:
            raise InputDeviceError(
                f"configured microphone {self.input_device!r} is unavailable"
            )
        if len(matches) > 1:
            matched_names = "; ".join(
                f"{description} ({name})" for name, description in matches
            )
            raise InputDeviceError(
                f"configured microphone {self.input_device!r} is ambiguous; matched {matched_names}"
            )

        source_name, description = matches[0]
        try:
            _ = subprocess.run(
                ["pactl", "set-default-source", source_name],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            )
        except subprocess.SubprocessError as e:
            raise InputDeviceError(
                f"could not select configured microphone {self.input_device!r}: {e}"
            ) from e

        if self.debug:
            elapsed_ms = int((time.monotonic() - selection_start) * 1000)
            self._log(
                "debug",
                f"using configured input device: {description} ({source_name}), selected in {elapsed_ms}ms",
            )

    def record(
        self,
        stop_event: threading.Event | None = None,
        auto_stop_on_silence: bool = True,
        on_chunk: Callable[[AudioArray], None] | None = None,
        test_input: str | None = None,
        save_audio: str | None = None,
        monitor: CaptureMonitor | None = None,
        test_capture_fault: str | None = None,
    ) -> AudioArray | None:
        """Capture one recording and abort the input stream when capture ends.

        `monitor` reports callback and teardown progress but does not transfer
        stream ownership. The caller must terminate the process when this
        operation does not return after a requested stop because Python cannot
        safely kill a thread blocked inside PortAudio.
        """
        if test_capture_fault not in (None, "stall", "hang"):
            raise ValueError("test_capture_fault must be 'stall' or 'hang'")
        if test_capture_fault is not None:
            # Integration tests enqueue fixture audio before simulating either
            # a stream that closes on request or a native owner that stays stuck.
            fixture_audio = self.from_file(test_input) if test_input else None
            if fixture_audio is not None and on_chunk:
                on_chunk(fixture_audio)
            if monitor:
                now = time.monotonic()
                monitor.mark_stream_started(now)
                monitor.mark_callback(now)
            if test_capture_fault == "stall":
                wait_event = stop_event or threading.Event()
                _ = wait_event.wait()
                return fixture_audio
            while True:
                time.sleep(1)

        if test_input:
            audio = self.from_file(test_input)
            if on_chunk and len(audio) > 0:
                on_chunk(audio)
            if save_audio:
                self.to_file(audio, save_audio)
            return audio

        self._select_input_device()
        block_samples = int(self.sample_rate * 0.1)
        pre_buffer_chunks_count = max(1, self.pre_buffer_samples // block_samples)
        pre_buffer: deque[AudioArray] = deque(maxlen=pre_buffer_chunks_count)
        audio_chunks: list[AudioArray] = []
        is_recording = False
        silence_samples = 0
        speech_samples = 0
        silence_samples_needed = int(self.silence_duration * self.sample_rate)
        speech_samples_needed = int(self.speech_start_duration * self.sample_rate)
        done = False
        self._start_time = time.time()
        self._waveform_history = []
        expected_callback_interval = 0.1
        callback_status_counts: dict[str, int] = {}
        delayed_callbacks = 0
        max_callback_gap = 0.0
        last_callback_time: float | None = None
        recording_wall_start: list[float] = []

        def callback(
            indata: AudioArray,
            _frames: int,
            _time_info: object,
            status: object,
        ) -> None:
            nonlocal is_recording, silence_samples, speech_samples, done
            nonlocal delayed_callbacks, last_callback_time, max_callback_gap

            if done or wait_event.is_set():
                return

            now = time.monotonic()
            if monitor:
                monitor.mark_callback(now)
            if last_callback_time is not None:
                gap = now - last_callback_time
                if gap > expected_callback_interval * 2.5:
                    delayed_callbacks += 1
                    max_callback_gap = max(max_callback_gap, gap)
            last_callback_time = now

            status_text = str(status)
            if status_text:
                callback_status_counts[status_text] = callback_status_counts.get(status_text, 0) + 1

            # PortAudio reuses `indata` after this callback. One float32 copy
            # preserves the chunk without allocating Python float objects.
            audio: AudioArray = indata[:, 0].copy()
            level = float(np.abs(audio).mean())  # pyright: ignore[reportAny]
            is_speech = level > self.silence_threshold

            if self.debug and self.is_tty and self._start_time is not None:
                elapsed = time.time() - self._start_time
                line = self._format_live_display(
                    level, is_speech, is_recording, speech_samples, silence_samples, elapsed
                )
                width = get_terminal_width() - 1
                _ = sys.stdout.write(f"\r{line:<{width}}")
                _ = sys.stdout.flush()

            if not is_recording:
                pre_buffer.append(audio)
                if not is_speech:
                    speech_samples = 0
                    return

                speech_samples += len(audio)
                if speech_samples < speech_samples_needed:
                    return

                is_recording = True
                recording_wall_start.append(now)
                seed = np.concatenate(tuple(pre_buffer))
                pre_buffer.clear()
                audio_chunks.append(seed)
                if self.debug:
                    self._log("debug", "speech detected, recording started")
                if on_chunk:
                    on_chunk(seed)
                return

            audio_chunks.append(audio)
            if on_chunk:
                on_chunk(audio)
            if is_speech:
                silence_samples = 0
                return

            silence_samples += len(audio)
            if auto_stop_on_silence and silence_samples >= silence_samples_needed:
                if self.debug:
                    self._log("debug", f"silence detected ({self.silence_duration}s), stopping")
                done = True

        wait_event = stop_event or threading.Event()
        stream: sd.InputStream | None = None
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=block_samples,
                latency="high",
                callback=callback,
            )
            stream.start()
            if monitor:
                monitor.mark_stream_started(time.monotonic())

            while not done:
                if wait_event.wait(0.05):
                    break
        except sd.PortAudioError as error:
            device_description = (
                repr(self.input_device)
                if self.input_device is not None
                else "the system default microphone"
            )
            raise InputDeviceError(f"could not open {device_description}: {error}") from error
        finally:
            if stream is not None:
                self._log("debug", "aborting input stream")
                try:
                    stream.abort(ignore_errors=False)
                except sd.PortAudioError as error:
                    self._log("warn", f"input stream abort failed: {error}")
                self._log("debug", "closing input stream")
                try:
                    stream.close(ignore_errors=False)
                except sd.PortAudioError as error:
                    self._log("warn", f"input stream close failed: {error}")
                self._log("debug", "input stream closed")

        if self.debug and self.is_tty:
            width = get_terminal_width() - 1
            _ = sys.stdout.write(f"\r{' ' * width}\r")
            _ = sys.stdout.flush()

        result = np.concatenate(audio_chunks) if audio_chunks else None
        if result is not None:
            audio_duration = len(result) / self.sample_rate
            if recording_wall_start:
                recording_wall_duration = time.monotonic() - recording_wall_start[0]
                if audio_duration + 0.25 < recording_wall_duration:
                    self._log(
                        "warn",
                        f"captured audio is shorter than recording wall time ({audio_duration:.1f}s audio vs {recording_wall_duration:.1f}s wall); CPU/audio scheduling may have dropped input",
                    )
            if delayed_callbacks:
                self._log(
                    "warn",
                    f"audio callback delayed {delayed_callbacks} times (max gap {max_callback_gap:.2f}s); CPU load may have starved recording",
                )
            for status_text, count in callback_status_counts.items():
                self._log("warn", f"audio callback status: {status_text} ({count}x)")

        if save_audio:
            self._log("debug", f"save_audio set: {save_audio}")
            if result is not None:
                self.to_file(result, save_audio)

        return result
