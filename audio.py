"""
Audio recording module for voiced.
Handles microphone input and silence detection.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import wave
from collections.abc import Callable
from typing import cast

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Type alias for audio arrays - float32 audio samples
AudioArray = NDArray[np.float32]


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
    channels: int
    debug: bool
    is_tty: bool
    pre_buffer_seconds: float
    pre_buffer_samples: int
    _start_time: float | None
    _waveform_history: list[str]

    def __init__(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.6,
        sample_rate: int = 16000,
        speech_start_duration: float = 0.2,
        debug: bool = False,
        is_tty: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.speech_start_duration = speech_start_duration  # Sustained speech needed to start
        self.channels = 1
        self.debug = debug
        self.is_tty = is_tty
        self._start_time = None
        self._waveform_history = []

        # Pre-buffer to catch start of speech (1 second)
        self.pre_buffer_seconds = 1.0
        self.pre_buffer_samples = int(self.pre_buffer_seconds * sample_rate)

    def _elapsed_ms(self) -> int:
        """Get elapsed milliseconds since recording started."""
        if self._start_time is None:
            return 0
        return int((time.time() - self._start_time) * 1000)

    def _log(self, level: str, message: str) -> None:
        """Log a message with audio namespace."""
        # Clear any in-progress live display line first
        if self.is_tty:
            width = get_terminal_width() - 1
            _ = sys.stdout.write(f"\r{' ' * width}\r")
            _ = sys.stdout.flush()
        print(f"[audio] [{level}] {message}", flush=True)

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
            # Convert float32 [-1.0, 1.0] to int16
            audio_int16 = (audio * 32767).astype(np.int16)

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

    def record(
        self,
        stop_event: threading.Event | None = None,
        auto_stop_on_silence: bool = True,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        test_input: str | None = None,
        save_audio: str | None = None,
    ) -> AudioArray | None:
        """
        Record audio from microphone.

        If test_input is provided, reads from that file instead of recording
        from microphone (short-circuits for testing).

        Stops when:
        - stop_event.is_set() (if provided)
        - OR silence detected for silence_duration (if auto_stop_on_silence=True)

        Args:
            stop_event: threading.Event to signal stop (optional)
            auto_stop_on_silence: if True, stop after silence_duration of quiet
            on_start: callback when speech first detected
            on_stop: callback when recording stops
            test_input: path to WAV file to use instead of microphone
            save_audio: path to save recorded audio as WAV file

        Returns:
            numpy array of audio samples (float32)
        """
        # Short-circuit: if test_input is provided, read from file instead of mic
        if test_input:
            return self.from_file(test_input)
        audio_buffer: list[float] = []
        pre_buffer: list[float] = []
        is_recording = False
        silence_samples = 0
        speech_samples = 0  # Track sustained speech before recording starts
        silence_samples_needed = int(self.silence_duration * self.sample_rate)
        speech_samples_needed = int(self.speech_start_duration * self.sample_rate)
        done = False
        self._start_time = time.time()
        self._waveform_history = []

        def callback(
            indata: AudioArray,
            _frames: int,
            _time_info: object,
            _status: object,
        ) -> None:
            nonlocal audio_buffer, pre_buffer, is_recording, silence_samples, speech_samples, done

            if done:
                return

            audio: AudioArray = indata[:, 0].copy()
            audio_list: list[float] = cast(list[float], audio.tolist())

            # Maintain pre-buffer
            pre_buffer.extend(audio_list)
            if len(pre_buffer) > self.pre_buffer_samples:
                pre_buffer = pre_buffer[-self.pre_buffer_samples :]

            level = float(np.abs(audio).mean())  # pyright: ignore[reportAny]
            is_speech = level > self.silence_threshold

            # Live display (only in TTY mode with debug) - updates every callback (~100ms)
            if self.debug and self.is_tty and self._start_time is not None:
                elapsed = time.time() - self._start_time
                line = self._format_live_display(
                    level, is_speech, is_recording, speech_samples, silence_samples, elapsed
                )
                # Carriage return + padded line to overwrite previous
                width = get_terminal_width() - 1
                _ = sys.stdout.write(f"\r{line:<{width}}")
                _ = sys.stdout.flush()

            if is_speech:
                if not is_recording:
                    speech_samples += len(audio)
                    # Only start recording after sustained speech
                    if speech_samples >= speech_samples_needed:
                        is_recording = True
                        audio_buffer = pre_buffer.copy()
                        if self.debug:
                            self._log("debug", "speech detected, recording started\n")
                        if on_start:
                            on_start()
                else:
                    audio_buffer.extend(audio_list)
                    silence_samples = 0
            else:
                # Reset speech counter if audio drops below threshold
                if not is_recording:
                    speech_samples = 0

                if is_recording:
                    audio_buffer.extend(audio_list)
                    silence_samples += len(audio)

                    # Auto-stop on silence (only if enabled)
                    if auto_stop_on_silence and silence_samples >= silence_samples_needed:
                        if self.debug:
                            self._log(
                                "debug", f"silence detected ({self.silence_duration}s), stopping"
                            )
                        done = True
                        if on_stop:
                            on_stop()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=int(self.sample_rate * 0.1),
            callback=callback,
        ):
            while not done:
                # Check for external stop signal
                if stop_event and stop_event.is_set():
                    if on_stop:
                        on_stop()
                    break
                sd.sleep(50)  # pyright: ignore[reportUnknownMemberType]

        # Clear live display line when done
        if self.debug and self.is_tty:
            width = get_terminal_width() - 1
            _ = sys.stdout.write(f"\r{' ' * width}\r")
            _ = sys.stdout.flush()

        result: AudioArray | None = (
            np.array(audio_buffer, dtype=np.float32) if audio_buffer else None
        )

        # Save audio to file if save_audio is set
        if save_audio:
            self._log("debug", f"save_audio set: {save_audio}")
            if result is not None:
                self.to_file(result, save_audio)

        return result
