"""
Transcription module for voiced.
Wraps faster-whisper for speech-to-text.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from numpy.typing import NDArray


class Transcriber:
    model_size: str
    device: str
    language: str
    model: WhisperModel | None
    debug: bool

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = "en",
        debug: bool = False,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self.debug = debug

    def load(self, on_progress: Callable[[str], None] | None = None) -> None:
        """Load the model into memory."""
        compute_type = "int8" if self.device == "cpu" else "float16"

        if on_progress:
            on_progress(f"Loading whisper-{self.model_size} ({compute_type})...")

        self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)

        # Warm up
        if on_progress:
            on_progress("Warming up...")
        dummy = np.zeros(16000, dtype=np.float32)
        _ = list(self.model.transcribe(dummy, language=self.language))  # pyright: ignore[reportUnknownMemberType]

        if on_progress:
            on_progress("Ready")

    def transcribe(self, audio_data: NDArray[np.float32]) -> str | None:
        """
        Transcribe audio data to text.

        Args:
            audio_data: numpy array of audio samples (float32, 16kHz)

        Returns:
            Transcribed text string, or None if no speech detected
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(audio_data) < 8000:
            return None

        segments, info = self.model.transcribe(  # pyright: ignore[reportUnknownMemberType]
            audio_data,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 200},
        )

        segments_list = list(segments)

        if self.debug:
            print(
                f"[transcriber] [debug] language={info.language} (prob={info.language_probability:.2f}), segments={len(segments_list)}",
                flush=True,
            )

        text = " ".join(seg.text.strip() for seg in segments_list if seg.text.strip())

        return text if text else None
