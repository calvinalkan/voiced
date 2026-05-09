"""
Transcription module for voiced.

Two engines, same input contract (numpy float32 mono @ 16kHz → str | None):

- Whisper via faster-whisper (CTranslate2): broad multilingual coverage,
  slower on CPU, prone to silence hallucinations.
- Moonshine: fast on CPU, English-focused, hallucinates less in silence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from moonshine_voice.transcriber import Transcriber as _MoonshineCoreTranscriber


WHISPER_MODELS: tuple[str, ...] = ("tiny", "base", "small", "medium", "large-v3")
MOONSHINE_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "tiny-streaming",
    "small-streaming",
    "base-streaming",
    "medium-streaming",
)
TRANSCRIBER_ENGINES: tuple[str, ...] = ("whisper", "moonshine")


class TranscriberProtocol(Protocol):
    debug: bool

    def load(self, on_progress: Callable[[str], None] | None = None) -> None: ...
    def transcribe(self, audio_data: NDArray[np.float32]) -> str | None: ...


class WhisperTranscriber:
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
        if model_size not in WHISPER_MODELS:
            raise ValueError(
                f"whisper model must be one of {WHISPER_MODELS}, got '{model_size}'"
            )
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self.debug = debug

    def load(self, on_progress: Callable[[str], None] | None = None) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        compute_type = "int8" if self.device == "cpu" else "float16"

        if on_progress:
            on_progress(f"Loading whisper-{self.model_size} ({compute_type})...")

        self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)

        if on_progress:
            on_progress("Warming up...")
        dummy = np.zeros(16000, dtype=np.float32)
        _ = list(self.model.transcribe(dummy, language=self.language))  # pyright: ignore[reportUnknownMemberType]

        if on_progress:
            on_progress("Ready")

    def transcribe(self, audio_data: NDArray[np.float32]) -> str | None:
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


_MOONSHINE_ARCH_BY_NAME: dict[str, str] = {
    "tiny": "TINY",
    "base": "BASE",
    "tiny-streaming": "TINY_STREAMING",
    "small-streaming": "SMALL_STREAMING",
    "base-streaming": "BASE_STREAMING",
    "medium-streaming": "MEDIUM_STREAMING",
}


class MoonshineTranscriber:
    model_size: str
    language: str
    debug: bool
    transcriber: _MoonshineCoreTranscriber | None

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        debug: bool = False,
    ) -> None:
        if model_size not in MOONSHINE_MODELS:
            raise ValueError(
                f"moonshine model must be one of {MOONSHINE_MODELS}, got '{model_size}'"
            )
        self.model_size = model_size
        self.language = language
        self.debug = debug
        self.transcriber = None

    def load(self, on_progress: Callable[[str], None] | None = None) -> None:
        from moonshine_voice import ModelArch, get_model_for_language
        from moonshine_voice.transcriber import Transcriber as MoonshineCoreTranscriber

        arch_name = _MOONSHINE_ARCH_BY_NAME[self.model_size]
        arch = ModelArch[arch_name]

        if on_progress:
            on_progress(f"Loading moonshine-{self.model_size} ({self.language})...")

        model_path, _ = get_model_for_language(self.language, arch)
        self.transcriber = MoonshineCoreTranscriber(model_path=model_path, model_arch=arch)

        if on_progress:
            on_progress("Warming up...")
        # 1 second of silence to prime the runtime; Moonshine is fine with all-zero input
        dummy = [0.0] * 16000
        _ = self.transcriber.transcribe_without_streaming(dummy, sample_rate=16000)

        if on_progress:
            on_progress("Ready")

    def transcribe(self, audio_data: NDArray[np.float32]) -> str | None:
        if self.transcriber is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(audio_data) < 8000:
            return None

        # Moonshine's C API takes a Python list of floats, not numpy
        audio_list = cast(list[float], audio_data.tolist())
        transcript = self.transcriber.transcribe_without_streaming(
            audio_list, sample_rate=16000
        )

        if self.debug:
            latencies = [
                line.last_transcription_latency_ms for line in transcript.lines
            ]
            print(
                f"[transcriber] [debug] moonshine lines={len(transcript.lines)}, latencies_ms={latencies}",
                flush=True,
            )

        text = " ".join(line.text.strip() for line in transcript.lines if line.text.strip())
        return text if text else None


def make_transcriber(
    engine: str,
    model: str,
    device: str,
    language: str = "en",
    debug: bool = False,
) -> TranscriberProtocol:
    """Construct the configured transcriber. Validates engine/model pairing."""
    if engine == "whisper":
        return WhisperTranscriber(
            model_size=model, device=device, language=language, debug=debug
        )
    if engine == "moonshine":
        return MoonshineTranscriber(model_size=model, language=language, debug=debug)
    raise ValueError(
        f"transcriber_engine must be one of {TRANSCRIBER_ENGINES}, got '{engine}'"
    )
