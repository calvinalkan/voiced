"""
Transcription module for voiced.

Two engines, same input contract (numpy float32 mono @ 16kHz → str | None):

- Whisper via faster-whisper (CTranslate2): broad multilingual coverage,
  slower on CPU, prone to silence hallucinations.
- Moonshine: fast on CPU, English-focused, hallucinates less in silence.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from moonshine_voice.transcriber import Stream as _MoonshineCoreStream
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


class MoonshineStream:
    """Wraps moonshine's Stream so the daemon can feed chunks during recording
    and finalize once at end-of-speech. Most of the inference work happens
    incrementally inside add_chunk; finalize only processes the trailing buffer."""

    debug: bool
    _stream: _MoonshineCoreStream
    _started: bool
    _closed: bool
    _chunks_fed: int
    _samples_fed: int
    _open_time: float

    def __init__(self, stream: _MoonshineCoreStream, debug: bool = False) -> None:
        self._stream = stream
        self._started = False
        self._closed = False
        self._chunks_fed = 0
        self._samples_fed = 0
        self._open_time = time.time()
        self.debug = debug

    def _log(self, level: str, msg: str) -> None:
        if level == "debug" and not self.debug:
            return
        print(f"[transcriber] [{level}] {msg}", flush=True)

    def add_chunk(self, audio: NDArray[np.float32]) -> None:
        """Feed a chunk to the encoder. Safe to call from the audio callback thread."""
        if self._closed:
            return
        if not self._started:
            self._stream.start()
            self._started = True
            self._log("debug", "stream started")

        chunk_list = cast(list[float], audio.tolist())
        self._stream.add_audio(chunk_list, sample_rate=16000)
        self._chunks_fed += 1
        self._samples_fed += len(chunk_list)

        if self.debug:
            wall_elapsed = time.time() - self._open_time
            buffered_sec = self._samples_fed / 16000
            self._log(
                "debug",
                f"chunk #{self._chunks_fed}: {len(chunk_list)} samples (buffered: {buffered_sec:.2f}s @ wall {wall_elapsed:.2f}s)",
            )

    def finalize(self) -> str | None:
        """Stop the stream and return the final transcript. Idempotent — closes
        underlying resources, safe to call once even if never started."""
        if self._closed:
            return None
        if not self._started:
            self._log("debug", "finalize called with no chunks fed")
            self.close()
            return None

        try:
            t0 = time.time()
            # Stream.stop() runs a final update_transcription internally and
            # returns the resulting Transcript (or None on internal error).
            transcript = self._stream.stop()
            finalize_ms = int((time.time() - t0) * 1000)

            if transcript is None:
                self._log("warn", "stream stop returned no transcript")
                return None

            text = " ".join(
                line.text.strip() for line in transcript.lines if line.text.strip()
            )

            if self.debug:
                self._log(
                    "debug",
                    f"finalized: {self._chunks_fed} chunks, {self._samples_fed / 16000:.2f}s audio, finalize {finalize_ms}ms, {len(transcript.lines)} lines",
                )
            return text if text else None
        finally:
            self.close()

    def close(self) -> None:
        """Release native resources. Idempotent."""
        if self._closed:
            return
        try:
            self._stream.close()
        except Exception as e:
            self._log("warn", f"error closing stream: {e}")
        self._closed = True


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

    def start_stream(self) -> MoonshineStream:
        """Open a new streaming inference session. Caller must call finalize()
        (or close()) when done to release resources."""
        if self.transcriber is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        core_stream = self.transcriber.create_stream()
        return MoonshineStream(core_stream, debug=self.debug)


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
