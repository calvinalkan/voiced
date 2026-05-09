"""
Transcription module for voiced.

Two engines, same threaded interface. Both engines own one persistent
worker thread for the daemon's lifetime, started in load() and joined in
shutdown(). The audio capture thread only enqueues chunks (microsecond
op); all model work runs on the worker thread, so frames are never
dropped by callback latency.

- WhisperTranscriber  (faster-whisper / CTranslate2)
  Buffer-and-transcribe; Whisper has no streaming inference API.
  Chunks accumulate on the worker's buffer and a single transcribe call
  runs at finalize.

- MoonshineTranscriber  (moonshine-voice)
  streaming=False → buffer-and-transcribe via transcribe_without_streaming
  streaming=True  → live Stream during recording; finalize reads
                    accumulated LineCompleted events from a listener.

Per-recording protocol:
    transcriber.start_session()
    transcriber.feed(chunk) × N         # safe to call from any thread
    text = transcriber.finalize()       # blocks until worker drains
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from moonshine_voice.transcriber import (
        Stream as _MoonshineCoreStream,
    )
    from moonshine_voice.transcriber import (
        Transcriber as _MoonshineCoreTranscriber,
    )
    from moonshine_voice.transcriber import (
        TranscriptEvent as _MoonshineTranscriptEvent,
    )


WHISPER_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    # English-only variants — typically more accurate on English at the same
    # size, since they don't have to share capacity with 98 other languages.
    # No `.en` for large-v3.
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
)
MOONSHINE_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "tiny-streaming",
    "small-streaming",
    "medium-streaming",
)
TRANSCRIBER_ENGINES: tuple[str, ...] = ("whisper", "moonshine")


class TranscriberProtocol(Protocol):
    debug: bool

    def load(self, on_progress: Callable[[str], None] | None = None) -> None: ...
    def start_session(self) -> None: ...
    def feed(self, audio: NDArray[np.float32]) -> None: ...
    def finalize(self) -> str | None: ...
    def shutdown(self) -> None: ...


class _Msg(NamedTuple):
    """Worker-thread inbox message.

    kind="start"     : begin a fresh session
    kind="chunk"     : audio chunk (audio field set)
    kind="finalize"  : drain session, append text to out_box, set done_event
    kind="shutdown"  : exit worker loop

    enqueue_time is the time.time() at queue.put — used by the worker to
    log how long a finalize message waited in the queue behind audio
    chunk backlog. That wait is the dominant component of user-perceived
    end-of-speech latency when the encoder runs slower than real-time.
    """

    kind: str
    audio: NDArray[np.float32] | None = None
    done_event: threading.Event | None = None
    out_box: list[str | None] | None = None
    enqueue_time: float = 0.0


def _signal_done(msg: _Msg, text: str | None) -> None:
    if msg.out_box is not None:
        msg.out_box.append(text)
    if msg.done_event is not None:
        msg.done_event.set()


class WhisperTranscriber:
    """Buffer-and-transcribe. Whisper has no streaming inference API."""

    debug: bool
    model_size: str
    device: str
    language: str
    model: WhisperModel | None
    _queue: queue.Queue[_Msg]
    _worker: threading.Thread | None
    _buffer: list[NDArray[np.float32]]

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
        self._queue = queue.Queue()
        self._worker = None
        self._buffer = []

    def _log(self, level: str, msg: str) -> None:
        if level == "debug" and not self.debug:
            return
        print(f"[transcriber] [{level}] {msg}", flush=True)

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

        self._worker = threading.Thread(
            target=self._worker_loop, name="whisper-worker", daemon=True
        )
        self._worker.start()

        if on_progress:
            on_progress("Ready")

    def _worker_loop(self) -> None:
        while True:
            msg = self._queue.get()
            try:
                self._dispatch(msg)
            except Exception as e:
                self._log("error", f"worker error on '{msg.kind}': {e}")
                if msg.kind == "finalize":
                    _signal_done(msg, None)
                self._buffer = []
            if msg.kind == "shutdown":
                return

    def _dispatch(self, msg: _Msg) -> None:
        if msg.kind == "start":
            self._buffer = []
        elif msg.kind == "chunk":
            if msg.audio is not None:
                self._buffer.append(msg.audio)
        elif msg.kind == "finalize":
            queue_wait_ms = (
                int((time.time() - msg.enqueue_time) * 1000)
                if msg.enqueue_time
                else 0
            )
            try:
                text = self._finalize_buffer()
            finally:
                self._buffer = []
            if self.debug and queue_wait_ms > 0:
                self._log("debug", f"finalize queue wait: {queue_wait_ms}ms")
            _signal_done(msg, text)
        elif msg.kind == "shutdown":
            self._buffer = []

    def _finalize_buffer(self) -> str | None:
        if not self._buffer:
            return None
        audio = np.concatenate(self._buffer)
        if len(audio) < 8000:
            return None
        return self._transcribe_full(audio)

    def _transcribe_full(self, audio: NDArray[np.float32]) -> str | None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.time()
        segments, info = self.model.transcribe(  # pyright: ignore[reportUnknownMemberType]
            audio,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 200},
        )
        segments_list = list(segments)
        elapsed_ms = int((time.time() - t0) * 1000)

        if self.debug:
            self._log(
                "debug",
                f"finalized: {len(audio) / 16000:.2f}s audio, transcribe {elapsed_ms}ms, language={info.language} (prob={info.language_probability:.2f}), segments={len(segments_list)}",
            )
        text = " ".join(seg.text.strip() for seg in segments_list if seg.text.strip())
        return text if text else None

    def start_session(self) -> None:
        self._queue.put(_Msg(kind="start"))

    def feed(self, audio: NDArray[np.float32]) -> None:
        self._queue.put(_Msg(kind="chunk", audio=audio))

    def finalize(self) -> str | None:
        done = threading.Event()
        out_box: list[str | None] = []
        self._queue.put(
            _Msg(
                kind="finalize",
                done_event=done,
                out_box=out_box,
                enqueue_time=time.time(),
            )
        )
        _ = done.wait()
        return out_box[0] if out_box else None

    def shutdown(self) -> None:
        self._queue.put(_Msg(kind="shutdown"))
        if self._worker is not None:
            self._worker.join(timeout=2)


_MOONSHINE_ARCH_BY_NAME: dict[str, str] = {
    "tiny": "TINY",
    "base": "BASE",
    "tiny-streaming": "TINY_STREAMING",
    "small-streaming": "SMALL_STREAMING",
    "medium-streaming": "MEDIUM_STREAMING",
}


class MoonshineTranscriber:
    """Persistent worker; streaming or buffered controlled by `streaming` flag.

    streaming=True  → live Stream during recording; finalize reads accumulated
                      LineCompleted events from a listener.
    streaming=False → chunks accumulate on a buffer; finalize calls
                      transcribe_without_streaming on the full audio.
    """

    debug: bool
    streaming: bool
    model_size: str
    language: str
    transcriber: _MoonshineCoreTranscriber | None
    _queue: queue.Queue[_Msg]
    _worker: threading.Thread | None
    _stream: _MoonshineCoreStream | None
    _buffer: list[NDArray[np.float32]]
    _session_lines: list[tuple[int, str]]
    _seen_line_ids: set[int]
    _session_chunks: int
    _session_samples: int
    _session_open_time: float

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        debug: bool = False,
        streaming: bool = False,
    ) -> None:
        if model_size not in MOONSHINE_MODELS:
            raise ValueError(
                f"moonshine model must be one of {MOONSHINE_MODELS}, got '{model_size}'"
            )
        self.model_size = model_size
        self.language = language
        self.debug = debug
        self.streaming = streaming
        self.transcriber = None
        self._queue = queue.Queue()
        self._worker = None
        self._stream = None
        self._buffer = []
        self._session_lines = []
        self._seen_line_ids = set()
        self._session_chunks = 0
        self._session_samples = 0
        self._session_open_time = 0.0

    def _log(self, level: str, msg: str) -> None:
        if level == "debug" and not self.debug:
            return
        print(f"[transcriber] [{level}] {msg}", flush=True)

    def load(self, on_progress: Callable[[str], None] | None = None) -> None:
        from moonshine_voice import ModelArch, get_model_for_language
        from moonshine_voice.transcriber import Transcriber as MoonshineCoreTranscriber

        arch_name = _MOONSHINE_ARCH_BY_NAME[self.model_size]
        arch = ModelArch[arch_name]

        if on_progress:
            mode = "streaming" if self.streaming else "buffered"
            on_progress(
                f"Loading moonshine-{self.model_size} ({self.language}, {mode})..."
            )

        model_path, _ = get_model_for_language(self.language, arch)
        self.transcriber = MoonshineCoreTranscriber(model_path=model_path, model_arch=arch)

        if on_progress:
            on_progress("Warming up...")
        # 1 second of zeros to prime the runtime; Moonshine accepts all-zero input.
        _ = self.transcriber.transcribe_without_streaming([0.0] * 16000, sample_rate=16000)

        self._worker = threading.Thread(
            target=self._worker_loop, name="moonshine-worker", daemon=True
        )
        self._worker.start()

        if on_progress:
            on_progress("Ready")

    def _worker_loop(self) -> None:
        while True:
            msg = self._queue.get()
            try:
                self._dispatch(msg)
            except Exception as e:
                self._log("error", f"worker error on '{msg.kind}': {e}")
                if msg.kind == "finalize":
                    _signal_done(msg, None)
                # Reset on any error so the next session starts clean.
                self._cleanup_session()
            if msg.kind == "shutdown":
                return

    def _dispatch(self, msg: _Msg) -> None:
        if msg.kind == "start":
            # Defensive: drop any leftover state from an interrupted session
            # (e.g. caller exception between start_session and finalize).
            self._cleanup_session()
            self._begin_session()
        elif msg.kind == "chunk":
            if msg.audio is not None:
                self._handle_chunk(msg.audio)
        elif msg.kind == "finalize":
            # Time spent waiting in the queue behind audio-chunk backlog. When
            # the encoder runs slower than real-time, this dominates the
            # user-perceived end-of-speech latency — the inner stop()/transcribe
            # call runs fast once the worker actually reaches it.
            queue_wait_ms = (
                int((time.time() - msg.enqueue_time) * 1000)
                if msg.enqueue_time
                else 0
            )
            try:
                text = self._handle_finalize(queue_wait_ms)
            finally:
                self._cleanup_session()
            _signal_done(msg, text)
        elif msg.kind == "shutdown":
            self._cleanup_session()

    def _begin_session(self) -> None:
        self._session_lines.clear()
        self._seen_line_ids.clear()
        self._session_chunks = 0
        self._session_samples = 0
        self._session_open_time = time.time()
        if self.streaming:
            assert self.transcriber is not None
            self._stream = self.transcriber.create_stream()
            self._stream.add_listener(self._on_event)
            self._stream.start()
            self._log("debug", "session opened (streaming)")
        else:
            self._buffer = []
            self._log("debug", "session opened (buffered)")

    def _handle_chunk(self, audio: NDArray[np.float32]) -> None:
        self._session_chunks += 1
        self._session_samples += len(audio)

        if self.streaming:
            if self._stream is None:
                self._log("warn", "chunk arrived without active streaming session")
                return
            self._stream.add_audio(cast(list[float], audio.tolist()), sample_rate=16000)
        else:
            self._buffer.append(audio)

        if self.debug:
            wall = time.time() - self._session_open_time
            buffered = self._session_samples / 16000
            self._log(
                "debug",
                f"chunk #{self._session_chunks}: {len(audio)} samples (buffered {buffered:.2f}s @ wall {wall:.2f}s)",
            )

    def _handle_finalize(self, queue_wait_ms: int) -> str | None:
        if self.streaming:
            return self._finalize_streaming(queue_wait_ms)
        return self._finalize_buffered(queue_wait_ms)

    def _finalize_streaming(self, queue_wait_ms: int) -> str | None:
        if self._stream is None:
            self._log("debug", "streaming finalize without active session")
            return None
        t0 = time.time()
        # Stream.stop() runs a final update_transcription internally and emits
        # any remaining LineCompleted events to our listener before returning.
        _ = self._stream.stop()
        stop_ms = int((time.time() - t0) * 1000)
        self._session_lines.sort(key=lambda p: p[0])
        text = " ".join(t for _, t in self._session_lines)
        if self.debug:
            total_ms = queue_wait_ms + stop_ms
            self._log(
                "debug",
                f"finalized streaming: {self._session_chunks} chunks, {self._session_samples / 16000:.2f}s audio, queue_wait {queue_wait_ms}ms + stop {stop_ms}ms = {total_ms}ms, {len(self._session_lines)} lines",
            )
        return text if text else None

    def _finalize_buffered(self, queue_wait_ms: int) -> str | None:
        if not self._buffer:
            return None
        audio = np.concatenate(self._buffer)
        if len(audio) < 8000:
            return None
        t0 = time.time()
        assert self.transcriber is not None
        audio_list = cast(list[float], audio.tolist())
        transcript = self.transcriber.transcribe_without_streaming(
            audio_list, sample_rate=16000
        )
        transcribe_ms = int((time.time() - t0) * 1000)
        text = " ".join(
            line.text.strip() for line in transcript.lines if line.text.strip()
        )
        if self.debug:
            total_ms = queue_wait_ms + transcribe_ms
            self._log(
                "debug",
                f"finalized buffered: {self._session_chunks} chunks, {len(audio) / 16000:.2f}s audio, queue_wait {queue_wait_ms}ms + transcribe {transcribe_ms}ms = {total_ms}ms, {len(transcript.lines)} lines",
            )
        return text if text else None

    def _cleanup_session(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as e:
                self._log("warn", f"error closing stream: {e}")
            self._stream = None
        self._buffer = []

    def _on_event(self, event: _MoonshineTranscriptEvent) -> None:
        # Lazy import keeps the module load light if someone imports the
        # transcriber module without using moonshine.
        from moonshine_voice.transcriber import LineCompleted

        if not isinstance(event, LineCompleted):
            return
        line = event.line
        if line.line_id in self._seen_line_ids:
            return
        self._seen_line_ids.add(line.line_id)
        text = line.text.strip()
        if text:
            self._session_lines.append((line.line_id, text))
            if self.debug:
                self._log("debug", f"line #{line.line_id} completed: {text!r}")

    def start_session(self) -> None:
        self._queue.put(_Msg(kind="start"))

    def feed(self, audio: NDArray[np.float32]) -> None:
        self._queue.put(_Msg(kind="chunk", audio=audio))

    def finalize(self) -> str | None:
        done = threading.Event()
        out_box: list[str | None] = []
        self._queue.put(
            _Msg(
                kind="finalize",
                done_event=done,
                out_box=out_box,
                enqueue_time=time.time(),
            )
        )
        _ = done.wait()
        return out_box[0] if out_box else None

    def shutdown(self) -> None:
        self._queue.put(_Msg(kind="shutdown"))
        if self._worker is not None:
            self._worker.join(timeout=2)


def make_transcriber(
    engine: str,
    model: str,
    device: str,
    language: str = "en",
    debug: bool = False,
    streaming: bool = False,
) -> TranscriberProtocol:
    """Construct the configured transcriber. Validates engine/model/streaming."""
    if engine == "whisper":
        if streaming:
            raise ValueError(
                "whisper has no streaming inference API; use transcriber_engine=moonshine for streaming"
            )
        return WhisperTranscriber(
            model_size=model, device=device, language=language, debug=debug
        )
    if engine == "moonshine":
        return MoonshineTranscriber(
            model_size=model, language=language, debug=debug, streaming=streaming
        )
    raise ValueError(
        f"transcriber_engine must be one of {TRANSCRIBER_ENGINES}, got '{engine}'"
    )
