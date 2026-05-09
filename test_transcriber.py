"""Smoke test for the persistent-worker transcriber architecture.

Exercises:
  1. WhisperTranscriber (buffered) — single + sequential sessions, empty session
  2. MoonshineTranscriber buffered  — same
  3. MoonshineTranscriber streaming — same
  4. Concurrent feed from a separate thread (simulates audio callback)

Run via `make test` (alongside the bash integration test) or directly:
    uv run python test_transcriber.py

First run will download any models that aren't cached yet.

Tests assert that the transcribed text contains "hello" — the fixture is
test-fixtures/hello_world.wav. Add more fixtures and assertions as needed.
"""

from __future__ import annotations

import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from transcriber import (  # noqa: E402
    MoonshineTranscriber,
    TranscriberProtocol,
    WhisperTranscriber,
)


def load_fixture() -> tuple[NDArray[np.float32], int]:
    p = REPO_ROOT / "test-fixtures" / "hello_world.wav"
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    int16: NDArray[np.int16] = np.frombuffer(raw, dtype=np.int16)
    audio = int16.astype(np.float32) / 32768.0
    return audio, sr


def feed_in_chunks(
    t: TranscriberProtocol, audio: NDArray[np.float32], sr: int, chunk_ms: int = 100
) -> None:
    chunk = int(sr * chunk_ms / 1000)
    for i in range(0, len(audio), chunk):
        t.feed(audio[i : i + chunk])


def expect_contains(text: str | None, needle: str, label: str) -> None:
    if not text:
        raise AssertionError(f"[{label}] expected text containing {needle!r}, got {text!r}")
    if needle.lower() not in text.lower():
        raise AssertionError(
            f"[{label}] expected text containing {needle!r}, got {text!r}"
        )
    print(f"  PASS {label}: {text!r}")


def expect_none(text: str | None, label: str) -> None:
    if text is not None:
        raise AssertionError(f"[{label}] expected None, got {text!r}")
    print(f"  PASS {label}: None")


def run_session(
    t: TranscriberProtocol, audio: NDArray[np.float32], sr: int, label: str, needle: str
) -> None:
    t.start_session()
    feed_in_chunks(t, audio, sr)
    t0 = time.time()
    text = t.finalize()
    elapsed_ms = int((time.time() - t0) * 1000)
    expect_contains(text, needle, f"{label} ({elapsed_ms}ms)")


def empty_session(t: TranscriberProtocol, label: str) -> None:
    t.start_session()
    text = t.finalize()
    expect_none(text, label)


def test_whisper(audio: NDArray[np.float32], sr: int) -> None:
    print("\n--- whisper (buffered) ---")
    t = WhisperTranscriber(model_size="base")
    t.load()
    try:
        run_session(t, audio, sr, "whisper run 1", "hello")
        run_session(t, audio, sr, "whisper run 2", "hello")
        empty_session(t, "whisper empty session")
    finally:
        t.shutdown()
    print("  PASS shutdown clean")


def test_moonshine_buffered(audio: NDArray[np.float32], sr: int) -> None:
    print("\n--- moonshine (buffered) ---")
    t = MoonshineTranscriber(model_size="base", streaming=False)
    t.load()
    try:
        run_session(t, audio, sr, "moonshine buffered run 1", "hello")
        run_session(t, audio, sr, "moonshine buffered run 2", "hello")
        empty_session(t, "moonshine buffered empty session")
    finally:
        t.shutdown()
    print("  PASS shutdown clean")


def test_moonshine_streaming(audio: NDArray[np.float32], sr: int) -> None:
    print("\n--- moonshine (streaming) ---")
    t = MoonshineTranscriber(model_size="tiny-streaming", streaming=True)
    t.load()
    try:
        run_session(t, audio, sr, "moonshine streaming run 1", "hello")
        run_session(t, audio, sr, "moonshine streaming run 2", "hello")
        empty_session(t, "moonshine streaming empty session")
    finally:
        t.shutdown()
    print("  PASS shutdown clean")


def test_concurrent_feed(audio: NDArray[np.float32], sr: int) -> None:
    """Audio-thread simulation: feed from a separate thread, finalize from main."""
    print("\n--- concurrent feed (audio thread simulation) ---")
    t = WhisperTranscriber(model_size="base")
    t.load()
    try:
        t.start_session()

        feeder = threading.Thread(target=feed_in_chunks, args=(t, audio, sr), name="audio-sim")
        feeder.start()
        feeder.join()  # ensure all chunks queued before finalize

        text = t.finalize()
        expect_contains(text, "hello", "concurrent feed")
    finally:
        t.shutdown()


def main() -> int:
    audio, sr = load_fixture()
    print(f"fixture: {len(audio) / sr:.1f}s of audio, {sr} Hz")
    try:
        test_whisper(audio, sr)
        test_moonshine_buffered(audio, sr)
        test_moonshine_streaming(audio, sr)
        test_concurrent_feed(audio, sr)
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1
    print("\nALL TRANSCRIBER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
