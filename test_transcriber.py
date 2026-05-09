"""Smoke test for the persistent-worker transcriber architecture.

Two flavors of test, measuring different things:

- Synthetic batched feed (default): chunks are pushed onto the worker
  queue as fast as Python can produce them. Total wall-clock time is
  dominated by the worker draining its queue — i.e. it measures the
  total cost of encoder work for that audio. Use to verify an engine
  works at all and produces sensible output.

  CAUTION: the "finalize" wall-clock from a batched test is NOT the
  user-perceived end-of-speech latency. In production the audio thread
  feeds chunks at real-time pace, so encoder work is spread across the
  recording window and finalize only handles the trailing buffer.

- Real-time paced feed (`--realtime`): chunks are pushed at 100ms
  wall-clock intervals, matching sounddevice's actual chunk delivery
  rate. Measures the user-perceived end-of-speech latency. Catches
  regressions in the worker decoupling (e.g. encoder work leaking back
  onto the audio thread, queue starvation).

Run via:
  make test               # batched tests (fast, runs on every check)
  make test-realtime      # adds the slow real-time-paced suite
  uv run python test_transcriber.py [--realtime]

First run will download any models that aren't cached yet.
"""

from __future__ import annotations

import sys
import threading
import time
import wave
from collections.abc import Callable
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


def load_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    int16: NDArray[np.int16] = np.frombuffer(raw, dtype=np.int16)
    audio = int16.astype(np.float32) / 32768.0
    return audio, sr


def test_paragraph_fixture() -> None:
    """Long-form paragraph (~30s, named-checkpoint paragraph used during dev).

    Synthetic batched feed — measures TOTAL ENCODER WORK for the fixture.
    Streaming-mode timings here are not the user-perceived end-of-speech
    latency (see test_paragraph_realtime for that). Asserts only that each
    engine returns non-empty text; counts how many section markers each
    captured so we can compare quality at a glance.
    """
    print("\n--- paragraph fixture (batched feed) ---")
    p = REPO_ROOT / "test-fixtures" / "paragraph.wav"
    if not p.exists():
        print(f"  SKIP (no fixture at {p})")
        return

    audio, sr = load_wav(p)
    print(f"  fixture: {len(audio) / sr:.1f}s, {sr} Hz")

    markers = ["alpha", "bravo", "charlie", "delta", "echo"]

    cases: list[tuple[str, Callable[[], TranscriberProtocol]]] = [
        ("whisper buffered (base)", lambda: WhisperTranscriber(model_size="base")),
        ("moonshine buffered (base)", lambda: MoonshineTranscriber(model_size="base", streaming=False)),
        ("moonshine streaming (tiny-streaming)", lambda: MoonshineTranscriber(model_size="tiny-streaming", streaming=True)),
    ]

    for label, factory in cases:
        t = factory()
        t.load()
        try:
            t.start_session()
            feed_in_chunks(t, audio, sr)
            t0 = time.time()
            text = t.finalize()
            elapsed_ms = int((time.time() - t0) * 1000)
            if not text:
                raise AssertionError(f"[{label}] returned empty text")
            text_lower = text.lower()
            found = [m for m in markers if m in text_lower]
            print(f"  {label}")
            print(f"    total_encoder_work: {elapsed_ms}ms, {len(text)} chars")
            print(f"    markers found: {found} ({len(found)}/{len(markers)})")
            print(f"    text: {text!r}")
        finally:
            t.shutdown()


def test_paragraph_realtime() -> None:
    """Real-time-paced streaming test: feeds the paragraph fixture at the same
    100ms-per-chunk rate sounddevice would deliver chunks during a live
    recording. Measures user-perceived end-of-speech latency, and catches
    regressions where encoder work leaks back onto the audio thread or the
    persistent worker falls behind real-time.

    Asserts:
      - feed phase ≈ audio duration (the test thread, simulating the audio
        callback, was never blocked by the worker)
      - finalize phase < MAX_FINALIZE_MS (worker kept up with real-time)
      - output non-empty
    """
    print("\n--- paragraph fixture (real-time pacing, streaming) ---")
    p = REPO_ROOT / "test-fixtures" / "paragraph.wav"
    if not p.exists():
        print(f"  SKIP (no fixture at {p})")
        return

    audio, sr = load_wav(p)
    duration_s = len(audio) / sr
    print(f"  fixture: {duration_s:.1f}s, {sr} Hz")

    chunk_ms = 100
    chunk_samples = int(sr * chunk_ms / 1000)
    # Generous threshold: tiny-streaming on a typical laptop CPU should finish
    # well under 1s. If this fails, encoder work is leaking onto the wrong
    # thread or the worker is fundamentally slower than real-time.
    max_finalize_ms = 1000
    # Tolerate a bit of overhead from time.sleep granularity / scheduling.
    max_feed_ratio = 1.10

    t = MoonshineTranscriber(model_size="tiny-streaming", streaming=True)
    t.load()
    try:
        t.start_session()

        feed_t0 = time.time()
        for i in range(0, len(audio), chunk_samples):
            t.feed(audio[i : i + chunk_samples])
            time.sleep(chunk_ms / 1000.0)
        feed_ms = int((time.time() - feed_t0) * 1000)

        finalize_t0 = time.time()
        text = t.finalize()
        finalize_ms = int((time.time() - finalize_t0) * 1000)
    finally:
        t.shutdown()

    expected_feed_ms = int(duration_s * 1000)
    feed_ratio = feed_ms / expected_feed_ms

    print(
        f"  feed:     {feed_ms}ms (audio={expected_feed_ms}ms, ratio {feed_ratio:.2f}x)"
    )
    print(f"  finalize: {finalize_ms}ms")
    print(f"  text:     {text!r}")

    if not text:
        raise AssertionError("realtime: returned empty text")
    if feed_ratio > max_feed_ratio:
        raise AssertionError(
            f"realtime: feed phase {feed_ms}ms vs expected ~{expected_feed_ms}ms (ratio {feed_ratio:.2f}x > {max_feed_ratio:.2f}x); the audio thread was blocked by encoder work"
        )
    if finalize_ms > max_finalize_ms:
        raise AssertionError(
            f"realtime: finalize {finalize_ms}ms exceeds {max_finalize_ms}ms threshold; the persistent-worker queue is falling behind real-time"
        )

    print("  PASS realtime streaming")


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
    import argparse
    from typing import cast

    ap = argparse.ArgumentParser()
    _ = ap.add_argument(
        "--realtime",
        action="store_true",
        help="Include real-time-paced tests (slow; ~one wall-clock second per second of fixture)",
    )
    args = ap.parse_args()
    realtime = cast(bool, args.realtime)

    audio, sr = load_fixture()
    print(f"fixture: {len(audio) / sr:.1f}s of audio, {sr} Hz")
    try:
        test_whisper(audio, sr)
        test_moonshine_buffered(audio, sr)
        test_moonshine_streaming(audio, sr)
        test_concurrent_feed(audio, sr)
        test_paragraph_fixture()
        if realtime:
            test_paragraph_realtime()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1
    print("\nALL TRANSCRIBER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
