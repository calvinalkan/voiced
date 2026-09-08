# Audio fixtures

This directory contains fixed audio fixtures for testing the pure Zig transcription runtime and comparing inference implementations. Every WAV file has an adjacent reference transcript:

```text
audio/<id>.wav
audio/<id>.transcript
```

The WAV files are mono, 16 kHz, signed 16-bit PCM. Transcript files contain the official LibriSpeech reference followed by one newline.

## Selection

The fixtures contain 32 LibriSpeech test clips:

```text
16 from test-clean
16 from test-other
```

Each split contains two female and two male speakers in each duration range:

```text
1–5 seconds
5–10 seconds
10–20 seconds
20–29.5 seconds
```

A speaker occurs at most once in each split. The selected utterance IDs are fixed
in `manifest.tsv`; migrating checksum algorithms does not resample the corpus.

`manifest.tsv` records source and BLAKE3-256 checksum provenance for `verify.sh`; the Zig test does not parse it. The test discovers `.wav` files directly from `audio/` and derives each adjacent `.transcript` filename.

Run every audio fixture with the default `base.en` model from the repository root:

```bash
zig test zig/inference/transcription_test.zig -OReleaseFast -mcpu=native
```

Select `small.en` with:

```bash
VOICED_RUNTIME_MODEL=small.en zig test zig/inference/transcription_test.zig -OReleaseFast -mcpu=native
```

`VOICED_RUNTIME_MODEL` accepts `base.en` and `small.en`. Select the per-transcription encoder tail with `VOICED_RUNTIME_TRAILING_PADDING_SECONDS`; accepted values are `5`, `10`, and the default `30`:

```bash
VOICED_RUNTIME_MODEL=small.en \
VOICED_RUNTIME_TRAILING_PADDING_SECONDS=10 \
zig test zig/inference/transcription_test.zig -OReleaseFast -fllvm -mcpu=native
```

The runtime always retains capacity for `Policy.samples_count_max`. Shorter trailing padding changes only the logical feature and encoder lengths for that call; it does not resize or allocate runtime memory.

Debug inference must use `-fllvm` because Zig's self-hosted x86 backend does not support the runtime's AVX-VNNI inline assembly:

```bash
VOICED_RUNTIME_AUDIO_FIXTURE=librispeech-test-clean-6829-68769-0026 zig test zig/inference/transcription_test.zig -ODebug -fllvm -mcpu=native
```

Select one fixture during development:

```bash
VOICED_RUNTIME_AUDIO_FIXTURE=librispeech-test-clean-6829-68769-0026 zig test zig/inference/transcription_test.zig -OReleaseFast -mcpu=native
```

An external directory can supply additional fixtures through `VOICED_RUNTIME_AUDIO_FIXTURES_DIRECTORY`. It must follow the same adjacent `.wav` and `.transcript` convention.

The verifier requires `ffprobe` and Python 3.11+ with the `blake3` package. From
the repository root, use the optional reference environment:

```bash
.venv/bin/python -m pip install blake3
PYTHON=.venv/bin/python zig/audio-fixtures/verify.sh
```

This validates the recorded BLAKE3-256 checksums and WAV representation. `PYTHON`
defaults to `python3` when that interpreter already has `blake3` installed.

## Transcription comparison

For ordinary word-error-rate comparison, normalize the hypothesis and reference consistently by removing case and punctuation. Retain the original strings as well: punctuation and formatting changes remain useful diagnostics even though LibriSpeech references do not contain punctuation.

Variable-length Whisper validation should run each WAV under all three trailing-padding settings:

```text
30 seconds: actual content + 3000 frames, capped at 3000
10 seconds: actual content + 1000 frames, capped at 3000
5 seconds:  actual content + 500 frames, capped at 3000
```

The result is aligned to eight Mel frames. A 30-second tail therefore produces Whisper's standard fixed 3000-frame input, while the shorter policies remove encoder work only when content plus padding remains below the cap.

Compare the hypotheses with the human transcript and with each other. Record word errors, repeated text, missing endings, token count, no-speech probability, average log probability, encoder positions, and latency.

## Source and license

The clips come from the official LibriSpeech `test-clean` and `test-other` archives:

```text
https://www.openslr.org/12

test-clean.tar.gz BLAKE3-256 c4f1173bb85312ed40f04f2477d7e3a2e9158c15174225187cd3b7ec8e4879f6
test-other.tar.gz BLAKE3-256 31e9d274a778b4d73e01e873229cf3e691afa2922e4a616a3025a94a1744324e
```

LibriSpeech is Copyright 2014 Vassil Panayotov and licensed under Creative Commons Attribution 4.0 International. See `LIBRISPEECH-LICENSE.txt`.
