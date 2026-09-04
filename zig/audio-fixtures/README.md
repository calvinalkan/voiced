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

A speaker occurs at most once in each split. Selection is deterministic from the SHA-256 ordering of eligible LibriSpeech utterance IDs.

`manifest.tsv` records source and checksum provenance for `verify.sh`; the Zig test does not parse it. The test discovers `.wav` files directly from `audio/` and derives each adjacent `.transcript` filename.

Run every audio fixture with the default `base.en` model from the repository root:

```bash
zig test zig/runtime/transcription_test.zig -OReleaseFast -mcpu=native
```

Select `small.en` with:

```bash
VOICED_RUNTIME_MODEL=small.en zig test zig/runtime/transcription_test.zig -OReleaseFast -mcpu=native
```

`VOICED_RUNTIME_MODEL` accepts `base.en` and `small.en`. Debug inference must use `-fllvm` because Zig's self-hosted x86 backend does not support the runtime's AVX-VNNI inline assembly:

```bash
VOICED_RUNTIME_AUDIO_FIXTURE=librispeech-test-clean-6829-68769-0026 zig test zig/runtime/transcription_test.zig -ODebug -fllvm -mcpu=native
```

Select one fixture during development:

```bash
VOICED_RUNTIME_AUDIO_FIXTURE=librispeech-test-clean-6829-68769-0026 zig test zig/runtime/transcription_test.zig -OReleaseFast -mcpu=native
```

Run `./verify.sh` to validate the recorded checksums and WAV representation.

## Transcription comparison

For ordinary word-error-rate comparison, normalize the hypothesis and reference consistently by removing case and punctuation. Retain the original strings as well: punctuation and formatting changes remain useful diagnostics even though LibriSpeech references do not contain punctuation.

Variable-length Whisper validation should run each WAV with both:

```text
fixed:     3000 Mel frames
candidate: actual Mel frames + configured trailing padding, aligned to 8
```

Compare both hypotheses with the human transcript and with each other. Record word errors, repeated text, missing endings, token count, no-speech probability, average log probability, encoder length, and latency.

## Source and license

The clips come from the official LibriSpeech `test-clean` and `test-other` archives:

```text
https://www.openslr.org/12

test-clean.tar.gz MD5 32fa31d27d2e1cad72775fee3f4849a9
test-other.tar.gz MD5 fb5a50374b501bb3bac4815ee91d3135
```

LibriSpeech is Copyright 2014 Vassil Panayotov and licensed under Creative Commons Attribution 4.0 International. See `LIBRISPEECH-LICENSE.txt`.
