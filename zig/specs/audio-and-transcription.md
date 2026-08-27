# Audio and Transcription

## User-visible requirement

Transcription should begin while a longer recording is still in progress, but
voiced should produce one final clipboard value and optional paste only after
the user stops normally. A two-minute dictation should not incur the cost of
transcribing the complete two minutes after release.

Under normal model throughput, stop-to-output latency should depend on the final
partial chunk rather than total session duration. The first target is at most
three seconds on the target machine when Whisper has kept up with capture; the
prototype should replace this provisional target with measured distributions.

This is eager chunk transcription, not live captions. Intermediate text remains
private to the active session and may be discarded by cancellation.

## Audio capture

The audio process connects directly to PipeWire and requests mono float32 audio
at 16 kHz where graph negotiation permits it. PipeWire may own or recycle its
small graph buffer pool. The process copies complete callback blocks into an
active shared audio slot.

The callback publishes `samples_count` only after the corresponding sample
bytes are readable. If the process dies during a copy, the supervisor and
Whisper process consume only the previously published prefix.

The callback performs bounded work. It does not allocate, send control
messages, transcribe, persist, notify, or wait for another thread.

Direct PipeWire state and registry events are the primary source of device
removal and stream failure. A monotonic callback-progress deadline remains a
fallback for failures that produce no useful event.

## Eager chunk pipeline

```text
free → filling → sealed → transcribing → produced → free
```

Only the audio process writes a filling slot. Only the Whisper process reads a
sealed or transcribing slot. The supervisor authorizes every transition and
does not release a slot until its transcription result is safely represented in
the session transcript.

A provisional boundary policy is:

```text
before 20 seconds   continue filling
20–28 seconds       prefer the next detected silence
at 30 seconds       force a boundary
```

These numbers are starting points, not compatibility promises. Fixture quality,
stop-to-output latency, and real dictation traces should determine them.

A silence boundary should not need audio overlap. A forced boundary may retain
a short overlap and deduplicate repeated boundary text. The implementation
should add this complexity only after a fixture demonstrates that a hard split
harms output.

If all slots are occupied, voiced stops capture and drains the retained slots.
It must never overwrite untranscribed audio or allocate an additional slot.

## Whisper inference

The selected backend is CTranslate2 4.6.2 with the English Whisper `small.en`
model and an intentionally narrow C ABI bridge. Fixture measurements retained
the current faster-whisper transcription quality while outperforming the
other measured native candidates on longer technical dictation.

CTranslate2, cpu_features, and spdlog sources are pinned and compiled with
voiced. Static Intel oneMKL and OpenMP archives provide the measured CPU
performance. Model weights remain separate runtime data. The Whisper process
loads the configured converted-model directory so a user may install or replace
a model without rebuilding voiced. The daemon performs no implicit network
access; a setup command downloads to a temporary location, verifies pinned
digests, and atomically installs the complete model.

The Whisper process loads its model and creates reusable inference state during
startup. It processes sealed chunks sequentially by ordinal while capture fills
a different slot. Previous-text conditioning remains enabled so adjacent
chunks preserve spelling and sentence continuity. If the bridge does not carry
that context reliably across calls, the process retains and supplies a fixed
tail of token IDs explicitly.

CTranslate2 reads log-Mel features derived directly from shared-slot PCM. It may
allocate internally; that allocation remains confined to the worker process.
Zig-owned model-worker buffers and queues remain fixed after initialization.

## Stop and cancel

Normal stop seals the current partial slot, stops audio, drains every sealed
chunk in order, persists the resulting transcript, and performs output.

Cancel marks the session as discard-only, requests audio stop, sets Whisper's
shared cancellation word, and waits until neither worker can access the
session buffers. It then resets the slots and transcript without persistence or
output.

The CTranslate2 bridge does not pretend a running native inference call was
canceled. The worker checks the shared cancellation word before publishing any
result, while the supervisor applies a cancellation deadline and kills the
Whisper process if native inference does not return.

## Failure recovery

When the microphone disappears, the supervisor freezes the current published
prefix, asks audio to stop, and enforces a teardown deadline. If teardown
blocks, it kills audio and treats the published prefix as the final chunk. The
prefix remains eligible for transcription and output on a normal-stop path.

When Whisper fails, the current slot remains sealed. The supervisor may restart
the model and retry that slot once. Repeated failures terminate the session
with an explicit error; retry loops are always bounded.

Worker events from an earlier generation cannot publish samples, text, or
lifecycle transitions into the current session.

## Bounds to validate

The implementation must choose and assert explicit values for:

- audio slots count;
- samples per slot;
- maximum total session duration;
- transcript byte capacity;
- pending worker-event capacity;
- capture setup, progress, and teardown deadlines;
- inference, cancellation, and model-startup deadlines; and
- retry counts.

The provisional fifteen-minute session maximum protects against accidental
recording. Reaching it should stop and finalize successfully rather than
truncate or discard the session.
