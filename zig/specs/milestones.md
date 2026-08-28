# Milestones and Open Questions

This sequence is provisional. A milestone exists to answer a risk or produce a
usable vertical slice; it is not a commitment to preserve an early internal
API.

## 1. Model and audio feasibility

The ASR backend decision is complete. CTranslate2 4.6.2, compiled from source
through `build.zig` and linked with static oneMKL and OpenMP, retained the
current faster-whisper `small.en` fixture quality with the best measured latency
and acceptable resident memory. Whisper.cpp and the other measured native
backends remain rejected prototype evidence rather than configurable engines.

The remaining milestone work is:

- select the converted `small.en` model source and digests for a separate,
  atomic setup download; the daemon itself performs no implicit network access;
- bind enough of PipeWire to select the configured MV7 source and capture 16 kHz
  mono float32 PCM; and
- confirm whether application-allocated PipeWire buffers help the callback path
  without coupling retained recordings to PipeWire's cyclic pool.

## 2. Supervisor and fake workers

The deterministic supervisor now establishes:

- one-binary role dispatch;
- fixed audio and transcript exchanges with narrow role protocols;
- SCM_RIGHTS descriptor handoff;
- one epoll loop over seqpackets, pidfds, timerfd, signalfd, and eventfd;
- count-based publication and slot-conservation assertions;
- absolute deadlines, cancellation, transactional text production, and bounded
  worker retry;
- deferred process replacement without worker-incarnation tags; and
- one session identity at the exchange boundary rather than in every slot.

The remaining milestone work is the resident CTranslate2 role, long-running
public service loop, and normal stop through persistence/output. Real PipeWire
already replaces deterministic audio without changing this ownership model;
the deterministic role remains for repeatable process failures.

## 3. Recoverable audio process

- Drive direct PipeWire capture from the epoll supervisor.
- Publish complete PCM blocks into shared slots.
- Detect explicit device removal and missing callback progress.
- Demonstrate that killing blocked audio preserves the published prefix.
- Exercise physical MV7 disconnect and re-enumeration.

## 4. Resident eager Whisper process

- Load and warm the model once.
- Transcribe sealed chunks while audio fills another slot.
- Measure and tune silence-preferred chunk boundaries.
- Preserve previous-text context across chunks.
- Implement cooperative cancellation and hard process deadlines.
- Demonstrate bounded behavior when inference falls behind capture.

## 5. Output and daily operation

- Persist the latest transcript atomically.
- Offer clipboard content through a supervised `wl-copy --foreground` process
  and optionally send the paste key.
- Provide concise notifications and journal events without transcript text.
- Install one systemd user service and bind the intended compositor hotkey.
- Validate startup, shutdown, suspend/resume, and repeated daily operation.

## 6. Hardening

- Run process-level fault scenarios for every phase and worker boundary.
- Measure stop-to-output latency for short and multi-minute recordings.
- Verify all capacities and deadline policies at their boundaries.
- Audit shared-memory layout, protocol decoding, file permissions, and session
  reset only after every prior worker is reaped.
- Remove abstractions and configuration that the vertical slice did not need.

## Open questions

The implementation should answer these through prototypes or measurements:

- Which exact converted CTranslate2 `small.en` artifact and digests should the
  setup command install?
- What chunk target, silence window, and hard maximum produce the best balance
  of boundary quality and stop latency?
- Do forced chunk boundaries require overlap and deduplication in real
  dictation?
- How many audio slots provide sufficient inference headroom on the target
  machine? Three is the provisional starting point.
- What transcript and total-session capacities are generous without hiding an
  accidental runaway recording?
- Can PipeWire reliably negotiate the desired 16 kHz mono float32 format on the
  target desktop, or should the audio process own bounded conversion?
- Which output key sequence works consistently across the target applications?
- Which exact public commands should remain from the Python CLI?
- Which PipeWire revision or system version should be treated as the supported
  baseline?

None of these questions requires a plugin system or generalized abstraction in
advance. The selected answer should be implemented directly and recorded in the
owning specification.
