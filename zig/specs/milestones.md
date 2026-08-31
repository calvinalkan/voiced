# Milestones and Open Questions

This sequence is provisional. A milestone exists to answer a risk or produce a
usable vertical slice; it is not a commitment to preserve an early internal
API.

## 1. Model and audio feasibility

The ASR backend decision is complete. CTranslate2 4.6.2, compiled from source
through `build.zig` and linked with static oneMKL and OpenMP, retained the
current faster-whisper `small.en` fixture quality with the best measured latency
and acceptable resident memory. The named Systran `base.en` variant remains an
explicit lower-memory, lower-latency alternative to the default `small.en`.
Whisper.cpp and the other measured native backends remain rejected prototype
evidence rather than configurable engines.

This milestone is complete. The setup command installs the pinned converted
model atomically, PipeWire resolves and verifies the configured physical Device,
and the callback copies bounded blocks into Voiced-owned shared slots rather
than retaining PipeWire's cyclic buffers.

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

The real CTranslate2 role now replaces deterministic transcription without
changing this ownership model. The remaining milestone work is retaining that
warmed process across sessions, adding the long-running public service loop, and
carrying normal stop through persistence and output. Deterministic roles remain
for repeatable process failures.

## 3. Recoverable audio process

- Drive direct PipeWire capture from the epoll supervisor.
- Publish complete PCM blocks into shared slots.
- Detect explicit device removal and missing callback progress.
- Demonstrate that killing blocked audio preserves the published prefix.
- Exercise physical MV7 disconnect and re-enumeration.

## 4. Resident eager Whisper process

One worker now loads and warms the model, transcribes sealed slots while capture
continues, publishes fixed-mailbox text in order, and remains killable during a
blocked native call. Measurements establish ample three-slot headroom at a
20-second boundary and pipeline exhaustion under deliberately unsuitable
one-second chunks plus model restart.

The implemented production policy uses 30-second physical slots, waits at least
20 seconds before publishing on 300 ms of quiet, forces publication at 30
seconds, and ends automatic listening after observed activity followed by 800
ms of quiet. If the natural boundary already published the utterance, the worker
discards the remaining quiet confirmation tail. Activity and model confidence
now suppress ordinary no-speech chunks and turn disagreement into a structured
failure. Broader probes rejected both previous-text prompting and PCM overlap
because they increased word errors. The remaining work is measured deadlines,
cross-microphone validation, a possible speech-specific VAD, and retaining the
warmed process across sessions.

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

- Can a speech-specific VAD reduce explicit activity/model conflicts without
  publishing non-speech hallucinations or silently dropping quiet speech?
- Can timestamp-aware boundary reconciliation outperform independent chunks on
  broad quality fixtures? Text-only overlap and deduplication did not.
- Should the roughly 403 MiB warmed model remain resident for the service's
  whole lifetime, or unload after an idle interval?
- Does the 4 KiB chunk mailbox cover the decoder's maximum output before the
  duration-derived session allocation is treated as final?
- What startup, inference, cancellation, and total-session deadlines preserve
  useful work under CPU contention while containing a blocked worker?
- Which output key sequence works consistently across the target applications?
- Which exact public commands should remain from the Python CLI?
- Which PipeWire revision or system version should be treated as the supported
  baseline?

None of these questions requires a plugin system or generalized abstraction in
advance. The selected answer should be implemented directly and recorded in the
owning specification.
