Here is the merged, current master list. It supersedes the older status list.

# Voiced Zig roadmap

## 1. Build, dependencies, and model feasibility — **done**

- CTranslate2 4.6.2 selected.
- CTranslate2, cpu_features, and spdlog compiled through Zig.
- Verified oneMKL/OpenMP downloads.
- Verified and atomic installation of named Systran `base.en` and `small.en` models.
- Model source revision and file digests pinned.
- Model fixture quality and performance measured.
- Build produces binaries without running fixtures.

The model backend and one-session resident worker are proven. Retaining that
worker across multiple supervisor sessions remains open.

---

## 2. PipeWire capture and shared audio exchange — **done**

- Direct PipeWire capture.
- 16 kHz, mono, float32 negotiation.
- Three fixed shared-memory slots.
- Bounded callback copying.
- No allocations or blocking operations in the callback.
- Complete-block publication with release/acquire ownership.
- Pipeline-full behavior.
- NaN/Inf rejection and finite overrange clamping.
- PipeWire buffers returned before callback completion.
- Decision made not to retain PipeWire’s cyclic buffers.

---

## 3. Granular setup diagnostics — **done**

Setup failures carry:

- exact stage;
- error domain;
- native error code;
- native message;
- linked PipeWire version;
- requested target;
- processing mode.

The C boundary exposes semantic operations while preserving the failed internal substage.

---

## 4. Structured runtime and teardown failures — **done**

Runtime reports now distinguish:

- stream errors and disconnection;
- Format removal and malformed Format pods;
- unexpected negotiated formats;
- malformed or corrupted buffers;
- Header discontinuities;
- timeline failures;
- buffer-return failures;
- pipeline exhaustion;
- teardown/disconnect failures.

Reports retain stage, domain, code, detailed context, and the valid captured prefix.

---

## 5. SPA Header metadata — **done**

When PipeWire supplies Header metadata:

- `CORRUPTED` stops capture.
- `DISCONT` stops capture rather than inventing audio.
- `GAP` inserts the exact represented duration as silence.
- Header shape and contents are validated.

Important finding: PipeWire may accept the Header request but provide no Headers.

---

## 6. Timeline continuity fallback — **done where PipeWire permits it**

### PipeWire 1.0.5 and newer

Full validation is implemented using:

- the buffer’s graph-cycle timestamp;
- current graph clock and rate;
- received sample duration;
- integer duration comparison with bounded rounding tolerance.

It detects graph time that advances farther than the received samples can
cover, retaining the valid prefix. PipeWire may legitimately combine multiple
ready graph quanta into one dequeued block; because `pw_buffer.time` identifies
the cycle that queued the combined buffer rather than its first sample, a block
whose sample duration exceeds its timestamp delta is accepted rather than
misclassified as duplicated audio.

### PipeWire 0.3.48

Only Header-based validation is available. A 500 ms worker pause completed with a 519 ms callback gap that the old API could not classify.

Therefore old builds explicitly report:

```text
Timeline validation: header-only
Warning: this PipeWire build cannot detect every dropped audio interval
```

The remaining question is product policy: whether this degraded mode is acceptable as an officially supported baseline.

---

## 7. PipeWire version compatibility — **done**

Validated using an actual isolated Ubuntu 22.04 environment:

```text
headers/library/server: 0.3.48/0.3.48/0.3.48
```

Also validated on the current PipeWire 1.0.5 desktop.

Compatibility work includes:

- guarded post-0.3.48 APIs and struct fields;
- no unavailable imports in old builds;
- initial null Format handling on 0.3.48;
- zero `spa_chunk.stride` handling;
- version reporting;
- Header-only versus full timeline capability selection.

---

## 8. Default and configured source selection — **done**

- No configuration resolves the current default source for each recording.
- `--device-serial` inventories current Audio/Source Nodes and Audio/Devices
  before creating a capture stream.
- A stable Device serial must resolve to exactly one current source node.
- Missing and ambiguous configured devices fail without opening the default
  microphone and report the bounded available-source catalog.
- The post-connect observer independently verifies the linked Device serial
  before opening the realtime sample gate.
- The configured MV7 and a non-default dock microphone both resolved and
  completed capture without changing the desktop default; a one-second MV7 run
  took 1.03 seconds wall-clock including discovery.
- Explicit node names remain a low-level spike control on modern PipeWire and
  Ubuntu 22.04's old media-session.
- Invalid explicit targets cannot silently fall back to another microphone.
- Old builds conditionally provide the legacy `node.target` property; modern
  builds use `target.object`.

This closes both silent-fallback windows: failure during pre-capture resolution
and a session manager linking a different Device after resolution.

---

## 9. Resolved source identity — **done**

The worker now observes PipeWire's registry and joins:

```text
actual source Node -- incoming Link --> Voiced stream Node
       |
       `-- device.id --> Device --> device.serial
```

Every report retains:

- requested target;
- actual source node name and description;
- source global ID and object serial;
- underlying Device global ID and object serial when present;
- stable `device.serial` and device description when present;
- PipeWire header, client-library, and server versions.

The first incoming Link opens the realtime copy gate. Link removal, source-Node
removal, Device removal, or a Link from another source closes the gate and stops
the recording with a structured source-identity error. Binding the selected Node
and Device retrieves identity properties omitted from registry globals on both
modern PipeWire and Ubuntu 22.04's 0.3.48 build.

PipeWire global IDs and object serials change after unplug/replug. The physical
MV7 retained `device.serial=Shure_Inc_Shure_MV7`; configured selection now
resolves that stable value anew before each recording rather than storing a
graph ID.

---

## 10. Worker containment and deadlines — **done**

- Separate audio worker process.
- `SOCK_SEQPACKET` control link.
- Sealed shared memfd.
- Parent-death signaling.
- pidfd-based observation and termination.
- Setup deadline.
- Callback-progress deadline.
- Forced kill containment.
- Worker signal/crash classification.
- Published-prefix retention.
- FD-limit preflight.

Validated:

- limits below 64 fail before entering PipeWire;
- limit 64 succeeds;
- killing the worker is classified;
- killing the supervisor terminates the worker;
- killing PipeWire retains captured audio.

---

## 11. Realtime and memory-lock policy — **done**

- Realtime capture selects PipeWire’s `client-rt.conf` and requests
  `RT_PROCESS`.
- The client configuration loads PipeWire’s realtime module, which uses direct
  scheduler permission or RTKit to promote the callback data thread.
- Scheduler reporting removes `SCHED_RESET_ON_FORK` from the returned policy
  flags and reports the underlying FIFO, round-robin, or normal mode.
- Missing realtime promotion produces a warning rather than failing capture.
- The worker touches every shared-exchange page before capture and attempts to
  lock the mapping.
- Failed memory locking produces a warning with Linux errno and the inherited
  `RLIMIT_MEMLOCK` soft limit rather than failing capture.

Validated on the target desktop:

- RTKit promoted the callback to round-robin priority 20.
- Under forced single-core contention, a low-priority normal callback stalled
  after one block and hit its progress deadline; the promoted callback completed
  with a 21 ms largest gap.
- With `RLIMIT_MEMLOCK=0`, capture completed and reported `EPERM` plus the zero
  soft limit.
- An isolated Ubuntu 22.04 environment without RTKit continued under the normal
  scheduler and printed the expected warning.

---

## 12. Physical hardware disconnect and re-enumeration — **done**

Validated:

- virtual source destruction;
- PipeWire server death;
- missing server during setup;
- valid physical microphone capture on the current desktop;
- physical MV7 unplug, reconnect, and fresh capture;
- an isolated PipeWire 0.3.48 source removal produces `source_disconnected`,
  retains the prefix, and a fresh worker selects the recreated source under its
  new ID.

The physical test before source observation exposed the motivating defect:
WirePlumber silently rerouted the still-running stream to another microphone.
With source observation enabled, the same 60-second test stopped at unplug after
20 seconds and retained its valid prefix. Full timeline validation reached the
first-wins terminal latch before the registry removal and reported
`timeline_discontinuity`; it did not continue on the replacement default source.
The isolated 0.3.48 test separately proves the graph observer's
`source_disconnected` path when registry removal is the first signal.

After reconnect, PipeWire reused source ID `110` and Device ID `114`, demonstrating
that numeric IDs alone cannot establish continuity. Their object serials changed
to `28238` and `28236`, while `device.serial=Shure_Inc_Shure_MV7` remained stable.
A fresh two-second worker resolved that stable identity and completed normally.
Supervisor recovery remains separate.

---

## 13. Production stop and cancel path — **done**

- The supervisor sends one fixed normal-stop or cancel record over the existing
  `SOCK_SEQPACKET` worker link.
- PipeWire watches control on its owner loop; the realtime callback performs no
  control polling or socket work.
- Control, callback completion, and capture failures share one first-wins atomic
  terminal latch.
- Normal stop destroys the stream to synchronize with an in-progress callback,
  then publishes the final slot and reports `stopped`.
- Cancel performs the same bounded teardown, abandons the final private slot,
  reports separate captured and published counts, and produces no output even
  when earlier slots were already consumed.
- The parent gives commanded teardown two seconds, then forcibly kills and reaps
  the worker and reports forced termination.

Validated with zero-sample commands, realtime and main-loop processing, complete
and partially filled slots, earlier publications, a deliberately delayed
consumer, stop/cancel versus natural-completion races, forty short callback-race
runs, and a SIGSTOP-induced teardown timeout. Normal-stop WAV sizes matched the
reported sample counts; cancelled runs left the output absent. The same source
also built and exercised both commands against isolated PipeWire 0.3.48.

---

## 14. Production assertion mode — **done**

Production uses `ReleaseSafe`.

- Trusted lifecycle, worker-protocol, slot-ownership, publication-order, and
  callback pre/postcondition assertions remain active.
- Zig's bounds, overflow, optional, pointer, and enum runtime safety remains
  active around those contracts.
- External PipeWire formats, metadata, timelines, and sample data continue to
  use explicit recoverable validation rather than assertions.
- `ReleaseFast` remains available only for deliberate comparison; it removes
  `std.debug.assert` and must not be used for the production service.

Ten one-second captures per mode completed 16,014 samples each. Under the normal
scheduler, ReleaseSafe versus ReleaseFast had median longest callbacks of 7.5 µs
versus 6 µs and maxima of 27 µs versus 93 µs. With a low-priority runnable load
on all 24 logical CPUs, medians were 6.5 µs versus 7 µs, maxima were 9 µs versus
8 µs, and maximum callback gaps were 25 ms versus 22 ms. This isolated PipeWire
1.0.5 server had no RTKit, so the comparison conservatively exercised the normal
scheduler. ReleaseSafe also passed twenty stop/cancel races and the
pipeline-full path. Its observed cost is immaterial beside the roughly 20 ms
graph interval.

---

## 15. Real supervisor and role launcher — **in progress**

The one-binary process and event-loop boundary is implemented under
`voiced supervisor-spike`. It supports deterministic roles for repeatable fault
scenarios and a real PipeWire-to-CTranslate2 path. It now proves:

- one executable dispatching supervisor, audio, and transcription roles;
- one epoll loop over role seqpackets, pidfds, timerfd, signalfd, and audio
  publication eventfd;
- role-specific launch and runtime protocols;
- explicit memfd/eventfd transfer with `SCM_RIGHTS` and close-on-exec receipt;
- deferred worker replacement without worker-incarnation state;
- current absolute deadline evaluation without deadline-incarnation state;
- role operations that carry their own deadline, with separate capture,
  cancellation, report-exit, and forced-termination meanings;
- count-based PCM and UTF-8 publication with `_atomic` field naming;
- no shared filling, transcribing, or produced slot states;
- one retained in-flight retry count rather than a per-chunk retry table;
- transcript recovery when a worker dies after publishing text but before its
  notification packet;
- bounded retry, pipeline-full drain, cancellation, signal shutdown, and
  parent-death containment;
- real PipeWire descriptor handoff and PCM publication through the same event
  loop, including setup, callback-progress, teardown, and session deadlines;
- real completion and valid-prefix failure drain, including a reportless audio
  worker death after earlier slot publication;
- setup-failure discard, supervisor cancellation, worker-crash containment,
  default and explicit target selection, and realtime or main-loop capture;
- pidfd-authoritative signaling and reaping without a second stored PID;
- one tagged source choice instead of mutually exclusive target optionals;
- four-byte transcription commands, transcription notifications, and fake-audio
  notifications whose payload state remains authoritative in shared memory;
- one CTranslate2 model load and warm-up followed by repeated direct
  shared-slot inference without a WAV intermediary;
- bounded startup and inference diagnostics, including model load, log-Mel,
  inference, no-speech, and average-log-probability observations;
- ReleaseSafe repository code around upstream CTranslate2 and cpu_features
  compiled with their supported native release assumptions.

The reduced-state process suite completed fifty runs of each normal, burst,
slow-consumer, crash-before-result, crash-after-result, repeated-crash, and hang
scenario without failure. Burst publication combined three eventfd increments
into one wakeup while preserving all three slot ordinals. The real PipeWire
worker also retained normal, stop, cancel, and pipeline-full behavior after the
same count-based audio exchange and publication eventfd were installed.

Still needed before this milestone is complete:

- retain the transcription process across more than one session;
- add the public control socket and long-running idle/session loop;
- implement normal-stop drain through final persistence/output rather than
  printing the fake transcript;
- finish recovery policy above the supervisor's now-integrated PipeWire setup,
  progress, teardown, session, report, and process-exit observations.

---

## 16. Supervisor recovery policy — **open**

The audio worker deliberately never reconnects itself.

The supervisor must decide:

- PipeWire unavailable → capped retry/backoff.
- Microphone removed → retain prefix, stop the session, re-enumerate.
- Permission failure → no blind retry.
- Unsupported format → no blind retry.
- Worker crash or hang → kill, reap, and replace.
- Pipeline full → terminate and drain retained chunks.
- Timeline discontinuity → terminate with diagnosis.
- Repeated startup failure → circuit-break rather than looping forever.

These policies cannot be completed correctly inside the standalone audio spike.

---

## 17. Resident eager Whisper worker — **in progress**

Implemented and measured:

- one model load and warm-up per worker;
- direct sealed-slot log-Mel extraction and CTranslate2 inference;
- concurrent capture and sequential inference;
- ordered fixed-mailbox publication;
- cancellation publication race and forced process containment;
- exact-slot retry after worker failure;
- no WAV intermediary;
- no-speech and average-log-probability observations;
- explicit `session` and `service` residency policy at the session boundary;
- structured startup, load, warm-up, feature, inference, audio, drain, and
  complete-command timings.

Still needed:

- retain one warmed worker across supervisor sessions;
- separate end-session from service shutdown;
- keep independent chunks unless timestamp-aware reconciliation makes forced overlap safe;
- replace provisional startup and inference deadlines with measured policy.

A four-thread `small.en` worker loaded in about 430–650 ms, warmed in about
0.9–1.5 seconds, and transcribed production-shaped chunks in about 0.9 seconds
on the target machine. Seven fresh `base.en` production workers loaded in
151–236 ms and completed their silent warm-up in 327–548 ms; evicting their
model files from Linux's page cache moved load to 222–300 ms. A separate
first-call probe measured roughly 456 ms for the first real inference versus
405 ms once warm, with about 289 MiB peak RSS. Session warm-up is therefore
usually hidden after roughly 0.6–0.8 seconds of base-model speech, while service
residency removes it before capture.

The `small.en` worker retained roughly 403 MiB proportional set size when idle
after inference and peaked near 675 MiB. Full equal-priority CPU saturation raised
four-thread inference to 11.8–15.2 seconds and made the current 10-second
inference deadline too short.

A subsequent native bridge sweep compared `base.en` and `small.en`, beam sizes
one and five, and one through sixteen CPU threads. On the six quality fixtures,
`small.en` beam one retained every normalized word produced by beam five while
reducing four-thread inference by roughly 8–17%. `base.en` beam one finished in
0.30–0.47 seconds instead of `small.en`'s 0.90–1.45 seconds and reduced peak RSS
from about 660 MiB to 290 MiB, but changed `letters` to `data` in the original
paragraph and changed six of 27 normalized words in the numbers fixture even
though it retained the required numeric values.

On a representative ten-second final chunk, idle beam-one timings were:

```text
threads          1       2       4       8      16
base.en        933     515     348     268     370 ms
small.en      3122    1751    1128     856    1089 ms
```

Eight threads won only on an idle host. Under equal-priority saturation of every
logical CPU, `base.en` medians were 1.63, 1.72, 1.77, and 8.69 seconds for one,
two, four, and eight threads respectively. Four remains the balanced default;
beam one is now the selected production decoding policy, while choosing
`base.en` instead of `small.en` remains an explicit
latency-versus-dictation-quality decision.

---

## 18. Chunking and capacity measurements — **in progress**

Selected policy, defined as named constants in `audio_policy.zig`:

```text
physical slot capacity       30 seconds
minimum internal chunk       20 seconds
natural boundary candidate   300 ms quiet
forced boundary              30 seconds
automatic stop               800 ms quiet
```

Measured findings:

- three slots have large headroom at a 20-second boundary: one 20-second chunk
  took about 0.86 seconds to transcribe;
- artificial one-second chunks took about 0.9 seconds each, and a model restart
  filled all three slots before recovery;
- a silence boundary at 20.03 seconds preserved every word in the paragraph
  fixture, while a hard 20.30-second split removed the divided word;
- a forced experimental split at 20.30 seconds dropped `Echo`, and 250–500 ms
  overlap restored it. Broader probes rejected overlap: both durations added
  more word errors than they removed, and on dense speech a zero-overlap
  30-second split retained every reference word while 250 ms lost one and
  500 ms changed three;
- a final 43 ms silent slot produced the hallucination `you`;
- no-speech scores separated the current speech fixtures (0.005–0.656) from
  silence and generated noise (0.828–0.944), but a steady tone scored 0.781 and
  faster-whisper's combined default confidence rule did not reject several
  high-confidence silence hallucinations;
- four threads balanced steady-state speed and contention behavior; eight were
  faster on an idle host but substantially slower under full CPU contention;
- an adaptive volume detector runs inside the realtime callback. It calibrates
  300 ms of background, applies separate active and quiet RMS thresholds,
  reports bounded session measurements, publishes 20-second-minimum chunks
  after 300 ms of quiet, and optionally stops after activity followed by 800 ms
  of quiet;
- the first purely relative detector was rejected because quiet startup noise
  falsely started activity and repeatedly reset silence. Absolute RMS floors
  removed those failures on the MV7 ambient recording while still detecting a
  replay attenuated to roughly ten percent of the speech fixture. A replay
  attenuated to roughly two percent fell below the floor, and sustained sound
  remains indistinguishable from speech by volume alone;
- ReleaseSafe observation left the measured callback maximum at 15–17 µs on
  ambient and replayed audio, within the pre-detector measurements;
- a tempo-preserving stretch expanded the 24.3-second paragraph fixture to 30
  seconds without changing its whole-window transcript. The exact native bridge
  measured 1.56 seconds for that whole window, 2.52 seconds of total work with a
  fixed 20-second split, and 3.30 seconds with fixed 10-second splits. Because
  earlier chunks complete during capture, simulated stop latency fell only from
  1.56 to 1.19 and 1.17 seconds respectively;
- fixed 10-second cuts changed about 5.2% of the stretched paragraph's words and
  7.5% of a 30-second composite dictation's words. Fixed 20-second cuts retained
  every paragraph word but changed 3.8% of the composite words. Aligning a
  20-second-minimum boundary to 300 ms of observed quiet retained every word in
  both recordings, with only capitalization or punctuation changes;
- 200, 300, and 500 ms quiet windows all retained every word with a 20-second
  minimum in the broader same-model sweep. An 800 ms window found no useful
  internal pause in the dense paragraph until 29.7 seconds and therefore hid
  almost none of the final inference;
- previous-text prompting did not rescue hard 10-second cuts and made one result
  substantially worse. Chunks therefore remain independent unless a future
  timestamp-aware design proves that it can reconcile overlap safely;
- acceptance metadata now commits activity, no-speech probability, average log
  probability, and text under the mailbox's one atomic publication. Inactive
  chunks at or above 0.60 become normal `no_speech`; activity/model disagreement
  and empty output after activity are structured failures. Silence, generated
  noise, continuous and post-calibration tones, both named models, reportless
  result recovery, and ordinary fake-worker scenarios exercised the policy.
  Isolated PipeWire WAV replay accepted `Hello world, hello my agent` with both
  named models, accepted the two-percent-volume replay, produced normal
  `no_speech` for quiet non-speech, and produced a structured conflict for a
  tone introduced after calibration;
- the shared exchange now stores only signed 16-bit PCM, reducing its three
  slots from 5.49 MiB to 2.75 MiB. Across twenty `small.en` float/int16 pairs
  down to 0.5% fixture gain, eighteen produced identical normalized words; the
  two differences restored `12th` from the full-volume result instead of `12`.
  `base.en` matched nine of ten pairs at 2% and 0.5% gain, but changed several
  words in the 0.5%-gain paragraph whose RMS was about -74 dBFS. Six noise/tone
  pairs retained the same rejected text and stayed well above the no-speech
  threshold. A complete PipeWire-to-Whisper int16 exchange returned the same
  fixture transcript; ten live captures moved median/maximum callback time from
  10/12 µs to 12/17 µs. Capture validates and quantizes PipeWire's F32 bytes
  directly into the slot, and the log-Mel extractor converts int16 samples into
  its reusable centered DFT workspace instead of allocating a separate float
  waveform;
- the three signed 16-bit PCM slots occupy about 2.75 MiB independent of
  recording duration. Session transcript storage is allocated once on the heap
  at 64 UTF-8 bytes per configured second plus one full 4 KiB mailbox result. The
  one-hour default reserves about 229 KiB; with the 4 KiB mailbox, total fixed
  recording/session storage is about 2.98 MiB before process and library
  overhead. The `u16` duration ceiling of 65,535 seconds reserves about 4 MiB
  for the transcript alone. A pitch-preserving synthetic speech sweep found
  7.5–14.7 bytes/second on normal fixtures and 62.9 bytes/second only after a
  4× speed-up had already caused repeated hallucinated text. The extra 4 KiB is
  one complete result of headroom, not another sustained-rate allowance.
  Exceeding the fixed aggregate capacity is a structured failure rather than an
  assertion, allocation, or truncation.

The selected timing policy is implemented. A natural publication starts model
work at 300 ms of quiet; if automatic listening confirms the same pause at 800
ms, capture discards the private confirmation tail instead of publishing a
second silence-only chunk. Explicit experimental slot boundaries below 20
seconds remain fixed for pressure testing.

Still to determine:

- whether the adaptive activity observations remain reliable across real quiet
  speech, background noise, and other microphones;
- whether a speech-specific VAD can replace explicit activity/model conflicts
  without reintroducing silent hallucinations or dropping quiet speech;
- whether timestamp-aware forced-boundary reconciliation ever beats independent
  chunks on broader quality fixtures;
- verification that the 4 KiB per-chunk mailbox covers maximum decoder output;
- integration of the configurable one-hour default into the future service configuration;
- inference, startup, and cancellation deadlines;
- actual stop-to-output latency with a worker retained across sessions.

---

## 19. Output and daily operation — **open**

Still needed:

- atomic latest-transcript persistence;
- supervised `wl-copy --foreground`;
- optional paste key;
- notifications;
- journal events without transcript text;
- public control socket and CLI;
- systemd user service;
- compositor hotkey;
- startup/shutdown and suspend/resume behavior.

---

## 20. Integration and final hardening — **open**

- Replace the spike parent with the supervisor.
- Feed exchange slots to the model process.
- Add process-level scenarios to `test.sh`.
- Automate Ubuntu 22.04 and 24.04 compatibility checks.
- Exercise every worker boundary and supervisor phase.
- Test multi-minute recording and concurrent inference.
- Verify all capacities and deadlines at their boundaries.
- Audit file permissions and persistence.
- Remove spike-only controls and unnecessary abstractions.

# Recommended order from here

1. Validate the implemented activity policy across microphones and noisy rooms,
   then choose the remaining resident-memory and deadline policies from the
   measured worker results.
2. Retain the model worker across sessions in
   the long-running control loop.
3. Add persistence, clipboard, CLI, and systemd operation.
4. Perform final integration hardening.

The audio worker's data path, source selection, identity, physical-device
behavior, and error contract are now largely complete. The remaining audio work
is supervisor control rather than basic capture or buffer handling.
