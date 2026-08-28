Here is the merged, current master list. It supersedes the older status list.

# Voiced Zig roadmap

## 1. Build, dependencies, and model feasibility — **done**

- CTranslate2 4.6.2 selected.
- CTranslate2, cpu_features, and spdlog compiled through Zig.
- Verified oneMKL/OpenMP downloads.
- Verified and atomic `small.en` model installation.
- Model source revision and file digests pinned.
- Model fixture quality and performance measured.
- Build produces binaries without running fixtures.

The model **backend** is proven; the resident model worker is not yet implemented.

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

It detected dropped cycles under worker pauses and CPU pressure while retaining the valid prefix.

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
scenarios and the real PipeWire role with deterministic transcription. It now
proves:

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
  notifications whose payload state remains authoritative in shared memory.

The reduced-state process suite completed fifty runs of each normal, burst,
slow-consumer, crash-before-result, crash-after-result, repeated-crash, and hang
scenario without failure. Burst publication combined three eventfd increments
into one wakeup while preserving all three slot ordinals. The real PipeWire
worker also retained normal, stop, cancel, and pipeline-full behavior after the
same count-based audio exchange and publication eventfd were installed.

Still needed before this milestone is complete:

- replace deterministic transcription with the resident CTranslate2 role;
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

## 17. Resident eager Whisper worker — **open**

Still needed:

- load and warm the model once;
- consume sealed shared audio slots;
- transcribe while recording continues;
- preserve chunk order;
- previous-text conditioning;
- cooperative cancellation word;
- inference deadline and forced worker kill;
- bounded model restart/retry;
- fixed transcript exchange;
- no WAV intermediary.

The current `model-spike` proves inference but reads a complete WAV and exits.

---

## 18. Chunking and capacity measurements — **open**

Provisional policy:

```text
before 20 seconds   keep filling
20–28 seconds       prefer silence
at 30 seconds       force boundary
```

Still to determine:

- silence detector and window;
- forced-boundary overlap;
- text deduplication;
- whether three slots survive real inference load;
- transcript capacity;
- total-session capacity;
- fifteen-minute stop behavior;
- inference and cancellation deadlines;
- actual stop-to-output latency.

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

1. Attach the resident Whisper worker and measure eager chunking against real
   PipeWire publications.
2. Retain that model worker across sessions and add the long-running control
   loop.
3. Add persistence, clipboard, CLI, and systemd operation.
4. Perform final integration hardening.

The audio worker's data path, source selection, identity, physical-device
behavior, and error contract are now largely complete. The remaining audio work
is supervisor control rather than basic capture or buffer handling.
