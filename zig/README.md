# Voiced Zig implementation

Three executable entry points establish the native boundaries and supervisor:

- `voiced supervisor-spike` runs the one-binary supervisor with deterministic
  roles, real PipeWire audio, or the complete PipeWire-to-CTranslate2 path.
- `audio-spike` independently exercises the same PipeWire worker and writes
  retained PCM to WAV.
- `model-spike` computes Whisper log-Mel features and measures CTranslate2
  transcription.

The model spike compiles pinned CTranslate2, cpu_features, and spdlog sources
through Zig's bundled C++ toolchain. It statically links verified Intel oneMKL
and OpenMP archives. The audio spike dynamically links the desktop's PipeWire
client library.

The supervisor implementation remains shallow:

```text
main.zig
└── supervisor.zig
    ├── audio_process.zig
    │   ├── pipewire.zig
    │   └── audio_exchange.zig
    ├── transcription_process.zig
    └── descriptor_handoff.zig
```

`main.zig` chooses the user-facing supervisor spike or one private worker role.
`supervisor.zig` owns epoll ordering, session outcome, absolute deadlines,
retries, slot release, and final transcript acceptance. Deterministic roles use
exactly the seqpacket, pidfd, memfd, eventfd, descriptor-handoff, and
parent-death contracts used by the attached PipeWire and resident CTranslate2
roles.

`audio-spike` remains a focused adapter over five operations from
`audio_process.zig`: `start`, `requestStop`, `requestCancel`, `receiveReport`,
and `killAndReap`. `pipewire.zig` owns the real stream and callbacks;
`audio_exchange.zig` owns count-based PCM publication. The transcription result
mailbox belongs to `transcription_process.zig`.

Install PipeWire's development package once so pkg-config can supply its public
headers and linker name:

```bash
sudo apt install libpipewire-0.3-dev
```

Install the verified Intel archives required by the build and both named
CTranslate2 English Whisper models:

```bash
zig build setup
```

They can also be installed independently:

```bash
zig build setup-native
zig build setup-models
```

Build the supervisor, both boundary spikes, and the private real-audio worker
without running any of them:

```bash
zig build -Doptimize=ReleaseSafe
```

Exercise the complete process boundary without opening a microphone or loading a
model:

```bash
./zig-out/bin/voiced supervisor-spike normal
./zig-out/bin/voiced supervisor-spike burst_publications
./zig-out/bin/voiced supervisor-spike transcription_crash_after_result
```

Replace only deterministic audio with the real default PipeWire source:

```bash
./zig-out/bin/voiced supervisor-spike pipewire \
  --seconds 5 \
  --slot-seconds 1
```

`--seconds` configures the maximum duration of this recording. It defaults to
one hour and accepts values through 65,535 seconds—the natural `u16` limit of
18 hours, 12 minutes, and 15 seconds. Reaching it performs a normal stop and
final drain. Add `--listen` to end early after speech followed by 800 ms of
quiet. Without it, quiet affects internal chunk boundaries but never ends
capture before that configured limit.

Without `--model`, this command drives real PCM publications through the
supervisor while transcription emits deterministic `chunk-N` text. Add a named model to exercise direct shared-memory inference:

```bash
./zig-out/bin/voiced supervisor-spike pipewire \
  --seconds 25 \
  --slot-seconds 20 \
  --model Systran/faster-whisper-small.en \
  --threads 4
```

`Systran/faster-whisper-small.en` is the default production choice for
technical dictation quality. `Systran/faster-whisper-base.en` uses less memory
and returns sooner at the measured cost of more changed words. Setup installs
both under `$XDG_DATA_HOME/voiced/models/Systran/`, or under
`~/.local/share/voiced/models/Systran/` when `XDG_DATA_HOME` is unset.

`--model-residency session` is the default: model loading and one silent warm-up
start alongside capture, so ordinary dictation hides most startup work.
`--model-residency service` requires the model to be ready before capture. The
one-session spike then shuts it down with the command; the future long-running
service keeps that same ready worker across recordings.

The model worker reuses its loaded model for every sealed slot. Each mailbox
commit includes the audio-activity observation and Whisper confidence values
used to accept its text. Inactive
chunks with no-speech probability at least 0.60 become a normal `no_speech`
result. Any disagreement between activity and model confidence fails explicitly
instead of publishing a likely hallucination or silently discarding possible
quiet speech. It accepts the same
`--device-serial`, low-level `--target`, and `--main-loop` source controls as the
audio experiment. Capture setup, sample-progress, teardown, session, inference,
and worker-exit deadlines belong to the epoll supervisor rather than either
worker.

Model-ready and result packets already carry nanosecond timings as structured
fields. The supervisor aggregates worker startup, model load, warm-up, feature
extraction, and inference totals and maxima into `SessionTimingReport`, together
with audio-start-to-finish and audio-finish-to-session-completion durations.
Console output is only a rendering of that report; later persistence and control
responses can consume the structured value directly.

ReleaseSafe is the production policy. It retains `std.debug.assert`, overflow,
bounds, enum, and other Zig runtime-safety checks around trusted lifecycle and
shared-exchange contracts. PipeWire formats and buffers remain explicitly
validated in every mode. ReleaseFast is reserved for deliberate performance
comparison because it removes the internal assertions that diagnose Voiced
defects.

In ten one-second runs per mode, ReleaseSafe's median longest callback was 7.5
µs versus ReleaseFast's 6 µs; maxima were 27 µs and 93 µs. With one sustained
low-priority load on every logical CPU, medians were 6.5 µs and 7 µs and the
largest callback gaps were 25 ms and 22 ms. All runs completed 16,014 samples.
The isolated server could not provide RTKit, so these are conservative
normal-scheduler measurements rather than a claim that safety checks improve
speed. They show no meaningful ReleaseSafe cost against a roughly 20 ms graph
interval.

Record five seconds from the default PipeWire source:

```bash
./zig-out/bin/audio-spike --output /tmp/voiced-audio.wav
```

By default, the worker selects PipeWire's `client-rt.conf` and requests its
realtime process callback. That client configuration loads PipeWire's realtime
module, which can ask RTKit to promote the callback data thread when ordinary
process limits do not permit realtime scheduling. Capture continues with a
warning when promotion is unavailable; setup and progress deadlines still
contain a starved worker. `--main-loop` provides the non-realtime comparison
path. `--device-serial <serial>` selects a configured physical microphone by
stable PipeWire `device.serial`; for example:

```bash
./zig-out/bin/audio-spike \
  --device-serial Shure_Inc_Shure_MV7 \
  --output /tmp/voiced-audio.wav
```

Before creating a capture stream, the worker inventories current source Nodes
and Devices and resolves that serial to exactly one current node name. An absent
or ambiguous device fails without opening the default microphone and reports
the available sources. The observer then verifies that PipeWire actually linked
the resolved Device before permitting sample copies.

`--target <pipewire-node-name>` remains a mutually exclusive low-level spike
control for virtual sources and graph experiments. `--seconds` defaults to 5
and accepts values from 1 through 90. `--stop-after-ms <0-90000>` makes the
parent send a normal-stop command: the worker closes its sample gate, waits for
stream destruction to synchronize with any callback already running, and
publishes the final partial slot before reporting `stopped`.
`--cancel-after-ms <0-90000>` sends the mutually exclusive cancel command. It
abandons the private final slot, reports `cancelled`, and leaves the requested
output path absent even when earlier complete slots had already been consumed.
The parent allows two seconds for either command to finish teardown, then kills
and reaps an unresponsive worker and reports forced termination.

`--slot-seconds <1-30>` shortens the publication boundary without shrinking the
production 30-second slot capacity; use it to exercise concurrent publication
and physical-slot reuse quickly. `--consumer-delay-ms <0-10000>` keeps each
publication occupied for a controlled interval, simulating a slow model process
and eventually producing `pipeline_full` when all three slots remain
unavailable.

The activity lines in the final report come from the adaptive volume detector
that drives natural chunking and optional automatic stop. It calibrates the
first 300 ms, then reports active and quiet sample counts, run lengths, the
learned noise floor, and its hysteresis thresholds. `audio_policy.zig` fixes the
production policy at a 30-second slot capacity, a 20-second internal minimum, a
natural boundary after 300 ms of quiet, and automatic stop after 800 ms of quiet.

Capture publishes a natural boundary only after the current slot contains at
least 20 seconds. `--listen` arms automatic stop only after activity has been
observed, so an idle microphone still reaches the explicit duration limit. When
a 300 ms boundary starts Whisper early and the same quiet run reaches 800 ms,
the worker discards the private 500 ms confirmation tail rather than sending a
second silence-only chunk to Whisper. Explicit `--slot-seconds` values below 20
remain fixed experimental boundaries for pipeline-pressure tests.

Forced 30-second boundaries currently use no PCM overlap. A 250–500 ms overlap
recovered one deliberately divided word, but broader cuts added more word errors
than they removed, including on dense speech designed to force the boundary.
Voiced therefore keeps the simpler independent chunks until timestamp-aware
reconciliation or stronger evidence can prevent overlap from deleting or
repeating dictated words.

Neither the requested node nor the discovery result is trusted as the source
identity. PipeWire represents
one capture as a source-node Link into Voiced's stream node, and WirePlumber can
replace that Link after a microphone disappears. The worker observes the graph,
locks the first concrete source before permitting realtime sample copies, and
stops the recording if its Link, source Node, or Device disappears or a different
source is linked. Reports include the source name and description, ephemeral
PipeWire IDs and object serials, and the underlying `device.serial` when the
source belongs to a device. That stable device serial survives the unplug/replug
case tested with the MV7; PipeWire IDs and object serials do not.

The audio worker rejects callback blocks larger than 100 ms and rejects NaN or
infinite PipeWire samples before publication. It clamps finite samples to
PipeWire's normalized `[-1.0, +1.0]` range, reports the number clipped, and
quantizes them directly into the shared signed 16-bit slot. These checks keep
validation and conversion bounded on PipeWire's realtime thread; no second
float PCM block is retained.

Before capture, the worker touches every shared-exchange page and attempts to
lock the complete mapping in memory. Locking is best effort: failure does not
make the exchange unsafe, but the report warns with the Linux errno and
`RLIMIT_MEMLOCK` soft limit because reclaimed pages could delay a callback.

The same source builds against Ubuntu 22.04's PipeWire 0.3.48 headers and
Ubuntu 24.04's PipeWire 1.0.5 headers. Build the audio executable on the Ubuntu
release where it will run: it links that host's PipeWire and libc rather than
pretending one desktop binary covers both releases.

Every build requests SPA Header metadata during buffer negotiation. When
PipeWire supplies it, Voiced rejects corrupted or discontinuous blocks and
inserts zero samples for the exact duration of Header-marked gaps. Header
metadata remains optional: PipeWire can accept the request but omit it on a
valid audio-converter stream.

PipeWire 1.0.5 adds the per-buffer timestamp needed for full timeline
validation. When both the build headers and loaded library provide it, Voiced
also compares every buffer's cycle with the graph clock and received sample
duration. Graph time advancing beyond the duration represented by received
samples then stops capture while retaining the valid prefix, even without a
Header warning. PipeWire may combine multiple ready graph quanta in one buffer;
the validator accepts that longer block because `pw_buffer.time` identifies its
queue cycle rather than the first contained sample. Older builds continue in
`header-only` mode. The worker publishes that already-resolved capability before
the first callback, so the supervisor warns once when capture starts without
parsing the PipeWire version again. Every successful report names the
build-header, client-library, and server versions plus the selected validation
mode.

The three shared signed 16-bit PCM slots occupy about 2.75 MiB in total
regardless of recording length: consumed slots are reused rather than retaining
the complete recording.
The supervisor allocates the session transcript once at 64 UTF-8 bytes per
configured audio second plus one 4 KiB result. The extra 4 KiB guarantees that
one maximum-sized mailbox result fits beyond the duration-based budget. The
one-hour default reserves about 229 KiB; the natural `u16` duration ceiling
reserves about 4 MiB. At the one-hour default, the audio exchange, transcript
mailbox, and aggregate transcript together occupy about 2.98 MiB of fixed
recording/session storage, excluding the Whisper model and ordinary process or
library overhead. If accepted model output exceeds this fixed capacity, the
supervisor reports `transcript_capacity_exceeded` and discards the session
instead of reallocating or truncating text. Allocation failure remains an
ordinary session-start failure.

The parent requires the first copied samples within three seconds and continued
sample progress at least every two seconds. It uses kernel parent-death signaling
and unconditional `SIGKILL` containment so a crashed, stopped, or orphaned
worker cannot retain the microphone. This process split is a lifecycle boundary,
not a security boundary: Voiced asserts its own packet and shared-memory
contracts on both sides, while validating formats and buffers received from
PipeWire. Setup failures report the exact stage, native error domain and code,
platform message, linked PipeWire version, requested target, and processing
mode. Runtime reports separately preserve the actual linked source and classify
source removal or replacement. Their captured and published sample counts differ
only when cancel discards the final unpublished slot. Deadlines report the stage
and last observed callback/sample counters. The memfd is sealed against
resizing, and the worker rejects a descriptor soft
limit below 64 before entering PipeWire.

Transcribe either that recording or the checked-in fixture:

```bash
./zig-out/bin/model-spike \
  --model ~/.local/share/voiced/models/Systran/faster-whisper-small.en \
  --audio ../test-fixtures/hello_world.wav \
  --threads 4 \
  --beam-size 1 \
  --runs 3
```

`--threads` defaults to 4. `--beam-size` defaults to the selected greedy value
of 1 and accepts 1 through 16 so decoding policy can be measured without
rebuilding. `--runs` defaults to 3 and must be an odd number between 1 and 9.
`--input-gain <0-1>` scales the signed 16-bit fixture in place before feature
extraction. Unlike the supervisor's named-model option, this measurement binary
deliberately accepts a model directory so it can probe uninstalled conversions.

`-Dmkl-prefix=/path/to/prefix` overrides the downloaded
`zig-pkg/mkl` installation. The prefix must contain `include/`,
`opt/compiler/include/`, and the required static archives under `lib/`.
