# Voiced Zig spikes

Two standalone binaries isolate the native boundaries selected for the Zig
implementation:

- `audio-spike` launches an isolated audio worker that records PipeWire blocks
  into a supervisor-created shared exchange.
- `model-spike` computes Whisper log-Mel features and measures CTranslate2
  transcription.

The model spike compiles pinned CTranslate2, cpu_features, and spdlog sources
through Zig's bundled C++ toolchain. It statically links verified Intel oneMKL
and OpenMP archives. The audio spike dynamically links the desktop's PipeWire
client library.

The audio implementation has four explicit module boundaries:

```text
audio_spike.zig
    └── audio_process.zig
          ├── pipewire.zig
          └── audio_exchange.zig
```

`pipewire.zig` owns stream setup, source observation, realtime callbacks,
validation, and teardown. `audio_exchange.zig` owns only the fixed shared-memory
layout and slot transitions. `audio_process.zig` owns the memfd, worker launch,
fixed packets, process deadlines, stop/cancel transmission, pidfd waiting, slot
consumption, and forced termination. Its supervisor-facing operations are
`start`, `requestStop`, `requestCancel`, `receiveReport`, and `killAndReap`.
`start` resolves and launches the installed sibling `audio-process` worker, so
the spike contains no worker-role dispatch or process arguments.
`audio_spike.zig` imports only `audio_process.zig` and owns only argument parsing,
scheduled experiment control, WAV output, and human-readable measurements.

Install PipeWire's development package once so pkg-config can supply its public
headers and linker name:

```bash
sudo apt install libpipewire-0.3-dev
```

Install the verified Intel archives required by the build and the CTranslate2
`small.en` model required at runtime:

```bash
zig build setup
```

They can also be installed independently:

```bash
zig build setup-native
zig build setup-models
```

Build both spikes and the private `audio-process` worker without running any of
them:

```bash
zig build -Doptimize=ReleaseSafe
```

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
infinite float32 samples before publication. It clamps finite samples to
PipeWire's normalized `[-1.0, +1.0]` range and reports the number clipped, so an
overbearing source cannot overflow Whisper's spectral calculations. These
checks keep validation and copying bounded on PipeWire's realtime thread.

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
duration. A skipped or duplicated interval then stops capture while retaining
the valid prefix, even without a Header warning. Older builds continue in
`header-only` mode and print a warning that some dropped intervals cannot be
detected. Every successful report names the build-header, client-library, and
server versions plus the selected validation mode.

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
  --model ~/.local/share/voiced/models/faster-whisper-small.en \
  --audio ../test-fixtures/hello_world.wav \
  --threads 4 \
  --runs 3
```

`--threads` defaults to 4. `--runs` defaults to 3 and must be an odd number
between 1 and 9. `setup-models` installs under `$XDG_DATA_HOME/voiced/models`
when `XDG_DATA_HOME` is set; otherwise it uses `~/.local/share/voiced/models`.

`-Dmkl-prefix=/path/to/prefix` overrides the downloaded
`zig-pkg/mkl` installation. The prefix must contain `include/`,
`opt/compiler/include/`, and the required static archives under `lib/`.
