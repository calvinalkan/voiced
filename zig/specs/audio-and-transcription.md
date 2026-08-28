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

The supervisor uses `audio_process.zig`, never `pipewire.zig` directly.
`audio_process.zig` owns worker launch, the control/report protocol, shared memfd
lifetime, publication consumption, deadlines, and forced termination. The child
then enters `pipewire.zig`, which owns one PipeWire stream and its callback
lifecycle. Both sides access slot state only through `audio_exchange.zig`.

The audio process connects directly to PipeWire and requests mono float32 audio
at 16 kHz where graph negotiation permits it. PipeWire may own or recycle its
small graph buffer pool. The process copies complete callback blocks into an
active shared audio slot.

The callback publishes `samples_count` only after the corresponding sample
bytes are readable. If the process dies during a copy, the supervisor and
Whisper process consume only the previously published prefix.

The callback performs bounded work. It does not allocate, send control
messages, transcribe, persist, notify, or wait for another thread.

Realtime capture selects PipeWire's `client-rt.conf` as well as requesting its
realtime process callback. The configuration loads PipeWire's realtime module,
which uses direct scheduler permission or RTKit to promote the data thread.
Voiced reports the observed scheduler mode and priority. Missing promotion is a
warning rather than a setup failure: callback-progress deadlines still contain
a starved worker, and available Header or graph-cycle information still detects
discontinuous audio.

Before PipeWire can start the callback, the audio process touches every page in
the shared exchange and attempts to lock the mapping. A failed `mlock` is also a
warning rather than a setup failure. The report retains its Linux errno and the
worker's `RLIMIT_MEMLOCK` soft limit because the unlocked pages remain usable
but may be reclaimed under memory pressure.

Production binaries use Zig's ReleaseSafe mode. Internal lifecycle, protocol,
and slot-ownership contracts remain assertions because a violation is a Voiced
defect; bounds, overflow, pointer, optional, and enum safety also remain active.
PipeWire-owned formats, metadata, timelines, and samples are external operating
input and continue through explicit validation and structured errors. Measured
ReleaseSafe callback work remained in the single-digit microseconds at the
median under both idle and all-CPU contention, far below the approximately 20 ms
graph interval.

A target name is a routing request, not proof of the source that produced a
buffer. The audio process observes PipeWire's registry and finds the incoming
Link whose input node is Voiced's stream node. That Link's output node becomes
the immutable source for the recording. The realtime callback cannot copy until
this first relationship is known. Removing the Link, source Node, or underlying
Device stops the recording; linking a different source also stops it rather
than allowing one session to mix microphones.

The report joins the source Node to its Device and retains the node name and
description, graph IDs and object serials, and `device.serial` when supplied.
Graph IDs and object serials are diagnostics for one PipeWire incarnation. A
configured physical microphone is resolved again by stable `device.serial`
before each recording because unplug/replug creates new graph IDs. Discovery
opens no capture stream: absence or ambiguity fails with the available-source
catalog instead of touching the default microphone. After connect, the graph
observer independently verifies that the linked Device carries the configured
serial before opening the realtime sample gate.

Direct PipeWire state and registry events are therefore the primary source of
device removal and stream failure. A monotonic callback-progress deadline
remains a fallback for failures that produce no useful event.

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

The supervisor sends one fixed command over the audio worker's existing
`SOCK_SEQPACKET` control link. PipeWire watches that descriptor on its owner
loop; the realtime callback never polls for control. Stop, cancel, callback
completion, and capture failure compete through the same first-wins terminal
latch.

Normal stop closes the source gate and quits the PipeWire loop. Destroying the
stream then synchronizes with any callback already in progress. Only after that
boundary does the worker publish the final slot, so every complete callback
block accepted before stop belongs to the retained prefix. The supervisor drains
every published chunk in order, persists the resulting transcript, and performs
output.

Cancel follows the same bounded stream teardown but abandons the final private
slot rather than publishing it. Earlier slots may already have been published or
consumed; the `cancelled` result tells the supervisor that the entire session is
discard-only, so none becomes persistence or user output. The audio report keeps
separate captured and published sample counts to make the abandoned final prefix
observable.

The supervisor applies a teardown deadline after either command. An audio worker
that does not report and exit in time is killed and reaped, and that path is
classified as forced termination rather than normal stop or cancellation.

At the session level, cancel also sets Whisper's shared cancellation word and
waits until neither worker can access the session buffers. It then resets the
slots and transcript without persistence or output.

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
