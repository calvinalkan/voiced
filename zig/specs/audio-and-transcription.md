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

The supervisor uses `audio_process.zig`, never `pipewire.zig` directly. In the
production path the supervisor creates the audio memfd and publication eventfd,
then `audio_process.zig` launches the worker and transfers those descriptors in
its fixed launch record. `pipewire.zig` owns one stream and its callback
lifecycle. Every process interprets PCM publication through
`audio_exchange.zig`.

The audio process connects directly to PipeWire and requests mono float32 audio
at 16 kHz where graph negotiation permits it. PipeWire may own or recycle its
small graph buffer pool. The process validates each complete callback block and
quantizes it directly into an active signed 16-bit shared audio slot.

The callback release-publishes `published_samples_count_atomic` only after the
corresponding sample bytes and ordinal are readable. It then performs one
nonblocking eventfd increment when the slot becomes complete. If the process
dies during a copy, the count remains zero and consumers see no private prefix.

The callback performs bounded work. It does not allocate, send structured
control messages, transcribe, persist, or wait for another thread. The eventfd
increment is only a doorbell; several slot publications may coalesce into one
counter value.

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

An audio slot has one shared publication value:

```text
published_samples_count_atomic = 0   no complete PCM is visible
published_samples_count_atomic > 0   one immutable PCM prefix is visible
```

The audio process privately tracks its filling slot. The supervisor privately
tracks the one in-flight transcription. Whisper never changes audio-slot state:
it reads the published prefix named by a supervisor command. The supervisor
restores the count to zero only after the corresponding text is represented in
the session transcript.

The transcript mailbox uses one atomic state so cancellation and publication
cannot both win. Zero is empty, one is cancelled, and values from two encode a
committed UTF-8 byte count plus two. The worker writes text and metadata first,
then release-CASes empty to published; cancellation atomically exchanges any
state to cancelled. A valid empty transcription is therefore value two.

The production timing policy is:

```text
physical slot capacity       30 seconds
minimum internal chunk       20 seconds
natural boundary candidate   300 ms quiet
automatic stop               800 ms quiet
forced boundary              30 seconds
```

`audio_policy.zig` is the authoritative source for these named constants and
their sample-count conversions. A slot continues filling until it has at least
20 seconds of audio and the activity detector has observed 300 ms of continuous
quiet. Capture forces publication at 30 seconds when no qualifying pause occurs.
In `listen` mode, observed activity followed by 800 ms of continuous quiet ends
the recording independently of the internal chunk minimum. If the same quiet
run already caused a natural publication at 300 ms, its remaining private
confirmation tail is discarded instead of becoming a silence-only model chunk.

Natural and forced boundaries currently publish independent PCM ranges. A hard
experimental split demonstrated one recoverable divided word, but 250 ms and
500 ms overlaps increased total word errors across broader cuts and dense speech.
Text-only deduplication could not distinguish repeated overlap from intentional
repetition or repair a different word chosen by the preceding chunk. Voiced must
not add overlap without timestamp-aware reconciliation or stronger quality
evidence.

If all slots are occupied, voiced stops capture and drains the retained slots.
It must never overwrite untranscribed audio or allocate an additional slot.

## Whisper inference

The selected backend is CTranslate2 4.6.2 with an intentionally narrow C ABI
bridge. Runtime configuration names a model by publisher and variant rather
than by filesystem path. `Systran/faster-whisper-small.en` is the default for
technical dictation quality; `Systran/faster-whisper-base.en` is the lower-memory,
lower-latency alternative. The setup command installs and verifies both.

CTranslate2, cpu_features, and spdlog sources are pinned and compiled with
voiced. Static Intel oneMKL and OpenMP archives provide the measured CPU
performance. Model weights remain separate runtime data. The Whisper process
loads the configured converted-model directory so a user may install or replace
a model without rebuilding voiced. The daemon performs no implicit network
access; a setup command downloads to a temporary location, verifies pinned
digests, and atomically installs the complete model.

Model residency is explicit. `session` starts loading and one silent warm-up
alongside capture, then destroys the worker after output. `service` requires a
ready worker before capture and keeps it across recordings once the long-running
control loop owns multiple sessions. The one-session supervisor spike waits for
readiness but necessarily shuts the worker down when its command exits.

The Whisper process handles sealed chunks sequentially by ordinal while capture
fills a different slot. Chunks are decoded independently: measured previous-text
prompting did not repair hard splits and worsened one fixture, while PCM overlap
introduced more errors than it removed in the broader boundary probe.

CTranslate2 reads float32 log-Mel features derived directly from shared-slot
signed 16-bit PCM. The extractor converts samples into its centered DFT
workspace and reuses its maximum-sized scratch and feature buffers across
warm-up and every transcription; it never materializes a second complete float
waveform. CTranslate2 may allocate internally, but Zig-owned model-worker
buffers and queues remain fixed after initialization.

The audio slot records whether its own PCM crossed the adaptive activity
threshold. The model worker commits that observation, no-speech probability,
average log probability, and text before atomically publishing the mailbox.
Result recovery after worker death therefore makes the same acceptance decision
as ordinary packet delivery.

Worker-ready and result packets carry model-load, warm-up, feature-extraction,
and inference durations as integer nanoseconds. The supervisor returns their
counts, totals, and maxima in one `SessionTimingReport`, along with the complete
command duration, audio duration, and post-audio drain duration. Human-readable
logs render that report but are not its authoritative representation.

Voiced accepts nonempty text only when activity was observed and Whisper's
no-speech probability is below 0.60. An inactive chunk at or above 0.60 is normal
`no_speech`; a disagreement becomes `speech_detection_conflict`, and empty model
output after observed activity becomes `speech_unrecognized`. These failures
report the model evidence without logging transcript text. The disagreement path
is intentional: a volume detector cannot distinguish very quiet speech from all
non-speech sounds, so Voiced neither publishes a hallucination nor silently
discards uncertain speech.

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

A replacement starts only after the prior worker is reaped and its complete
epoll batch is consumed. No prior-session process can therefore publish samples,
text, or lifecycle transitions into reset storage.

## Bounds

Three 30-second signed 16-bit slots use about 2.75 MiB regardless of session
duration; the supervisor reuses them after transcription instead of retaining
session PCM. The maximum recording duration defaults to one hour and remains a
configuration value. Reaching it stops and finalizes successfully rather than
truncating or discarding the session. Its `u16` representation provides a
natural fail-safe ceiling of 65,535 seconds, or 18 hours, 12 minutes, and 15
seconds.

The supervisor allocates transcript storage once when the session starts. Its
capacity is 64 UTF-8 bytes per configured audio second plus one complete 4 KiB
mailbox result. The rate budget is independent of publication boundaries, while
the extra result guarantees that one maximum-sized publication fits beyond the
rate-derived capacity. One hour reserves 234,496 bytes, or about 229 KiB, on the
heap rather than inside the supervisor's stack. The natural duration ceiling
reserves about 4 MiB. Accepted output beyond this fixed capacity is a structured
`transcript_capacity_exceeded` failure; the supervisor never grows or truncates
the transcript.

Still to validate and lock:

- the 4 KiB result mailbox against the decoder's maximum output;
- capture setup, progress, and teardown deadlines;
- inference, cancellation, and model-startup deadlines; and
- bounded retry counts.
