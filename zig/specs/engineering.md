# Engineering Principles

## Scope of the style

The Zig implementation follows TigerStyle-inspired constraints where voiced
controls the code and representation. PipeWire, CTranslate2, oneMKL, libc, and
other native dependencies may allocate internally. Their behavior is
confined to worker processes and bounded by supervisor deadlines.

The goal is not allocation purity. The goal is predictable ownership, bounded
resource use, visible control flow, and reliable recovery from dependencies
that do not cooperate.

## Allocation and bounds

Zig-owned memory is allocated during initialization from fixed backing storage.
After a role reports ready, its normal path receives no general-purpose
allocator.

Every queue, array, protocol packet, recording, transcript, retry, and deadline
has an explicit maximum and overflow policy. Loops are bounded unless they are
a role's deliberate top-level event loop.

Overflow never silently truncates user data. A full audio pipeline stops and
drains. A full transcript fails explicitly. A full command queue rejects the
new request.

## Assertions

Assertions should be dense around state transitions and representation
boundaries. They should establish both expected state and impossible negative
space.

The implementation should assert at least:

- compile-time capacity arithmetic, field widths, offsets, sizes, and atomic
  alignment;
- one authoritative session generation across active buffers and messages;
- valid source and destination states for every transition;
- conservation of every audio slot across free, filling, ready, and busy sets;
- sample, text, queue, and protocol counts within their capacities;
- payload visibility before release publication;
- transcript write offsets matching the produced prefix;
- deadlines belonging to the phase and worker generation they may terminate;
- no active worker access before a canceled session reuses its buffers; and
- no output before normal session commit.

Assertions represent programmer errors and violated internal invariants.
Malformed client input, unsupported protocol versions, missing devices, full
external filesystems, and native-library failures are handled and reported
rather than asserted.

## Source shape and naming

The source follows the semantic-batch style exemplified by
`snicco-js-toolchain/zig/parser-one-shot-token-tape/src/numeric_literal.zig`.
A file begins with a short ownership preamble when multiple declarations depend
on a non-obvious shared contract, then imports, aliases, caller-facing types,
and its principal operation or central state type.

Principal operations expose their control flow from top to bottom. Substantial
inline phases may use short dividers and preambles that establish hidden entry
state, ordering, or invariants. Extracted operations follow the central
operation they support; they exist only for an independent contract, failure
policy, lifetime, or a bounded region that would obscure the principal flow.
The implementation does not build helper call trees merely to shorten
functions.

Comments preserve caller obligations, representation invariants, indirect
callback effects, required ordering, failure consequences, and reasons an
apparently simpler change is unsafe. They do not narrate visible statements.
Assertions stay adjacent to the values and transitions they establish.

Each file begins with imports and concise aliases for externally owned types it
uses repeatedly:

```zig
const std = @import("std");
const assert = std.debug.assert;

const exchange = @import("exchange.zig");
const AudioExchange = exchange.AudioExchange;
```

Central types and operations appear before their local implementation. Principal
control flow remains visible rather than being divided into chains of helpers.
A function is extracted only when it owns a meaningful contract, invariant,
failure policy, or independent lifetime.

Names retain domain and units where confusion is possible, such as
`stop_deadline_monotonic_ns`, `published_samples_count`, and
`transcript_bytes_capacity`. Index, count, offset, size, capacity, minimum, and
maximum are not interchangeable.

The initial source layout should remain shallow:

```text
zig/src/
├── main.zig
├── supervisor.zig
├── pipewire.zig
├── audio_process.zig
├── audio_exchange.zig
├── transcriber.zig
├── output.zig
└── config.zig
```

A new file requires an independent lifecycle, state model, representation, or
contract. Generic `helpers`, `utils`, `manager`, and `service` layers are not
part of the intended design.

## Shared memory and protocols

Shared structs use an externally defined layout and fixed-width fields. Their
critical offsets, sizes, and alignments have compile-time assertions. They do
not contain process-local pointers.

Socket messages use explicit bounded encoding. Logical Zig unions should not be
sent by treating their compiler-selected memory layout as a wire format.

Payload publication follows one rule:

```text
write complete payload
publish count or completion with release ordering
reader acquires publication
read only the published prefix
```

Generation checks protect every delayed event and shared region reuse.

## Dependencies

The initial implementation targets Zig 0.16.0, the compiler installed on the
target machine when this specification was written. A toolchain change is an
explicit repository decision rather than an unresolved portability goal.

The implementation targets x86-64 Linux and PipeWire directly. Portability
abstractions are not required. Pinned CTranslate2, cpu_features, and spdlog
sources are compiled directly by `build.zig` with Zig's bundled Clang. Generic
code targets an SSE4.1 baseline; AVX, AVX2/FMA, and AVX-512 kernels are separate
objects selected by CTranslate2 at runtime. The build does not require CMake or
a separate system C/C++ compiler.

Pinned Intel oneMKL and OpenMP wheels supply verified static archives and
headers because oneMKL is not source-available. They are linked into the
executable rather than discovered as system shared libraries. The upstream
CTranslate2 Python wheel is not a build input: it supplies a shared library with
a libstdc++ C++ ABI and packaged runtime search-path requirements, not the
static libc++ configuration voiced ships. CTranslate2 is exposed only through
the voiced C ABI bridge behind the Whisper process boundary; model weights
remain separately installable runtime data.

Dependencies should be few, pinned, and justified by a capability that would be
riskier to reproduce. A dependency does not weaken supervisor validation or
remove its deadline.

## Testing and fault injection

Lifecycle and recovery behavior is tested at process level with real sockets,
shared mappings, worker exits, and observable output. Tests should cover:

- normal short and multi-chunk dictation;
- stop during every lifecycle phase;
- cancel during capture and inference;
- microphone removal and callback stall;
- audio setup and teardown hangs;
- Whisper failure, hang, cancellation, and restart;
- stale generation messages;
- slot and transcript capacity boundaries;
- persistence, clipboard, and paste failures;
- shutdown with active work; and
- systemd-style restart recovery.

Fault injection is available only to named test instances and cannot be enabled
accidentally in the user's service.

Audio fixtures compare transcript quality across chunk boundaries and against
the current implementation. Performance measurements report capture overhead,
chunk inference speed, stop-to-output latency, model startup, and resident
memory.
