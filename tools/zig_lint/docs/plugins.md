# Native lint plugins

This document defines the contract for native Zig lint plugins. The implementation must preserve
this contract unless the plugin compatibility epoch changes.

Plugins extend the linter with repository-specific `Rule.Definition` values. The host loads only
explicitly selected shared libraries, parses each source file once, and lends the resulting AST and
line indexes to plugin rules. The SDK adapts the raw dynamic-library boundary back into the same
`Rule.Context` API used by built-in rules.

The format is a trusted, version-locked Zig-to-Zig integration. It uses C-compatible declarations
only to transport values between two Zig images; it is not a stable general-purpose C API and does
not make Zig's native AST portable across compiler versions.

## Author-facing shape

A plugin defines ordinary rules and exports one immutable descriptor:

```zig
const std = @import("std");
const zig_lint = @import("zig_lint");
const Plugin = zig_lint.Plugin;
const Rule = zig_lint.Rule;

const rules = [_]Rule.Definition{
    Rule.define("company_name", lintCompanyName),
};

fn lintCompanyName(context: *Rule.Context) Rule.Error!void {
    for (0..context.ast.tokens.len) |token_index| {
        const token: std.zig.Ast.TokenIndex = @intCast(token_index);
        if (context.ast.tokenTag(token) != .identifier or
            !std.mem.eql(u8, Rule.Context.identifierText(context.ast, token), "todo_value"))
        {
            continue;
        }

        try context.report(.{
            .token = token,
            .message = "`todo_value` is not an accepted project name",
            .help = "use a name that describes the value's role",
            .fix = context.tokenReplacement(token, "descriptive_value"),
        });
    }
}

export const zig_lint_plugin_v1 = Plugin.define("company", &rules);
```

`Plugin.define` performs compile-time validation and produces descriptor storage with static
lifetime. Plugin authors do not write C trampolines, structure-size checks, allocator adapters, or
report callbacks. The SDK generates those details from `Rule.Definition`.

The host qualifies each rule with its plugin namespace. The example reports the external rule ID
`company/company_name`.

## Trust and scope

A plugin has the same trust level as the linter process and the repository that selected it. The
host does not sandbox plugins or attempt to contain malicious native code. A plugin can corrupt
memory, block forever, terminate the process, or violate every pointer contract in this document.
Descriptor validation catches accidental incompatibility and produces useful diagnostics; it is
not a security boundary.

Version 1 provides syntax-tree inspection. It does not provide compiler semantic analysis, type
resolution, package resolution, or whole-program indexing.

Successful findings and fixes may depend only on immutable plugin implementation data and the
semantic values of tracked inputs supplied by the host. Pointer addresses, allocator behavior,
allocation outcomes, timing, and execution infrastructure are not semantic inputs. A rule may
recover from allocation refusal only when recovery preserves its findings and fixes; otherwise it
fails the invocation.

Plugins must not discover additional result-affecting inputs themselves. Reading arbitrary files,
environment variables, time, randomness, network responses, invocation history, or mutable process
state to change lint results is outside the contract. Future host APIs may expose configuration,
environment, or file snapshots as tracked inputs; the host must include those values in lint-cache
identity.

### Source language contract

`LintOptions.target_zig_version` selects the Zig language and standard-library contract that rules
target. It defaults to the exact `builtin.zig_version` of the toolchain that built the linter. The
linter passes the selected semantic version, including prerelease and build metadata, to every
built-in and plugin rule as `Rule.Context.target_zig_version`. It does not infer the value from source
text, `build.zig.zon`, the environment, or other project state.

Version-sensitive rules must use `target_zig_version` when deciding whether to report a diagnostic
or offer a fix. Version-neutral rules may ignore it. For example, a rule that recommends the Zig
0.15.1 writer API can gate that recommendation explicitly:

```zig
const writer_api_version: std.SemanticVersion = .{
    .major = 0,
    .minor = 15,
    .patch = 1,
};

if (context.target_zig_version.order(writer_api_version) == .lt) {
    return;
}
```

Zig 0.14 code commonly obtained stdout through a syntactically ordinary field access:

```zig
const stdout_file = std.io.getStdOut().writer();
```

A newer parser can accept that expression without resolving whether `std.io.getStdOut` exists. Zig
0.15.1 replaced this API with the new `std.Io` model, so the gated rule may recommend
`std.fs.File.stdout().writer(&buffer)` for a target of Zig 0.15.1 or newer while leaving an
intentionally Zig 0.14 repository alone. The API migration is documented in the
[Zig 0.15.1 release notes](https://ziglang.org/download/0.15.1/release-notes.html#Upgrading-stdiogetStdOutwriterprint).

The selected target does not switch parsers, AST helpers, or formatters. Each linter build parses and
interprets every source file with the toolchain used to compile it. Zig source files do not declare a
language version, and `std.zig.Ast.parse` does not accept one. Any source accepted by the host parser
is eligible for linting; when parsing fails, the host reports parse diagnostics and does not invoke
rules that require a valid AST. Formatting remains the canonical formatting of the linter's
toolchain.

During cached directory scans, changing `target_zig_version` establishes a distinct lint-cache
identity, including changes only to prerelease or build metadata. Cached lint-clean facts are never
reused across target Zig versions. Explicit file inputs do not use persistent cache state.
Token and AST cache entries remain eligible because the selected target does not change the parser
that produced them.

For authoritative parser and formatter behavior, use the linter compatibility epoch built for the
Zig version targeted by the repository. Supporting several Zig grammars or formatters in one process
would require several implementations or a project-owned normalized representation and is outside
this native zero-copy design.

### Why plugins use the host's Zig toolchain

Only three designs can make a plugin consume an AST produced by another compilation:

1. Zig publishes and preserves a stable native AST ABI.
2. The linter defines and maintains a project-owned stable AST representation or host-query API.
3. The host and plugin use the same Zig compiler, standard library, and native AST contract.

Zig does not provide the first guarantee. The second design requires converting every native AST or
routing AST access through a broad callback API; it adds representation maintenance plus copying,
memory, or callback overhead and abandons the direct native `Rule.Context` model.

Version 1 chooses the third design. The linter remains an independently buildable artifact that does
not compile the daemon or inference runtime, and the SDK remains a small source dependency. A
repository without a matching prebuilt linter can build the linter and its plugins with its pinned
Zig toolchain. Keeping that local build path fast is part of the compatibility strategy, not merely
a developer convenience.

## Compatibility model

Compatibility has two levels:

1. A hard compatibility epoch determines whether the host may interpret and invoke the plugin.
2. Append-only host API fields allow limited growth within one epoch.

### Hard compatibility epoch

`abi_major` is a compatibility epoch, not merely a revision of the transport declarations. The
project increments it for any change that requires plugins to be rebuilt, including:

- a change to the pinned Zig compiler or standard library;
- an incompatible `RawAst` or `RawAllocator` change;
- a rule or report callback signature change;
- a status-code meaning change;
- a change to existing ownership, lifetime, threading, or reporting semantics; or
- a change to any fixed structure's field type, offset, or meaning.

One published epoch therefore has one pinned Zig identity and one native AST contract. The plugin
descriptor also records the complete `builtin.zig_version_string`. The host compares that string
byte-for-byte so an incorrectly built plugin is rejected even if its SDK forgot to advance
`abi_major`. Prerelease and build identifiers are part of the comparison.

A patched compiler or substituted standard library that retains the same version string is outside
the supported contract. The supported build path uses the project's pinned compiler and vendored
SDK. Rebuilding the host and plugin with the same epoch, SDK, compiler and standard library, target,
and compatible `Ast.Node.Data` representation restores binary compatibility. Source changes may
still be required when the author-facing rule API changes.

Every change to the published pinned Zig compiler or standard library advances `abi_major`, even
when the observed AST layout is unchanged. The explicit Zig string remains an enforced consistency
check and improves diagnostics for incorrect builds.

### Build configurations

Build mode is not part of the descriptor. In Zig 0.16, a module's `builtin.mode` can differ from the
root compilation setting that determines the physical `Ast.Node.Data` representation. Recording
that value could therefore misdiagnose compatibility.

The host compares the actual `Ast.Node.Data` size and alignment instead. This permits combinations
whose native representation matches, such as the verified Zig 0.16 Debug/ReleaseSafe pair, and
rejects combinations whose representation differs, such as the verified safety-enabled and
ReleaseFast/ReleaseSmall layouts. The supported build helper should default the plugin's root
optimization mode to the host's mode.

Matching size and alignment is sufficient only under the exact pinned-compiler contract. Those
values do not independently prove union tag encoding, `extra_data` semantics, or AST helper
behavior.

### Soft API growth

Only the plugin descriptor and host invocation are append-only extension points. Each carries a
producer-written size in its frozen prefix: `header.descriptor_size` or `header.struct_size`. The
producer writes the complete size of the structure it emitted:

```text
producer size = @sizeOf(the complete emitted structure)
```

A field's required extent is a different value:

```text
field extent = @offsetOf(Structure, "field") + @sizeOf(FieldType)
```

Consumers access a field only when the producer's size covers that field's complete extent. The
producer never writes a field extent as its complete structure size. A consumer first reads only the
frozen size header and checks the mandatory extent; it never loads or copies its newest complete
structure from a shorter producer object.

An appended field must begin at or after the previous published structure's complete `@sizeOf`.
Implementations must not reuse a previous version's tail padding, increase the alignment required
for its prefix, move existing fields, or change existing semantics. Versioned extension declarations
embed the complete previous declaration at offset zero, or use equivalent byte-and-offset access,
and compile-time checks preserve every published offset and alignment.

New fields must be genuinely optional for older consumers. An operation field whose extent is
advertised contains a valid callable function pointer; a separate appended value represents whether
the operation is enabled for one invocation. A plugin that requires an appended invocation
operation advertises a larger `minimum_invocation_size`; an older host rejects it during
registration.

`RuleDescriptor`, `RawAst`, `RawAllocator`, `Finding`, and `Fix` remain fixed for an epoch. The format
does not use per-rule variable records, generic capability masks, flag-bit growth within an epoch,
or size fields on every nested structure.

## ABI shape

The declarations below lock in the semantic shape of the boundary. All aggregates are `extern
struct`; all callbacks use `callconv(.c)`. Implementations may add explicit padding where required
to preserve the documented offsets and append-only boundaries.

### Discovery envelope

The exported symbol name is stable:

```zig
pub const descriptor_symbol: [:0]const u8 = "zig_lint_plugin_v1";
pub const descriptor_magic = "ZLINTPLG".*;

pub const DescriptorHeader = extern struct {
    magic: [8]u8,
    abi_major: u32,
    descriptor_size: u32,
};
```

The symbol suffix identifies this discovery envelope. It does not change with each compatibility
epoch. A stable envelope lets a host report the plugin's incompatible `abi_major` instead of
reporting only a missing symbol.

`descriptor_size` equals the producer's complete plugin descriptor size. The format publishes a
resource ceiling for that size; the ceiling is independent of the current host descriptor's size.
The producer's size must cover the frozen mandatory descriptor prefix.

The host initially reads only `DescriptorHeader`. It establishes alignment before a typed header
read or copies the header through byte-addressable storage. It then preserves the producer's size and
copies at most `min(descriptor_size, @sizeOf(HostDescriptor))` bytes into zero-initialized host
storage. Zero fill does not make an absent or partially supplied field present; the host still checks
each field's complete extent. It must not cast the symbol to its newest descriptor type and load
`descriptor.*`, because an older descriptor may be shorter.

### Shared bytes

```zig
pub const Bytes = extern struct {
    pointer: ?[*]const u8,
    length: usize,
};
```

A nonzero length requires a non-null pointer. A zero length denotes an empty value and does not
require a dereference. Every byte range is borrowed for the duration stated by the containing
operation.

### Compatibility

```zig
pub const Compatibility = extern struct {
    zig: ZigCompatibility,
    target: TargetCompatibility,
    ast: AstCompatibility,
};

pub const ZigCompatibility = extern struct {
    version: [64]u8,
};

pub const TargetCompatibility = extern struct {
    triple: [64]u8,
};

pub const AstCompatibility = extern struct {
    node_data_size: u16,
    node_data_alignment: u16,
};
```

`version` and `triple` contain at most 63 content bytes followed by a NUL; every remaining byte is
zero. Consumers decode them only within their 64-byte arrays and reject noncanonical padding. The
target uses the canonical `architecture-operating_system-abi` spelling, for example
`x86_64-linux-gnu`, rather than a version-qualified or implementation-dependent form. The supported
target mapping determines pointer width and endianness, so the descriptor does not repeat them.
CPU feature compatibility remains a build and deployment responsibility; a plugin built for
unsupported instructions can still fault when called.

The SDK and host compile only when the pinned Zig types satisfy the raw column contract:

```zig
comptime {
    if (@sizeOf(std.zig.Token.Tag) != 1)
        @compileError("the raw AST requires byte-sized token tags");
    if (@sizeOf(std.zig.Ast.Node.Tag) != 1)
        @compileError("the raw AST requires byte-sized node tags");
    if (@sizeOf(std.zig.Ast.TokenIndex) != 4 or
        @sizeOf(std.zig.Ast.Node.Index) != 4 or
        @sizeOf(std.zig.Ast.ByteOffset) != 4)
    {
        @compileError("the raw AST requires 32-bit indexes and byte offsets");
    }
}
```

Production declarations use explicit `@compileError` branches for descriptive failures. They also
check the required column alignments, integer backing representations, and optional-index sentinel
values rather than relying on size alone.

### Plugin and rule descriptors

```zig
pub const PluginDescriptor = extern struct {
    header: DescriptorHeader,
    compatibility: Compatibility,

    minimum_invocation_size: u32,
    rule_descriptor_size: u32,

    name: Bytes,
    rules: ?[*]const RuleDescriptor,
    rules_length: u32,

    // Explicit padding may consume tail padding before future fields.
};

pub const RuleDescriptor = extern struct {
    name: Bytes,
    run: ?RuleFn,
};
```

The plugin owns the descriptor, rule array, names, and function pointers until process exit. The host
copies plugin and rule names during registration and retains the dynamic-library handle.

`rule_descriptor_size` must equal the host's `@sizeOf(RuleDescriptor)`. Rule descriptors do not carry
individual sizes and do not use a variable stride. Changing `RuleDescriptor` advances the hard
compatibility epoch.

The SDK validates plugin and local rule names at compile time. Before forming a rule slice, the host
validates the table pointer's alignment and checks `rules_length * rule_descriptor_size` for
representability. It then validates bounded names, callbacks, duplicate local names, and
namespace-qualified collisions before publishing the plugin.

### Raw AST view

```zig
pub const RawAst = extern struct {
    source_pointer: [*:0]const u8,
    source_length: u32,

    token_tags_pointer: [*]const u8,
    token_starts_pointer: [*]const u32,
    token_count: u32,

    node_tags_pointer: [*]const u8,
    node_main_tokens_pointer: [*]const u32,
    node_data_pointer: [*]const u8,
    node_count: u32,

    extra_data_pointer: ?[*]const u32,
    extra_data_length: u32,

    line_start_offsets_pointer: [*]const u32,
    line_count: u32,

    // This column contains exactly token_count entries.
    token_line_indexes_pointer: [*]const u32,
};
```

All AST indexes are zero-based. Source offsets, token starts, and fix ranges count bytes. The token
columns contain `token_count` entries, including the final EOF token; the node columns contain
`node_count` entries. The host checks count-to-byte extent arithmetic and pointer alignment before
forming any slice or sentinel view.

Line zero begins at source byte zero. Each LF byte appends the following byte offset as another line
start, including `source_length` after a trailing LF. `token_line_indexes_pointer` contains one
zero-based line index for every token, including EOF; EOF belongs to the final line, which is empty
after a trailing LF.

A nonzero `extra_data_length` requires a non-null and suitably aligned `extra_data_pointer`. A zero
length reconstructs an empty slice without dereferencing that pointer.

The host invokes plugin rules only for a parse-valid Zig source snapshot. `source_pointer` references
the exact sentinel-terminated buffer used to produce every token and node column. The sentinel zero
at `source_pointer[source_length]` is accessible for the complete invocation.

The node-data pointer references the host parser's live `Ast.Node.Data` objects. The host does not
normalize those unions into integer pairs or copy their raw bytes. Their inactive bytes and padding
can be uninitialized in every build mode, so implementations must not compare, serialize, or hash
the complete node-data byte column.

The SDK reconstructs plugin-local `std.zig.Ast.TokenList.Slice` and
`std.zig.Ast.NodeList.Slice` values whose field pointers reference these host columns. It sets the
local AST mode to `.zig` and errors to an empty slice because parsing succeeded before invocation.
This operation parses nothing, builds no indexes, copies no columns, and allocates no storage.

The reconstructed AST is a borrowed read-only view, not an owning AST. Plugin code must not:

- call `Ast.deinit` or any column deinitializer;
- call `toMultiArrayList` or another ownership reconstruction;
- mutate a column through a method that exposes a mutable slice;
- grow, resize, or transfer ownership of a token or node collection; or
- retain the AST, source, column, or line-index pointers after the callback returns.

The SDK documentation must keep those restrictions adjacent to the exposed `Rule.Context.ast`.

### Raw scratch allocator

```zig
pub const RawAllocator = extern struct {
    state: *anyopaque,
    alloc: AllocFn,
    resize: ResizeFn,
    remap: RemapFn,
    free: FreeFn,
};
```

The callbacks mirror the pinned `std.mem.Allocator.VTable` using only raw C-compatible arguments:

```text
alloc(state, length, alignment_log2, return_address) -> nullable pointer
resize(state, pointer, old_length, alignment_log2, new_length, return_address) -> u8
remap(state, pointer, old_length, alignment_log2, new_length, return_address) -> nullable pointer
free(state, pointer, length, alignment_log2, return_address) -> void
```

The host backs this table with the file's existing scratch arena. The SDK constructs a local
`std.mem.Allocator` facade over the callbacks; that facade is a small value and does not create a
plugin allocator, arena, or page allocation.

The host range-checks `alignment_log2` before converting it to `std.mem.Alignment`. `resize` returns
exactly zero or one. Failed allocation, resize, or remap operations retain their ordinary allocator
meaning and may be handled by rule code; they do not automatically latch a protocol failure. An
invalid alignment or otherwise malformed raw allocator request latches an invocation protocol
failure that later plugin success cannot clear.

Scratch allocations and the allocator facade must not escape the invocation. Plugin code must not
free memory obtained from another allocator through this table, and neither image frees memory
owned by the other image directly. Each image constructs its native allocator facade and vtable
locally because their Zig representation and calling convention are not part of this transport
contract. DSO boundaries do not guarantee isolation of runtime state or shared dependencies.

### Target Zig version

```zig
pub const RawZigVersion = extern struct {
    major: u64,
    minor: u64,
    patch: u64,
    prerelease: Bytes,
    build: Bytes,
};
```

The host converts `LintOptions.target_zig_version` into `RawZigVersion` for every invocation. Its
prerelease and build slices continue to borrow the caller's option storage, which must remain valid
until `lint` returns; the host lends those slices to the plugin only for the synchronous callback.
The SDK reconstructs a plugin-local `std.SemanticVersion` and exposes it as
`Rule.Context.target_zig_version`; plugins do not consume `RawZigVersion` directly.

The target version is a semantic rule input, not a plugin compatibility claim. Plugin loading still
requires the plugin and host to share the compatibility epoch, Zig toolchain, target, and native AST
contract. Selecting an older target version changes how version-sensitive rules behave but does not
permit a plugin built with that older Zig release to load into a newer host.

### Invocation

```zig
pub const InvocationHeader = extern struct {
    struct_size: u32,
};

pub const Invocation = extern struct {
    header: InvocationHeader,
    fixes_enabled: u32,
    target_zig_version: RawZigVersion,

    allocator: *const RawAllocator,
    ast: *const RawAst,

    report_context: *anyopaque,
    report_finding: ReportFindingFn,

    // Future optional host operations append here.
};

pub const RuleFn = *const fn (
    invocation: *const InvocationHeader,
) callconv(.c) c_int;
```

The host sets `header.struct_size` to the complete invocation size it supplies and passes the
address of that frozen header. `fixes_enabled` is exactly zero or one. `target_zig_version` is a
mandatory tracked input in the v1 prefix. The SDK first reads only the header, checks that its size
covers the v1 mandatory extent, and then reads covered fields
individually. It never loads or copies its newest complete `Invocation` from a shorter producer
object. Future extension types preserve `Invocation` at offset zero before appending fields.

The plugin descriptor's `minimum_invocation_size` normally names the v1 mandatory prefix. A future
plugin that requires an appended host operation advertises the extent of that operation. The SDK
cannot infer such a requirement from arbitrary rule code; the future author-facing API must make it
explicit.

### Findings, fixes, and statuses

```zig
pub const Finding = extern struct {
    token: u32,
    message: Bytes,
    help: Bytes,
    note: Bytes,
    fix: ?*const Fix,
};

pub const Fix = extern struct {
    start_offset: u32,
    end_offset: u32,
    replacement: Bytes,
};

pub const ReportFindingFn = *const fn (
    report_context: *anyopaque,
    finding: *const Finding,
) callconv(.c) c_int;
```

A fix range is half-open and refers to the invocation's original immutable source:

```text
0 <= start_offset <= end_offset <= source_length
```

The finding and optional fix may live on the plugin stack. All strings and replacement bytes remain
borrowed only until `report_finding` returns. The host validates the finding completely and copies
every retained byte before returning success.

Statuses are `c_int` constants rather than transported Zig enums or error sets:

```zig
pub const status_ok: c_int = 0;
pub const status_out_of_memory: c_int = 1;
pub const status_finding_rejected: c_int = 2;
pub const status_host_protocol_violation: c_int = 3;
pub const status_rule_failed: c_int = 4;
```

The rule callback may return `status_ok`, `status_out_of_memory`,
`status_finding_rejected`, `status_host_protocol_violation`, or `status_rule_failed`. The reporting
callback may return `status_ok`, `status_out_of_memory`, `status_finding_rejected`, or
`status_host_protocol_violation`; `status_rule_failed` is not a valid reporting response. The host
and SDK handle every unknown or context-invalid integer as a protocol violation.

Malformed allocator requests, reporting failures, and invalid host responses latch the first
invocation failure. Catching an error or later returning `status_ok` cannot clear that failure.
Ordinary allocator refusal remains nonsticky.

The host validates at least:

- bounded finding counts and byte lengths;
- a nonempty message;
- `token < token_count`;
- nullable pointer and length consistency;
- rejection of a non-null fix when fixes are disabled; and
- half-open fix ranges within the source snapshot.

The host binds the file and qualified rule identity through `report_context`; a plugin cannot
attribute a finding to another rule.

## Loading and registration

The host resolves only paths explicitly supplied by a caller or CLI option. It does not scan plugin
directories, inspect environment variables, or load by bare library name. The host resolves a
relative input against its documented base and passes an absolute or slash-containing path to
`dlopen`, avoiding the loader's bare-name search rules.

On Linux the host uses libc directly:

```text
dlopen(path, RTLD_NOW | RTLD_LOCAL)
dlsym(handle, "zig_lint_plugin_v1")
dlclose(handle) only after process work has ended, if at all
```

In Zig's Linux `std.c.RTLD`, `RTLD_LOCAL` is represented by the absence of `GLOBAL`, so the actual
mode is `.{ .NOW = true }`. `std.DynLib.open` is not used because Zig 0.16 selects lazy binding and
collapses useful loader failures.

The host clears and reads `dlerror()` around each loader operation. It copies an error string before
another `dl*` call, including cleanup, can invalidate the loader-owned buffer. The `dlsym` call uses
the platform-safe non-tail-call form required by Zig's dynamic loader implementation.

`dlsym(handle, ...)` may search the selected object's dependency closure. The selected top-level DSO
must define the reserved descriptor symbol, and ordinary dependencies must not export it. Version 1
does not add loader-specific provenance machinery for this trusted build contract.

Registration proceeds in this order:

1. Validate and resolve the explicit path.
2. Enter the loader/registry lifecycle region.
3. Open with immediate, local symbol resolution.
4. Resolve the immutable descriptor data symbol.
5. Read only the stable descriptor header.
6. Establish header alignment or copy the header as bytes, then validate magic, `abi_major`, and the
   published descriptor-size bounds.
7. Preserve the producer size and copy at most the host descriptor size into zero-initialized host
   storage; continue to gate every field by its complete extent.
8. Compare the complete Zig version, target triple, and native AST layout.
9. Validate `minimum_invocation_size` and the exact rule descriptor size.
10. Validate rule-table alignment and extent arithmetic, then validate names, rule counts, callbacks,
    and qualified identity collisions.
11. Copy names, retain function pointers and the DSO handle, then publish the registration atomically.

`dlopen` can execute ELF constructors, dependency initialization, and platform resolver code before
the descriptor is available. The guarantee is therefore that the host calls no plugin callback
before validation, not that loading executes no plugin code.

The descriptor's pointers and lengths remain producer obligations. Native validation cannot prove
that arbitrary plugin-supplied memory is mapped for a claimed extent. That limitation is accepted
because plugins are trusted and normally use SDK-generated descriptors.

The host loads plugins before starting lint work and retains each DSO until process exit. Version 1
does not support hot unload or reload. Removing those operations avoids stale function pointers,
TLS/destructor hazards, and callback quiescence protocols that provide no value to the command-line
linter.

## File and rule flow

For each file, the host:

1. Establishes one immutable sentinel-terminated source snapshot.
2. Parses or reconstructs one AST from the host's compatible token cache.
3. Reports parse failures without invoking built-in or plugin rules that require a valid AST.
4. Builds line-start offsets and token-line indexes once.
5. Runs built-in and plugin rules against that same snapshot.
6. Finalizes fixes only after every rule has observed the original source coordinates.
7. Publishes owned diagnostics in deterministic file and rule order.

Any failed invocation makes the file pass unsuccessful. The host does not publish a successful
cached result or apply pending fixes from that pass, and it waits for every outstanding callback
before releasing file resources.

A plugin invocation uses the same file-scoped scratch allocator and derived indexes as built-in
rules. The host may construct small raw views and SDK facades per callback; those values allocate no
backing storage and perform no whole-file work.

The host never reparses source, rebuilds line indexes, or creates a page-backed plugin arena as part
of dynamic dispatch. Native incompatibility rejects the plugin; it does not fall back to plugin-side
parsing.

## Parallel execution contract

Every plugin must be safe for concurrent and reentrant invocation. The host may invoke:

- the same rule concurrently for different files;
- different rules from the same DSO concurrently;
- callbacks on arbitrary worker threads; and
- files and rules in an order that changes between runs.

The host does not provide thread affinity or serialized plugin execution. Plugins and their reachable
dependencies must be compiled for multithreaded execution; the supported build path does not use
`-fsingle-threaded`.

Plugin results must not depend on callback order, worker identity, thread identity, or prior
invocations. Rules must not maintain mutable process-global or thread-local cross-invocation state,
including memoization, even when intended to be observationally transparent. Immutable compile-time
data and incidental runtime bookkeeping that never becomes a result input remain permitted.
Temporary indexes and buffers belong in invocation-local variables or the supplied scratch
allocator.

A plugin must return only after all work using invocation state has completed. It must not retain or
asynchronously use the AST, source, allocator, report callback, `Rule.Context`, or any value derived
from them. It must not recursively invoke plugin linting or mutate the plugin registry from a rule
callback.

Every invocation has independent bridge and sticky-failure state. Any file arena or report collector
shared by concurrent invocations supports that concurrency or is explicitly synchronized. Shared
backing allocators likewise support calls from different file jobs. A plugin calls host allocator
and reporting callbacks synchronously as part of the invoking rule callback; it does not call them
concurrently from plugin-created threads.

Overlapping independent invocations are supported. Recursive plugin linting and registry mutation
from within a rule remain prohibited. The SDK explicitly rejects `builtin.single_threaded`; the
supported build recipe also configures every plugin root and dependency for multithreaded execution.

The host does not hold a registry mutex while executing rules. A lint run takes an immutable snapshot
of selected registrations and retains their DSO ownership until all callbacks complete. Per-file
collectors gather diagnostics and fixes; the host merges them by logical file and rule order rather
than callback completion order.

Parallel file scheduling must bound the number or retained bytes of in-flight source snapshots. A
worker or queued job keeps its file arena and every borrowed pointer alive until its plugin callbacks
finish.

## Lint caching

`LintOptions.Cache.base_dir` selects only where the linter may keep private cache state. The linter
manages everything beneath that directory and may use multiple files or subdirectories, replace
formats, or discard entries without notice. Callers must treat its contents as private and unstable
and must not depend on their names or representation. Cache state is scoped to the selected
directory: the linter never consults another base directory, so selecting a fresh directory
guarantees a full cache miss.

The ABI contains no cache-policy field. Every plugin is cacheable by contract because every
result-affecting input must flow through the host.

Before the host reuses lint-clean cache state influenced by plugins, its cache identity must include
at least:

- the host and cache-format identity;
- the source identity and every relevant invocation option;
- the exact `target_zig_version`, including prerelease and build metadata;
- the ordered selected plugin and rule set;
- each plugin namespace; and
- a canonical content hash of each exact selected plugin artifact loaded into the process.

Registration retains the identity of the loaded artifact. The artifact remains unchanged while the
host fingerprints and loads it; replacing its pathname later does not change an existing
registration's identity. A dynamic dependency or external configuration that can change results is
an untracked input and violates the v1 plugin contract. A future host-owned input API must snapshot
or otherwise stabilize its values and add their canonical representation to the cache identity.

Until the complete plugin-influenced cache identity is implemented, selecting any plugin neither
consumes nor records lint-clean facts. A successful complete plugin scan commits an empty lint-cache
generation, while a failed scan leaves the previous generation intact. Token and AST caches remain
eligible because they do not cache rule outcomes and remain governed by their own parser
compatibility contract.

Parallel scheduling does not change cache semantics. A cached or newly computed result must not
depend on invocation order, and the host publishes cache updates through its normal synchronized,
atomic cache path.

## Plugin and host ownership

The following lifetimes are normative:

| Value | Owner | Validity |
|---|---|---|
| Descriptor and rule table | Plugin DSO | Until process exit |
| Descriptor and rule names copied at registration | Host | Until registry shutdown |
| Source, AST columns, and line indexes | Host file job | Until all callbacks for that file return |
| Raw invocation and bridge state | Host rule invocation | Until the rule callback returns |
| Scratch allocations | Host file arena | Invocation-only by contract |
| Finding, fix, and their byte slices | Plugin callback | Until `report_finding` returns |
| Retained diagnostic and replacement bytes | Host | Report/fix lifecycle |

Published text and fixes never retain plugin-owned pointers. Plugin descriptors never retain
host-owned pointers. Rule callbacks do not transfer ownership in either direction.

## Build and distribution contract

The linter distribution must ship or identify:

- the plugin compatibility epoch;
- the exact pinned Zig compiler distribution;
- the matching plugin SDK sources;
- a supported build recipe that uses the host target and a compatible AST storage mode; and
- the author-facing `Rule` API used by both built-in and plugin rules.

A repository may vendor that SDK and build its plugin beside the linter. The normal migration from an
incompatible linter release is to rebuild the plugin with the new vendored SDK and pinned compiler.
Diagnostics report both epochs when a valid discovery header is available. They report compiler,
target, and AST-layout details only after the accepted epoch and supplied descriptor prefix establish
that those fields can be interpreted.

Plugin builds must preserve the exported descriptor object in the dynamic symbol table under
ReleaseSmall, stripping, and section garbage collection. The supported build recipe must verify that
property.

## Testing expectations

Rule behavior should normally be tested without a DSO. A plugin testing module can run ordinary
`Rule.Definition` values with fresh `Rule.Context` state and the same report/fix semantics as the
host. Integration coverage separately builds and loads a real shared library to exercise the ABI.

Before publishing an epoch, integration tests must cover:

- delivery of the exact target Zig version to built-in and plugin rules;
- lint-cache separation across target Zig versions;
- immediate loader failure for unresolved dependencies and preservation of `dlerror()` text;
- descriptor magic, epoch, size, target, Zig, and AST-layout rejection;
- older/newer append-only descriptor and invocation prefixes without full-structure overreads;
- supported and rejected host/plugin AST storage-mode combinations;
- one host parse and one line-index build shared by all plugin rules;
- allocator allocation, resize, remap, free, ordinary refusal, and invalid alignment;
- borrowed finding/fix bytes copied before return and sticky reporting failures;
- concurrent invocation of the same plugin and rule across many files;
- deterministic diagnostics across worker counts and completion orders;
- source sentinel and original-snapshot fix coordinates;
- lint-cache bypass until plugin identity is keyed; and
- descriptor symbol retention in optimized and stripped plugins.

Automated plugin tests use private fixtures and never interact with the user's live service,
microphone, keyboard, or clipboard.

## Explicitly excluded designs

Version 1 deliberately excludes:

- passing `std.zig.Ast`, `std.mem.Allocator`, Zig slices, error sets, or automatic-layout structs
  directly across the DSO boundary;
- reparsing source or rebuilding line indexes inside each plugin callback;
- a plugin-owned page allocator or per-rule arena;
- compatibility across different Zig compiler or standard-library identities;
- fallback reparsing after native compatibility fails;
- reflection-based FNV, Wyhash, or other AST schema digests;
- a build-mode compatibility field;
- generic required/optional capability masks;
- separate plugin execution-policy or cache-policy fields;
- serialized or thread-affine plugin callback guarantees;
- per-rule variable-size records or rule-table strides;
- independently extensible `RawAst`, allocator, finding, or fix structures;
- plugin-defined result inputs that bypass host cache identity;
- directory scanning, environment-based discovery, or bare-name library loading;
- hot unload and reload;
- sandboxing, crash isolation, pointer provenance validation, or hostile-plugin containment; and
- a second plugin-specific rule definition API.

These exclusions keep the contract aligned with its actual use: trusted repository-local rules,
compiled against a pinned Zig toolchain, sharing one native parse without duplicate whole-file work.
