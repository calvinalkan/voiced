//! Version-locked C ABI between the linter host and trusted native plugins.
//!
//! The boundary carries primitive C-compatible values and borrowed columnar
//! AST storage. Zig containers and allocator vtables are reconstructed on the
//! side that owns the matching Zig standard library; they never cross the DSO
//! boundary directly.

const std = @import("std");
const builtin = @import("builtin");

pub const descriptor_name: [:0]const u8 = "zig_lint_plugin_v1";
pub const descriptor_magic = [_]u8{ 'Z', 'L', 'I', 'N', 'T', 'P', 'L', 'G' };
pub const version: u32 = 1;
pub const sdk_abi_id: u64 = 0x564f494345445031;

pub const max_plugin_name_bytes: u64 = 128;
pub const max_rule_name_bytes: u64 = 128;
pub const max_rules_per_plugin: u32 = 256;
pub const max_findings_per_rule: u32 = 100_000;
pub const max_message_bytes: u64 = 64 * 1024;
pub const max_help_bytes: u64 = 64 * 1024;
pub const max_note_bytes: u64 = 64 * 1024;
pub const max_replacement_bytes: u64 = 16 * 1024 * 1024;

pub const capability_scratch_allocator: u64 = 1 << 0;
pub const capability_ast_view: u64 = 1 << 1;
pub const capability_line_indexes: u64 = 1 << 2;
pub const capability_report_findings: u64 = 1 << 3;
pub const capability_fixes: u64 = 1 << 4;
pub const required_sdk_capabilities: u64 = capability_scratch_allocator |
    capability_ast_view |
    capability_line_indexes |
    capability_report_findings;
pub const host_capabilities: u64 = required_sdk_capabilities | capability_fixes;

pub const RawStatus = c_int;

pub const status = struct {
    pub const ok: RawStatus = 0;
    pub const out_of_memory: RawStatus = 1;
    pub const finding_rejected: RawStatus = 2;
    pub const host_protocol_violation: RawStatus = 3;
    pub const rule_failed: RawStatus = 4;
};

/// Nullable pointer plus byte length. A zero length always denotes an empty
/// slice and does not require a non-null pointer.
pub const Bytes = extern struct {
    pointer: ?[*]const u8 = null,
    length: u64 = 0,
};

/// Semantic Zig version selected by the lint caller. Prerelease and build
/// slices remain valid for the complete synchronous rule invocation.
pub const ZigVersionV1 = extern struct {
    major: u64,
    minor: u64,
    patch: u64,
    prerelease: Bytes = .{},
    build: Bytes = .{},
};

pub const SourceRange = extern struct {
    start_offset: u32,
    end_offset: u32,
};

pub const FixV1 = extern struct {
    struct_size: u32,
    reserved: u32 = 0,
    range: SourceRange,
    replacement: Bytes,
};

/// Finding fields are borrowed only for the duration of `report_finding`.
pub const FindingV1 = extern struct {
    struct_size: u32,
    token: u32,
    message: Bytes,
    help: Bytes = .{},
    note: Bytes = .{},
    fix: FixV1 = .{
        .struct_size = @sizeOf(FixV1),
        .range = .{ .start_offset = 0, .end_offset = 0 },
        .replacement = .{},
    },
    has_fix: u8 = 0,
    reserved: [7]u8 = @splat(0),
};

/// Allocator operations mirror `std.mem.Allocator.VTable`, but use only raw C
/// ABI values. The host owns the backing allocator and keeps it alive for the
/// complete synchronous rule callback.
pub const AllocatorV1 = extern struct {
    state: *anyopaque,
    alloc: *const fn (
        state: *anyopaque,
        length: usize,
        alignment_log2: u8,
        return_address: usize,
    ) callconv(.c) ?[*]u8,
    resize: *const fn (
        state: *anyopaque,
        memory: [*]u8,
        old_length: usize,
        alignment_log2: u8,
        new_length: usize,
        return_address: usize,
    ) callconv(.c) u8,
    remap: *const fn (
        state: *anyopaque,
        memory: [*]u8,
        old_length: usize,
        alignment_log2: u8,
        new_length: usize,
        return_address: usize,
    ) callconv(.c) ?[*]u8,
    free: *const fn (
        state: *anyopaque,
        memory: [*]u8,
        length: usize,
        alignment_log2: u8,
        return_address: usize,
    ) callconv(.c) void,
};

/// Borrowed representation of the host's one parsed AST and its one set of
/// derived line indexes. Columns point directly at host storage.
pub const AstV1 = extern struct {
    struct_size: u32,
    reserved: u32 = 0,

    source_pointer: [*:0]const u8,
    source_length: u32,

    token_tags_pointer: [*]const u8,
    token_starts_pointer: [*]const u32,
    token_count: u32,

    node_tags_pointer: [*]const u8,
    node_main_tokens_pointer: [*]const u32,
    node_data_pointer: [*]const u8,
    node_count: u32,

    extra_data_pointer: [*]const u32,
    extra_data_length: u32,

    line_start_offsets_pointer: [*]const u32,
    line_start_offsets_length: u32,
    token_line_indexes_pointer: [*]const u32,
    token_line_indexes_length: u32,
};

pub const ReportFindingFn = *const fn (
    host_context: ?*anyopaque,
    finding: *const FindingV1,
) callconv(.c) RawStatus;

/// One synchronous rule invocation. The plugin must not retain any pointer
/// reachable from this value after its callback returns.
pub const InvocationV1 = extern struct {
    struct_size: u32,
    abi_version: u32,
    host_capabilities: u64,
    target_zig_version: ZigVersionV1,
    scratch_allocator: *const AllocatorV1,
    ast: *const AstV1,
    host_context: ?*anyopaque,
    report_finding: ReportFindingFn,
    fixes_enabled: u8,
    reserved: [7]u8 = @splat(0),
};

pub const RuleFnV1 = *const fn (invocation: *const InvocationV1) callconv(.c) RawStatus;

/// Immutable rule metadata retained by the plugin for the DSO lifetime.
pub const RuleDescriptorV1 = extern struct {
    struct_size: u32,
    reserved: u32 = 0,
    required_host_capabilities: u64,
    optional_host_capabilities: u64,
    name: Bytes,
    run: ?RuleFnV1,
};

pub const BuildMode = enum(u8) {
    debug,
    release_safe,
    release_fast,
    release_small,
};

pub const build_mode: BuildMode = switch (builtin.mode) {
    .Debug => .debug,
    .ReleaseSafe => .release_safe,
    .ReleaseFast => .release_fast,
    .ReleaseSmall => .release_small,
};

/// Compatibility facts for reconstructing a local `std.zig.Ast` view over the
/// borrowed columns. Build mode is intentionally not part of this gate.
pub const Compatibility = extern struct {
    zig_major: u16,
    zig_minor: u16,
    zig_patch: u16,
    architecture: u16,
    operating_system: u16,
    target_abi: u16,
    pointer_bits: u8,
    endian: u8,
    token_tag_size: u8,
    node_tag_size: u8,
    node_data_size: u8,
    node_data_alignment: u8,
    ast_schema_id: u64,
};

pub const compatibility: Compatibility = .{
    .zig_major = builtin.zig_version.major,
    .zig_minor = builtin.zig_version.minor,
    .zig_patch = builtin.zig_version.patch,
    .architecture = @intFromEnum(builtin.target.cpu.arch),
    .operating_system = @intFromEnum(builtin.target.os.tag),
    .target_abi = @intFromEnum(builtin.target.abi),
    .pointer_bits = @intCast(builtin.target.ptrBitWidth()),
    .endian = @intFromEnum(builtin.target.cpu.arch.endian()),
    .token_tag_size = @sizeOf(std.zig.Token.Tag),
    .node_tag_size = @sizeOf(std.zig.Ast.Node.Tag),
    .node_data_size = @sizeOf(std.zig.Ast.Node.Data),
    .node_data_alignment = @alignOf(std.zig.Ast.Node.Data),
    .ast_schema_id = astSchemaId(),
};

pub const DescriptorHeader = extern struct {
    magic: [8]u8,
    abi_version: u32,
    descriptor_size: u32,
};

/// The exported `zig_lint_plugin_v1` data symbol. The host validates every
/// compatibility field before it calls any rule function pointer.
pub const PluginV1 = extern struct {
    header: DescriptorHeader,
    sdk_abi: u64,
    compatibility: Compatibility,
    build_mode: u8,
    reserved: [7]u8 = @splat(0),
    required_host_capabilities: u64,
    optional_host_capabilities: u64,
    name: Bytes,
    rules_pointer: ?[*]const RuleDescriptorV1,
    rules_length: u32,
    reserved_length: u32 = 0,
};

fn astSchemaId() u64 {
    @setEvalBranchQuota(20_000);

    var hash: u64 = 0xcbf29ce484222325;

    inline for (@typeInfo(std.zig.Token.Tag).@"enum".fields) |field| {
        hashBytes(&hash, field.name);
        hashInteger(&hash, field.value);
    }

    inline for (@typeInfo(std.zig.Ast.Node.Tag).@"enum".fields) |field| {
        hashBytes(&hash, field.name);
        hashInteger(&hash, field.value);
    }

    inline for (@typeInfo(std.zig.Ast.Node.Data).@"union".fields) |field| {
        hashBytes(&hash, field.name);
        hashBytes(&hash, @typeName(field.type));
        hashInteger(&hash, @sizeOf(field.type));
        hashInteger(&hash, @alignOf(field.type));
    }

    hashInteger(&hash, @sizeOf(std.zig.Ast.ByteOffset));
    hashInteger(&hash, @sizeOf(std.zig.Ast.TokenIndex));
    hashInteger(&hash, @sizeOf(std.zig.Ast.Node.Index));
    hashInteger(&hash, @intFromEnum(std.zig.Ast.OptionalTokenIndex.none));
    hashInteger(&hash, @intFromEnum(std.zig.Ast.Node.OptionalIndex.none));
    hashInteger(&hash, @sizeOf(std.zig.Ast.Node.For));
    hashInteger(&hash, @bitOffsetOf(std.zig.Ast.Node.For, "inputs"));
    hashInteger(&hash, @bitOffsetOf(std.zig.Ast.Node.For, "has_else"));

    if (builtin.zig_version.pre) |pre| {
        hashBytes(&hash, pre);
    }

    if (builtin.zig_version.build) |build| {
        hashBytes(&hash, build);
    }

    return hash;
}

fn hashBytes(hash: *u64, value: []const u8) void {
    for (value) |byte| {
        hash.* = (hash.* ^ byte) *% 0x100000001b3;
    }
}

fn hashInteger(hash: *u64, value: anytype) void {
    var remaining: u64 = @intCast(value);

    for (0..@sizeOf(u64)) |_| {
        hashBytes(hash, &.{@truncate(remaining)});

        remaining >>= 8;
    }
}

pub fn nameIsValid(name: []const u8) bool {
    if (name.len == 0) {
        return false;
    }

    for (name) |byte| {
        switch (byte) {
            'a'...'z', 'A'...'Z', '0'...'9', '_', '-', '.' => {},
            else => return false,
        }
    }

    return true;
}

comptime {
    if (@sizeOf(?*anyopaque) != @sizeOf(*anyopaque)) {
        @compileError("the plugin ABI requires pointer-sized nullable pointers");
    }

    if (@sizeOf(std.zig.Token.Tag) != 1 or @sizeOf(std.zig.Ast.Node.Tag) != 1) {
        @compileError("the plugin AST ABI requires byte-sized token and node tags");
    }
}
