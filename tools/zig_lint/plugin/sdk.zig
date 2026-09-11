//! Zig SDK adapter for the version-locked native plugin ABI.
//!
//! Plugin rules use the ordinary `Rule.Definition` and `Rule.Context` API.
//! `define` creates the exported immutable ABI descriptor and each generated
//! trampoline reconstructs only local Zig views over host-owned storage.

const std = @import("std");
const abi = @import("abi.zig");
const Rule = @import("../Rule.zig");

pub const PluginDescriptor = abi.PluginV1;

/// Build an immutable descriptor for export from a plugin root:
///
/// ```zig
/// export const zig_lint_plugin_v1 = Plugin.define("example", &rules);
/// ```
pub fn define(
    comptime plugin_name: []const u8,
    comptime definitions: []const Rule.Definition,
) abi.PluginV1 {
    const Storage = ManifestStorage(plugin_name, definitions);

    return Storage.manifest;
}

fn ManifestStorage(
    comptime plugin_name: []const u8,
    comptime definitions: []const Rule.Definition,
) type {
    if (!abi.nameIsValid(plugin_name) or plugin_name.len > abi.max_plugin_name_bytes) {
        @compileError("plugin name is invalid or exceeds the ABI limit");
    }

    if (definitions.len == 0 or definitions.len > abi.max_rules_per_plugin) {
        @compileError("plugin rule count is outside the ABI limit");
    }

    for (definitions, 0..) |definition, definition_index| {
        if (!abi.nameIsValid(definition.name) or definition.name.len > abi.max_rule_name_bytes) {
            @compileError("plugin rule name is invalid or exceeds the ABI limit");
        }

        for (definitions[0..definition_index]) |previous| {
            if (std.mem.eql(u8, previous.name, definition.name)) {
                @compileError("plugin contains duplicate rule names");
            }
        }
    }

    return struct {
        const descriptors: [definitions.len]abi.RuleDescriptorV1 = descriptors: {
            var result: [definitions.len]abi.RuleDescriptorV1 = undefined;
            for (definitions, 0..) |definition, index| {
                const Adapter = RuleAdapter(definition);

                result[index] = .{
                    .struct_size = @sizeOf(abi.RuleDescriptorV1),
                    .required_host_capabilities = abi.required_sdk_capabilities,
                    .optional_host_capabilities = abi.capability_fixes,
                    .name = bytes(definition.name),
                    .run = Adapter.run,
                };
            }

            break :descriptors result;
        };

        const manifest: abi.PluginV1 = .{
            .header = .{
                .magic = abi.descriptor_magic,
                .abi_version = abi.version,
                .descriptor_size = @sizeOf(abi.PluginV1),
            },
            .sdk_abi = abi.sdk_abi_id,
            .compatibility = abi.compatibility,
            .build_mode = @intFromEnum(abi.build_mode),
            .required_host_capabilities = abi.required_sdk_capabilities,
            .optional_host_capabilities = abi.capability_fixes,
            .name = bytes(plugin_name),
            .rules_pointer = &descriptors,
            .rules_length = definitions.len,
        };
    };
}

fn RuleAdapter(comptime definition: Rule.Definition) type {
    return struct {
        fn run(invocation: *const abi.InvocationV1) callconv(.c) abi.RawStatus {
            if (!validInvocation(invocation)) {
                return abi.status.host_protocol_violation;
            }

            const raw_ast = invocation.ast;
            var bridge: ReportBridge = .{ .invocation = invocation };

            var context: Rule.Context = .{
                .scratch_allocator = allocatorFromRaw(invocation.scratch_allocator),
                .target_zig_version = zigVersionFromRaw(invocation.target_zig_version),
                .ast = astFromRaw(raw_ast),
                .fixes_enabled = invocation.fixes_enabled == 1,
                .line_start_offsets = raw_ast.line_start_offsets_pointer[0..raw_ast.line_start_offsets_length],
                .token_line_indexes = raw_ast.token_line_indexes_pointer[0..raw_ast.token_line_indexes_length],
                .report_state = &bridge,
                .report_fn = ReportBridge.report,
            };

            definition.lint(&context) catch |failure| {
                return statusFromRuleError(failure);
            };

            if (bridge.report_failure) |failure| {
                return statusFromRuleError(failure);
            }

            return abi.status.ok;
        }
    };
}

const ReportBridge = struct {
    invocation: *const abi.InvocationV1,
    report_failure: ?Rule.Error = null,

    fn report(opaque_bridge: *anyopaque, finding: Rule.Finding) Rule.Error!void {
        const bridge: *ReportBridge = @ptrCast(@alignCast(opaque_bridge));
        if (bridge.report_failure) |failure| {
            return failure;
        }

        const abi_finding = abi.FindingV1{
            .struct_size = @sizeOf(abi.FindingV1),
            .token = finding.token,
            .message = bytes(finding.message),
            .help = bytes(finding.help),
            .note = bytes(finding.note),
            .fix = if (finding.fix) |fix|
                .{
                    .struct_size = @sizeOf(abi.FixV1),
                    .range = .{
                        .start_offset = fix.range.start_offset,
                        .end_offset = fix.range.end_offset,
                    },
                    .replacement = bytes(fix.replacement),
                }
            else
                .{
                    .struct_size = @sizeOf(abi.FixV1),
                    .range = .{ .start_offset = 0, .end_offset = 0 },
                    .replacement = .{},
                },
            .has_fix = @intFromBool(finding.fix != null),
        };

        const raw_status = bridge.invocation.report_finding(
            bridge.invocation.host_context,
            &abi_finding,
        );

        const failure: ?Rule.Error = switch (raw_status) {
            abi.status.ok => null,
            abi.status.out_of_memory => error.OutOfMemory,
            abi.status.finding_rejected => error.FindingRejected,
            abi.status.host_protocol_violation => error.HostProtocolViolation,
            else => error.HostProtocolViolation,
        };

        if (failure) |report_failure| {
            bridge.report_failure = report_failure;

            return report_failure;
        }
    }
};

fn validInvocation(invocation: *const abi.InvocationV1) bool {
    if (invocation.struct_size != @sizeOf(abi.InvocationV1) or
        invocation.abi_version != abi.version or
        invocation.host_capabilities & abi.required_sdk_capabilities !=
            abi.required_sdk_capabilities or
        invocation.fixes_enabled > 1 or
        !std.mem.allEqual(u8, &invocation.reserved, 0) or
        !zigVersionIsValid(invocation.target_zig_version))
    {
        return false;
    }

    if (invocation.fixes_enabled == 1 and
        invocation.host_capabilities & abi.capability_fixes == 0)
    {
        return false;
    }

    const ast = invocation.ast;

    return ast.struct_size == @sizeOf(abi.AstV1) and
        ast.reserved == 0 and
        ast.token_count != 0 and
        ast.node_count != 0 and
        ast.line_start_offsets_length != 0 and
        ast.token_line_indexes_length == ast.token_count;
}

fn zigVersionIsValid(version: abi.ZigVersionV1) bool {
    return version.major <= std.math.maxInt(usize) and
        version.minor <= std.math.maxInt(usize) and
        version.patch <= std.math.maxInt(usize) and
        optionalBytesIsValid(version.prerelease) and
        optionalBytesIsValid(version.build);
}

fn optionalBytesIsValid(value: abi.Bytes) bool {
    return value.length == 0 or
        (value.pointer != null and value.length <= std.math.maxInt(usize));
}

fn zigVersionFromRaw(version: abi.ZigVersionV1) std.SemanticVersion {
    return .{
        .major = @intCast(version.major),
        .minor = @intCast(version.minor),
        .patch = @intCast(version.patch),
        .pre = optionalBytesSlice(version.prerelease),
        .build = optionalBytesSlice(version.build),
    };
}

fn optionalBytesSlice(value: abi.Bytes) ?[]const u8 {
    if (value.length == 0) {
        return null;
    }

    return (value.pointer orelse {
        unreachable;
    })[0..@intCast(value.length)];
}

fn allocatorFromRaw(raw: *const abi.AllocatorV1) std.mem.Allocator {
    return .{
        .ptr = @ptrCast(@constCast(raw)),
        .vtable = &allocator_vtable,
    };
}

const allocator_vtable: std.mem.Allocator.VTable = .{
    .alloc = adapterAlloc,
    .resize = adapterResize,
    .remap = adapterRemap,
    .free = adapterFree,
};

fn adapterAlloc(
    opaque_allocator: *anyopaque,
    length: usize,
    alignment: std.mem.Alignment,
    return_address: usize,
) ?[*]u8 {
    const raw: *const abi.AllocatorV1 = @ptrCast(@alignCast(opaque_allocator));

    return raw.alloc(raw.state, length, @intFromEnum(alignment), return_address);
}

fn adapterResize(
    opaque_allocator: *anyopaque,
    memory: []u8,
    alignment: std.mem.Alignment,
    new_length: usize,
    return_address: usize,
) bool {
    const raw: *const abi.AllocatorV1 = @ptrCast(@alignCast(opaque_allocator));

    return raw.resize(
        raw.state,
        memory.ptr,
        memory.len,
        @intFromEnum(alignment),
        new_length,
        return_address,
    ) != 0;
}

fn adapterRemap(
    opaque_allocator: *anyopaque,
    memory: []u8,
    alignment: std.mem.Alignment,
    new_length: usize,
    return_address: usize,
) ?[*]u8 {
    const raw: *const abi.AllocatorV1 = @ptrCast(@alignCast(opaque_allocator));

    return raw.remap(
        raw.state,
        memory.ptr,
        memory.len,
        @intFromEnum(alignment),
        new_length,
        return_address,
    );
}

fn adapterFree(
    opaque_allocator: *anyopaque,
    memory: []u8,
    alignment: std.mem.Alignment,
    return_address: usize,
) void {
    const raw: *const abi.AllocatorV1 = @ptrCast(@alignCast(opaque_allocator));

    raw.free(
        raw.state,
        memory.ptr,
        memory.len,
        @intFromEnum(alignment),
        return_address,
    );
}

fn astFromRaw(raw: *const abi.AstV1) std.zig.Ast {
    var tokens = std.zig.Ast.TokenList.Slice{
        .ptrs = undefined,
        .len = raw.token_count,
        .capacity = raw.token_count,
    };

    tokens.ptrs[@intFromEnum(std.zig.Ast.TokenList.Field.tag)] =
        @ptrCast(@constCast(raw.token_tags_pointer));
    tokens.ptrs[@intFromEnum(std.zig.Ast.TokenList.Field.start)] =
        @ptrCast(@constCast(raw.token_starts_pointer));

    var nodes = std.zig.Ast.NodeList.Slice{
        .ptrs = undefined,
        .len = raw.node_count,
        .capacity = raw.node_count,
    };

    nodes.ptrs[@intFromEnum(std.zig.Ast.NodeList.Field.tag)] =
        @ptrCast(@constCast(raw.node_tags_pointer));
    nodes.ptrs[@intFromEnum(std.zig.Ast.NodeList.Field.main_token)] =
        @ptrCast(@constCast(raw.node_main_tokens_pointer));
    nodes.ptrs[@intFromEnum(std.zig.Ast.NodeList.Field.data)] =
        @ptrCast(@constCast(raw.node_data_pointer));

    return .{
        .source = raw.source_pointer[0..raw.source_length :0],
        .tokens = tokens,
        .nodes = nodes,
        .extra_data = @constCast(raw.extra_data_pointer[0..raw.extra_data_length]),
        .mode = .zig,
        .errors = &.{},
    };
}

fn statusFromRuleError(failure: Rule.Error) abi.RawStatus {
    return switch (failure) {
        error.OutOfMemory => abi.status.out_of_memory,
        error.FindingRejected => abi.status.finding_rejected,
        error.HostProtocolViolation => abi.status.host_protocol_violation,
    };
}

fn bytes(value: []const u8) abi.Bytes {
    return .{
        .pointer = if (value.len == 0) null else value.ptr,
        .length = value.len,
    };
}

test "target Zig version is reconstructed without loss" {
    const prerelease = "dev.123";
    const build = "custom.456";

    const version = zigVersionFromRaw(.{
        .major = 0,
        .minor = 16,
        .patch = 0,
        .prerelease = bytes(prerelease),
        .build = bytes(build),
    });

    const decoded_prerelease = version.pre orelse {
        return error.MissingPrerelease;
    };

    const decoded_build = version.build orelse {
        return error.MissingBuild;
    };

    try std.testing.expectEqual(@as(usize, 0), version.major);
    try std.testing.expectEqual(@as(usize, 16), version.minor);
    try std.testing.expectEqual(@as(usize, 0), version.patch);
    try std.testing.expectEqualStrings(prerelease, decoded_prerelease);
    try std.testing.expectEqualStrings(build, decoded_build);
}

test "define materializes one complete plugin descriptor" {
    const manifest = comptime define("example", &.{
        Rule.define("sample", testRule),
    });

    const descriptors = manifest.rules_pointer orelse {
        return error.MissingRuleDescriptors;
    };

    const descriptor = descriptors[0];

    try std.testing.expectEqualSlices(u8, &abi.descriptor_magic, &manifest.header.magic);
    try std.testing.expectEqual(abi.version, manifest.header.abi_version);
    try std.testing.expectEqual(abi.sdk_abi_id, manifest.sdk_abi);
    try std.testing.expectEqual(@as(u32, 1), manifest.rules_length);
    try std.testing.expectEqualStrings("example", bytesSlice(manifest.name));
    try std.testing.expectEqualStrings("sample", bytesSlice(descriptor.name));
    try std.testing.expect(descriptor.run != null);
}

fn testRule(_: *Rule.Context) Rule.Error!void {}

fn bytesSlice(value: abi.Bytes) []const u8 {
    if (value.length == 0) {
        return "";
    }

    const pointer = value.pointer orelse {
        unreachable;
    };

    return pointer[0..@intCast(value.length)];
}
