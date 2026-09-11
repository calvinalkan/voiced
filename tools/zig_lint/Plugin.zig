//! Explicit process-wide registry for trusted native lint plugins.
//!
//! Registering a path executes code from that dynamic library. The registry
//! never discovers or loads plugins implicitly. One mutex protects lookup,
//! invocation, unloading, and registry mutation, which also serializes all
//! plugin callbacks and prevents unload while plugin code is running.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const LintContext = @import("LintContext.zig");
const Rule = @import("Rule.zig");
pub const abi = @import("plugin/abi.zig");
pub const sdk = @import("plugin/sdk.zig");
pub const define = sdk.define;

const registry_allocator = std.heap.page_allocator;
const max_registered_plugins: usize = 64;
const max_plugin_path_bytes: usize = 4096;

var registry_mutex: std.Io.Mutex = .init;
var registered_plugins: std.ArrayList(RegisteredPlugin) = .empty;
var next_handle_value: u64 = 1;
threadlocal var plugin_code_is_active = false;

/// `Handle` identifies one registered plugin until it is unregistered.
pub const Handle = enum(u64) { _ };

/// The registry rejects same-thread reentry from plugin code before locking.
pub const RegistryError = error{PluginRegistryReentry};

pub const RegisterError = Allocator.Error || RegistryError || error{
    InvalidPluginPath,
    PluginOpenFailed,
    MissingDescriptor,
    UnsupportedAbiVersion,
    UnsupportedCapabilities,
    IncompatibleSdk,
    IncompatibleZigVersion,
    IncompatiblePluginTarget,
    IncompatibleAstStorage,
    IncompatibleAstSchema,
    InvalidPluginManifest,
    InvalidPluginName,
    InvalidRuleDescriptor,
    InvalidRuleName,
    DuplicatePlugin,
    DuplicateRule,
    PluginLimitReached,
};

pub const RunError = Rule.Error || RegistryError || error{
    PluginNotRegistered,
    TooManyPluginsSelected,
    DuplicatePluginSelection,
    PluginRuleFailed,
    PluginProtocolViolation,
};

/// Every `Info` slice belongs to the containing `List` snapshot. `rule_names`
/// contains plugin-local names; diagnostics qualify them as `name/rule_name`.
pub const Info = struct {
    handle: Handle,
    path: []const u8,
    name: []const u8,
    rule_names: []const []const u8,
};

pub const List = struct {
    items: []const Info,

    /// `deinit` invalidates every `Info` and slice in the snapshot.
    pub fn deinit(plugin_list: *List, allocator: Allocator) void {
        for (plugin_list.items) |info| {
            for (info.rule_names) |rule_name| {
                allocator.free(rule_name);
            }

            allocator.free(info.rule_names);
            allocator.free(info.name);
            allocator.free(info.path);
        }

        allocator.free(plugin_list.items);

        plugin_list.* = undefined;
    }
};

const DynamicLibrary = struct {
    handle: *anyopaque,

    fn openNowLocal(path: []const u8) error{PluginOpenFailed}!DynamicLibrary {
        var path_buffer: [max_plugin_path_bytes + 1]u8 = undefined;
        @memcpy(path_buffer[0..path.len], path);

        path_buffer[path.len] = 0;

        const path_z = path_buffer[0..path.len :0];

        _ = std.c.dlerror();

        const handle = std.c.dlopen(path_z.ptr, .{ .NOW = true }) orelse {
            return error.PluginOpenFailed;
        };

        return .{ .handle = handle };
    }

    fn lookup(library: DynamicLibrary, name: [:0]const u8) ?*anyopaque {
        _ = std.c.dlerror();

        return @call(.never_tail, std.c.dlsym, .{ library.handle, name.ptr });
    }

    fn close(library: *DynamicLibrary) void {
        _ = std.c.dlclose(library.handle);
        library.* = undefined;
    }
};

const RegisteredRule = struct {
    local_name: []u8,
    qualified_name: []u8,
    run: abi.RuleFnV1,
};

const RegisteredPlugin = struct {
    handle: Handle,
    path: []u8,
    name: []u8,
    library: DynamicLibrary,
    rules: []RegisteredRule,

    fn deinit(plugin: *RegisteredPlugin) void {
        for (plugin.rules) |rule| {
            registry_allocator.free(rule.qualified_name);
            registry_allocator.free(rule.local_name);
        }

        registry_allocator.free(plugin.rules);
        registry_allocator.free(plugin.name);
        registry_allocator.free(plugin.path);
        plugin.library.close();

        plugin.* = undefined;
    }
};

/// `register` loads one trusted dynamic library and returns its process-local
/// handle. It validates the exported data descriptor before invoking any rule.
/// No source path, environment variable, or directory is searched implicitly.
pub fn register(io: std.Io, path: []const u8) RegisterError!Handle {
    if (plugin_code_is_active) {
        return error.PluginRegistryReentry;
    }

    if (comptime builtin.os.tag == .linux and !builtin.link_libc) {
        @compileError("Linux plugin hosts must link libc for dlopen/dlsym/dlclose");
    }

    if (path.len == 0 or path.len > max_plugin_path_bytes or
        std.mem.indexOfScalar(u8, path, 0) != null)
    {
        return error.InvalidPluginPath;
    }

    registry_mutex.lockUncancelable(io);
    defer registry_mutex.unlock(io);

    // ── Load And Validate Descriptor ──

    var library: ?DynamicLibrary = try DynamicLibrary.openNowLocal(path);
    errdefer if (library) |*owned_library| {
        owned_library.close();
    };

    const loaded_library = library orelse {
        unreachable;
    };

    const descriptor_symbol = loaded_library.lookup(abi.descriptor_name) orelse {
        return error.MissingDescriptor;
    };

    const header: *const abi.DescriptorHeader = @ptrCast(@alignCast(descriptor_symbol));

    if (!std.mem.eql(u8, &header.magic, &abi.descriptor_magic) or
        header.abi_version != abi.version)
    {
        return error.UnsupportedAbiVersion;
    }

    if (header.descriptor_size != @sizeOf(abi.PluginV1)) {
        return error.InvalidPluginManifest;
    }

    const descriptor: *const abi.PluginV1 = @ptrCast(@alignCast(descriptor_symbol));
    const candidate = try copyAndValidatePlugin(path, loaded_library, descriptor.*);

    library = null;

    var owned_candidate = candidate;
    errdefer owned_candidate.deinit();

    // ── Publish Registration ──

    if (registered_plugins.items.len >= max_registered_plugins) {
        return error.PluginLimitReached;
    }

    for (registered_plugins.items) |registered| {
        if (std.mem.eql(u8, registered.path, candidate.path) or
            std.mem.eql(u8, registered.name, candidate.name))
        {
            return error.DuplicatePlugin;
        }
    }

    const handle: Handle = @enumFromInt(next_handle_value);

    next_handle_value +%= 1;

    if (next_handle_value == 0) {
        next_handle_value = 1;
    }

    owned_candidate.handle = handle;

    try registered_plugins.append(registry_allocator, owned_candidate);

    return handle;
}

/// `unregister` closes the selected dynamic library after active plugin code
/// leaves the registry's serialized critical section. It returns false when
/// the handle is not registered.
pub fn unregister(io: std.Io, handle: Handle) RegistryError!bool {
    if (plugin_code_is_active) {
        return error.PluginRegistryReentry;
    }

    registry_mutex.lockUncancelable(io);
    defer registry_mutex.unlock(io);

    for (registered_plugins.items, 0..) |plugin, index| {
        if (plugin.handle != handle) {
            continue;
        }

        var removed = registered_plugins.orderedRemove(index);

        removed.deinit();

        return true;
    }

    return false;
}

/// `unregisterAll` closes every plugin for deterministic process teardown and
/// invalidates every previously returned handle.
pub fn unregisterAll(io: std.Io) RegistryError!void {
    if (plugin_code_is_active) {
        return error.PluginRegistryReentry;
    }

    registry_mutex.lockUncancelable(io);
    defer registry_mutex.unlock(io);

    while (registered_plugins.pop()) |plugin| {
        var removed = plugin;

        removed.deinit();
    }

    registered_plugins.deinit(registry_allocator);

    registered_plugins = .empty;
}

/// `list` returns an allocator-owned snapshot which remains valid across
/// subsequent registration and unloading. The caller must invoke `List.deinit`
/// with the same allocator.
pub fn list(allocator: Allocator, io: std.Io) (Allocator.Error || RegistryError)!List {
    if (plugin_code_is_active) {
        return error.PluginRegistryReentry;
    }

    registry_mutex.lockUncancelable(io);
    defer registry_mutex.unlock(io);

    const result = try allocator.alloc(Info, registered_plugins.items.len);

    var initialized_count: usize = 0;
    errdefer {
        for (result[0..initialized_count]) |info| {
            for (info.rule_names) |rule_name| {
                allocator.free(rule_name);
            }

            allocator.free(info.rule_names);
            allocator.free(info.name);
            allocator.free(info.path);
        }

        allocator.free(result);
    }

    for (registered_plugins.items, result) |plugin, *info| {
        const path = try allocator.dupe(u8, plugin.path);
        errdefer allocator.free(path);

        const name = try allocator.dupe(u8, plugin.name);
        errdefer allocator.free(name);

        const rule_names = try allocator.alloc([]const u8, plugin.rules.len);

        var rule_count: usize = 0;
        errdefer {
            for (rule_names[0..rule_count]) |rule_name| {
                allocator.free(rule_name);
            }

            allocator.free(rule_names);
        }

        for (plugin.rules, rule_names) |rule, *rule_name| {
            rule_name.* = try allocator.dupe(u8, rule.local_name);
            rule_count += 1;
        }

        info.* = .{
            .handle = plugin.handle,
            .path = path,
            .name = name,
            .rule_names = rule_names,
        };
        initialized_count += 1;
    }

    return .{ .items = result };
}

/// Run only the explicitly selected registered plugins. The global lock is
/// intentionally held through every callback: plugin execution is serialized,
/// unregister cannot invalidate code, and a rule callback must not mutate the
/// registry or start a nested lint operation that selects plugins.
pub fn runConfigured(
    io: std.Io,
    handles: []const Handle,
    context: *LintContext,
) RunError!void {
    if (handles.len == 0) {
        return;
    }

    if (plugin_code_is_active) {
        return error.PluginRegistryReentry;
    }

    if (handles.len > max_registered_plugins) {
        return error.TooManyPluginsSelected;
    }

    registry_mutex.lockUncancelable(io);
    defer registry_mutex.unlock(io);

    var selected_plugins: [max_registered_plugins]*const RegisteredPlugin = undefined;
    for (handles, 0..) |handle, handle_index| {
        for (handles[0..handle_index]) |previous| {
            if (previous == handle) {
                return error.DuplicatePluginSelection;
            }
        }

        selected_plugins[handle_index] = findRegistered(handle) orelse {
            return error.PluginNotRegistered;
        };
    }

    for (selected_plugins[0..handles.len]) |plugin| {
        for (plugin.rules) |rule| {
            try runExternalRule(context, rule);
        }
    }
}

fn copyAndValidatePlugin(
    path: []const u8,
    library: DynamicLibrary,
    manifest: abi.PluginV1,
) RegisterError!RegisteredPlugin {
    // ── Validate Manifest ──

    if (manifest.sdk_abi != abi.sdk_abi_id) {
        return error.IncompatibleSdk;
    }

    try validateCompatibility(manifest.compatibility);

    if (manifest.build_mode > @intFromEnum(abi.BuildMode.release_small) or
        !std.mem.allEqual(u8, &manifest.reserved, 0) or
        manifest.reserved_length != 0)
    {
        return error.InvalidPluginManifest;
    }

    if (manifest.required_host_capabilities & ~abi.host_capabilities != 0 or
        manifest.required_host_capabilities & abi.capability_report_findings == 0)
    {
        return error.UnsupportedCapabilities;
    }

    if (manifest.rules_length == 0 or manifest.rules_length > abi.max_rules_per_plugin) {
        return error.InvalidPluginManifest;
    }

    const plugin_name = checkedBytes(manifest.name, abi.max_plugin_name_bytes) orelse {
        return error.InvalidPluginName;
    };

    if (!abi.nameIsValid(plugin_name)) {
        return error.InvalidPluginName;
    }

    const descriptors_pointer = manifest.rules_pointer orelse {
        return error.InvalidPluginManifest;
    };

    const descriptors = descriptors_pointer[0..manifest.rules_length];

    // ── Copy Load-Lifetime State ──

    const owned_path = try registry_allocator.dupe(u8, path);
    errdefer registry_allocator.free(owned_path);

    const owned_name = try registry_allocator.dupe(u8, plugin_name);
    errdefer registry_allocator.free(owned_name);

    const rules = try registry_allocator.alloc(RegisteredRule, descriptors.len);

    var initialized_rules: usize = 0;
    errdefer {
        for (rules[0..initialized_rules]) |rule| {
            registry_allocator.free(rule.qualified_name);
            registry_allocator.free(rule.local_name);
        }

        registry_allocator.free(rules);
    }

    for (descriptors, 0..) |descriptor, descriptor_index| {
        if (descriptor.struct_size < @sizeOf(abi.RuleDescriptorV1) or
            descriptor.reserved != 0 or
            descriptor.required_host_capabilities & ~abi.host_capabilities != 0 or
            descriptor.required_host_capabilities & abi.capability_report_findings == 0)
        {
            return error.InvalidRuleDescriptor;
        }

        const rule_name = checkedBytes(descriptor.name, abi.max_rule_name_bytes) orelse {
            return error.InvalidRuleName;
        };

        if (!abi.nameIsValid(rule_name)) {
            return error.InvalidRuleName;
        }

        for (rules[0..initialized_rules]) |previous| {
            if (std.mem.eql(u8, previous.local_name, rule_name)) {
                return error.DuplicateRule;
            }
        }

        const run = descriptor.run orelse {
            return error.InvalidRuleDescriptor;
        };

        const local_name = try registry_allocator.dupe(u8, rule_name);
        errdefer registry_allocator.free(local_name);

        const qualified_name = try std.fmt.allocPrint(
            registry_allocator,
            "{s}/{s}",
            .{ plugin_name, rule_name },
        );
        errdefer registry_allocator.free(qualified_name);

        rules[descriptor_index] = .{
            .local_name = local_name,
            .qualified_name = qualified_name,
            .run = run,
        };
        initialized_rules += 1;
    }

    return .{
        .handle = @enumFromInt(0),
        .path = owned_path,
        .name = owned_name,
        .library = library,
        .rules = rules,
    };
}

fn validateCompatibility(plugin: abi.Compatibility) RegisterError!void {
    const host = abi.compatibility;
    if (plugin.zig_major != host.zig_major or
        plugin.zig_minor != host.zig_minor or
        plugin.zig_patch != host.zig_patch)
    {
        return error.IncompatibleZigVersion;
    }

    if (plugin.architecture != host.architecture or
        plugin.operating_system != host.operating_system or
        plugin.target_abi != host.target_abi or
        plugin.pointer_bits != host.pointer_bits or
        plugin.endian != host.endian)
    {
        return error.IncompatiblePluginTarget;
    }

    if (plugin.token_tag_size != host.token_tag_size or
        plugin.node_tag_size != host.node_tag_size or
        plugin.node_data_size != host.node_data_size or
        plugin.node_data_alignment != host.node_data_alignment)
    {
        return error.IncompatibleAstStorage;
    }

    if (plugin.ast_schema_id != host.ast_schema_id) {
        return error.IncompatibleAstSchema;
    }
}

fn findRegistered(handle: Handle) ?*const RegisteredPlugin {
    for (registered_plugins.items) |*plugin| {
        if (plugin.handle == handle) {
            return plugin;
        }
    }

    return null;
}

const ExternalInvocation = struct {
    context: *LintContext,
    rule_name: []const u8,
    finding_count: u32 = 0,
    report_failure: ?Rule.Error = null,
};

fn runExternalRule(context: *LintContext, rule: RegisteredRule) RunError!void {
    var external: ExternalInvocation = .{
        .context = context,
        .rule_name = rule.qualified_name,
    };

    const rule_context = &context.rule_context;
    var scratch_allocator = rule_context.scratch_allocator;

    const allocator: abi.AllocatorV1 = .{
        .state = &scratch_allocator,
        .alloc = pluginAllocatorAlloc,
        .resize = pluginAllocatorResize,
        .remap = pluginAllocatorRemap,
        .free = pluginAllocatorFree,
    };

    const token_tags = rule_context.ast.tokens.items(.tag);
    const token_starts = rule_context.ast.tokens.items(.start);
    const node_tags = rule_context.ast.nodes.items(.tag);
    const node_main_tokens = rule_context.ast.nodes.items(.main_token);
    const node_data = rule_context.ast.nodes.items(.data);

    const ast: abi.AstV1 = .{
        .struct_size = @sizeOf(abi.AstV1),
        .source_pointer = rule_context.ast.source.ptr,
        .source_length = @intCast(rule_context.ast.source.len),
        .token_tags_pointer = @ptrCast(token_tags.ptr),
        .token_starts_pointer = token_starts.ptr,
        .token_count = @intCast(token_tags.len),
        .node_tags_pointer = @ptrCast(node_tags.ptr),
        .node_main_tokens_pointer = @ptrCast(node_main_tokens.ptr),
        .node_data_pointer = @ptrCast(node_data.ptr),
        .node_count = @intCast(node_tags.len),
        .extra_data_pointer = @ptrCast(rule_context.ast.extra_data.ptr),
        .extra_data_length = @intCast(rule_context.ast.extra_data.len),
        .line_start_offsets_pointer = rule_context.line_start_offsets.ptr,
        .line_start_offsets_length = @intCast(rule_context.line_start_offsets.len),
        .token_line_indexes_pointer = rule_context.token_line_indexes.ptr,
        .token_line_indexes_length = @intCast(rule_context.token_line_indexes.len),
    };

    const invocation: abi.InvocationV1 = .{
        .struct_size = @sizeOf(abi.InvocationV1),
        .abi_version = abi.version,
        .host_capabilities = abi.host_capabilities,
        .target_zig_version = zigVersionToRaw(rule_context.target_zig_version),
        .scratch_allocator = &allocator,
        .ast = &ast,
        .host_context = &external,
        .report_finding = reportExternalFinding,
        .fixes_enabled = @intFromBool(context.fixesEnabled()),
    };

    plugin_code_is_active = true;
    defer plugin_code_is_active = false;

    const raw_status = rule.run(&invocation);

    if (external.report_failure) |failure| {
        return failure;
    }

    return switch (raw_status) {
        abi.status.ok => {},
        abi.status.out_of_memory => error.OutOfMemory,
        abi.status.rule_failed => error.PluginRuleFailed,
        abi.status.finding_rejected, abi.status.host_protocol_violation => error.PluginProtocolViolation,
        else => error.PluginProtocolViolation,
    };
}

fn reportExternalFinding(
    host_context: ?*anyopaque,
    abi_finding: *const abi.FindingV1,
) callconv(.c) abi.RawStatus {
    const invocation_pointer = host_context orelse {
        return abi.status.host_protocol_violation;
    };

    const invocation: *ExternalInvocation = @ptrCast(@alignCast(invocation_pointer));

    if (invocation.report_failure) |failure| {
        return statusFromRuleError(failure);
    }

    if (invocation.finding_count >= abi.max_findings_per_rule) {
        return rejectFinding(invocation);
    }

    if (abi_finding.struct_size < @sizeOf(abi.FindingV1) or
        abi_finding.has_fix > 1 or
        !std.mem.allEqual(u8, &abi_finding.reserved, 0))
    {
        return rejectFinding(invocation);
    }

    const message = checkedBytes(abi_finding.message, abi.max_message_bytes) orelse {
        return rejectFinding(invocation);
    };

    const help = checkedBytes(abi_finding.help, abi.max_help_bytes) orelse {
        return rejectFinding(invocation);
    };

    const note = checkedBytes(abi_finding.note, abi.max_note_bytes) orelse {
        return rejectFinding(invocation);
    };

    if (message.len == 0 or
        @as(usize, abi_finding.token) >= invocation.context.rule_context.ast.tokens.len)
    {
        return rejectFinding(invocation);
    }

    const fix: ?Rule.Fix = if (abi_finding.has_fix == 1) fix: {
        if (!invocation.context.fixesEnabled() or
            abi_finding.fix.struct_size < @sizeOf(abi.FixV1) or
            abi_finding.fix.reserved != 0)
        {
            return rejectFinding(invocation);
        }

        const replacement = checkedBytes(
            abi_finding.fix.replacement,
            abi.max_replacement_bytes,
        ) orelse {
            return rejectFinding(invocation);
        };

        if (abi_finding.fix.range.start_offset > abi_finding.fix.range.end_offset or
            @as(usize, abi_finding.fix.range.end_offset) > invocation.context.rule_context.ast.source.len)
        {
            return rejectFinding(invocation);
        }

        break :fix .{
            .range = .{
                .start_offset = abi_finding.fix.range.start_offset,
                .end_offset = abi_finding.fix.range.end_offset,
            },
            .replacement = replacement,
        };
    } else null;

    invocation.context.reportExternalFinding(invocation.rule_name, .{
        .token = abi_finding.token,
        .message = message,
        .help = help,
        .note = note,
        .fix = fix,
    }) catch |failure| {
        invocation.report_failure = failure;

        return statusFromRuleError(failure);
    };

    invocation.finding_count += 1;

    return abi.status.ok;
}

fn rejectFinding(invocation: *ExternalInvocation) abi.RawStatus {
    invocation.report_failure = error.FindingRejected;

    return abi.status.finding_rejected;
}

fn statusFromRuleError(failure: Rule.Error) abi.RawStatus {
    return switch (failure) {
        error.OutOfMemory => abi.status.out_of_memory,
        error.FindingRejected => abi.status.finding_rejected,
        error.HostProtocolViolation => abi.status.host_protocol_violation,
    };
}

fn pluginAllocatorAlloc(
    opaque_allocator: *anyopaque,
    length: usize,
    alignment_log2: u8,
    return_address: usize,
) callconv(.c) ?[*]u8 {
    if (alignment_log2 >= @bitSizeOf(usize)) {
        return null;
    }

    const alignment: std.mem.Alignment = @enumFromInt(alignment_log2);
    const allocator: *const Allocator = @ptrCast(@alignCast(opaque_allocator));

    return allocator.rawAlloc(length, alignment, return_address);
}

fn pluginAllocatorResize(
    opaque_allocator: *anyopaque,
    memory: [*]u8,
    old_length: usize,
    alignment_log2: u8,
    new_length: usize,
    return_address: usize,
) callconv(.c) u8 {
    if (alignment_log2 >= @bitSizeOf(usize)) {
        return 0;
    }

    const alignment: std.mem.Alignment = @enumFromInt(alignment_log2);
    const allocator: *const Allocator = @ptrCast(@alignCast(opaque_allocator));

    return @intFromBool(allocator.rawResize(
        memory[0..old_length],
        alignment,
        new_length,
        return_address,
    ));
}

fn pluginAllocatorRemap(
    opaque_allocator: *anyopaque,
    memory: [*]u8,
    old_length: usize,
    alignment_log2: u8,
    new_length: usize,
    return_address: usize,
) callconv(.c) ?[*]u8 {
    if (alignment_log2 >= @bitSizeOf(usize)) {
        return null;
    }

    const alignment: std.mem.Alignment = @enumFromInt(alignment_log2);
    const allocator: *const Allocator = @ptrCast(@alignCast(opaque_allocator));

    return allocator.rawRemap(
        memory[0..old_length],
        alignment,
        new_length,
        return_address,
    );
}

fn pluginAllocatorFree(
    opaque_allocator: *anyopaque,
    memory: [*]u8,
    length: usize,
    alignment_log2: u8,
    return_address: usize,
) callconv(.c) void {
    if (alignment_log2 >= @bitSizeOf(usize)) {
        return;
    }

    const alignment: std.mem.Alignment = @enumFromInt(alignment_log2);
    const allocator: *const Allocator = @ptrCast(@alignCast(opaque_allocator));

    allocator.rawFree(memory[0..length], alignment, return_address);
}

fn zigVersionToRaw(version: std.SemanticVersion) abi.ZigVersionV1 {
    return .{
        .major = @intCast(version.major),
        .minor = @intCast(version.minor),
        .patch = @intCast(version.patch),
        .prerelease = optionalBytes(version.pre),
        .build = optionalBytes(version.build),
    };
}

fn optionalBytes(value: ?[]const u8) abi.Bytes {
    const contents = value orelse {
        return .{};
    };

    return .{
        .pointer = if (contents.len == 0) null else contents.ptr,
        .length = contents.len,
    };
}

fn checkedBytes(view: abi.Bytes, maximum: u64) ?[]const u8 {
    if (view.length > maximum or view.length > std.math.maxInt(usize)) {
        return null;
    }

    if (view.length == 0) {
        return "";
    }

    const pointer = view.pointer orelse {
        return null;
    };

    return pointer[0..@intCast(view.length)];
}

test "target Zig version is encoded without loss" {
    const version: std.SemanticVersion = .{
        .major = 0,
        .minor = 16,
        .patch = 0,
        .pre = "dev.123",
        .build = "custom.456",
    };

    const raw = zigVersionToRaw(version);

    const prerelease = version.pre orelse {
        return error.MissingPrerelease;
    };

    const build = version.build orelse {
        return error.MissingBuild;
    };

    const raw_prerelease = checkedBytes(raw.prerelease, std.math.maxInt(u64)) orelse {
        return error.MissingRawPrerelease;
    };

    const raw_build = checkedBytes(raw.build, std.math.maxInt(u64)) orelse {
        return error.MissingRawBuild;
    };

    try std.testing.expectEqual(@as(u64, version.major), raw.major);
    try std.testing.expectEqual(@as(u64, version.minor), raw.minor);
    try std.testing.expectEqual(@as(u64, version.patch), raw.patch);
    try std.testing.expectEqualStrings(prerelease, raw_prerelease);
    try std.testing.expectEqualStrings(build, raw_build);
}
