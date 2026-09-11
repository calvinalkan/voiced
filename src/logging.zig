//! Process-owned diagnostics. Service records use either one bounded
//! nonblocking journal datagram or one explicit stderr line; recording and
//! inference hot paths must not call this API. Initialize before starting
//! threads, and close only after they have joined.
const std = @import("std");
const linux = std.os.linux;
const decimal = @import("decimal.zig");

pub const Level = enum(u8) { critical = 2, err = 3, warn = 4, info = 6, debug = 7 };
pub const Target = enum { auto, journal, stderr };
pub const Context = struct { recording_id: ?u64 = null };
const record_bytes_max = 4096;
const message_bytes_max = 3584;
const Sink = union(enum) { disabled, journal: struct { fd: i32, connected: bool }, stderr };
var sink: Sink = .disabled;
var threshold: Level = .info;
var dropped: std.atomic.Value(u64) = .init(0);

/// The auto target selects stderr when stderr is a terminal and journal
/// otherwise. The journal target sends to journald's pathname on each call, so
/// a restart requires no reconnect state. A connected datagram on stderr
/// supplies an isolated journal receiver for tests. The stderr target writes one
/// bounded human-readable line to the inherited descriptor; a pipe or slow
/// terminal can therefore block its calling thread and should be actively drained.
pub fn init(io: std.Io, target: Target, configured_level: Level) linux.E {
    std.debug.assert(sink == .disabled);
    threshold = configured_level;
    const resolved_target: Target = if (target == .auto)
        if (std.Io.File.stderr().isTty(io) catch false) .stderr else .journal
    else
        target;
    if (resolved_target == .stderr) {
        sink = .stderr;
        return .SUCCESS;
    }

    var socket_type: i32 = 0;
    var length: linux.socklen_t = @sizeOf(i32);
    if (linux.errno(linux.getsockopt(2, linux.SOL.SOCKET, linux.SO.TYPE, @ptrCast(&socket_type), &length)) == .SUCCESS and socket_type == linux.SOCK.DGRAM) {
        const result = linux.fcntl(2, linux.F.DUPFD_CLOEXEC, 3);
        if (linux.errno(result) != .SUCCESS) return linux.errno(result);
        sink = .{ .journal = .{ .fd = @intCast(result), .connected = true } };
    } else {
        const result = linux.socket(linux.AF.UNIX, linux.SOCK.DGRAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0);
        if (linux.errno(result) != .SUCCESS) return linux.errno(result);
        sink = .{ .journal = .{ .fd = @intCast(result), .connected = false } };
    }
    return .SUCCESS;
}

/// Offline CLI tools may use stderr; they do not own service deadlines. Help,
/// argument errors, and command results need not go through the logger.
pub fn initCli(configured_level: Level) void {
    std.debug.assert(sink == .disabled);
    threshold = configured_level;
    sink = .stderr;
}

pub fn deinit() void {
    if (sink == .journal) _ = linux.close(sink.journal.fd);
    sink = .disabled;
    dropped.store(0, .monotonic);
}

pub fn activeTarget() Target {
    return switch (sink) {
        .journal => .journal,
        .stderr => .stderr,
        .disabled => unreachable,
    };
}

pub fn parseLevel(text: []const u8) ?Level {
    if (std.mem.eql(u8, text, "error")) return .err;
    if (std.mem.eql(u8, text, "warn")) return .warn;
    if (std.mem.eql(u8, text, "critical")) return .critical;
    if (std.mem.eql(u8, text, "info")) return .info;
    if (std.mem.eql(u8, text, "debug")) return .debug;
    return null;
}

pub fn parseTarget(text: []const u8) ?Target {
    return std.meta.stringToEnum(Target, text);
}

pub fn levelName(value: Level) []const u8 {
    return if (value == .err) "error" else @tagName(value);
}

/// Returns a `{f}` formatter for a named Linux errno, with the same text as
/// `{t}` and no width or alignment. Like `{t}`, unnamed enum values are invalid.
/// Name lookup happens only when formatted; writer errors propagate unchanged.
pub fn fmtErrno(value: linux.E) ErrnoFormat {
    return .{ .value = value };
}

const ErrnoFormat = struct {
    value: linux.E,

    // PERFORMANCE: Separate name selection from writing, and share both on
    // diagnostic paths. Formatting linux.E directly with `{t}` lets LLVM copy
    // writer logic into its many tag branches. The separate lookup returns one
    // slice to one writer instead, at the cost of a call. Keep conversion lazy
    // so a filtered log never performs the lookup.
    pub noinline fn format(self: ErrnoFormat, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.writeAll(errnoName(self.value));
    }

    noinline fn errnoName(value: linux.E) []const u8 {
        return @tagName(value);
    }
};

/// The event function filters before formatting. Guard expensive argument
/// preparation at the call site with enabled(); Zig evaluates arguments first.
pub fn enabled(value: Level) bool {
    return sink != .disabled and @intFromEnum(value) <= @intFromEnum(threshold);
}

pub const Field = union(enum) {
    /// Writes a lower-case ASCII identifier without quotes.
    name: []const u8,
    /// Writes arbitrary bytes inside quotes with Zig-style escapes.
    str: []const u8,
    /// Writes slices consecutively as one quoted, escaped string.
    str_joined: []const []const u8,
    /// Writes a trusted, nonempty ASCII value without quotes. The value must not
    /// contain whitespace, controls, quotes, or `=`.
    verbatim: []const u8,
    u: u64,
    i: i64,
    octal: u64,
    f: struct { value: f64, digits: u3 },
    /// Preserves f32 decimal rounding; widening to f64 can change the last digit.
    f32: struct { value: f32, digits: u3 },
    b: bool,
    errno: linux.E,
};
pub const Entry = struct { []const u8, Field };

/// Builds a bounded field list whose storage and names come from a dense,
/// zero-based enum. Each name has one slot; duplicate additions assert in safe
/// builds and overwrite that slot otherwise. `items()` compacts populated slots
/// into enum declaration order.
pub fn EnumFieldSet(comptime Name: type) type {
    const names = std.meta.fields(Name);
    comptime {
        // Large schemas validate every identifier byte and can exceed Zig's
        // default compile-time branch quota.
        @setEvalBranchQuota(10_000);
        if (@typeInfo(Name) != .@"enum") @compileError("logging field names must be an enum");
        if (names.len > @bitSizeOf(u64)) @compileError("logging field enum exceeds the presence mask");
        for (names, 0..) |name, index| {
            if (name.value != index) @compileError("logging field enum must be dense and zero-based");
            assertIdentifier(name.name);
        }
    }

    return struct {
        const Self = @This();
        pub const Value = struct { Name, Field };

        storage: [names.len]Entry = undefined,
        present_mask: u64 = 0,

        pub fn init(fields: *Self) void {
            fields.present_mask = 0;
        }

        pub fn addAll(fields: *Self, values: []const Value) void {
            for (values) |value| fields.add(value[0], value[1]);
        }

        pub fn add(fields: *Self, name: Name, value: Field) void {
            const index = @intFromEnum(name);
            const mask = @as(u64, 1) << @intCast(index);
            std.debug.assert(fields.present_mask & mask == 0);
            fields.storage[index] = .{ @tagName(name), value };
            fields.present_mask |= mask;
        }

        pub fn items(fields: *Self) []const Entry {
            var count: usize = 0;
            for (std.meta.tags(Name)) |name| {
                const index = @intFromEnum(name);
                if (fields.present_mask & (@as(u64, 1) << @intCast(index)) == 0) continue;
                fields.storage[count] = fields.storage[index];
                count += 1;
            }
            return fields.storage[0..count];
        }
    };
}

pub fn scoped(comptime component: @EnumLiteral()) type {
    return struct {
        /// Appends ordered `key=value` fields to one event. Event names, keys,
        /// and `.name` values must be nonempty lower-case ASCII identifiers
        /// (`[a-z_][a-z0-9_]*`). Integers are decimal unless marked `.octal`;
        /// floats use the requested fractional digits, and `.errno` requires a
        /// named Linux value. Inputs are borrowed only during the call.
        /// Filtering precedes formatting, not field-array construction; use
        /// `enabled()` before costly preparation.
        pub fn event(severity: Level, context: Context, event_name: []const u8, fields: []const Entry) void {
            comptime std.debug.assert(@tagName(component).len <= 64);
            comptime assertIdentifier(@tagName(component));
            emitEvent(severity, @tagName(component), event_name, context, fields);
        }
    };
}

// PERFORMANCE: Keep this loop non-generic and out-of-line. Specializing on
// field shapes or inlining into callers duplicates formatting across events.
// The tagged entries and extra call trade stack space and cold-path dispatch
// for shared code; no allocation or capture/inference hot-path work is added.
noinline fn emitEvent(severity: Level, component: []const u8, event: []const u8, context: Context, fields: []const Entry) void {
    if (!enabled(severity)) return;
    std.debug.assert(event.len <= 64);
    assertIdentifier(event);
    var message_buffer: [message_bytes_max]u8 = undefined;
    var message = std.Io.Writer.fixed(&message_buffer);
    const truncated = writeMessage(&message, event, context, fields);
    sendRecord(severity, component, event, context, &message, truncated);
}

fn writeMessage(message: *std.Io.Writer, event: []const u8, context: Context, fields: []const Entry) bool {
    message.writeAll(event) catch return true;
    if (context.recording_id) |recording_id| {
        message.print(" recording_id={d}", .{recording_id}) catch return true;
    }
    for (fields) |entry| {
        const key, const field = entry;
        assertIdentifier(key);
        message.writeByte(' ') catch return true;
        message.writeAll(key) catch return true;
        message.writeByte('=') catch return true;
        switch (field) {
            .name => |value| {
                assertIdentifier(value);
                message.writeAll(value) catch return true;
            },
            .str => |value| {
                message.writeByte('"') catch return true;
                std.zig.stringEscape(value, message) catch return true;
                message.writeByte('"') catch return true;
            },
            .str_joined => |values| {
                message.writeByte('"') catch return true;
                for (values) |value| std.zig.stringEscape(value, message) catch return true;
                message.writeByte('"') catch return true;
            },
            .verbatim => |value| {
                assertVerbatim(value);
                message.writeAll(value) catch return true;
            },
            .u => |value| message.printInt(value, 10, .lower, .{}) catch return true,
            .i => |value| message.printInt(value, 10, .lower, .{}) catch return true,
            .octal => |value| message.printInt(value, 8, .lower, .{}) catch return true,
            .f => |value| decimal.fmt(value.value, value.digits).format(message) catch return true,
            .f32 => |value| decimal.fmt(value.value, value.digits).format(message) catch return true,
            .b => |value| message.writeAll(if (value) "true" else "false") catch return true,
            .errno => |value| fmtErrno(value).format(message) catch return true,
        }
    }
    return false;
}

test "event emits escaped fields in one journal datagram" {
    try std.testing.expect(sink == .disabled);
    var sockets: [2]i32 = undefined;
    try std.testing.expectEqual(.SUCCESS, linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.DGRAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0, &sockets)));
    defer _ = linux.close(sockets[1]);
    const previous_threshold = threshold;
    sink = .{ .journal = .{ .fd = sockets[0], .connected = true } };
    threshold = .debug;
    defer {
        deinit();
        threshold = previous_threshold;
    }

    const test_log = scoped(.logging);
    test_log.event(.warn, .{ .recording_id = 7 }, "escaped_message", &.{
        .{ "detail", .{ .str = "line \"one\"\npriority=0\t\x01" } },
    });

    var record: [record_bytes_max]u8 = undefined;
    const received = linux.read(sockets[1], &record, record.len);
    try std.testing.expectEqual(.SUCCESS, linux.errno(received));
    const record_size: usize = received;
    const journal_prefix =
        "PRIORITY=4\n" ++
        "SYSLOG_IDENTIFIER=voiced\n" ++
        "VOICED_COMPONENT=logging\n" ++
        "VOICED_EVENT=escaped_message\n" ++
        "VOICED_RECORDING_ID=7\n" ++
        "MESSAGE\n";
    const expected_message = "escaped_message recording_id=7 detail=\"line \\\"one\\\"\\npriority=0\\t\\x01\"";
    const message_size_offset = journal_prefix.len;
    const message_offset = message_size_offset + @sizeOf(u64);
    const message_end = message_offset + expected_message.len;

    try std.testing.expectEqual(message_end + 1, record_size);
    try std.testing.expectEqualStrings(journal_prefix, record[0..journal_prefix.len]);
    try std.testing.expectEqual(@as(u64, expected_message.len), std.mem.readInt(u64, record[message_size_offset..message_offset], .little));
    try std.testing.expectEqualStrings(expected_message, record[message_offset..message_end]);
    try std.testing.expectEqual(@as(u8, '\n'), record[message_end]);
}

fn assertIdentifier(value: []const u8) void {
    std.debug.assert(value.len > 0);
    for (value, 0..) |byte, position| {
        std.debug.assert(std.ascii.isLower(byte) or byte == '_' or
            (position > 0 and std.ascii.isDigit(byte)));
    }
}

fn assertVerbatim(value: []const u8) void {
    std.debug.assert(value.len > 0);
    for (value) |byte| {
        std.debug.assert(byte > ' ' and byte < 0x7f and byte != '"' and byte != '=');
    }
}

// Keep delivery non-generic and out-of-line so journal framing and syscalls are
// not copied into every message-format specialization. The extra call is on the
// logging cold path, not capture/inference hot paths. Measured with Zig 0.16 on
// x86-64 ReleaseSafe: ~151 KiB saved, a ~10% smaller whole stripped daemon.
noinline fn sendRecord(severity: Level, component: []const u8, event: []const u8, context: Context, message: *std.Io.Writer, truncated: bool) void {
    // PERFORMANCE: Finalize here, not in generic emit: otherwise every message
    // format can get another copy of trimming and truncation code. Measured
    // 2026-09-07 with stock Zig 0.16.0/LLVM, host x86-64, ReleaseSafe application
    // and inference, static PIE, crash diagnostics disabled, GNU strip --strip-all:
    // this change alone reduced 1,161,416 to 1,147,032 bytes (14,384 saved).
    // Together with decimal.zig, the applied worktree went from 1,323,960 to
    // 1,301,832 bytes (22,128 saved). The isolated baseline also had initializer
    // optimizations; do not add savings across baselines or compiler versions.
    // The combined prototype passed control, logging and real-model output
    // integration, including truncation, metadata injection and receiver drops.
    // Only emitEvent's fixed writer enters. Append the marker before trimming so a
    // truncated message retains embedded/trailing newlines before the marker.
    if (truncated) {
        const marker = " [truncated]";
        const end = @min(message.end, message.buffer.len - marker.len);
        @memcpy(message.buffer[end..][0..marker.len], marker);
        message.end = end + marker.len;
    }
    const text = std.mem.trimEnd(u8, message.buffered(), "\n");
    if (sink == .stderr) {
        var line_buffer: [record_bytes_max]u8 = undefined;
        const line = std.fmt.bufPrint(&line_buffer, "{s}: {s}\n", .{ levelName(severity), text }) catch unreachable;
        var remaining = line;
        while (remaining.len != 0) {
            const result = linux.write(2, remaining.ptr, remaining.len);
            switch (linux.errno(result)) {
                .SUCCESS => {
                    if (result == 0) return;
                    remaining = remaining[result..];
                },
                .INTR => continue,
                else => return,
            }
        }
        return;
    }
    const lost = dropped.swap(0, .monotonic);
    var buffer: [record_bytes_max]u8 = undefined;
    var record = std.Io.Writer.fixed(&buffer);
    record.print("PRIORITY={d}\nSYSLOG_IDENTIFIER=voiced\nVOICED_COMPONENT={s}\nVOICED_EVENT={s}\n", .{ @intFromEnum(severity), component, event }) catch unreachable;
    if (context.recording_id) |value| record.print("VOICED_RECORDING_ID={d}\n", .{value}) catch unreachable;
    if (lost != 0) record.print("VOICED_DROPPED={d}\n", .{lost}) catch unreachable;
    // Always use the native binary MESSAGE encoding. Newlines in native error
    // details remain one message and cannot inject journal metadata fields.
    record.writeAll("MESSAGE\n") catch unreachable;
    record.writeInt(u64, text.len, .little) catch unreachable;
    record.writeAll(text) catch unreachable;
    record.writeByte('\n') catch unreachable;
    const sent = switch (sink) {
        .journal => |journal| blk: {
            var address: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) };
            const path = "/run/systemd/journal/socket";
            @memcpy(address.path[0..path.len], path);
            const bytes = record.buffered();
            const result = linux.sendto(journal.fd, bytes.ptr, bytes.len, linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL, if (journal.connected) null else @ptrCast(&address), if (journal.connected) 0 else @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1));
            break :blk linux.errno(result) == .SUCCESS and result == bytes.len;
        },
        .disabled => false,
        .stderr => unreachable,
    };
    if (!sent) countDropped(lost +| 1);
}

fn countDropped(count: u64) void {
    var previous = dropped.load(.monotonic);
    while (dropped.cmpxchgWeak(previous, previous +| count, .monotonic, .monotonic)) |actual| previous = actual;
}
