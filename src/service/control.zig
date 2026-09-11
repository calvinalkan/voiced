//! Owns bounded fixed-record control connections, not command or lifecycle policy.
//! CLI and daemon share one little-endian wire ABI. Earlier layouts are never
//! accepted; changing the layout requires replacing both ends together.
//! The supervisor supplies epoll tags and absolute time, dispatches requests,
//! and includes client deadlines in its existing timerfd schedule.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const logging = @import("../logging.zig");
const log = logging.scoped(.control);
const assert = std.debug.assert;
pub const clients_count_max = 16;
const request_timeout_ns = std.time.ns_per_s;

pub const Request = struct {
    cmd: enum(u8) { record = 2, stop = 3, cancel = 4, status = 5, kill = 6 },
    toggle: bool = false,
};

pub const Status = struct {
    ignored: bool,
    phase: Phase,
    model: ModelState,
    model_kind: ModelKind,
    recording_id: u64,
    model_idle_seconds_max: u64,
    daemon_uptime_seconds: u64,
    recording_elapsed_seconds: ?u64,
    model_idle_seconds_remaining: ?u64,
};

pub const Phase = enum(u8) { unavailable = 0, idle = 1, capturing = 2, stopping = 3, transcribing = 4, delivering = 5, _ };
pub const ModelState = enum(u8) { unavailable = 0, unloaded = 1, loading = 2, loaded = 3, unloading = 4, _ };
pub const ModelKind = enum(u8) {
    unavailable = 0,
    whisper_base_en = 1,
    whisper_small_en = 2,
    whisper_medium_en = 3,
    _,

    pub fn name(kind: ModelKind) []const u8 {
        return switch (kind) {
            .whisper_base_en => "whisper.base.en",
            .whisper_small_en => "whisper.small.en",
            .whisper_medium_en => "whisper.medium.en",
            else => unreachable,
        };
    }
};

// Non-exhaustive wire enums represent every incoming byte. Validate them before
// dispatch or @tagName; unknown values must not become invalid Zig enums.
const WireRequest = extern struct {
    magic: [4]u8 = "VCDQ".*,
    version: u16 = 1,
    command: Command,
    flags: u8 = 0,

    const Command = enum(u8) { record = 2, stop = 3, cancel = 4, status = 5, kill = 6, _ };
};

const WireResult = enum(u8) { accepted = 0, ignored = 1, invalid_request = 2, unsupported_version = 3, _ };
const seconds_unavailable = std.math.maxInt(u64);

const WireReply = extern struct {
    magic: [4]u8 = "VCDR".*,
    version: u16 = 1,
    result: WireResult,
    command: WireRequest.Command,
    recording_id: u64 = 0,
    model_idle_seconds_max: u64 = 0,
    daemon_uptime_seconds: u64 = 0,
    recording_elapsed_seconds: u64 = seconds_unavailable,
    model_idle_seconds_remaining: u64 = seconds_unavailable,
    phase: Phase = .unavailable,
    model: ModelState = .unavailable,
    model_kind: ModelKind = .unavailable,
    reserved: [13]u8 = @splat(0),
};

comptime {
    if (builtin.target.cpu.arch.endian() != .little) @compileError("control wire ABI requires little endian");
    assert(@sizeOf(WireRequest) == 8);
    for (std.meta.fields(WireRequest), [_]usize{ 0, 4, 6, 7 }) |field, offset| assert(@offsetOf(WireRequest, field.name) == offset);
    assert(@sizeOf(WireReply) == 64);
    for (std.meta.fields(WireReply), [_]usize{ 0, 4, 6, 7, 8, 16, 24, 32, 40, 48, 49, 50, 51 }) |field, offset| assert(@offsetOf(WireReply, field.name) == offset);
    for (std.meta.fields(@FieldType(Request, "cmd"))) |field| assert(field.value == @intFromEnum(@field(WireRequest.Command, field.name)));
}

/// `Server` holds the runtime-directory lock until deinit. Accept new clients
/// only after consuming the current epoll batch, so a reused client slot cannot
/// receive an event belonging to a closed connection. Each connection permits
/// one request and one response, with bounded bytes and a one-second deadline.
pub const Server = struct {
    directory: std.Io.Dir,
    listener: std.posix.fd_t,
    epoll_fd: std.posix.fd_t,
    client_event_tag: u64,
    accepting: bool = true,
    clients: [clients_count_max]Client = @splat(.{}),

    pub fn open(init: std.process.Init, epoll_fd: std.posix.fd_t, listener_tag: u64, client_tag: u64) !Server {
        const path = try socketPath(init);
        defer init.gpa.free(path);
        const runtime_root = init.environ_map.get("XDG_RUNTIME_DIR").?;
        const runtime_dir = std.Io.Dir.cwd().openDir(init.io, runtime_root, .{}) catch |err| {
            log.event(.err, .{}, "runtime_directory_unavailable", &.{
                .{ "path", .{ .str = runtime_root } },
                .{ "error", .{ .verbatim = @errorName(err) } },
            });
            return error.UnsafeRuntimeDirectory;
        };
        defer runtime_dir.close(init.io);
        try requireOwnedPrivateDirectory(runtime_dir.handle, runtime_root);
        const directory_path = std.fs.path.dirname(path).?;
        const directory = try std.Io.Dir.cwd().createDirPathOpen(init.io, directory_path, .{
            .permissions = .fromMode(0o700),
            .open_options = .{ .iterate = true, .follow_symlinks = false },
        });
        errdefer directory.close(init.io);
        try requireOwnedPrivateDirectory(directory.handle, directory_path);
        const lock: std.Io.File = .{ .handle = directory.handle, .flags = .{ .nonblocking = false } };
        if (!try lock.tryLock(init.io, .exclusive)) {
            return error.DaemonAlreadyRunning;
        }

        const listener = try createSocket(true);
        errdefer close(listener);
        const address = try unixAddress(path);
        const address_size: u32 = @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1);
        const bind_result = linux.bind(listener, @ptrCast(&address), address_size);
        switch (linux.errno(bind_result)) {
            .SUCCESS => {},
            .ADDRINUSE => {
                // EPROTOTYPE can mean a live older stream listener. Only
                // ECONNREFUSED establishes staleness; never unlink on mismatch.
                const probe = try createSocket(true);
                defer close(probe);
                switch (linux.errno(linux.connect(probe, @ptrCast(&address), address_size))) {
                    .CONNREFUSED => {},
                    else => {
                        return error.DaemonAlreadyRunning;
                    },
                }
                const entry = try directory.statFile(init.io, "control.sock", .{ .follow_symlinks = false });
                if (entry.kind != .unix_domain_socket) {
                    return error.UnsafeControlSocket;
                }
                try directory.deleteFile(init.io, "control.sock");
                const retry_result = linux.bind(listener, @ptrCast(&address), address_size);
                if (linux.errno(retry_result) != .SUCCESS) {
                    log.event(.err, .{}, "control_bind_failed", &.{
                        .{ "operation", .{ .name = "bind" } },
                        .{ "system_error", .{ .errno = linux.errno(retry_result) } },
                    });
                    return error.ControlBindFailed;
                }
            },
            else => {
                log.event(.err, .{}, "control_bind_failed", &.{
                    .{ "operation", .{ .name = "bind" } },
                    .{ "system_error", .{ .errno = linux.errno(bind_result) } },
                });
                return error.ControlBindFailed;
            },
        }
        errdefer directory.deleteFile(init.io, "control.sock") catch |err| {
            if (err != error.FileNotFound) log.event(.err, .{}, "control_cleanup_failed", &.{
                .{ "operation", .{ .name = "unlink" } },
                .{ "error", .{ .verbatim = @errorName(err) } },
            });
        };
        try directory.setFilePermissions(init.io, "control.sock", .fromMode(0o600), .{});
        const listen_result = linux.listen(listener, clients_count_max);
        if (linux.errno(listen_result) != .SUCCESS) {
            log.event(.err, .{}, "control_listen_failed", &.{
                .{ "operation", .{ .name = "listen" } },
                .{ "system_error", .{ .errno = linux.errno(listen_result) } },
            });
            return error.ControlListenFailed;
        }
        try register(epoll_fd, listener, listener_tag, linux.EPOLL.IN);
        return .{ .directory = directory, .listener = listener, .epoll_fd = epoll_fd, .client_event_tag = client_tag };
    }

    pub fn deinit(server: *Server, io: std.Io) void {
        for (0..server.clients.len) |index| server.closeClient(index);
        close(server.listener);
        server.directory.deleteFile(io, "control.sock") catch |err| {
            if (err != error.FileNotFound) log.event(.err, .{}, "control_cleanup_failed", &.{
                .{ "operation", .{ .name = "unlink" } },
                .{ "error", .{ .verbatim = @errorName(err) } },
            });
        };
        server.directory.close(io);
    }

    /// Stop accepting new commands while existing replies drain to their deadlines.
    pub fn stopAccepting(server: *Server) !void {
        if (!server.accepting) return;
        // A queued connection keeps a level-triggered listener readable. Remove
        // readiness too, or shutdown spins while a blocked reply awaits expiry.
        const errno = linux.errno(linux.epoll_ctl(server.epoll_fd, linux.EPOLL.CTL_DEL, server.listener, null));
        if (errno != .SUCCESS) {
            log.event(.err, .{}, "control_shutdown_failed", &.{
                .{ "operation", .{ .name = "epoll_ctl_del" } },
                .{ "system_error", .{ .errno = errno } },
            });
            return error.ControlUnregisterFailed;
        }
        server.accepting = false;
    }

    pub fn acceptClients(server: *Server, now_ns: u64) !void {
        assert(server.accepting);
        for (0..clients_count_max) |_| {
            const result = linux.accept4(server.listener, null, null, linux.SOCK.CLOEXEC | linux.SOCK.NONBLOCK);
            switch (linux.errno(result)) {
                .SUCCESS => {},
                .INTR => continue,
                .AGAIN => {
                    return;
                },
                else => {
                    log.event(.err, .{}, "control_accept_failed", &.{
                        .{ "operation", .{ .name = "accept4" } },
                        .{ "system_error", .{ .errno = linux.errno(result) } },
                    });
                    return error.ControlAcceptFailed;
                },
            }
            const descriptor: std.posix.fd_t = @intCast(result);
            const index = for (&server.clients, 0..) |*client, index| {
                if (client.descriptor == null) break index;
            } else {
                close(descriptor);
                continue;
            };
            register(server.epoll_fd, descriptor, server.client_event_tag, linux.EPOLL.IN) catch |err| {
                close(descriptor);
                return err;
            };
            server.clients[index] = .{ .descriptor = descriptor, .deadline_ns = now_ns + request_timeout_ns };
        }
    }

    pub fn deadline(server: *const Server) ?u64 {
        var earliest: ?u64 = null;
        for (&server.clients) |*client| {
            if (client.descriptor != null) earliest = @min(earliest orelse std.math.maxInt(u64), client.deadline_ns);
        }
        return earliest;
    }

    pub fn expire(server: *Server, now_ns: u64) void {
        for (&server.clients, 0..) |*client, index| {
            if (client.descriptor != null and now_ns >= client.deadline_ns) server.closeClient(index);
        }
    }

    /// Shutdown drains already-dispatched replies, without waiting for new requests.
    pub fn hasPendingReplies(server: *const Server) bool {
        for (&server.clients) |*client| {
            if (client.descriptor != null and client.response != null) return true;
        }
        return false;
    }

    pub fn receive(server: *Server, index: usize) ?Request {
        const client = &server.clients[index];
        const descriptor = client.descriptor orelse {
            return null;
        };
        if (client.response != null) {
            server.flush(index);
            return null;
        }
        var wire: WireRequest = undefined;
        while (true) {
            // Without MSG_TRUNC, an oversized packet with a valid prefix passes.
            const result = linux.recvfrom(descriptor, std.mem.asBytes(&wire).ptr, @sizeOf(WireRequest), linux.MSG.DONTWAIT | linux.MSG.TRUNC, null, null);
            switch (linux.errno(result)) {
                .SUCCESS => {},
                .INTR => continue,
                .AGAIN => {
                    return null;
                },
                else => {
                    server.closeClient(index);
                    return null;
                },
            }
            if (result == 0) {
                server.closeClient(index);
                return null;
            }
            if (result != @sizeOf(WireRequest)) {
                server.reject(index, .invalid_request, @enumFromInt(0));
                return null;
            }
            if (!std.mem.eql(u8, &wire.magic, "VCDQ")) {
                server.reject(index, .invalid_request, wire.command);
                return null;
            }
            if (wire.version != 1) {
                server.reject(index, .unsupported_version, wire.command);
                return null;
            }
            const cmd = std.enums.fromInt(@FieldType(Request, "cmd"), @intFromEnum(wire.command)) orelse {
                server.reject(index, .invalid_request, wire.command);
                return null;
            };
            if (wire.flags > 1 or (wire.flags == 1 and cmd != .record)) {
                server.reject(index, .invalid_request, wire.command);
                return null;
            }
            client.command = wire.command;
            return .{ .cmd = cmd, .toggle = wire.flags == 1 };
        }
    }

    pub fn respond(server: *Server, index: usize, status: Status) void {
        const client = &server.clients[index];
        if (client.descriptor == null) return;
        client.response = .{
            .result = if (status.ignored) .ignored else .accepted,
            .command = client.command,
            .recording_id = status.recording_id,
            .model_idle_seconds_max = status.model_idle_seconds_max,
            .daemon_uptime_seconds = status.daemon_uptime_seconds,
            .recording_elapsed_seconds = status.recording_elapsed_seconds orelse seconds_unavailable,
            .model_idle_seconds_remaining = status.model_idle_seconds_remaining orelse seconds_unavailable,
            .phase = status.phase,
            .model = status.model,
            .model_kind = status.model_kind,
        };
        server.flush(index);
    }

    fn reject(server: *Server, index: usize, result: WireResult, command: WireRequest.Command) void {
        server.clients[index].response = .{ .result = result, .command = command };
        server.flush(index);
    }

    fn flush(server: *Server, index: usize) void {
        const client = &server.clients[index];
        const descriptor = client.descriptor orelse {
            return;
        };
        const bytes = std.mem.asBytes(&client.response.?);
        while (true) {
            const result = linux.sendto(descriptor, bytes.ptr, bytes.len, linux.MSG.NOSIGNAL | linux.MSG.DONTWAIT, null, 0);
            switch (linux.errno(result)) {
                // A short record is a failure, never a reason to send a suffix.
                .SUCCESS => {
                    if (result != bytes.len) log.event(.err, .{}, "control_reply_failed", &.{
                        .{ "reason", .{ .name = "short_write" } },
                        .{ "write_size", .{ .u = result } },
                        .{ "expected_size", .{ .u = bytes.len } },
                    });
                    break;
                },
                .INTR => continue,
                .AGAIN => {
                    var event: linux.epoll_event = .{ .events = linux.EPOLL.OUT | linux.EPOLL.RDHUP, .data = .{ .u64 = server.client_event_tag } };
                    if (linux.errno(linux.epoll_ctl(server.epoll_fd, linux.EPOLL.CTL_MOD, descriptor, &event)) != .SUCCESS) server.closeClient(index);
                    return;
                },
                else => {
                    server.closeClient(index);
                    return;
                },
            }
        }
        server.closeClient(index);
    }

    fn closeClient(server: *Server, index: usize) void {
        const descriptor = server.clients[index].descriptor orelse {
            return;
        };
        _ = linux.epoll_ctl(server.epoll_fd, linux.EPOLL.CTL_DEL, descriptor, null);
        close(descriptor);
        server.clients[index].descriptor = null;
    }
};

const Client = struct {
    descriptor: ?std.posix.fd_t = null,
    deadline_ns: u64 = 0,
    command: WireRequest.Command = @enumFromInt(0),
    response: ?WireReply = null,
};

/// `sendRequest` writes status data to stdout and human acknowledgements to
/// stderr. A lost reply leaves the outcome unknown. Never automatically retry a toggle.
pub fn sendRequest(init: std.process.Init, request: Request) !void {
    const path = try socketPath(init);
    defer init.gpa.free(path);
    const socket = try createSocket(false);
    defer close(socket);
    const timeout: linux.timeval = .{ .sec = 3, .usec = 0 };
    for ([_]u32{ linux.SO.RCVTIMEO, linux.SO.SNDTIMEO }) |option| {
        if (linux.errno(linux.setsockopt(socket, linux.SOL.SOCKET, option, std.mem.asBytes(&timeout).ptr, @sizeOf(linux.timeval))) != .SUCCESS) {
            return error.ControlTimeoutSetupFailed;
        }
    }
    const address = try unixAddress(path);
    switch (linux.errno(linux.connect(socket, @ptrCast(&address), @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1)))) {
        .SUCCESS => {},
        .PROTOTYPE => return error.IncompatibleControlProtocol,
        else => return error.DaemonNotRunning,
    }
    const wire: WireRequest = .{ .command = @enumFromInt(@intFromEnum(request.cmd)), .flags = @intFromBool(request.toggle) };
    while (true) {
        const result = linux.sendto(socket, std.mem.asBytes(&wire).ptr, @sizeOf(WireRequest), linux.MSG.NOSIGNAL, null, 0);
        if (linux.errno(result) == .INTR) continue;
        if (linux.errno(result) != .SUCCESS or result != @sizeOf(WireRequest)) return error.ControlSendFailed;
        break;
    }
    var wire_reply: WireReply = undefined;
    const reply_size = while (true) {
        const result = linux.recvfrom(socket, std.mem.asBytes(&wire_reply).ptr, @sizeOf(WireReply), linux.MSG.TRUNC, null, null);
        if (linux.errno(result) == .INTR) continue;
        if (linux.errno(result) != .SUCCESS or result == 0) return error.ControlReceiveFailed;
        break result;
    };
    if (reply_size != @sizeOf(WireReply)) return error.InvalidControlReply;
    if (!std.mem.eql(u8, &wire_reply.magic, "VCDR")) return error.InvalidControlReply;
    if (wire_reply.version != 1) return error.IncompatibleControlProtocol;
    if (!std.mem.allEqual(u8, &wire_reply.reserved, 0)) return error.InvalidControlReply;
    if (wire_reply.command != wire.command) return error.InvalidControlReply;

    const recording_elapsed_seconds = optionalSeconds(wire_reply.recording_elapsed_seconds);
    const model_idle_seconds_remaining = optionalSeconds(wire_reply.model_idle_seconds_remaining);
    switch (wire_reply.result) {
        .accepted, .ignored => {
            if (wire_reply.result == .ignored and request.cmd != .record) return error.InvalidControlReply;
            switch (wire_reply.phase) {
                .idle, .capturing, .stopping, .transcribing, .delivering => {},
                else => return error.InvalidControlReply,
            }
            switch (wire_reply.model) {
                .unloaded, .loading, .loaded, .unloading => {},
                else => return error.InvalidControlReply,
            }
            switch (wire_reply.model_kind) {
                .whisper_base_en, .whisper_small_en, .whisper_medium_en => {},
                else => return error.InvalidControlReply,
            }
            if (wire_reply.model_idle_seconds_max > std.math.maxInt(u32)) return error.InvalidControlReply;
            if (recording_elapsed_seconds != null and wire_reply.phase != .capturing) return error.InvalidControlReply;
            if (model_idle_seconds_remaining) |remaining| {
                if (wire_reply.phase != .idle or wire_reply.model != .loaded or remaining > wire_reply.model_idle_seconds_max)
                    return error.InvalidControlReply;
            }
            if (recording_elapsed_seconds) |elapsed| {
                if (elapsed > wire_reply.daemon_uptime_seconds) return error.InvalidControlReply;
            }
        },
        .invalid_request, .unsupported_version => {
            if (wire_reply.recording_id != 0 or wire_reply.model_idle_seconds_max != 0 or wire_reply.daemon_uptime_seconds != 0 or
                recording_elapsed_seconds != null or model_idle_seconds_remaining != null or
                wire_reply.phase != .unavailable or wire_reply.model != .unavailable or
                wire_reply.model_kind != .unavailable)
                return error.InvalidControlReply;
            return if (wire_reply.result == .unsupported_version) error.IncompatibleControlProtocol else error.CommandRejected;
        },
        else => return error.InvalidControlReply,
    }

    const status: Status = .{
        .ignored = wire_reply.result == .ignored,
        .phase = wire_reply.phase,
        .model = wire_reply.model,
        .model_kind = wire_reply.model_kind,
        .recording_id = wire_reply.recording_id,
        .model_idle_seconds_max = wire_reply.model_idle_seconds_max,
        .daemon_uptime_seconds = wire_reply.daemon_uptime_seconds,
        .recording_elapsed_seconds = recording_elapsed_seconds,
        .model_idle_seconds_remaining = model_idle_seconds_remaining,
    };
    if (request.cmd == .status)
        try writeStatus(init.io, status)
    else
        try writeAcknowledgement(init.io, request, status);
}

fn optionalSeconds(value: u64) ?u64 {
    return if (value == seconds_unavailable) null else value;
}

fn writeStatus(io: std.Io, status: Status) !void {
    var buffer: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try writer.print("phase={s}\nrecording_id={d}\n", .{ @tagName(status.phase), status.recording_id });
    try writeOptionalSeconds(&writer, "recording_elapsed_seconds", status.recording_elapsed_seconds);
    try writer.print("model={s}\nmodel_state={s}\n", .{ status.model_kind.name(), @tagName(status.model) });
    try writeOptionalSeconds(&writer, "model_idle_seconds_remaining", status.model_idle_seconds_remaining);
    try writer.print("model_idle_seconds_max={d}\ndaemon_uptime_seconds={d}\n", .{ status.model_idle_seconds_max, status.daemon_uptime_seconds });
    try std.Io.File.stdout().writeStreamingAll(io, writer.buffered());
}

fn writeOptionalSeconds(writer: *std.Io.Writer, name: []const u8, value: ?u64) error{WriteFailed}!void {
    if (value) |seconds|
        try writer.print("{s}={d}\n", .{ name, seconds })
    else
        try writer.print("{s}=unavailable\n", .{name});
}

fn writeAcknowledgement(io: std.Io, request: Request, status: Status) !void {
    var buffer: [128]u8 = undefined;
    const text = if (status.ignored)
        try std.fmt.bufPrint(&buffer, "voiced: recording command ignored; daemon phase is {s}.\n", .{@tagName(status.phase)})
    else switch (request.cmd) {
        .record => if (request.toggle)
            try std.fmt.bufPrint(&buffer, "voiced: recording toggle accepted; daemon phase is {s}.\n", .{@tagName(status.phase)})
        else
            "voiced: recording requested.\n",
        .stop => "voiced: recording stop requested.\n",
        .cancel => "voiced: recording cancellation requested.\n",
        .kill => "voiced: daemon shutdown requested.\n",
        .status => unreachable,
    };
    try std.Io.File.stderr().writeStreamingAll(io, text);
}

// World-writable XDG_RUNTIME_DIR lets another uid create `voiced/` first.
// Leaf 0700 cannot see that parent. Require the same owner and no group/other
// bits on the runtime root before creating the service directory.
fn requireOwnedPrivateDirectory(handle: std.posix.fd_t, path: []const u8) !void {
    var stat: linux.Statx = undefined;
    const stat_errno = linux.errno(linux.statx(handle, "", linux.AT.EMPTY_PATH, .BASIC_STATS, &stat));
    if (stat_errno != .SUCCESS) {
        log.event(.err, .{}, "runtime_directory_inspection_failed", &.{
            .{ "path", .{ .str = path } },
            .{ "operation", .{ .name = "statx" } },
            .{ "system_error", .{ .errno = stat_errno } },
        });
        return error.ControlDirectoryStatFailed;
    }
    if (!stat.mask.TYPE or stat.mode & linux.S.IFMT != linux.S.IFDIR) {
        log.event(.err, .{}, "runtime_directory_invalid", &.{
            .{ "reason", .{ .name = "not_directory" } },
            .{ "path", .{ .str = path } },
        });
        return error.UnsafeRuntimeDirectory;
    }
    if (!stat.mask.UID or !stat.mask.MODE or stat.uid != linux.geteuid() or stat.mode & 0o077 != 0) {
        log.event(.err, .{}, "runtime_directory_invalid", &.{
            .{ "reason", .{ .name = "not_private" } },
            .{ "path", .{ .str = path } },
            .{ "uid", .{ .u = stat.uid } },
            .{ "uid_expected", .{ .u = linux.geteuid() } },
            .{ "mode", .{ .octal = stat.mode & 0o777 } },
            .{ "uid_available", .{ .b = stat.mask.UID } },
            .{ "mode_available", .{ .b = stat.mask.MODE } },
        });
        return error.UnsafeRuntimeDirectory;
    }
}

fn socketPath(init: std.process.Init) ![]u8 {
    const root = init.environ_map.get("XDG_RUNTIME_DIR") orelse {
        return error.RuntimeDirectoryNotSet;
    };
    if (!std.fs.path.isAbsolute(root)) {
        return error.RuntimeDirectoryNotAbsolute;
    }
    return std.fs.path.join(init.gpa, &.{ root, "voiced/control.sock" });
}

fn unixAddress(path: []const u8) !linux.sockaddr.un {
    var address: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) };
    if (path.len >= address.path.len) {
        return error.SocketPathTooLong;
    }
    @memcpy(address.path[0..path.len], path);
    return address;
}

fn createSocket(nonblocking: bool) !std.posix.fd_t {
    const result = linux.socket(linux.AF.UNIX, linux.SOCK.SEQPACKET | linux.SOCK.CLOEXEC | (if (nonblocking) @as(u32, linux.SOCK.NONBLOCK) else 0), 0);
    if (linux.errno(result) != .SUCCESS) {
        log.event(.err, .{}, "control_socket_failed", &.{
            .{ "operation", .{ .name = "socket" } },
            .{ "system_error", .{ .errno = linux.errno(result) } },
        });
        return error.ControlSocketFailed;
    }
    return @intCast(result);
}

fn register(epoll_fd: std.posix.fd_t, descriptor: std.posix.fd_t, tag: u64, events: u32) !void {
    var event: linux.epoll_event = .{ .events = events | linux.EPOLL.RDHUP, .data = .{ .u64 = tag } };
    const result = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, descriptor, &event);
    if (linux.errno(result) != .SUCCESS) {
        log.event(.err, .{}, "control_registration_failed", &.{
            .{ "operation", .{ .name = "epoll_ctl_add" } },
            .{ "descriptor", .{ .i = descriptor } },
            .{ "system_error", .{ .errno = linux.errno(result) } },
        });
        return error.ControlRegisterFailed;
    }
}

fn close(descriptor: std.posix.fd_t) void {
    const errno = linux.errno(linux.close(descriptor));
    if (errno != .SUCCESS) log.event(.err, .{}, "control_cleanup_failed", &.{
        .{ "operation", .{ .name = "close" } },
        .{ "descriptor", .{ .i = descriptor } },
        .{ "system_error", .{ .errno = errno } },
    });
}
