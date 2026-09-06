//! Owns bounded newline-JSON connections, not command or lifecycle policy.
//! The supervisor supplies epoll tags and absolute time, dispatches requests,
//! and includes client deadlines in its existing timerfd schedule.

const std = @import("std");
const linux = std.os.linux;
const log = @import("logging.zig").scoped(.control);
const assert = std.debug.assert;
pub const clients_count_max = 16;
const request_bytes_max = 4096;
const response_bytes_max = 1024;
const request_timeout_ns = std.time.ns_per_s;

pub const Request = struct {
    cmd: enum { listen, record, stop, cancel, status, kill },
    toggle: bool = false,
};

/// `Server` holds the runtime-directory lock until deinit. Accept new clients
/// only after consuming the current epoll batch, so a reused client slot cannot
/// receive an event belonging to a closed connection. Each connection permits
/// one request and one response, with bounded bytes and a one-second deadline.
pub const Server = struct {
    directory: std.Io.Dir,
    listener: std.posix.fd_t,
    epoll_fd: std.posix.fd_t,
    client_event_tag: u64,
    clients: [clients_count_max]Client = @splat(.{}),

    pub fn open(init: std.process.Init, epoll_fd: std.posix.fd_t, listener_tag: u64, client_tag: u64) !Server {
        const path = try socketPath(init);
        defer init.gpa.free(path);
        const runtime_root = init.environ_map.get("XDG_RUNTIME_DIR").?;
        const runtime_dir = std.Io.Dir.cwd().openDir(init.io, runtime_root, .{}) catch |err| {
            log.err(.{}, "Runtime directory unavailable: path=\"{f}\", error={s}", .{ std.zig.fmtString(runtime_root), @errorName(err) });
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
                // Do not unlink a live socket, including one owned by the Python
                // daemon, which does not participate in this directory lock.
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
                    log.err(.{}, "Control bind failed: operation=bind, errno={t}", .{linux.errno(retry_result)});
                    return error.ControlBindFailed;
                }
            },
            else => {
                log.err(.{}, "Control bind failed: operation=bind, errno={t}", .{linux.errno(bind_result)});
                return error.ControlBindFailed;
            },
        }
        errdefer directory.deleteFile(init.io, "control.sock") catch |err| {
            if (err != error.FileNotFound) log.err(.{}, "Control socket cleanup failed: operation=unlink, error={s}", .{@errorName(err)});
        };
        try directory.setFilePermissions(init.io, "control.sock", .fromMode(0o600), .{});
        const listen_result = linux.listen(listener, clients_count_max);
        if (linux.errno(listen_result) != .SUCCESS) {
            log.err(.{}, "Control listen failed: operation=listen, errno={t}", .{linux.errno(listen_result)});
            return error.ControlListenFailed;
        }
        try register(epoll_fd, listener, listener_tag, linux.EPOLL.IN);
        return .{ .directory = directory, .listener = listener, .epoll_fd = epoll_fd, .client_event_tag = client_tag };
    }

    pub fn deinit(server: *Server, io: std.Io) void {
        for (0..server.clients.len) |index| server.closeClient(index);
        close(server.listener);
        server.directory.deleteFile(io, "control.sock") catch |err| {
            if (err != error.FileNotFound) log.err(.{}, "Control socket cleanup failed: operation=unlink, error={s}", .{@errorName(err)});
        };
        server.directory.close(io);
    }

    pub fn acceptClients(server: *Server, now_ns: u64) !void {
        for (0..clients_count_max) |_| {
            const result = linux.accept4(server.listener, null, null, linux.SOCK.CLOEXEC | linux.SOCK.NONBLOCK);
            switch (linux.errno(result)) {
                .SUCCESS => {},
                .INTR => continue,
                .AGAIN => {
                    return;
                },
                else => {
                    log.err(.{}, "Control accept failed: operation=accept4, errno={t}", .{linux.errno(result)});
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

    pub fn receive(server: *Server, index: usize) ?Request {
        const client = &server.clients[index];
        const descriptor = client.descriptor orelse {
            return null;
        };
        if (client.responding) {
            server.flush(index);
            return null;
        }
        while (client.request_size < client.request.len) {
            const remaining = client.request[client.request_size..];
            const result = linux.recvfrom(descriptor, remaining.ptr, remaining.len, linux.MSG.DONTWAIT, null, null);
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
            client.request_size += result;
            const bytes = client.request[0..client.request_size];
            if (std.mem.indexOfScalar(u8, bytes, '\n')) |newline| {
                var memory: [8192]u8 = undefined;
                var allocator = std.heap.FixedBufferAllocator.init(&memory);
                const parsed = std.json.parseFromSlice(Request, allocator.allocator(), bytes[0..newline], .{}) catch {
                    server.reject(index);
                    return null;
                };
                defer parsed.deinit();
                if (newline + 1 != bytes.len or (parsed.value.toggle and parsed.value.cmd != .record)) {
                    server.reject(index);
                    return null;
                }
                return parsed.value;
            }
        }
        server.reject(index);
        return null;
    }

    pub fn respond(server: *Server, index: usize, value: anytype) void {
        const client = &server.clients[index];
        if (client.descriptor == null) {
            return;
        }
        var writer: std.Io.Writer = .fixed(&client.response);
        std.json.Stringify.value(value, .{}, &writer) catch unreachable;
        writer.writeByte('\n') catch unreachable;
        client.response_size = writer.end;
        client.responding = true;
        server.flush(index);
    }

    fn reject(server: *Server, index: usize) void {
        server.respond(index, .{ .ok = false, .err = "invalid_request" });
    }

    fn flush(server: *Server, index: usize) void {
        const client = &server.clients[index];
        const descriptor = client.descriptor orelse {
            return;
        };
        while (client.response_sent < client.response_size) {
            const bytes = client.response[client.response_sent..client.response_size];
            const result = linux.sendto(descriptor, bytes.ptr, bytes.len, linux.MSG.NOSIGNAL | linux.MSG.DONTWAIT, null, 0);
            switch (linux.errno(result)) {
                .SUCCESS => client.response_sent += result,
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
    request: [request_bytes_max]u8 = undefined,
    request_size: usize = 0,
    response: [response_bytes_max]u8 = undefined,
    response_size: usize = 0,
    response_sent: usize = 0,
    responding: bool = false,
};

/// `sendRequest` sends one command to the selected instance, prints its JSON
/// response, and fails on timeout, malformed responses, or command rejection.
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
    if (linux.errno(linux.connect(socket, @ptrCast(&address), @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1))) != .SUCCESS) {
        return error.DaemonNotRunning;
    }
    var buffer: [response_bytes_max]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try std.json.Stringify.value(request, .{}, &writer);
    try writer.writeByte('\n');
    var sent: usize = 0;
    while (sent < writer.end) {
        const bytes = buffer[sent..writer.end];
        const result = linux.sendto(socket, bytes.ptr, bytes.len, linux.MSG.NOSIGNAL, null, 0);
        if (linux.errno(result) == .INTR) continue;
        if (linux.errno(result) != .SUCCESS or result == 0) {
            return error.ControlSendFailed;
        }
        sent += result;
    }
    var received: usize = 0;
    while (received < buffer.len) {
        const remaining = buffer[received..];
        const result = linux.recvfrom(socket, remaining.ptr, remaining.len, 0, null, null);
        if (linux.errno(result) == .INTR) continue;
        if (linux.errno(result) != .SUCCESS or result == 0) {
            return error.ControlReceiveFailed;
        }
        received += result;
        if (std.mem.indexOfScalar(u8, buffer[0..received], '\n')) |newline| {
            const parsed = try std.json.parseFromSlice(struct { ok: bool }, init.gpa, buffer[0..newline], .{ .ignore_unknown_fields = true });
            defer parsed.deinit();
            try std.Io.File.stdout().writeStreamingAll(init.io, buffer[0 .. newline + 1]);
            if (!parsed.value.ok) {
                return error.CommandRejected;
            }
            return;
        }
    }
    return error.ControlResponseTooLarge;
}

// World-writable XDG_RUNTIME_DIR lets another uid create `voiced/` first.
// Leaf 0700 cannot see that parent. Require the same owner and no group/other
// bits on the runtime root before creating the instance directory.
fn requireOwnedPrivateDirectory(handle: std.posix.fd_t, path: []const u8) !void {
    var stat: linux.Statx = undefined;
    const stat_errno = linux.errno(linux.statx(handle, "", linux.AT.EMPTY_PATH, .BASIC_STATS, &stat));
    if (stat_errno != .SUCCESS) {
        log.err(.{}, "Runtime directory inspection failed: path=\"{f}\", operation=statx, errno={t}", .{ std.zig.fmtString(path), stat_errno });
        return error.ControlDirectoryStatFailed;
    }
    if (!stat.mask.TYPE or stat.mode & linux.S.IFMT != linux.S.IFDIR) {
        log.err(.{}, "Runtime directory is not a directory: path=\"{f}\"", .{std.zig.fmtString(path)});
        return error.UnsafeRuntimeDirectory;
    }
    if (!stat.mask.UID or !stat.mask.MODE or stat.uid != linux.geteuid() or stat.mode & 0o077 != 0) {
        log.err(.{}, "Runtime directory is not private: path=\"{f}\", uid={d}, expected_uid={d}, mode={o}, uid_available={}, mode_available={}", .{ std.zig.fmtString(path), stat.uid, linux.geteuid(), stat.mode & 0o777, stat.mask.UID, stat.mask.MODE });
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
    const instance = init.environ_map.get("VOICED_INSTANCE") orelse "";
    if (instance.len > 40) {
        return error.InvalidInstance;
    }
    for (instance) |byte| {
        if (!std.ascii.isAlphanumeric(byte) and byte != '-' and byte != '_') {
            return error.InvalidInstance;
        }
    }
    const directory = if (instance.len == 0) try init.gpa.dupe(u8, "voiced") else try std.fmt.allocPrint(init.gpa, "voiced-{s}", .{instance});
    defer init.gpa.free(directory);
    return std.fs.path.join(init.gpa, &.{ root, directory, "control.sock" });
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
    const result = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.CLOEXEC | (if (nonblocking) @as(u32, linux.SOCK.NONBLOCK) else 0), 0);
    if (linux.errno(result) != .SUCCESS) {
        log.err(.{}, "Control socket failed: operation=socket, errno={t}", .{linux.errno(result)});
        return error.ControlSocketFailed;
    }
    return @intCast(result);
}

fn register(epoll_fd: std.posix.fd_t, descriptor: std.posix.fd_t, tag: u64, events: u32) !void {
    var event: linux.epoll_event = .{ .events = events | linux.EPOLL.RDHUP, .data = .{ .u64 = tag } };
    const result = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, descriptor, &event);
    if (linux.errno(result) != .SUCCESS) {
        log.err(.{}, "Control registration failed: operation=epoll_ctl_add, fd={d}, errno={t}", .{ descriptor, linux.errno(result) });
        return error.ControlRegisterFailed;
    }
}

fn close(descriptor: std.posix.fd_t) void {
    const errno = linux.errno(linux.close(descriptor));
    if (errno != .SUCCESS) log.err(.{}, "Control cleanup failed: operation=close, fd={d}, errno={t}", .{ descriptor, errno });
}
