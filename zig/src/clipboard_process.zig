//! Own stock wl-copy's launcher, background selection owner, and paste readers.
//! A separate session/subreaper preserves wl-copy's parent-exit readiness signal
//! and contains blocked readers. The service must also be a child subreaper so
//! it can clean up this session if its guardian dies unexpectedly.
const std = @import("std");
const logging = @import("logging.zig");
const log = logging.scoped(.clipboard);
const linux = std.os.linux;
const fd_t = linux.fd_t;
const ms = std.time.ns_per_ms;

pub const Error = union(enum) {
    launch: SystemError,
    system: SystemError,
    worker: WorkerError,
    input_write: struct { errno: linux.E, bytes_sent: usize, bytes_total: usize },
    timed_out: struct { stage: enum { acquisition, guardian_exit, cleanup }, deadline_ns: u64, observed_ns: u64, bytes_sent: usize = 0, bytes_total: usize = 0 },
    invalid_report: struct { size: usize, byte: u8 },
};
pub fn Result(comptime T: type) type {
    return union(enum) { ok: T, err: Error };
}

pub const SystemError = struct {
    cause: anyerror = error.ClipboardSyscallFailed,
    native: ?struct { operation: []const u8, errno: linux.E } = null,

    fn check(self: *SystemError, result: usize, operation: []const u8) !void {
        const errno = linux.errno(result);
        if (errno != .SUCCESS) {
            self.native = .{ .operation = operation, .errno = errno };
            return error.ClipboardSyscallFailed;
        }
    }
};

// Value-only diagnostic packet, independent of the guardian's stack lifetime.
// stderr is bounded and drained even after the buffer fills; omitted bytes are
// explicit. This is tool diagnostic output, never the transcript on stdin.
pub const WorkerError = extern struct {
    kind: enum(u8) { tool_not_found, launcher_exited, owner_unavailable, system, cleanup },
    errno: u16,
    wait_status_present: u8,
    wait_status: u32,
    name_size: u8,
    operation_size: u8,
    stderr_size: u16,
    stderr_omitted: u64,
    name: [64]u8,
    operation: [64]u8,
    stderr: [2048]u8,
};
const Stderr = struct {
    bytes: [2048]u8 = @splat(0),
    size: u16 = 0,
    omitted: u64 = 0,
    open: bool = true,

    fn drain(self: *Stderr, fd: fd_t, context: *SystemError) !void {
        var buffer: [4096]u8 = undefined;
        // Bound work per poll iteration even when a faulty tool writes forever.
        for (0..16) |_| {
            const count = linux.read(fd, &buffer, buffer.len);
            switch (linux.errno(count)) {
                .SUCCESS => {},
                .AGAIN => return,
                .INTR => continue,
                else => {
                    try context.check(count, "stderr read");
                    unreachable;
                },
            }
            if (count == 0) {
                self.open = false;
                return;
            }
            const retained = @min(count, self.bytes.len - self.size);
            @memcpy(self.bytes[self.size..][0..retained], buffer[0..retained]);
            self.size += @intCast(retained);
            self.omitted +|= count - retained;
        }
    }
};

pub const Process = struct {
    pid: linux.pid_t,
    pid_fd: fd_t,
    socket: ?fd_t,
    input: ?fd_t,
    operation: union(enum) {
        copying: struct { sent: usize = 0, deadline_ns: u64 },
        owning,
        stopping: u64,
        terminating: u64,
        reaping: struct { next_ns: u64, deadline_ns: u64 },
    },

    pub fn start(io: std.Io, epoll_fd: fd_t, tag: u64, now_ns: u64) Result(Process) {
        var context: SystemError = .{};
        const process = startInternal(&context, io, epoll_fd, tag, now_ns) catch |err| {
            context.cause = err;
            return .{ .err = .{ .launch = context } };
        };
        return .{ .ok = process };
    }

    pub fn advance(self: *Process, epoll_fd: fd_t, text: []const u8, now_ns: u64) Result(Event) {
        var context: SystemError = .{};
        var reported: ?Error = null;
        const event = advanceInternal(&context, &reported, self, epoll_fd, text, now_ns) catch |err| {
            context.cause = err;
            return .{ .err = reported orelse .{ .system = context } };
        };
        return .{ .ok = event };
    }

    /// All three descriptors use one epoll tag. Call advance after consuming the
    /// complete event batch, and retain the transcript until acquisition/stop.
    fn startInternal(context: *SystemError, io: std.Io, epoll_fd: fd_t, tag: u64, now_ns: u64) !Process {
        var sockets: [2]fd_t = undefined;
        try context.check(linux.socketpair(linux.AF.UNIX, linux.SOCK.SEQPACKET, 0, &sockets), "socketpair");
        defer _ = linux.close(sockets[1]);
        errdefer _ = linux.close(sockets[0]);
        try context.check(linux.fcntl(sockets[0], linux.F.SETFD, linux.FD_CLOEXEC), "fcntl");
        var socket_buffer: [32]u8 = undefined;
        var parent_buffer: [32]u8 = undefined;
        var child = try std.process.spawn(io, .{
            .argv = &.{ "/proc/self/exe", "--internal-role", "clipboard", try std.fmt.bufPrint(&socket_buffer, "{d}", .{sockets[1]}), try std.fmt.bufPrint(&parent_buffer, "{d}", .{linux.getpid()}), logging.levelName(logging.level()) },
            .stdin = .pipe,
            .stdout = .ignore,
        });
        // The worker cannot fork wl-copy before our explicit start packet.
        // Thus rollback here never strands a background clipboard process.
        errdefer child.kill(io);
        const input = child.stdin.?.handle;
        child.stdin = null;
        errdefer _ = linux.close(input);
        try context.check(linux.fcntl(input, linux.F.SETFL, @as(u32, @bitCast(linux.O{ .NONBLOCK = true }))), "fcntl");
        const pid_fd = try descriptor(context, linux.pidfd_open(child.id.?, 0), "pidfd_open");
        errdefer _ = linux.close(pid_fd);
        try register(context, epoll_fd, sockets[0], linux.EPOLL.IN, tag);
        errdefer unregister(epoll_fd, sockets[0]);
        try register(context, epoll_fd, input, linux.EPOLL.OUT, tag);
        errdefer unregister(epoll_fd, input);
        try register(context, epoll_fd, pid_fd, linux.EPOLL.IN, tag);
        errdefer unregister(epoll_fd, pid_fd);
        try sendByte(context, sockets[0], @intFromEnum(Command.start));
        return .{
            .pid = child.id.?,
            .pid_fd = pid_fd,
            .socket = sockets[0],
            .input = input,
            .operation = .{ .copying = .{ .deadline_ns = now_ns + 2 * std.time.ns_per_s } },
        };
    }

    pub const Event = enum { none, acquired, finished };

    /// Normal selection loss retires the process; it never reacquires selection.
    /// Errors before retirement should be logged and followed by stop(). Errors
    /// during reaping are terminal cleanup failures. finished consumes this handle.
    fn advanceInternal(context: *SystemError, reported: *?Error, self: *Process, epoll_fd: fd_t, text: []const u8, now_ns: u64) !Event {
        if (self.operation == .reaping) {
            const progress = &self.operation.reaping;
            if (now_ns < progress.next_ns) return .none;
            if (try reapGroup(context, self.pid)) {
                var status: u32 = undefined;
                try context.check(linux.waitpid(self.pid, &status, 0), "waitpid");
                _ = linux.close(self.pid_fd);
                return .finished;
            }
            if (now_ns >= progress.deadline_ns) {
                reported.* = .{ .timed_out = .{ .stage = .cleanup, .deadline_ns = progress.deadline_ns, .observed_ns = now_ns } };
                return error.ClipboardCleanupTimedOut;
            }
            progress.next_ns = now_ns + 5 * ms;
            return .none;
        }

        var acquired = false;
        if (self.socket) |socket| {
            while (true) {
                const packet = receiveReport(context, socket) catch |err| switch (err) {
                    error.WouldBlock => break,
                    error.PeerClosed => {
                        if (self.operation == .copying or self.operation == .owning)
                            log.err(.{}, "Clipboard owner disconnected: pid={d}, stage={t}, reason=peer_closed_without_stop_report", .{ self.pid, std.meta.activeTag(self.operation) });
                        self.stop(epoll_fd, now_ns);
                        break;
                    },
                    else => return err,
                };
                const report = switch (packet) {
                    .event => |event| event,
                    .err => |err| {
                        reported.* = .{ .worker = err };
                        return error.ClipboardWorkerError;
                    },
                    .invalid => |invalid| {
                        reported.* = .{ .invalid_report = invalid };
                        return error.InvalidClipboardReport;
                    },
                };
                switch (report) {
                    .acquired => {
                        if (self.operation == .stopping or self.operation == .terminating) continue;
                        if (self.operation != .copying or self.input != null) return error.InvalidClipboardReport;
                        self.operation = .owning;
                        acquired = true;
                    },
                    .failed => return error.ClipboardAcquisitionFailed,
                    .selection_lost, .stopped => {
                        self.stop(epoll_fd, now_ns);
                        break;
                    },
                }
            }
        }

        // WNOWAIT keeps the guardian's numeric PID reserved. A pidfd alone does
        // not reserve it after reaping: kill(-pid) could then hit another group.
        var information = std.mem.zeroes(linux.siginfo_t);
        try context.check(linux.waitid(.PIDFD, self.pid_fd, &information, linux.W.EXITED | linux.W.NOHANG | linux.W.NOWAIT, null), "waitid");
        if (information.fields.common.first.piduid.pid != 0) {
            // Only the terminating phase has sent SIGKILL. A peer EOF can
            // initiate cooperative cleanup but cannot explain an external kill.
            const exited = information.code == @intFromEnum(linux.CLD.EXITED);
            log.processExited(.{}, "clipboard", information, self.operation == .terminating or (self.operation == .stopping and exited));
            self.closeEndpoint(epoll_fd, &self.input);
            self.closeEndpoint(epoll_fd, &self.socket);
            unregister(epoll_fd, self.pid_fd);
            // Before setsid succeeds the worker has no children and may share
            // the service's session. Never signal that inherited group.
            self.killSession();
            self.operation = .{ .reaping = .{ .next_ns = now_ns, .deadline_ns = now_ns + 2 * std.time.ns_per_s } };
            return .none;
        }

        switch (self.operation) {
            .copying => |*copying| {
                if (now_ns >= copying.deadline_ns) {
                    reported.* = .{ .timed_out = .{ .stage = .acquisition, .deadline_ns = copying.deadline_ns, .observed_ns = now_ns, .bytes_sent = copying.sent, .bytes_total = text.len } };
                    return error.ClipboardAcquisitionTimedOut;
                }
                if (self.input) |input| {
                    // One bounded nonblocking write per event batch preserves
                    // command responsiveness even for a multi-megabyte result.
                    const bytes = text[copying.sent..@min(text.len, copying.sent + 65536)];
                    const written = linux.write(input, bytes.ptr, bytes.len);
                    switch (linux.errno(written)) {
                        .SUCCESS => copying.sent += written,
                        .AGAIN, .INTR => return .none,
                        else => |errno| {
                            reported.* = .{ .input_write = .{ .errno = errno, .bytes_sent = copying.sent, .bytes_total = text.len } };
                            return error.ClipboardInputFailed;
                        },
                    }
                    if (copying.sent == text.len) self.closeEndpoint(epoll_fd, &self.input);
                }
            },
            .stopping => |deadline_ns| if (now_ns >= deadline_ns) {
                self.killSession();
                // Keep the exit deadline finite even if a task cannot respond
                // to SIGKILL. Reaping/group cleanup starts only after its exit.
                self.operation = .{ .terminating = now_ns + std.time.ns_per_s };
            },
            .terminating => |deadline_ns| if (now_ns >= deadline_ns) {
                {
                    reported.* = .{ .timed_out = .{ .stage = .guardian_exit, .deadline_ns = deadline_ns, .observed_ns = now_ns } };
                    return error.ClipboardGuardianExitTimedOut;
                }
            },
            .owning => {},
            .reaping => unreachable,
        }
        return if (acquired and self.operation == .owning) .acquired else .none;
    }

    pub fn stop(self: *Process, epoll_fd: fd_t, now_ns: u64) void {
        var diagnostic: SystemError = .{};
        const context = &diagnostic;
        if (self.operation == .stopping or self.operation == .terminating or self.operation == .reaping) return;
        if (self.socket) |socket| sendByte(context, socket, @intFromEnum(Command.stop)) catch {};
        self.closeEndpoint(epoll_fd, &self.input);
        self.closeEndpoint(epoll_fd, &self.socket);
        self.operation = .{ .stopping = now_ns + 750 * ms };
    }

    pub fn deadlineMonotonicNs(self: *const Process) ?u64 {
        return switch (self.operation) {
            .copying => |value| value.deadline_ns,
            .owning => null,
            .stopping, .terminating => |value| value,
            .reaping => |value| value.next_ns,
        };
    }

    /// Bounded fallback for supervisor stack unwinding, outside its event loop.
    pub fn deinit(self: *Process, epoll_fd: fd_t) void {
        self.stop(epoll_fd, now());
        const deadline_ns = now() + 3 * std.time.ns_per_s;
        var descriptors: [0]linux.pollfd = .{};
        while (now() < deadline_ns) {
            const event = switch (self.advance(epoll_fd, "", now())) {
                .ok => |event| event,
                .err => |err| {
                    log.err(.{}, "Clipboard cleanup error: {any}", .{err});
                    self.killSession();
                    _ = linux.poll(&descriptors, 0, 5);
                    continue;
                },
            };
            if (event == .finished) return;
            _ = linux.poll(&descriptors, 0, 5);
        }
        log.err(.{}, "Clipboard cleanup exceeded its exit deadline", .{});
        _ = linux.close(self.pid_fd);
    }

    fn closeEndpoint(_: *Process, epoll_fd: fd_t, endpoint: *?fd_t) void {
        const fd = endpoint.* orelse return;
        unregister(epoll_fd, fd);
        _ = linux.close(fd);
        endpoint.* = null;
    }

    fn killSession(self: *const Process) void {
        // The still-unreaped guardian anchors this ID even before its exit.
        // Kill descendants too if the guardian itself cannot leave kernel work.
        if (linux.getsid(self.pid) == @as(usize, @intCast(self.pid))) _ = linux.kill(-self.pid, .KILL);
        _ = linux.pidfd_send_signal(self.pid_fd, .KILL, null, 0);
    }
};

pub fn runWorker(init: std.process.Init, socket: fd_t, expected_parent: linux.pid_t) !void {
    var context: SystemError = .{};
    var stderr: Stderr = .{};
    var status: ?u32 = null;
    const completed = runWorkerInternal(&context, &stderr, &status, init, socket, expected_parent) catch |err| {
        reportError(socket, .system, err, context, status, &stderr);
        return err;
    };
    if (!completed) return error.ClipboardWorkerReportedError;
}

fn runWorkerInternal(context: *SystemError, stderr: *Stderr, status: *?u32, init: std.process.Init, socket: fd_t, expected_parent: linux.pid_t) !bool {
    // Undo inherited SIGCHLD=SIG_IGN before any fork; auto-reaping would destroy
    // the launcher-exit evidence. TERM remains blocked and wakes signalfd.
    for ([_]linux.SIG{ .CHLD, .TERM, .INT, .PIPE }) |signal|
        try context.check(linux.sigaction(signal, &.{ .handler = .{ .handler = linux.SIG.DFL }, .mask = linux.sigemptyset(), .flags = 0 }, null), "sigaction");
    var mask = std.posix.sigemptyset();
    for ([_]linux.SIG{ .CHLD, .TERM, .INT }) |signal| std.posix.sigaddset(&mask, signal);
    std.posix.sigprocmask(std.posix.SIG.BLOCK, &mask, null);
    const signal_fd = try std.posix.signalfd(-1, &mask, linux.SFD.CLOEXEC | linux.SFD.NONBLOCK);
    defer _ = linux.close(signal_fd);
    try context.check(linux.prctl(@intFromEnum(linux.PR.SET_CHILD_SUBREAPER), 1, 0, 0, 0), "prctl");
    try context.check(linux.prctl(@intFromEnum(linux.PR.SET_PDEATHSIG), @intFromEnum(linux.SIG.TERM), 0, 0, 0), "prctl");
    if (linux.getppid() != expected_parent) return error.ClipboardParentGone;
    // Own an entire session, not merely a shell pipeline's process group.
    if (linux.getsid(0) != @as(usize, @intCast(linux.getpid()))) try context.check(linux.setsid(), "setsid");
    try context.check(linux.fcntl(socket, linux.F.SETFD, linux.FD_CLOEXEC), "fcntl");

    if (!try waitForStart(context, socket, signal_fd)) return true;
    const executable = try findWlCopy(context);
    const identity = try statFile(context, executable);
    const devnull = try descriptor(context, linux.open("/dev/null", .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0), "open /dev/null");
    defer _ = linux.close(devnull);
    var errors: [2]fd_t = undefined;
    try context.check(linux.pipe2(&errors, .{ .CLOEXEC = true }), "stderr pipe2");
    defer _ = linux.close(errors[0]);
    var write_open = true;
    defer {
        if (write_open) _ = linux.close(errors[1]);
    }
    try context.check(linux.fcntl(errors[0], linux.F.SETFL, @as(u32, @bitCast(linux.O{ .NONBLOCK = true }))), "stderr fcntl");
    const argv = [_:null]?[*:0]const u8{ executable, "--type", "text/plain;charset=utf-8", "--" };
    if (try drainSignals(context, signal_fd) or linux.getppid() != expected_parent) return true;
    const fork_result = linux.fork();
    try context.check(fork_result, "fork");
    if (fork_result == 0) {
        // Keep the report fd until exec so an exec/setup error retains errno.
        // CLOEXEC still prevents every non-stdio descriptor reaching wl-copy.
        // Only raw syscalls and stack formatting are allowed after fork.
        childCheck(socket, errors[1], "dup2 stdout", linux.dup2(devnull, 1));
        childCheck(socket, errors[1], "dup2 stderr", linux.dup2(errors[1], 2));
        // Linux UAPI CLOSE_RANGE_CLOEXEC is 1 << 2. Zig 0.16's packed
        // CLOSE_RANGE puts it at bit 1 (UNSHARE), which would close the report
        // socket and lose exec errno. Keep the UAPI value until that ABI is fixed.
        childCheck(socket, 2, "close_range", linux.syscall3(.close_range, 3, std.math.maxInt(fd_t), 1 << 2));
        var empty = linux.sigemptyset();
        childCheck(socket, 2, "sigprocmask", linux.sigprocmask(linux.SIG.SETMASK, &empty, null));
        childCheck(socket, 2, "execve", linux.execve(executable, &argv, init.minimal.environ.block.slice.ptr));
        unreachable;
    }
    _ = linux.close(errors[1]);
    write_open = false;
    _ = linux.close(0); // wl-copy exclusively owns reading the inherited pipe.
    // Report before cleanup can change diagnostics; a cleanup error is a
    // second report and journal event, never a replacement for the first cause.
    var completed = true;
    serveOwner(context, socket, signal_fd, @intCast(fork_result), identity, errors[0], stderr, status) catch |err| {
        reportError(socket, .system, err, context.*, status.*, stderr);
        completed = false;
    };
    context.* = .{};
    cleanupChildren(context, signal_fd) catch |err| {
        reportError(socket, .cleanup, err, context.*, null, stderr);
        return false;
    };
    sendByte(context, socket, @intFromEnum(Report.stopped)) catch {};
    return completed;
}

const Command = enum(u8) { start = 1, stop = 2 };
const Report = enum(u8) { acquired = 1, selection_lost = 2, failed = 3, stopped = 4 };

fn serveOwner(context: *SystemError, socket: fd_t, signal_fd: fd_t, launcher: linux.pid_t, identity: linux.Statx, stderr_fd: fd_t, stderr: *Stderr, exit_status: *?u32) !void {
    var owner: ?linux.pid_t = null;
    while (true) {
        if (stderr.open) try stderr.drain(stderr_fd, context);
        var launcher_status: ?u32 = null;
        while (true) {
            var status: u32 = undefined;
            const child = linux.waitpid(-1, &status, linux.W.NOHANG);
            switch (linux.errno(child)) {
                .SUCCESS => if (child == 0) break,
                .CHILD => break,
                .INTR => continue,
                else => {
                    try context.check(child, "waitpid");
                    unreachable;
                },
            }
            if (child == launcher) {
                launcher_status = status;
                exit_status.* = status;
            }
            if (owner != null and child == owner.?) {
                exit_status.* = status;
                if (stderr.open) try stderr.drain(stderr_fd, context);
                if (status != 0) return error.WlCopyOwnerExited;
                sendByte(context, socket, @intFromEnum(Report.selection_lost)) catch {};
                return;
            }
        }
        if (launcher_status) |status| {
            if (stderr.open) try stderr.drain(stderr_fd, context);
            if (status != 0) return error.WlCopyLauncherFailed;
            // wl-copy also exits zero on cancellation before daemonization.
            // Require a surviving, adopted instance of the selected binary.
            var children: [64]linux.pid_t = undefined;
            for (try directChildren(context, &children)) |child| {
                var path: [64]u8 = undefined;
                const actual = statFile(context, try std.fmt.bufPrintZ(&path, "/proc/{d}/exe", .{child})) catch {
                    context.* = .{};
                    continue;
                };
                if (actual.ino != identity.ino or actual.dev_major != identity.dev_major or actual.dev_minor != identity.dev_minor) continue;
                if (owner != null) return error.AmbiguousClipboardOwner;
                owner = child;
            }
            if (owner == null) return error.NoClipboardOwner;
            try sendByte(context, socket, @intFromEnum(Report.acquired));
        }
        var descriptors = [_]linux.pollfd{
            .{ .fd = socket, .events = linux.POLL.IN, .revents = 0 },
            .{ .fd = signal_fd, .events = linux.POLL.IN, .revents = 0 },
            .{ .fd = if (stderr.open) stderr_fd else -1, .events = linux.POLL.IN, .revents = 0 },
        };
        const result = linux.poll(&descriptors, descriptors.len, -1);
        if (linux.errno(result) == .INTR) continue;
        try context.check(result, "poll");
        if (descriptors[0].revents != 0) return; // stop, EOF, or malformed control
        if (try drainSignals(context, signal_fd)) return;
    }
}

fn waitForStart(context: *SystemError, socket: fd_t, signal_fd: fd_t) !bool {
    var descriptors = [_]linux.pollfd{
        .{ .fd = socket, .events = linux.POLL.IN, .revents = 0 },
        .{ .fd = signal_fd, .events = linux.POLL.IN, .revents = 0 },
    };
    while (true) {
        const result = linux.poll(&descriptors, descriptors.len, -1);
        if (linux.errno(result) == .INTR) continue;
        try context.check(result, "poll");
        if (try drainSignals(context, signal_fd)) return false;
        if (descriptors[0].revents != 0) {
            const command = receiveByte(context, socket) catch return false;
            return command == @intFromEnum(Command.start);
        }
    }
}

fn cleanupChildren(context: *SystemError, signal_fd: fd_t) !void {
    _ = linux.kill(-linux.getpid(), .TERM); // our TERM is blocked
    const started_ns = now();
    while (true) {
        var status: u32 = undefined;
        const child = linux.waitpid(-1, &status, linux.W.NOHANG);
        switch (linux.errno(child)) {
            .SUCCESS => if (child > 0) continue,
            .CHILD => return,
            .INTR => continue,
            else => {
                try context.check(child, "waitpid");
                unreachable;
            },
        }
        const elapsed_ns = now() - started_ns;
        if (elapsed_ns >= 2 * std.time.ns_per_s) return error.ClipboardCleanupTimedOut;
        if (elapsed_ns >= 250 * ms) {
            // Keep this group anchor alive; kill/reap direct children, adopting
            // their descendants in turn. Unreaped child PIDs cannot be reused.
            var children: [64]linux.pid_t = undefined;
            for (try directChildren(context, &children)) |pid| _ = linux.kill(pid, .KILL);
        }
        var descriptors = [_]linux.pollfd{.{ .fd = signal_fd, .events = linux.POLL.IN, .revents = 0 }};
        _ = linux.poll(&descriptors, descriptors.len, 5);
        _ = try drainSignals(context, signal_fd);
    }
}

// Only the clipboard session's adopted children are eligible. A global waitpid
// in the supervisor would steal the audio/model workers' pidfd-owned exits.
fn reapGroup(context: *SystemError, guardian: linux.pid_t) !bool {
    var children: [64]linux.pid_t = undefined;
    var found = false;
    for (try directChildren(context, &children)) |pid| {
        if (pid == guardian or linux.getpgid(pid) != @as(usize, @intCast(guardian))) continue;
        found = true;
        _ = linux.kill(pid, .KILL);
        var status: u32 = undefined;
        const result = linux.waitpid(pid, &status, linux.W.NOHANG);
        if (linux.errno(result) != .INTR) try context.check(result, "poll");
    }
    return !found; // rescan after reaps before releasing the guardian PID
}

fn directChildren(context: *SystemError, storage: *[64]linux.pid_t) ![]const linux.pid_t {
    const fd = try descriptor(context, linux.open("/proc/thread-self/children", .{ .CLOEXEC = true }, 0), "open children");
    defer _ = linux.close(fd);
    var bytes: [2048]u8 = undefined;
    const count = linux.read(fd, &bytes, bytes.len);
    try context.check(count, "children read");
    if (count == bytes.len) return error.TooManyClipboardChildren;
    var words = std.mem.tokenizeScalar(u8, bytes[0..count], ' ');
    var used: usize = 0;
    while (words.next()) |word| {
        if (used == storage.len) return error.TooManyClipboardChildren;
        storage[used] = try std.fmt.parseInt(linux.pid_t, word, 10);
        if (storage[used] <= 1) return error.InvalidClipboardChild;
        used += 1;
    }
    return storage[0..used];
}

const wl_copy_path: [:0]const u8 = "/usr/bin/wl-copy";

fn findWlCopy(context: *SystemError) ![:0]const u8 {
    const stat = statFile(context, wl_copy_path) catch {
        context.* = .{};
        return error.WlCopyNotFound;
    };
    if (stat.mode & linux.S.IFMT != linux.S.IFREG) return error.WlCopyNotFound;
    if (linux.errno(linux.access(wl_copy_path, 1)) != .SUCCESS) return error.WlCopyNotFound;
    return wl_copy_path;
}

fn statFile(context: *SystemError, path: [*:0]const u8) !linux.Statx {
    var stat: linux.Statx = undefined;
    try context.check(linux.statx(linux.AT.FDCWD, path, 0, .{ .INO = true, .MODE = true, .TYPE = true }, &stat), "statx");
    if (!stat.mask.INO) return error.ClipboardIdentityUnavailable;
    return stat;
}

fn drainSignals(context: *SystemError, fd: fd_t) !bool {
    var stop = false;
    while (true) {
        var information: linux.signalfd_siginfo = undefined;
        const result = linux.read(fd, std.mem.asBytes(&information).ptr, @sizeOf(@TypeOf(information)));
        switch (linux.errno(result)) {
            .SUCCESS => {},
            .AGAIN => return stop,
            .INTR => continue,
            else => {
                try context.check(result, "signalfd read");
                unreachable;
            },
        }
        if (result != @sizeOf(@TypeOf(information))) return error.ClipboardSignalReadFailed;
        if (information.signo == @intFromEnum(linux.SIG.TERM) or information.signo == @intFromEnum(linux.SIG.INT)) stop = true;
    }
}

fn receiveByte(context: *SystemError, socket: fd_t) !u8 {
    var bytes: [2]u8 = undefined;
    const result = linux.recvfrom(socket, &bytes, bytes.len, linux.MSG.DONTWAIT, null, null);
    switch (linux.errno(result)) {
        .SUCCESS => {},
        .AGAIN, .INTR => return error.WouldBlock,
        .CONNRESET => return error.PeerClosed,
        else => {
            try context.check(result, "recvfrom");
            unreachable;
        },
    }
    if (result == 0) return error.PeerClosed;
    if (result != 1) return error.InvalidClipboardPacket;
    return bytes[0];
}

fn sendByte(context: *SystemError, socket: fd_t, byte: u8) !void {
    const result = linux.sendto(socket, @ptrCast(&byte), 1, linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL, null, 0);
    try context.check(result, "sendto");
    if (result != 1) return error.ClipboardSendFailed;
}

fn register(context: *SystemError, epoll_fd: fd_t, fd: fd_t, events: u32, tag: u64) !void {
    var event: linux.epoll_event = .{ .events = events, .data = .{ .u64 = tag } };
    try context.check(linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, fd, &event), "epoll_ctl");
}

fn unregister(epoll_fd: fd_t, fd: fd_t) void {
    _ = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_DEL, fd, null);
}

fn descriptor(context: *SystemError, result: usize, operation: []const u8) !fd_t {
    try context.check(result, operation);
    return @intCast(result);
}

fn now() u64 {
    var time: linux.timespec = undefined;
    std.debug.assert(linux.errno(linux.clock_gettime(.MONOTONIC, &time)) == .SUCCESS);
    return @as(u64, @intCast(time.sec)) * std.time.ns_per_s + @as(u64, @intCast(time.nsec));
}

fn reportError(socket: fd_t, kind: @FieldType(WorkerError, "kind"), cause: anyerror, context: SystemError, status: ?u32, stderr: *const Stderr) void {
    var detail = std.mem.zeroes(WorkerError);
    detail.kind = if (kind == .cleanup) .cleanup else switch (cause) {
        error.WlCopyNotFound => .tool_not_found,
        error.WlCopyLauncherFailed => .launcher_exited,
        error.WlCopyOwnerExited => .owner_unavailable,
        error.NoClipboardOwner, error.AmbiguousClipboardOwner => .owner_unavailable,
        else => .system,
    };
    const name = @errorName(cause);
    std.debug.assert(name.len <= detail.name.len);
    detail.name_size = @intCast(name.len);
    @memcpy(detail.name[0..name.len], name);
    if (context.native) |native| {
        detail.errno = @intFromEnum(native.errno);
        std.debug.assert(native.operation.len <= detail.operation.len);
        detail.operation_size = @intCast(native.operation.len);
        @memcpy(detail.operation[0..native.operation.len], native.operation);
    }
    detail.wait_status_present = @intFromBool(status != null);
    detail.wait_status = status orelse 0;
    detail.stderr = stderr.bytes;
    detail.stderr_size = stderr.size;
    detail.stderr_omitted = stderr.omitted;
    // The supervisor may already have closed its endpoint to cancel output.
    // Always retain both primary and cleanup reports in the inherited journal.
    logWorkerError(&detail);
    var packet: [1 + @sizeOf(WorkerError)]u8 = undefined;
    packet[0] = @intFromEnum(Report.failed);
    @memcpy(packet[1..], std.mem.asBytes(&detail));
    _ = linux.sendto(socket, &packet, packet.len, linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL, null, 0);
}

pub fn logWorkerError(detail: *const WorkerError) void {
    log.err(.{}, "Clipboard worker error: kind={t}, cause={s}, operation={s}, errno={d}, wait_status_present={d}, wait_status=0x{x}, stderr_omitted_size={d}, stderr=\"{f}\"", .{ detail.kind, detail.name[0..detail.name_size], detail.operation[0..detail.operation_size], detail.errno, detail.wait_status_present, detail.wait_status, detail.stderr_omitted, std.zig.fmtString(detail.stderr[0..detail.stderr_size]) });
}

fn receiveReport(context: *SystemError, socket: fd_t) !union(enum) {
    event: Report,
    err: WorkerError,
    invalid: @FieldType(Error, "invalid_report"),
} {
    var packet: [1 + @sizeOf(WorkerError)]u8 = undefined;
    const size = linux.recvfrom(socket, &packet, packet.len, linux.MSG.DONTWAIT | linux.MSG.TRUNC, null, null);
    switch (linux.errno(size)) {
        .SUCCESS => {},
        .AGAIN, .INTR => return error.WouldBlock,
        .CONNRESET => return error.PeerClosed,
        else => {
            try context.check(size, "report recvfrom");
            unreachable;
        },
    }
    if (size == 0) return error.PeerClosed;
    const invalid: @FieldType(Error, "invalid_report") = .{ .size = size, .byte = packet[0] };
    if (size == 1) return .{ .event = std.enums.fromInt(Report, packet[0]) orelse return .{ .invalid = invalid } };
    if (size != packet.len or packet[0] != @intFromEnum(Report.failed)) return .{ .invalid = invalid };
    const detail = std.mem.bytesToValue(WorkerError, packet[1..]);
    if (std.enums.fromInt(@FieldType(WorkerError, "kind"), @intFromEnum(detail.kind)) == null or
        detail.name_size > detail.name.len or detail.operation_size > detail.operation.len or
        detail.stderr_size > detail.stderr.len or detail.wait_status_present > 1) return .{ .invalid = invalid };
    return .{ .err = detail };
}

// Do not enter std.log/std.Io after fork: inherited library locks may be held.
fn childCheck(socket: fd_t, stderr_fd: fd_t, comptime operation: []const u8, result: usize) void {
    const errno = linux.errno(result);
    if (errno == .SUCCESS) return;
    var detail = std.mem.zeroes(WorkerError);
    detail.kind = .system;
    detail.errno = @intFromEnum(errno);
    const name = "WlCopyExecFailed";
    detail.name_size = name.len;
    @memcpy(detail.name[0..name.len], name);
    detail.operation_size = operation.len;
    @memcpy(detail.operation[0..operation.len], operation);
    var packet: [1 + @sizeOf(WorkerError)]u8 = undefined;
    packet[0] = @intFromEnum(Report.failed);
    @memcpy(packet[1..], std.mem.asBytes(&detail));
    _ = linux.sendto(socket, &packet, packet.len, linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL, null, 0);
    var buffer: [128]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "wl-copy {s}: errno={d}\n", .{ operation, @intFromEnum(errno) }) catch unreachable;
    _ = linux.write(stderr_fd, text.ptr, text.len);
    linux.exit(127);
}
