//! A single typed job/result mailbox for one persistent thread. Payloads never
//! allocate. The supervisor consumes completion before reusing borrowed storage;
//! eventfd wakeups coalesce without carrying ownership or result data.
const std = @import("std");
const linux = std.os.linux;
const assert = std.debug.assert;

pub fn Mailbox(comptime Job: type, comptime Result: type) type {
    return struct {
        const Self = @This();
        const State = enum(u8) { idle, job, result, shutdown, exited };
        const Payload = union { job: Job, result: Result };

        wake_fd: std.posix.fd_t,
        notification_fd: std.posix.fd_t,

        // PERFORMANCE: `state` is authoritative, while `payload` is valid only
        // in `.job` or `.result`. Combining them in a tagged union makes each
        // payload-free transition materialize and copy a value sized for the
        // largest payload. This mailbox has exactly one producer and one
        // consumer. Release/acquire publication protects the plain payload;
        // idle, shutdown, and exited leave stale payload bytes unread.
        state: std.atomic.Value(State) = .init(.idle),
        payload: Payload = undefined,

        // PERFORMANCE: Initialize in place so constructing a mailbox does not
        // materialize and copy a constant sized for its largest payload.
        pub fn init(self: *Self, notification_fd: std.posix.fd_t) !void {
            const result = linux.eventfd(0, linux.EFD.CLOEXEC | linux.EFD.NONBLOCK);
            if (linux.errno(result) != .SUCCESS) return error.WorkerEventCreateFailed;
            self.wake_fd = @intCast(result);
            self.notification_fd = notification_fd;
            self.state = .init(.idle);
        }

        /// Call after join, or before a thread was spawned.
        pub fn deinit(self: *Self) void {
            _ = linux.close(self.wake_fd);
        }

        pub fn submit(self: *Self, job: Job) void {
            assert(self.state.load(.acquire) == .idle);
            self.payload = .{ .job = job };
            self.state.store(.job, .release);
            wake(self.wake_fd);
        }

        /// Only the worker calls next/complete. A job remains outstanding until
        /// complete publishes its result; its borrowed inputs stay immutable.
        pub fn next(self: *Self) ?Job {
            while (true) {
                switch (self.state.load(.acquire)) {
                    .job => {
                        const job = self.payload.job;
                        drain(self.wake_fd);
                        return job;
                    },
                    .shutdown => return null,
                    .idle, .result => {},
                    .exited => unreachable,
                }
                var fd = [_]linux.pollfd{.{ .fd = self.wake_fd, .events = linux.POLL.IN, .revents = 0 }};
                const result = linux.poll(&fd, 1, -1);
                switch (linux.errno(result)) {
                    .INTR => continue,
                    .SUCCESS => drain(self.wake_fd),
                    else => @panic("Worker wake poll failed"),
                }
            }
        }

        pub fn complete(self: *Self, result: Result) void {
            assert(self.state.load(.monotonic) == .job);
            // A plain Zig union still tracks its active field in safe builds.
            // Field assignment would access the old variant instead of retagging.
            self.payload = .{ .result = result };
            self.state.store(.result, .release);
            wake(self.notification_fd);
        }

        pub fn receive(self: *Self) ?Result {
            if (self.state.load(.acquire) != .result) return null;
            const result = self.payload.result;
            self.state.store(.idle, .release);
            return result;
        }

        pub fn shutdown(self: *Self) void {
            assert(self.state.load(.acquire) == .idle);
            self.state.store(.shutdown, .release);
            wake(self.wake_fd);
        }

        /// Publish only after all worker-owned resources and compute threads
        /// have been released. The supervisor can then join without waiting on I/O.
        pub fn finish(self: *Self) void {
            assert(self.state.load(.acquire) == .shutdown);
            self.state.store(.exited, .release);
            wake(self.notification_fd);
        }

        pub fn exited(self: *Self) bool {
            return self.state.load(.acquire) == .exited;
        }
    };
}

pub fn wake(fd: std.posix.fd_t) void {
    const one: u64 = 1;
    while (true) switch (linux.errno(linux.write(fd, std.mem.asBytes(&one).ptr, @sizeOf(u64)))) {
        .SUCCESS, .AGAIN => return,
        .INTR => continue,
        else => @panic("Worker wake failed"),
    };
}

pub fn drain(fd: std.posix.fd_t) void {
    var count: u64 = undefined;
    while (true) switch (linux.errno(linux.read(fd, std.mem.asBytes(&count).ptr, @sizeOf(u64)))) {
        .SUCCESS, .AGAIN => return,
        .INTR => continue,
        else => @panic("Worker wake read failed"),
    };
}

pub fn name(value: [:0]const u8) void {
    assert(value.len <= 15);
    _ = linux.prctl(@intFromEnum(linux.PR.SET_NAME), @intFromPtr(value.ptr), 0, 0, 0);
}
