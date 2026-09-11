//! PipeWire native framing and SPA POD encoding. Message views borrow the
//! connection until consume(); received descriptors belong to that message.
//! Interface versions are selected by the client, independently of server
//! release numbers. Audio memory and scheduling live in the capture client.
const std = @import("std");
const linux = std.os.linux;
const endian = @import("builtin").cpu.arch.endian();
pub const Error = error{ InvalidMessage, MessageTooLarge, TooManyDescriptors, SocketFailed, ConnectFailed, ReadFailed, WriteFailed, Disconnected, QueueFull, DescriptorFailed, MappingFailed, InvalidMemory };
pub const bytes_max = 64 * 1024;
pub const descriptors_max = 64;

pub const Connection = struct {
    fd: linux.fd_t = -1,
    errno: linux.E = .SUCCESS,
    input: [bytes_max]u8 = undefined,
    input_size: usize = 0,
    descriptors: [descriptors_max]linux.fd_t = undefined,
    descriptors_count: usize = 0,
    // Input permits 64 KiB frames; our bounded request batches fit in 32 KiB.
    // Socket backpressure retains this queue and reports exhaustion explicitly.
    output: [32 * 1024]u8 = undefined,
    output_size: usize = 0,
    output_sent: usize = 0,
    sequence: u32 = 0,

    pub fn connect(self: *Connection, path: []const u8) Error!void {
        var address: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) };
        if (path.len == 0 or path.len >= address.path.len or std.mem.indexOfScalar(u8, path, 0) != null) {
            return error.ConnectFailed;
        }

        @memcpy(address.path[0..path.len], path);

        const result = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.CLOEXEC | linux.SOCK.NONBLOCK, 0);

        self.errno = linux.errno(result);

        if (self.errno != .SUCCESS) {
            return error.SocketFailed;
        }

        self.fd = @intCast(result);
        self.errno = linux.errno(linux.connect(self.fd, @ptrCast(&address), @intCast(2 + path.len + 1)));

        if (self.errno != .SUCCESS) {
            return error.ConnectFailed;
        }
    }

    pub fn deinit(self: *Connection) void {
        for (self.descriptors[0..self.descriptors_count]) |fd| {
            _ = linux.close(fd);
        }

        self.descriptors_count = 0;

        if (self.fd >= 0) {
            _ = linux.close(self.fd);
        }

        self.fd = -1;
    }

    /// Initialize fresh storage or reset it after deinit. This does not close
    /// descriptors; call before connect when using undefined storage.
    pub fn initEmpty(self: *Connection) void {
        // Counts guard buffer and descriptor-slot reads. Leave those arrays
        // untouched instead of materializing a whole-connection default value.
        inline for (std.meta.fields(Connection)) |field| {
            if (comptime std.mem.eql(u8, field.name, "input") or std.mem.eql(u8, field.name, "output") or std.mem.eql(u8, field.name, "descriptors")) {
                continue;
            }

            @field(self, field.name) = comptime field.defaultValue() orelse
                @compileError("Connection metadata requires a default: " ++ field.name);
        }
    }

    pub fn send(self: *Connection, object: u32, opcode: u8, payload: []const u8) Error!void {
        if (payload.len > bytes_max - 16) {
            return error.MessageTooLarge;
        }

        if (self.output_sent > 0) {
            std.mem.copyForwards(u8, &self.output, self.output[self.output_sent..self.output_size]);

            self.output_size -= self.output_sent;
            self.output_sent = 0;
        }

        if (16 + payload.len > self.output.len - self.output_size) {
            return error.QueueFull;
        }

        const dest = self.output[self.output_size..][0 .. 16 + payload.len];

        std.mem.writeInt(u32, dest[0..4], object, endian);
        std.mem.writeInt(u32, dest[4..8], @as(u32, opcode) << 24 | @as(u32, @intCast(payload.len)), endian);
        std.mem.writeInt(u32, dest[8..12], self.sequence, endian);
        std.mem.writeInt(u32, dest[12..16], 0, endian);
        @memcpy(dest[16..], payload);

        self.output_size += dest.len;
        self.sequence +%= 1;
    }

    pub fn flush(self: *Connection) Error!void {
        if (self.output_sent == self.output_size) {
            return;
        }

        const bytes = self.output[self.output_sent..self.output_size];
        const result = linux.sendto(self.fd, bytes.ptr, bytes.len, linux.MSG.NOSIGNAL, null, 0);

        self.errno = linux.errno(result);

        switch (self.errno) {
            .AGAIN, .INTR => return,
            .SUCCESS => {},
            else => return error.WriteFailed,
        }

        if (result == 0) {
            return error.Disconnected;
        }

        self.output_sent += result;

        if (self.output_sent == self.output_size) {
            self.output_sent = 0;
            self.output_size = 0;
        }
    }

    /// Read one frame at a time, retaining descriptors in their wire order.
    /// PipeWire batches frames in sendmsg: SCM_RIGHTS for later frames may
    /// arrive with the first byte of an earlier frame. n_fds partitions this
    /// descriptor queue when each complete frame is consumed.
    pub fn receive(self: *Connection) Error!?Message {
        var target: usize = 16;

        if (self.input_size >= 16) {
            target += std.mem.readInt(u32, self.input[4..8], endian) & 0xffffff;

            if (target > self.input.len) {
                return error.MessageTooLarge;
            }

            if (std.mem.readInt(u32, self.input[12..16], endian) > descriptors_max) {
                return error.TooManyDescriptors;
            }
        }

        if (self.input_size < target) {
            var control: [@sizeOf(linux.cmsghdr) + descriptors_max * 4]u8 align(@alignOf(linux.cmsghdr)) = undefined;
            var vector = [_]std.posix.iovec{.{ .base = self.input[self.input_size..].ptr, .len = target - self.input_size }};

            var message: linux.msghdr = .{ .name = null, .namelen = 0, .iov = &vector, .iovlen = 1, .control = &control, .controllen = control.len, .flags = 0 };
            const result = linux.recvmsg(self.fd, &message, linux.MSG.CMSG_CLOEXEC);

            self.errno = linux.errno(result);

            switch (self.errno) {
                .AGAIN, .INTR => return null,
                .SUCCESS => {},
                else => return error.ReadFailed,
            }

            var overflow = false;
            var cursor: usize = 0;

            while (message.controllen - cursor >= @sizeOf(linux.cmsghdr)) {
                const header: *const linux.cmsghdr = @ptrCast(@alignCast(control[cursor..].ptr));
                if (header.len < @sizeOf(linux.cmsghdr) or header.len > message.controllen - cursor) {
                    return error.InvalidMessage;
                }

                const bytes = control[cursor + @sizeOf(linux.cmsghdr) .. cursor + header.len];
                if (header.level != linux.SOL.SOCKET or header.type != linux.SCM.RIGHTS or bytes.len % 4 != 0) {
                    return error.InvalidMessage;
                }

                var offset: usize = 0;

                while (offset < bytes.len) : (offset += 4) {
                    const fd = std.mem.readInt(i32, bytes[offset..][0..4], endian);

                    if (self.descriptors_count == self.descriptors.len) {
                        _ = linux.close(fd);
                        overflow = true;
                    } else {
                        self.descriptors[self.descriptors_count] = fd;
                        self.descriptors_count += 1;
                    }
                }

                const next = std.mem.alignForward(usize, header.len, @alignOf(linux.cmsghdr));

                cursor += @min(next, message.controllen - cursor);
            }

            if (overflow or message.flags & linux.MSG.CTRUNC != 0) {
                return error.TooManyDescriptors;
            }

            if (result == 0) {
                return error.Disconnected;
            }

            self.input_size += result;

            if (target == 16 or self.input_size < target) {
                return null;
            }
        }

        if (self.descriptors_count < std.mem.readInt(u32, self.input[12..16], endian)) {
            return error.InvalidMessage;
        }

        var pods: Reader = .{ .bytes = self.input[16..self.input_size] };
        const body = try pods.structure();

        // Protocol footers are independent extension PODs, not part of the
        // interface payload. Validate framing without interpreting extensions.
        while (pods.bytes.len > 0) {
            _ = try pods.next();
        }

        return .{ .object = std.mem.readInt(u32, self.input[0..4], endian), .opcode = @intCast(std.mem.readInt(u32, self.input[4..8], endian) >> 24), .body = body };
    }

    pub fn duplicateDescriptor(self: *Connection, index: i64) Error!linux.fd_t {
        if (index < 0 or index >= std.mem.readInt(u32, self.input[12..16], endian)) {
            return error.InvalidMessage;
        }

        const result = linux.fcntl(self.descriptors[@intCast(index)], linux.F.DUPFD_CLOEXEC, 0);

        self.errno = linux.errno(result);

        if (self.errno != .SUCCESS) {
            return error.DescriptorFailed;
        }

        return @intCast(result);
    }

    pub fn descriptor(self: *const Connection, index: i64) Error!linux.fd_t {
        if (index < 0 or index >= std.mem.readInt(u32, self.input[12..16], endian)) {
            return error.InvalidMessage;
        }

        return self.descriptors[@intCast(index)];
    }

    pub fn consume(self: *Connection) void {
        const count: usize = std.mem.readInt(u32, self.input[12..16], endian);
        std.debug.assert(count <= self.descriptors_count);

        for (self.descriptors[0..count]) |fd| {
            _ = linux.close(fd);
        }

        std.mem.copyForwards(linux.fd_t, &self.descriptors, self.descriptors[count..self.descriptors_count]);

        self.descriptors_count -= count;
        self.input_size = 0;
    }
};

pub const Message = struct { object: u32, opcode: u8, body: Reader };
pub const Pod = struct {
    kind: u32,
    body: []const u8,
    encoded: []const u8,

    pub fn scalar(self: Pod, kind: u32) Error!u32 {
        // Negotiated SPA values may retain a Choice(None) wrapper.
        if (self.kind == 19) {
            if (self.body.len != 20 or std.mem.readInt(u32, self.body[0..4], endian) != 0 or std.mem.readInt(u32, self.body[8..12], endian) != 4 or std.mem.readInt(u32, self.body[12..16], endian) != kind) {
                return error.InvalidMessage;
            }

            return std.mem.readInt(u32, self.body[16..20], endian);
        }

        if (self.kind != kind or self.body.len != 4) {
            return error.InvalidMessage;
        }

        return std.mem.readInt(u32, self.body[0..4], endian);
    }
};

pub const Reader = struct {
    bytes: []const u8,

    pub fn next(self: *Reader) Error!Pod {
        if (self.bytes.len < 8) {
            return error.InvalidMessage;
        }

        const size: usize = std.mem.readInt(u32, self.bytes[0..4], endian);
        const padded = std.mem.alignForward(usize, size, 8);

        if (padded > self.bytes.len - 8) {
            return error.InvalidMessage;
        }

        const result: Pod = .{ .kind = std.mem.readInt(u32, self.bytes[4..8], endian), .body = self.bytes[8..][0..size], .encoded = self.bytes[0 .. 8 + padded] };

        self.bytes = self.bytes[8 + padded ..];

        return result;
    }
    pub fn structure(self: *Reader) Error!Reader {
        const p = try self.next();
        if (p.kind != 14) {
            return error.InvalidMessage;
        }

        return .{ .bytes = p.body };
    }
    pub fn int(self: *Reader) Error!u32 {
        return (try self.next()).scalar(4);
    }
    pub fn id(self: *Reader) Error!u32 {
        return (try self.next()).scalar(3);
    }
    pub fn long(self: *Reader, kind: u32) Error!i64 {
        const p = try self.next();
        if (p.kind != kind or p.body.len != 8) {
            return error.InvalidMessage;
        }

        return std.mem.readInt(i64, p.body[0..8], endian);
    }
    pub fn string(self: *Reader) Error![]const u8 {
        const p = try self.next();
        if (p.kind != 8 or p.body.len == 0 or p.body[p.body.len - 1] != 0) {
            return error.InvalidMessage;
        }

        const value = p.body[0 .. p.body.len - 1];
        if (std.mem.indexOfScalar(u8, value, 0) != null or !std.unicode.utf8ValidateSlice(value)) {
            return error.InvalidMessage;
        }

        return value;
    }
    pub fn end(self: Reader) Error!void {
        if (self.bytes.len != 0) {
            return error.InvalidMessage;
        }
    }
};

pub const Writer = struct {
    bytes: [16 * 1024]u8 = undefined,
    size: usize = 0,

    pub fn begin(self: *Writer, kind: u32) Error!usize {
        const start = self.size;

        try self.word(0);
        try self.word(kind);

        return start;
    }
    pub fn finish(self: *Writer, start: usize) Error!void {
        std.mem.writeInt(u32, self.bytes[start..][0..4], @intCast(self.size - start - 8), endian);

        const padded = std.mem.alignForward(usize, self.size, 8);
        if (padded > self.bytes.len) {
            return error.MessageTooLarge;
        }

        @memset(self.bytes[self.size..padded], 0);

        self.size = padded;
    }
    pub fn word(self: *Writer, value: u32) Error!void {
        if (self.bytes.len - self.size < 4) {
            return error.MessageTooLarge;
        }

        std.mem.writeInt(u32, self.bytes[self.size..][0..4], value, endian);

        self.size += 4;
    }
    pub fn raw(self: *Writer, value: []const u8) Error!void {
        if (value.len > self.bytes.len - self.size) {
            return error.MessageTooLarge;
        }

        @memcpy(self.bytes[self.size..][0..value.len], value);

        self.size += value.len;
    }
    pub fn scalar(self: *Writer, kind: u32, value: u32) Error!void {
        const start = try self.begin(kind);

        try self.word(value);
        try self.finish(start);
    }
    pub fn int(self: *Writer, value: u32) Error!void {
        try self.scalar(4, value);
    }
    pub fn id(self: *Writer, value: u32) Error!void {
        try self.scalar(3, value);
    }
    pub fn long(self: *Writer, value: u64) Error!void {
        const start = try self.begin(5);

        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, endian);

        try self.raw(&bytes);
        try self.finish(start);
    }
    pub fn string(self: *Writer, value: []const u8) Error!void {
        const start = try self.begin(8);

        try self.raw(value);
        try self.raw(&.{0});
        try self.finish(start);
    }
    pub fn dictionary(self: *Writer, pairs: []const [2][]const u8) Error!void {
        const start = try self.begin(14);

        try self.int(@intCast(pairs.len));

        for (pairs) |pair| {
            try self.string(pair[0]);
            try self.string(pair[1]);
        }

        try self.finish(start);
    }
    pub fn property(self: *Writer, key: u32, kind: u32, value: u32) Error!void {
        try self.word(key);
        try self.word(0);
        try self.scalar(kind, value);
    }
    pub fn data(self: *const Writer) []const u8 {
        return self.bytes[0..self.size];
    }
};

test "connection initialization survives poisoned storage, failed connect and reuse" {
    for ([_]u8{ 0, 0xa5, 0xff }) |poison| {
        var connection: Connection = undefined;
        @memset(std.mem.asBytes(&connection), poison);

        connection.initEmpty();
        try std.testing.expectError(error.ConnectFailed, connection.connect(""));
        connection.deinit();

        for (0..2) |_| {
            connection.initEmpty();

            var sockets: [2]linux.fd_t = undefined;
            try std.testing.expectEqual(.SUCCESS, linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0, &sockets)));
            defer _ = linux.close(sockets[1]);

            connection.fd = sockets[0];
            defer connection.deinit();

            try connection.send(7, 3, "abcd");
            try connection.flush();

            var frame: [20]u8 = undefined;
            try std.testing.expectEqual(frame.len, linux.read(sockets[1], &frame, frame.len));

            try std.testing.expectEqual(7, std.mem.readInt(u32, frame[0..4], endian));
            try std.testing.expectEqual(0x03000004, std.mem.readInt(u32, frame[4..8], endian));
            try std.testing.expectEqual(0, std.mem.readInt(u32, frame[8..12], endian));
            try std.testing.expectEqual(0, std.mem.readInt(u32, frame[12..16], endian));
            try std.testing.expectEqualStrings("abcd", frame[16..]);

            // Closing with queued output must not leak it into the next session.
            try connection.send(8, 4, "stale");
            connection.deinit();
            try std.testing.expectEqual(0, linux.read(sockets[1], &frame, frame.len));
        }
    }
}
