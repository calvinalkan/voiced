//! Bounded Wayland wire transport. The connection owns queued/received FDs;
//! event handlers take each FD in protocol-signature order. No client library.
const std = @import("std");
const linux = std.os.linux;
const endian = @import("builtin").cpu.arch.endian();
pub const Error = error{ System, Disconnected, InvalidMessage, QueueFull, TooManyDescriptors };
pub const Connection = struct {
    fd: i32 = -1,
    errno: linux.E = .SUCCESS,
    input: [65536]u8 = undefined,
    input_size: usize = 0,
    fds: [32]i32 = undefined,
    fds_count: usize = 0,
    output: [65536]u8 = undefined,
    output_size: usize = 0,
    output_sent: usize = 0,
    output_fd: ?struct { fd: i32, offset: usize } = null,

    pub fn connect(self: *Connection, path: []const u8) Error!void {
        var address: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) };
        if (path.len == 0 or path.len >= address.path.len or std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidMessage;
        @memcpy(address.path[0..path.len], path);
        self.fd = @intCast(try self.check(linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0)));
        _ = try self.check(linux.connect(self.fd, @ptrCast(&address), @intCast(2 + path.len + 1)));
    }
    pub fn deinit(self: *Connection) void {
        if (self.fd >= 0) _ = linux.close(self.fd);
        for (self.fds[0..self.fds_count]) |fd| _ = linux.close(fd);
        if (self.output_fd) |pending| _ = linux.close(pending.fd);
        self.fd = -1;
        self.fds_count = 0;
        self.output_fd = null;
    }
    pub fn check(self: *Connection, result: usize) Error!usize {
        self.errno = linux.errno(result);
        if (self.errno != .SUCCESS) return error.System;
        return result;
    }
    /// Copies request bytes; ownership of an optional FD transfers on success.
    pub fn send(self: *Connection, object: u32, opcode: u16, body: []const u8, fd: ?i32) Error!void {
        if (body.len % 4 != 0 or body.len + 8 > 65532) return error.InvalidMessage;
        if (fd != null and self.output_fd != null) return error.QueueFull;
        if (self.output_sent > 0) {
            std.mem.copyForwards(u8, &self.output, self.output[self.output_sent..self.output_size]);
            self.output_size -= self.output_sent;
            if (self.output_fd) |*pending| pending.offset -= self.output_sent;
            self.output_sent = 0;
        }
        if (body.len + 8 > self.output.len - self.output_size) return error.QueueFull;
        if (fd) |value| self.output_fd = .{ .fd = value, .offset = self.output_size };
        const dest = self.output[self.output_size..][0 .. body.len + 8];
        std.mem.writeInt(u32, dest[0..4], object, endian);
        std.mem.writeInt(u32, dest[4..8], @as(u32, @intCast(dest.len)) << 16 | opcode, endian);
        @memcpy(dest[8..], body);
        self.output_size += dest.len;
    }
    pub fn flush(self: *Connection) Error!void {
        if (self.output_size == 0) return;
        const end = if (self.output_fd) |pending| if (pending.offset > self.output_sent) pending.offset else self.output_size else self.output_size;
        var vectors = [_]std.posix.iovec_const{.{ .base = self.output[self.output_sent..].ptr, .len = end - self.output_sent }};
        var control: [24]u8 align(8) = @splat(0);
        var msg: linux.msghdr_const = .{ .name = null, .namelen = 0, .iov = &vectors, .iovlen = 1, .control = null, .controllen = 0, .flags = 0 };
        const with_fd = if (self.output_fd) |pending| pending.offset == self.output_sent else false;
        if (with_fd) {
            const header: *linux.cmsghdr = @ptrCast(&control);
            header.* = .{ .len = @sizeOf(linux.cmsghdr) + 4, .level = linux.SOL.SOCKET, .type = linux.SCM.RIGHTS };
            std.mem.writeInt(i32, control[@sizeOf(linux.cmsghdr)..][0..4], self.output_fd.?.fd, endian);
            msg.control = &control;
            msg.controllen = control.len;
        }
        const result = linux.sendmsg(self.fd, &msg, linux.MSG.NOSIGNAL);
        self.errno = linux.errno(result);
        if (self.errno == .AGAIN or self.errno == .INTR) return;
        _ = try self.check(result);
        if (result == 0) return error.Disconnected;
        // SCM_RIGHTS is delivered with the first byte, even on a partial send.
        if (with_fd) {
            _ = linux.close(self.output_fd.?.fd);
            self.output_fd = null;
        }
        self.output_sent += result;
        if (self.output_sent == self.output_size) {
            self.output_sent = 0;
            self.output_size = 0;
        }
    }
    pub fn receive(self: *Connection) Error!?Message {
        const target: usize = if (self.input_size < 8) 8 else std.mem.readInt(u32, self.input[4..8], endian) >> 16;
        if (target < 8 or target % 4 != 0) return error.InvalidMessage;
        if (self.input_size < target) {
            var control: [144]u8 align(8) = undefined;
            var vectors = [_]std.posix.iovec{.{ .base = self.input[self.input_size..].ptr, .len = target - self.input_size }};
            var msg: linux.msghdr = .{ .name = null, .namelen = 0, .iov = &vectors, .iovlen = 1, .control = &control, .controllen = control.len, .flags = 0 };
            const result = linux.recvmsg(self.fd, &msg, linux.MSG.CMSG_CLOEXEC);
            self.errno = linux.errno(result);
            if (self.errno == .AGAIN or self.errno == .INTR) return null;
            _ = try self.check(result);
            var cursor: usize = 0;
            var overflow = msg.flags & linux.MSG.CTRUNC != 0;
            while (msg.controllen - cursor >= @sizeOf(linux.cmsghdr)) {
                const header: *const linux.cmsghdr = @ptrCast(@alignCast(control[cursor..].ptr));
                if (header.len < @sizeOf(linux.cmsghdr) or header.len > msg.controllen - cursor) return error.InvalidMessage;
                const bytes = control[cursor + @sizeOf(linux.cmsghdr) .. cursor + header.len];
                if (header.level != linux.SOL.SOCKET or header.type != linux.SCM.RIGHTS or bytes.len % 4 != 0) return error.InvalidMessage;
                var offset: usize = 0;
                while (offset < bytes.len) : (offset += 4) {
                    const fd = std.mem.readInt(i32, bytes[offset..][0..4], endian);
                    if (self.fds_count == self.fds.len) {
                        _ = linux.close(fd);
                        overflow = true;
                    } else {
                        self.fds[self.fds_count] = fd;
                        self.fds_count += 1;
                    }
                }
                cursor += @min(std.mem.alignForward(usize, header.len, 8), msg.controllen - cursor);
            }
            if (overflow) return error.TooManyDescriptors;
            if (result == 0) return error.Disconnected;
            self.input_size += result;
            if (target == 8 or self.input_size < target) return null;
        }
        return .{ .object = std.mem.readInt(u32, self.input[0..4], endian), .opcode = @truncate(std.mem.readInt(u32, self.input[4..8], endian)), .body = .{ .bytes = self.input[8..target] } };
    }
    pub fn takeFd(self: *Connection) Error!i32 {
        if (self.fds_count == 0) return error.InvalidMessage;
        const fd = self.fds[0];
        self.fds_count -= 1;
        std.mem.copyForwards(i32, &self.fds, self.fds[1..][0..self.fds_count]);
        return fd;
    }
};
pub const Message = struct { object: u32, opcode: u16, body: Reader };
pub const Reader = struct {
    bytes: []const u8,
    pub fn word(self: *Reader) Error!u32 {
        if (self.bytes.len < 4) return error.InvalidMessage;
        const value = std.mem.readInt(u32, self.bytes[0..4], endian);
        self.bytes = self.bytes[4..];
        return value;
    }
    pub fn array(self: *Reader) Error![]const u8 {
        const size: usize = try self.word();
        const padded = std.mem.alignForward(usize, size, 4);
        if (padded > self.bytes.len) return error.InvalidMessage;
        const bytes = self.bytes[0..size];
        self.bytes = self.bytes[padded..];
        return bytes;
    }
    pub fn string(self: *Reader) Error![]const u8 {
        const bytes = try self.array();
        if (bytes.len == 0) return "";
        if (bytes[bytes.len - 1] != 0 or std.mem.indexOfScalar(u8, bytes[0 .. bytes.len - 1], 0) != null) return error.InvalidMessage;
        return bytes[0 .. bytes.len - 1];
    }
    pub fn end(self: Reader) Error!void {
        if (self.bytes.len != 0) return error.InvalidMessage;
    }
};
pub const Writer = struct {
    bytes: [1024]u8 = undefined,
    size: usize = 0,
    pub fn word(self: *Writer, value: u32) Error!void {
        if (self.bytes.len - self.size < 4) return error.QueueFull;
        std.mem.writeInt(u32, self.bytes[self.size..][0..4], value, endian);
        self.size += 4;
    }
    pub fn string(self: *Writer, value: []const u8) Error!void {
        const length = std.mem.alignForward(usize, value.len + 1, 4);
        if (length + 4 > self.bytes.len - self.size) return error.QueueFull;
        try self.word(@intCast(value.len + 1));
        @memset(self.bytes[self.size..][0..length], 0);
        @memcpy(self.bytes[self.size..][0..value.len], value);
        self.size += length;
    }
    pub fn data(self: *const Writer) []const u8 {
        return self.bytes[0..self.size];
    }
};
