//! Bounded X11 core transport. The connection authenticates a local Unix
//! display, assigns request sequences, and frames setup, replies, errors and
//! events without libX11, XCB, allocation or borrowed output.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const endian = builtin.cpu.arch.endian();
const authority_size_max = 64 * 1024;

pub const Error = error{
    System,
    Disconnected,
    InvalidAuthority,
    AuthorityMissing,
    AuthoritySystem,
    UnsupportedDisplay,
    InvalidMessage,
    QueueFull,
};

pub const Setup = union(enum) {
    success: struct {
        root: u32,
        resource_id_base: u32,
        resource_id_mask: u32,
        maximum_request_size: u32,
    },
    rejected: struct {
        status: u8,
        reason: [128]u8,
        reason_size: u8,
        truncated: bool,
    },
};

pub const Message = struct {
    bytes: []const u8,

    pub fn responseType(self: Message) u8 {
        return self.bytes[0] & 0x7f;
    }

    pub fn sequence(self: Message) u16 {
        return std.mem.readInt(u16, self.bytes[2..4], endian);
    }
};

pub const Connection = struct {
    fd: i32 = -1,
    errno: linux.E = .SUCCESS,
    input: [65536]u8 = undefined,
    input_size: usize = 0,
    output: [16384]u8 = undefined,
    output_size: usize = 0,
    output_sent: usize = 0,
    sequence_number: u32 = 0,
    maximum_request_size: u32 = 0,

    /// Initializes fresh storage or resets it after deinit. This does not close
    /// descriptors; call before connect when using undefined storage.
    pub fn initEmpty(self: *Connection) void {
        inline for (std.meta.fields(Connection)) |field| {
            if (comptime std.mem.eql(u8, field.name, "input") or std.mem.eql(u8, field.name, "output")) continue;
            @field(self, field.name) = comptime field.defaultValue() orelse @compileError("Connection metadata requires a default: " ++ field.name);
        }
    }

    pub fn connectDisplay(self: *Connection, display: []const u8, authority: ?[]const u8, home: ?[]const u8) Error!void {
        const display_number = try parseDisplay(display);
        var cookie: [16]u8 = undefined;
        const has_cookie = try self.readCookie(display_number, authority, home, &cookie);

        var socket_path: [108]u8 = undefined;
        const path = std.fmt.bufPrint(&socket_path, "/tmp/.X11-unix/X{d}", .{display_number}) catch return error.UnsupportedDisplay;
        var address: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) };
        @memcpy(address.path[0..path.len], path);
        self.fd = @intCast(try self.check(linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0)));
        errdefer {
            _ = linux.close(self.fd);
            self.fd = -1;
        }
        _ = try self.check(linux.connect(self.fd, @ptrCast(&address), @intCast(2 + path.len + 1)));

        const name = if (has_cookie) "MIT-MAGIC-COOKIE-1" else "";
        const data = if (has_cookie) cookie[0..] else "";
        const name_size = std.mem.alignForward(usize, name.len, 4);
        const data_size = std.mem.alignForward(usize, data.len, 4);
        const packet = self.output[0 .. 12 + name_size + data_size];
        @memset(packet, 0);
        packet[0] = if (endian == .little) 'l' else 'B';
        std.mem.writeInt(u16, packet[2..4], 11, endian);
        std.mem.writeInt(u16, packet[6..8], @intCast(name.len), endian);
        std.mem.writeInt(u16, packet[8..10], @intCast(data.len), endian);
        @memcpy(packet[12..][0..name.len], name);
        @memcpy(packet[12 + name_size ..][0..data.len], data);
        self.output_size = packet.len;
    }

    pub fn deinit(self: *Connection) void {
        if (self.fd >= 0) _ = linux.close(self.fd);
        self.fd = -1;
        self.input_size = 0;
        self.output_size = 0;
        self.output_sent = 0;
    }

    pub fn check(self: *Connection, result: usize) Error!usize {
        self.errno = linux.errno(result);
        if (self.errno != .SUCCESS) return error.System;
        return result;
    }

    pub fn flush(self: *Connection) Error!void {
        if (self.output_sent == self.output_size) {
            self.output_sent = 0;
            self.output_size = 0;
            return;
        }
        const result = linux.sendto(self.fd, self.output[self.output_sent..self.output_size].ptr, self.output_size - self.output_sent, linux.MSG.NOSIGNAL, null, 0);
        self.errno = linux.errno(result);
        if (self.errno == .AGAIN or self.errno == .INTR) return;
        _ = try self.check(result);
        if (result == 0) return error.Disconnected;
        self.output_sent += result;
        if (self.output_sent == self.output_size) {
            self.output_sent = 0;
            self.output_size = 0;
        }
    }

    /// Copies one padded core request and returns its low 16-bit sequence.
    pub fn send(self: *Connection, opcode: u8, data: u8, body: []const u8) Error!u16 {
        return self.sendParts(opcode, data, body, "");
    }

    /// Copies an aligned fixed body followed by bytes and zero padding.
    pub fn sendParts(self: *Connection, opcode: u8, data: u8, body: []const u8, trailing: []const u8) Error!u16 {
        if (body.len % 4 != 0) return error.InvalidMessage;
        if (self.output_sent > 0) {
            std.mem.copyForwards(u8, &self.output, self.output[self.output_sent..self.output_size]);
            self.output_size -= self.output_sent;
            self.output_sent = 0;
        }
        const request_size = std.mem.alignForward(usize, body.len + trailing.len + 4, 4);
        if (request_size > std.math.maxInt(u16) * 4) return error.InvalidMessage;
        if ((self.maximum_request_size != 0 and request_size > self.maximum_request_size) or request_size > self.output.len - self.output_size) return error.QueueFull;
        const destination = self.output[self.output_size..][0..request_size];
        destination[0] = opcode;
        destination[1] = data;
        std.mem.writeInt(u16, destination[2..4], @intCast(request_size / 4), endian);
        @memcpy(destination[4..][0..body.len], body);
        @memcpy(destination[4 + body.len ..][0..trailing.len], trailing);
        @memset(destination[4 + body.len + trailing.len ..], 0);
        self.output_size += request_size;
        self.sequence_number +%= 1;
        return @truncate(self.sequence_number);
    }

    pub fn receiveSetup(self: *Connection) Error!?Setup {
        if (!try self.readTo(8)) return null;
        const target = 8 + @as(usize, std.mem.readInt(u16, self.input[6..8], endian)) * 4;
        if (target > self.input.len) return error.InvalidMessage;
        if (!try self.readTo(target)) return null;

        const status = self.input[0];
        if (status != 1) {
            if (status == 0 and self.input[1] > target - 8) return error.InvalidMessage;
            const reason_size = if (status == 0) @as(usize, self.input[1]) else target - 8;
            var rejected: @FieldType(Setup, "rejected") = .{ .status = status, .reason = @splat(0), .reason_size = @intCast(@min(reason_size, 128)), .truncated = reason_size > 128 };
            @memcpy(rejected.reason[0..rejected.reason_size], self.input[8..][0..rejected.reason_size]);
            self.input_size = 0;
            return .{ .rejected = rejected };
        }
        if (target < 40 or std.mem.readInt(u16, self.input[2..4], endian) != 11 or std.mem.readInt(u16, self.input[4..6], endian) != 0) return error.InvalidMessage;
        const vendor_size = std.mem.alignForward(usize, std.mem.readInt(u16, self.input[24..26], endian), 4);
        const formats_size = @as(usize, self.input[29]) * 8;
        const root_offset = 40 + vendor_size + formats_size;
        if (self.input[28] == 0 or root_offset + 40 > target) return error.InvalidMessage;
        const maximum_request_size = @as(u32, std.mem.readInt(u16, self.input[26..28], endian)) * 4;
        if (maximum_request_size < 16384) return error.InvalidMessage;
        const setup: Setup = .{ .success = .{
            .root = std.mem.readInt(u32, self.input[root_offset..][0..4], endian),
            .resource_id_base = std.mem.readInt(u32, self.input[12..16], endian),
            .resource_id_mask = std.mem.readInt(u32, self.input[16..20], endian),
            .maximum_request_size = maximum_request_size,
        } };
        self.maximum_request_size = @min(maximum_request_size, self.output.len);
        self.input_size = 0;
        return setup;
    }

    pub fn receive(self: *Connection) Error!?Message {
        if (!try self.readTo(32)) return null;
        const response_type = self.input[0] & 0x7f;
        const target = if (response_type == 1 or response_type == 35)
            32 + @as(usize, std.mem.readInt(u32, self.input[4..8], endian)) * 4
        else
            32;
        if (target > self.input.len) return error.InvalidMessage;
        if (!try self.readTo(target)) return null;
        return .{ .bytes = self.input[0..target] };
    }

    pub fn consume(self: *Connection) void {
        self.input_size = 0;
    }

    fn readTo(self: *Connection, target: usize) Error!bool {
        if (self.input_size >= target) return true;
        const result = linux.read(self.fd, self.input[self.input_size..target].ptr, target - self.input_size);
        self.errno = linux.errno(result);
        if (self.errno == .AGAIN or self.errno == .INTR) return false;
        _ = try self.check(result);
        if (result == 0) return error.Disconnected;
        self.input_size += result;
        return self.input_size == target;
    }

    fn readCookie(self: *Connection, display_number: u16, authority: ?[]const u8, home: ?[]const u8, cookie: *[16]u8) Error!bool {
        var default_path: [std.posix.PATH_MAX]u8 = undefined;
        const path = authority orelse if (home) |directory|
            std.fmt.bufPrint(&default_path, "{s}/.Xauthority", .{directory}) catch return error.InvalidAuthority
        else
            return false;
        const path_z = std.posix.toPosixPath(path) catch return error.InvalidAuthority;
        const result = linux.open(&path_z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        self.errno = linux.errno(result);
        if (self.errno == .NOENT) {
            if (authority == null) return false;
            return error.AuthorityMissing;
        }
        if (self.errno != .SUCCESS) return error.AuthoritySystem;
        const fd: i32 = @intCast(result);
        defer _ = linux.close(fd);

        var reader: AuthorityReader = .{ .fd = fd, .errno = &self.errno };
        var display_buffer: [5]u8 = undefined;
        const display = std.fmt.bufPrint(&display_buffer, "{d}", .{display_number}) catch unreachable;
        var host: linux.utsname = undefined;
        const host_result = linux.uname(&host);
        self.errno = linux.errno(host_result);
        if (self.errno != .SUCCESS) return error.AuthoritySystem;
        const hostname = std.mem.sliceTo(&host.nodename, 0);
        var address: [256]u8 = undefined;
        var number: [16]u8 = undefined;
        var name: [32]u8 = undefined;
        var data: [64]u8 = undefined;

        while (try reader.readU16OrEof()) |family| {
            const address_field = try reader.readField(&address);
            const number_field = try reader.readField(&number);
            const name_field = try reader.readField(&name);
            const data_field = try reader.readField(&data);
            const family_matches = family == 256 or family == 65535;
            // Match XauGetBestAuthByAddr's FamilyLocal query: FamilyWild bypasses
            // the address, while an empty record number is a display wildcard.
            const address_matches = family == 65535 or (address_field.complete and std.mem.eql(u8, address_field.bytes, hostname));
            const number_matches = number_field.complete and (number_field.bytes.len == 0 or std.mem.eql(u8, number_field.bytes, display));
            if (family_matches and address_matches and number_matches and name_field.complete and std.mem.eql(u8, name_field.bytes, "MIT-MAGIC-COOKIE-1")) {
                if (!data_field.complete or data_field.bytes.len != cookie.len) return error.InvalidAuthority;
                @memcpy(cookie, data_field.bytes);
                return true;
            }
        }
        return error.AuthorityMissing;
    }
};

const AuthorityReader = struct {
    fd: i32,
    errno: *linux.E,
    size: usize = 0,

    fn readU16OrEof(self: *AuthorityReader) Error!?u16 {
        var bytes: [2]u8 = undefined;
        var size: usize = 0;
        while (size < bytes.len) {
            const part_size = try self.readSome(bytes[size..]);
            if (part_size == 0) {
                if (size == 0) return null;
                return error.InvalidAuthority;
            }
            size += part_size;
        }
        return std.mem.readInt(u16, &bytes, .big);
    }

    fn readField(self: *AuthorityReader, storage: []u8) Error!struct { bytes: []const u8, complete: bool } {
        const size = (try self.readU16OrEof()) orelse return error.InvalidAuthority;
        const retained_size = @min(@as(usize, size), storage.len);
        try self.readExact(storage[0..retained_size]);
        var remaining = @as(usize, size) - retained_size;
        var discarded: [256]u8 = undefined;
        while (remaining > 0) {
            const part_size = @min(remaining, discarded.len);
            try self.readExact(discarded[0..part_size]);
            remaining -= part_size;
        }
        return .{ .bytes = storage[0..retained_size], .complete = size <= storage.len };
    }

    fn readExact(self: *AuthorityReader, bytes: []u8) Error!void {
        var size: usize = 0;
        while (size < bytes.len) {
            const part_size = try self.readSome(bytes[size..]);
            if (part_size == 0) return error.InvalidAuthority;
            size += part_size;
        }
    }

    fn readSome(self: *AuthorityReader, destination: []u8) Error!usize {
        const available = authority_size_max -| self.size;
        if (available == 0) return error.InvalidAuthority;
        while (true) {
            const result = linux.read(self.fd, destination.ptr, @min(destination.len, available));
            self.errno.* = linux.errno(result);
            if (self.errno.* == .INTR) continue;
            if (self.errno.* != .SUCCESS) return error.AuthoritySystem;
            self.size += result;
            return result;
        }
    }
};

fn parseDisplay(display: []const u8) Error!u16 {
    const start = if (std.mem.startsWith(u8, display, ":"))
        @as(usize, 1)
    else if (std.mem.startsWith(u8, display, "unix:"))
        @as(usize, 5)
    else
        return error.UnsupportedDisplay;
    const suffix = display[start..];
    const end = std.mem.indexOfScalar(u8, suffix, '.') orelse suffix.len;
    if (end == 0 or (end != suffix.len and !std.mem.eql(u8, suffix[end..], ".0"))) return error.UnsupportedDisplay;
    return std.fmt.parseInt(u16, suffix[0..end], 10) catch error.UnsupportedDisplay;
}

test "local display parsing accepts only screen zero" {
    try std.testing.expectEqual(0, try parseDisplay(":0"));
    try std.testing.expectEqual(77, try parseDisplay("unix:77.0"));
    try std.testing.expectError(error.UnsupportedDisplay, parseDisplay("unix:77.2"));
    try std.testing.expectError(error.UnsupportedDisplay, parseDisplay("localhost:0"));
    try std.testing.expectError(error.UnsupportedDisplay, parseDisplay(":0."));
    try std.testing.expectError(error.UnsupportedDisplay, parseDisplay(":0.bad"));
}

test "requests are copied, padded and sequenced" {
    var connection: Connection = undefined;
    connection.initEmpty();
    const sequence = try connection.sendParts(18, 0, &.{ 1, 2, 3, 4 }, "abc");
    try std.testing.expectEqual(1, sequence);
    try std.testing.expectEqual(12, connection.output_size);
    try std.testing.expectEqualSlices(u8, &.{ 18, 0, 3, 0, 1, 2, 3, 4, 'a', 'b', 'c', 0 }, connection.output[0..12]);
}

test "nonblocking output survives backpressure and reports a closed peer" {
    var sockets: [2]i32 = undefined;
    try std.testing.expectEqual(.SUCCESS, linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0, &sockets)));
    var peer_open = true;
    defer if (peer_open) {
        _ = linux.close(sockets[1]);
    };
    var connection: Connection = undefined;
    connection.initEmpty();
    connection.fd = sockets[0];
    defer connection.deinit();

    const capacity: c_int = 1024;
    try std.testing.expectEqual(.SUCCESS, linux.errno(linux.setsockopt(sockets[0], linux.SOL.SOCKET, linux.SO.SNDBUF, std.mem.asBytes(&capacity), @sizeOf(c_int))));
    var bytes: [4096]u8 = @splat(0);
    while (true) {
        const result = linux.sendto(sockets[0], &bytes, bytes.len, linux.MSG.NOSIGNAL, null, 0);
        if (linux.errno(result) == .AGAIN) break;
        try std.testing.expectEqual(.SUCCESS, linux.errno(result));
    }
    @memcpy(connection.output[0..5], "hello");
    connection.output_size = 5;
    try connection.flush();
    try std.testing.expectEqual(0, connection.output_sent);
    try std.testing.expectEqual(5, connection.output_size);

    while (true) {
        const result = linux.read(sockets[1], &bytes, bytes.len);
        if (linux.errno(result) == .AGAIN) break;
        try std.testing.expectEqual(.SUCCESS, linux.errno(result));
    }
    try connection.flush();
    try std.testing.expectEqual(0, connection.output_size);
    try std.testing.expectEqual(5, linux.read(sockets[1], &bytes, bytes.len));
    try std.testing.expectEqualStrings("hello", bytes[0..5]);

    _ = linux.close(sockets[1]);
    peer_open = false;
    @memcpy(connection.output[0..5], "again");
    connection.output_size = 5;
    try std.testing.expectError(error.System, connection.flush());
    try std.testing.expectEqual(.PIPE, connection.errno);
}
