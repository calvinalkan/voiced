//! Bounded D-Bus wire I/O for the notification client. Unix sockets only; no
//! descriptor passing. Received message views borrow Connection.input until
//! consume/read changes it. Wire integers are decoded explicitly in either
//! byte order; outgoing messages use little endian.
const std = @import("std");
const linux = std.os.linux;

pub const Error = error{ InvalidAddress, UnsupportedAddress, AddressTooLong, SocketFailed, ConnectFailed, ReadFailed, WriteFailed, Disconnected, MessageTooLarge, InvalidMessage, BufferFull };
pub const input_bytes_max = 16 * 1024;
pub const output_bytes_max = 4096;

pub const Address = struct {
    value: linux.sockaddr.un = .{ .family = linux.AF.UNIX, .path = @splat(0) },
    length: linux.socklen_t = 0,

    /// Resolve one Unix address once. The caller may try successive addresses
    /// in a semicolon-separated session address list. Paths are percent-decoded.
    pub fn parse(text: []const u8) Error!Address {
        if (!std.mem.startsWith(u8, text, "unix:")) return error.UnsupportedAddress;
        var result: Address = .{};
        var found = false;
        var fields = std.mem.splitScalar(u8, text[5..], ',');
        while (fields.next()) |field| {
            const eq = std.mem.indexOfScalar(u8, field, '=') orelse return error.InvalidAddress;
            const key = field[0..eq];
            const abstract = std.mem.eql(u8, key, "abstract");
            if (!abstract and !std.mem.eql(u8, key, "path")) {
                if (!std.mem.eql(u8, key, "guid")) return error.UnsupportedAddress;
                continue;
            }
            if (found) return error.InvalidAddress;
            found = true;
            var used: usize = if (abstract) 1 else 0;
            var i = eq + 1;
            const start = used;
            while (i < field.len) {
                var byte = field[i];
                i += 1;
                if (byte == '%') {
                    if (field.len - i < 2) return error.InvalidAddress;
                    byte = std.fmt.parseInt(u8, field[i..][0..2], 16) catch return error.InvalidAddress;
                    i += 2;
                }
                if (byte == 0) return error.InvalidAddress;
                if (used == result.value.path.len - @as(usize, if (abstract) 0 else 1)) return error.AddressTooLong;
                result.value.path[used] = byte;
                used += 1;
            }
            if (used == start or (!abstract and result.value.path[0] != '/')) return error.InvalidAddress;
            result.length = @intCast(@offsetOf(linux.sockaddr.un, "path") + used + @as(usize, if (abstract) 0 else 1));
        }
        if (!found) return error.UnsupportedAddress;
        return result;
    }
};

/// Fixed storage bounds both queued output and untrusted input. Each read/write
/// attempts one syscall, leaving fairness and deadlines to the epoll owner.
pub const Connection = struct {
    fd: ?linux.fd_t = null,
    input: [input_bytes_max]u8 = undefined,
    input_size: usize = 0,
    output: [output_bytes_max]u8 = undefined,
    output_size: usize = 0,
    output_sent: usize = 0,
    errno: linux.E = .SUCCESS,

    pub fn connect(self: *Connection, address: *const Address) Error!void {
        const result = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0);
        self.errno = linux.errno(result);
        if (self.errno != .SUCCESS) return error.SocketFailed;
        self.fd = @intCast(result);
        self.errno = linux.errno(linux.connect(self.fd.?, @ptrCast(&address.value), address.length));
        // Unix nonblocking connect uses EAGAIN for a full listener queue. Retry
        // on a fresh socket later, never wait inside the supervisor.
        if (self.errno != .SUCCESS and self.errno != .INPROGRESS) return error.ConnectFailed;
    }

    pub fn close(self: *Connection) void {
        if (self.fd) |fd| _ = linux.close(fd);
        self.fd = null;
        self.input_size = 0;
        self.output_size = 0;
        self.output_sent = 0;
    }

    pub fn flush(self: *Connection) Error!void {
        if (self.output_size == 0) return;
        const bytes = self.output[self.output_sent..self.output_size];
        const result = linux.sendto(self.fd.?, bytes.ptr, bytes.len, linux.MSG.NOSIGNAL, null, 0);
        self.errno = linux.errno(result);
        switch (self.errno) {
            .AGAIN, .INTR => return,
            .SUCCESS => {},
            else => return error.WriteFailed,
        }
        if (result == 0) return error.Disconnected;
        self.output_sent += result;
        if (self.output_sent == self.output_size) {
            self.output_size = 0;
            self.output_sent = 0;
        }
    }

    pub fn read(self: *Connection) Error!bool {
        if (self.input_size == self.input.len) return error.MessageTooLarge;
        const bytes = self.input[self.input_size..];
        const result = linux.read(self.fd.?, bytes.ptr, bytes.len);
        self.errno = linux.errno(result);
        switch (self.errno) {
            .AGAIN, .INTR => return false,
            .SUCCESS => {},
            else => return error.ReadFailed,
        }
        if (result == 0) return error.Disconnected;
        self.input_size += result;
        return true;
    }

    pub fn consume(self: *Connection, count: usize) void {
        std.debug.assert(count <= self.input_size);
        std.mem.copyForwards(u8, &self.input, self.input[count..self.input_size]);
        self.input_size -= count;
    }
};

pub const Message = struct {
    kind: enum(u8) { call = 1, reply = 2, err = 3, signal = 4 },
    serial: u32,
    reply_serial: u32 = 0,
    path: []const u8 = "",
    interface: []const u8 = "",
    member: []const u8 = "",
    destination: []const u8 = "",
    sender: []const u8 = "",
    error_name: []const u8 = "",
    signature: []const u8 = "",
    body: Reader,
    size: usize,

    /// Null means more bytes are needed. Reject excessive lengths as soon as
    /// the fixed header arrives, without allocating or waiting for that body.
    pub fn parse(bytes: []const u8) Error!?Message {
        if (bytes.len < 16) return null;
        const endian: std.builtin.Endian = switch (bytes[0]) {
            'l' => .little,
            'B' => .big,
            else => return error.InvalidMessage,
        };
        if (bytes[3] != 1) return error.InvalidMessage;
        const body_size: usize = std.mem.readInt(u32, bytes[4..8], endian);
        const fields_size: usize = std.mem.readInt(u32, bytes[12..16], endian);
        if (fields_size > input_bytes_max - 16 or body_size > input_bytes_max) return error.MessageTooLarge;
        const body_start = std.mem.alignForward(usize, 16 + fields_size, 8);
        const size = body_start + body_size;
        if (size > input_bytes_max) return error.MessageTooLarge;
        if (bytes.len < size) return null;
        var message: Message = .{
            .kind = std.enums.fromInt(@FieldType(Message, "kind"), bytes[1]) orelse return error.InvalidMessage,
            .serial = std.mem.readInt(u32, bytes[8..12], endian),
            .body = .{ .bytes = bytes[body_start..size], .endian = endian },
            .size = size,
        };
        if (message.serial == 0) return error.InvalidMessage;
        var reader: Reader = .{ .bytes = bytes[0 .. 16 + fields_size], .position = 16, .endian = endian };
        var seen: u16 = 0;
        while (reader.position < reader.bytes.len) {
            try reader.alignTo(8);
            const code = (try reader.take(1))[0];
            const sig = try reader.signature();
            if (code == 0) return error.InvalidMessage;
            if (code <= 9) {
                const mask = @as(u16, 1) << @as(u4, @intCast(code));
                if (seen & mask != 0) return error.InvalidMessage;
                seen |= mask;
                const expected: []const u8 = switch (code) {
                    1 => "o",
                    5, 9 => "u",
                    8 => "g",
                    else => "s",
                };
                if (!std.mem.eql(u8, sig, expected)) return error.InvalidMessage;
            }
            switch (code) {
                1 => message.path = try reader.string(),
                2 => message.interface = try reader.string(),
                3 => message.member = try reader.string(),
                4 => message.error_name = try reader.string(),
                5 => message.reply_serial = try reader.uint32(),
                6 => message.destination = try reader.string(),
                7 => message.sender = try reader.string(),
                8 => message.signature = try reader.signature(),
                9 => if (try reader.uint32() != 0) {
                    return error.InvalidMessage;
                },
                else => try reader.skipVariant(sig, 0),
            }
        }
        var padding: Reader = .{ .bytes = bytes[0..body_start], .position = 16 + fields_size, .endian = endian };
        try padding.alignTo(8);
        switch (message.kind) {
            .call => if (message.path.len == 0 or message.member.len == 0) {
                return error.InvalidMessage;
            },
            .reply => if (message.reply_serial == 0) {
                return error.InvalidMessage;
            },
            .err => if (message.reply_serial == 0 or message.error_name.len == 0) {
                return error.InvalidMessage;
            },
            .signal => if (message.path.len == 0 or message.interface.len == 0 or message.member.len == 0) {
                return error.InvalidMessage;
            },
        }
        if (body_size != 0 and message.signature.len == 0) return error.InvalidMessage;
        return message;
    }
};

pub const Writer = struct {
    bytes: []u8,
    position: usize = 0,
    body_start: usize = 0,

    pub fn call(bytes: []u8, serial: u32, destination: []const u8, path: []const u8, interface: []const u8, member: []const u8, body_signature: []const u8) Error!Writer {
        var writer: Writer = .{ .bytes = bytes };
        try writer.append(&.{ 'l', 1, 0, 1 });
        try writer.uint32(0);
        try writer.uint32(serial);
        try writer.uint32(0);
        try writer.field(1, "o", path);
        try writer.field(2, "s", interface);
        try writer.field(3, "s", member);
        try writer.field(6, "s", destination);
        if (body_signature.len != 0) try writer.field(8, "g", body_signature);
        std.mem.writeInt(u32, bytes[12..16], @intCast(writer.position - 16), .little);
        try writer.alignTo(8);
        writer.body_start = writer.position;
        return writer;
    }

    pub fn finish(self: *Writer) usize {
        std.mem.writeInt(u32, self.bytes[4..8], @intCast(self.position - self.body_start), .little);
        return self.position;
    }

    fn field(self: *Writer, code: u8, sig: []const u8, value: []const u8) Error!void {
        try self.alignTo(8);
        try self.append(&.{code});
        try self.signature(sig);
        if (std.mem.eql(u8, sig, "g")) try self.signature(value) else try self.string(value);
    }

    pub fn uint32(self: *Writer, value: u32) Error!void {
        try self.alignTo(4);
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        try self.append(&bytes);
    }

    pub fn uint64(self: *Writer, value: u64) Error!void {
        try self.alignTo(8);
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        try self.append(&bytes);
    }

    pub fn string(self: *Writer, value: []const u8) Error!void {
        try self.uint32(std.math.cast(u32, value.len) orelse return error.BufferFull);
        try self.append(value);
        try self.append(&.{0});
    }

    fn signature(self: *Writer, value: []const u8) Error!void {
        try self.append(&.{std.math.cast(u8, value.len) orelse return error.BufferFull});
        try self.append(value);
        try self.append(&.{0});
    }

    pub fn alignTo(self: *Writer, alignment: usize) Error!void {
        const end = std.mem.alignForward(usize, self.position, alignment);
        if (end > self.bytes.len) return error.BufferFull;
        @memset(self.bytes[self.position..end], 0);
        self.position = end;
    }

    fn append(self: *Writer, bytes: []const u8) Error!void {
        if (bytes.len > self.bytes.len - self.position) return error.BufferFull;
        @memcpy(self.bytes[self.position..][0..bytes.len], bytes);
        self.position += bytes.len;
    }
};

pub const Reader = struct {
    bytes: []const u8,
    position: usize = 0,
    endian: std.builtin.Endian,

    pub fn uint32(self: *Reader) Error!u32 {
        try self.alignTo(4);
        return std.mem.readInt(u32, (try self.take(4))[0..4], self.endian);
    }

    pub fn uint64(self: *Reader) Error!u64 {
        try self.alignTo(8);
        return std.mem.readInt(u64, (try self.take(8))[0..8], self.endian);
    }

    pub fn string(self: *Reader) Error![]const u8 {
        const size = try self.uint32();
        return self.text(size);
    }

    pub fn signature(self: *Reader) Error![]const u8 {
        const size = (try self.take(1))[0];
        return self.text(size);
    }

    fn text(self: *Reader, size: usize) Error![]const u8 {
        const value = try self.take(size);
        if ((try self.take(1))[0] != 0 or std.mem.indexOfScalar(u8, value, 0) != null or !std.unicode.utf8ValidateSlice(value)) return error.InvalidMessage;
        return value;
    }

    pub fn end(self: *const Reader) Error!void {
        if (self.position != self.bytes.len) return error.InvalidMessage;
    }

    fn take(self: *Reader, size: usize) Error![]const u8 {
        if (size > self.bytes.len - self.position) return error.InvalidMessage;
        const value = self.bytes[self.position..][0..size];
        self.position += size;
        return value;
    }

    fn alignTo(self: *Reader, alignment: usize) Error!void {
        const end_position = std.mem.alignForward(usize, self.position, alignment);
        const padding = try self.take(end_position - self.position);
        for (padding) |byte| if (byte != 0) return error.InvalidMessage;
    }

    // Unknown header fields are variants and must be ignored for protocol
    // extension compatibility. Walk only their declared value, with bounded
    // nesting and byte ranges; never guess its length from the field number.
    fn skipVariant(self: *Reader, sig: []const u8, depth: u8) Error!void {
        var position: usize = 0;
        try self.skipValue(sig, &position, depth);
        if (position != sig.len) return error.InvalidMessage;
    }

    fn skipValue(self: *Reader, sig: []const u8, position: *usize, depth: u8) Error!void {
        if (depth >= 32 or position.* >= sig.len) return error.InvalidMessage;
        const code = sig[position.*];
        position.* += 1;
        switch (code) {
            'y' => _ = try self.take(1),
            'n', 'q' => {
                try self.alignTo(2);
                _ = try self.take(2);
            },
            'b', 'i', 'u', 'h' => _ = try self.uint32(),
            'x', 't', 'd' => {
                try self.alignTo(8);
                _ = try self.take(8);
            },
            's', 'o' => _ = try self.string(),
            'g' => _ = try self.signature(),
            'v' => try self.skipVariant(try self.signature(), depth + 1),
            '(', '{' => {
                try self.alignTo(8);
                const close: u8 = if (code == '(') ')' else '}';
                const start = position.*;
                while (position.* < sig.len and sig[position.*] != close) try self.skipValue(sig, position, depth + 1);
                if (position.* == start or position.* == sig.len) return error.InvalidMessage;
                position.* += 1;
            },
            'a' => {
                const size = try self.uint32();
                if (position.* == sig.len) return error.InvalidMessage;
                const alignment: usize = switch (sig[position.*]) {
                    'y', 'g', 'v' => 1,
                    'n', 'q' => 2,
                    'b', 'i', 'u', 'h', 's', 'o', 'a' => 4,
                    'x', 't', 'd', '(', '{' => 8,
                    else => return error.InvalidMessage,
                };
                try self.alignTo(alignment);
                _ = try self.take(size);
                // The array length lets us skip its contents, but its element
                // signature still needs a complete syntactic boundary.
                try skipType(sig, position, depth + 1);
            },
            else => return error.InvalidMessage,
        }
    }
};

fn skipType(sig: []const u8, position: *usize, depth: u8) Error!void {
    if (depth >= 32 or position.* >= sig.len) return error.InvalidMessage;
    const code = sig[position.*];
    position.* += 1;
    switch (code) {
        'y', 'b', 'n', 'q', 'i', 'u', 'x', 't', 'd', 'h', 's', 'o', 'g', 'v' => {},
        'a' => try skipType(sig, position, depth + 1),
        '(', '{' => {
            const close: u8 = if (code == '(') ')' else '}';
            const start = position.*;
            while (position.* < sig.len and sig[position.*] != close) try skipType(sig, position, depth + 1);
            if (position.* == start or position.* == sig.len) return error.InvalidMessage;
            position.* += 1;
        },
        else => return error.InvalidMessage,
    }
}
