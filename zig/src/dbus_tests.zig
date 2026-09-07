const std = @import("std");
const dbus = @import("dbus.zig");

// Independent wire fixture: METHOD_RETURN(serial=11, reply_serial=9),
// signature u, body 42. Header fields occupy 15 bytes plus one padding byte.
const reply_little = [_]u8{
    'l', 2, 0,   1, 4, 0, 0, 0, 11, 0, 0,   0, 15, 0,   0, 0,
    5,   1, 'u', 0, 9, 0, 0, 0, 8,  1, 'g', 0, 1,  'u', 0, 0,
    42,  0, 0,   0,
};
const reply_big = [_]u8{
    'B', 2, 0,   1,  0, 0, 0, 4, 0, 0, 0,   11, 0, 0,   0, 15,
    5,   1, 'u', 0,  0, 0, 0, 9, 8, 1, 'g', 0,  1, 'u', 0, 0,
    0,   0, 0,   42,
};

test "reply framing tolerates every split and both byte orders" {
    for ([_][]const u8{ &reply_little, &reply_big }) |bytes| {
        for (0..bytes.len) |end| try std.testing.expectEqual(null, try dbus.Message.parse(bytes[0..end]));
        const message = (try dbus.Message.parse(bytes)).?;
        try std.testing.expectEqual(.reply, message.kind);
        try std.testing.expectEqual(11, message.serial);
        try std.testing.expectEqual(9, message.reply_serial);
        try std.testing.expectEqualStrings("u", message.signature);
        var body = message.body;
        try std.testing.expectEqual(42, try body.uint32());
        try body.end();
    }
}

test "coalesced messages retain their individual byte boundaries" {
    const messages = reply_little ++ reply_big;
    const first = (try dbus.Message.parse(&messages)).?;
    const second = (try dbus.Message.parse(messages[first.size..])).?;
    try std.testing.expectEqual(reply_little.len, first.size);
    try std.testing.expectEqual(reply_big.len, second.size);
}

test "oversize frames fail at the fixed header before receiving a body" {
    for ([_]usize{ 4, 12 }) |offset| {
        var bytes = reply_little;
        @memset(bytes[offset..][0..4], 255);
        try std.testing.expectError(error.MessageTooLarge, dbus.Message.parse(bytes[0..16]));
    }
}

test "reject invalid frame tags, serials, field types and padding" {
    for ([_]struct { offset: usize, byte: u8 }{
        .{ .offset = 0, .byte = 'x' }, .{ .offset = 1, .byte = 0 },
        .{ .offset = 3, .byte = 2 },   .{ .offset = 8, .byte = 0 },
        .{ .offset = 16, .byte = 0 },  .{ .offset = 18, .byte = 's' },
        .{ .offset = 19, .byte = 1 },  .{ .offset = 20, .byte = 0 },
        .{ .offset = 30, .byte = 1 },  .{ .offset = 31, .byte = 1 },
    }) |change| {
        var bytes = reply_little;
        bytes[change.offset] = change.byte;
        try std.testing.expectError(error.InvalidMessage, dbus.Message.parse(&bytes));
    }
    var duplicate = reply_little;
    @memcpy(duplicate[24..32], duplicate[16..24]);
    std.mem.writeInt(u32, duplicate[12..16], 16, .little);
    try std.testing.expectError(error.InvalidMessage, dbus.Message.parse(&duplicate));
}

test "unknown header variants can be skipped without losing body alignment" {
    // Append an extension field containing an empty array of strings.
    var bytes: [52]u8 = @splat(0);
    @memcpy(bytes[0..31], reply_little[0..31]);
    @memcpy(bytes[32..38], &[_]u8{ 42, 2, 'a', 's', 0, 0 });
    std.mem.writeInt(u32, bytes[12..16], 28, .little);
    std.mem.writeInt(u32, bytes[48..52], 42, .little);
    const message = (try dbus.Message.parse(&bytes)).?;
    var body = message.body;
    try std.testing.expectEqual(42, try body.uint32());
}

test "Unix addresses decode escaped bytes and distinguish abstract names" {
    const path = try dbus.Address.parse("unix:path=/tmp/bus%20name%2c,guid=0123456789abcdef0123456789abcdef");
    try std.testing.expectEqualStrings("/tmp/bus name,", std.mem.sliceTo(&path.value.path, 0));
    const abstract = try dbus.Address.parse("unix:abstract=voiced%2ftest");
    try std.testing.expectEqual(0, abstract.value.path[0]);
    try std.testing.expectEqualStrings("voiced/test", abstract.value.path[1..12]);
    for ([_][]const u8{ "unix:path=relative", "unix:path=/tmp/%00", "unix:path=/tmp/%", "unix:path=/tmp/%zz", "unix:path=/a,abstract=b", "unix:path=" }) |text|
        try std.testing.expectError(error.InvalidAddress, dbus.Address.parse(text));
    try std.testing.expectError(error.UnsupportedAddress, dbus.Address.parse("tcp:host=localhost,port=9999"));
}

test "bounded writer reports capacity errors" {
    var bytes: [16]u8 = undefined;
    try std.testing.expectError(error.BufferFull, dbus.Writer.call(&bytes, 1, "org.example.Test", "/test", "org.example.Test", "Test", ""));
}

test "malformed input cannot escape bounded decoding" {
    var random = std.Random.DefaultPrng.init(0x8a6f1707);
    var bytes: [256]u8 = undefined;
    for (0..20000) |iteration| {
        random.random().bytes(&bytes);
        const size = random.random().uintLessThan(usize, bytes.len + 1);
        if (iteration % 2 == 0 and size >= reply_little.len) {
            @memcpy(bytes[0..reply_little.len], &reply_little);
            const offset = random.random().uintLessThan(usize, reply_little.len);
            bytes[offset] = random.random().int(u8);
        }
        _ = dbus.Message.parse(bytes[0..size]) catch continue;
    }
}

test "nonblocking writes retain bytes across backpressure and report a closed peer" {
    const linux = std.os.linux;
    var sockets: [2]i32 = undefined;
    try std.testing.expectEqual(.SUCCESS, linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0, &sockets)));
    var connection: dbus.Connection = undefined;
    @memset(std.mem.asBytes(&connection), 0xa5);
    connection.initEmpty();
    connection.close();
    connection.initEmpty();
    connection.fd = sockets[0];
    defer connection.close();
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
    connection.output_size = 5;
    try std.testing.expectError(error.WriteFailed, connection.flush());
    try std.testing.expectEqual(.PIPE, connection.errno);
}
