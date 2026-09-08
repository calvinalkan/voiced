fn branches(flag: bool) void {
    if (flag) {
        work();
    }
}

fn nested() !void {
    consume(try load());
}

fn work() void {}
fn consume(_: u8) void {}
fn load() !u8 {
    return 1;
}
fn eof() void {
    return;
}
