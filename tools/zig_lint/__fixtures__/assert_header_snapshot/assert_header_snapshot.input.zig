const std = @import("std");
const assert = std.debug.assert;
const foo = @import("std");
const debug = std.debug;
const debug_assert = std.debug.assert;

fn ok() void {
    assert(true);
}

fn qualifiedCall() void {
    std.debug.assert(true);
}

fn aliasedDebug() void {
    debug.assert(true);
}

fn renamedStd() void {
    foo.debug.assert(true);
}

fn imported() void {
    @import("std").debug.assert(true);
}

fn comptimeQualified() void {
    comptime std.debug.assert(true);
}

fn otherAssert(self: anytype) void {
    self.assert();
}

fn commented() void {
    // std.debug.assert(true);
    _ = "std.debug.assert";
}

const Wrapper = struct {
    const assert = std.debug.assert;
};

const late = std.debug.assert;
