fn implicitUnwrap(opt: ?i32) i32 {
    return opt.?;
}

fn implicitGuardedUnwrap(opt: ?i32) i32 {
    if (opt != null) {
        return opt.?;
    }

    return 0;
}

fn implicitNestedUnwrap(opt: ??i32) i32 {
    return opt.?.?;
}

fn explicitCapture(opt: ?i32) i32 {
    if (opt) |value| {
        return value;
    }

    return 0;
}

fn explicitDefault(opt: ?i32) i32 {
    return opt orelse 0;
}

fn explicitUnreachable(opt: ?i32) i32 {
    return opt orelse {
        unreachable;
    };
}

fn explicitEarlyExit(opt: ?i32) !i32 {
    const value = opt orelse {
        return error.MissingValue;
    };

    return value;
}

fn ignoredText() []const u8 {
    // opt.? is forbidden syntax, not forbidden comment text.
    return "opt.?";
}
