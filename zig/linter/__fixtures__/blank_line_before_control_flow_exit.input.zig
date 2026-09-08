fn invalidReturn() usize {
    const result = compute();
    return result;
}

fn invalidBreak() void {
    while (true) {
        work();
        break;
    }
}

fn invalidContinue() void {
    while (true) {
        work();
        continue;
    }
}

fn invalidUnreachable() void {
    work();
    unreachable;
}

fn invalidLabeledBreak() usize {
    return result: {
        const value = compute();
        break :result value;
    };
}

fn invalidSwitchBlock(value: u8) void {
    switch (value) {
        0 => {
            work();
            return;
        },

        else => {},
    }
}

fn invalidAttachedComment() void {
    work();
    // The caller retries this operation.
    return;
}

fn validReturnParagraph() usize {
    const result = compute();

    return result;
}

fn validOnlyReturn() usize {
    return compute();
}

fn validFirstExitInBlock() void {
    return;

    work();
}

fn validOnlyStatementInBlock(condition: bool) void {
    if (condition) {
        return;
    }
}

fn validBracedIfBody(condition: bool) void {
    if (condition) {
        return;
    }
}

fn validBracedWhileBody(condition: bool) void {
    while (condition) {
        break;
    }
}

fn validBracedForBody(items: []const u8) void {
    for (items) |_| {
        continue;
    }
}

fn validDirectSwitchArm(value: u8) void {
    switch (value) {
        0 => return,

        else => {},
    }
}

fn validDirectUnreachableArm(value: u8) void {
    switch (value) {
        0 => unreachable,

        else => {},
    }
}

fn validOnlyLabeledBreak() usize {
    return result: {
        break :result compute();
    };
}

fn validAttachedComment() void {
    work();

    // The caller retries this operation.
    return;
}

fn validOnlyStatementWithComment() void {
    // No earlier statement requires paragraph separation.
    return;
}

fn compute() usize {
    return 1;
}

fn work() void {}
