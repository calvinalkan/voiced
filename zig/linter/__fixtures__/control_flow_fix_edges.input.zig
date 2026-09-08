const std = @import("std");

// These cases also compile and execute as a standalone test, before and after
// fixing. Whitespace changes must preserve evaluation order and error paths.

fn choose(flag: bool) u8 {
    return if (flag) identity(1) else identity(2);
}

fn chooseChain(first: bool, second: bool) u8 {
    return if (first) identity(1) else if (second) identity(2) else identity(3);
}

fn chooseCaptured(optional: ?u8) u8 {
    return if (optional) |value| identity(value) else identity(4);
}

fn groupedExit(flag: bool) u8 {
    return if (flag) (return 5) else identity(6);
}

fn optionalExit(optional: ?u8) u8 {
    return optional orelse (return 7);
}

fn fallback(optional: ?u8) u8 {
    return optional orelse identity(8);
}

fn errorFallback(fail: bool) u8 {
    return attempt(fail) catch |err| recover(err);
}

fn errorExit(fail: bool) u8 {
    return attempt(fail) catch return 9;
}

fn optionalWork(optional: ?void, value: *u8) void {
    optional orelse increment(value);
}

fn errorWork(result: error{Failed}!void, value: *u8) void {
    result catch increment(value);
}

fn tryBody(flag: bool, value: *u8) error{Failed}!void {
    if (flag) try errorWorkBody(value);
}

fn errorWorkBody(value: *u8) error{Failed}!void {
    increment(value);

    return error.Failed;
}

fn simpleBranches(flag: bool, value: *u8) void {
    if (flag) value.* += 1 else value.* += 2;
}

fn loopBodies(value: *u8) void {
    while (value.* < 3) value.* += 1;
    for (0..2) |_| value.* += 1;
    inline for (0..2) |_| value.* += 1;
}

fn loopElse(value: *u8) void {
    while (false) value.* += 100 else value.* += 1;
    for ([_]u8{}) |_| value.* += 100 else value.* += 1;
}

fn loopExits(value: *u8) void {
    outer: for (0..3) |index| {
        if (index == 0) continue :outer;
        if (index == 2) break :outer;

        value.* += 1;
    }
}

fn captureHandler(value: *u8) error{Failed}!void {
    errdefer |err| value.* = recover(err);
    return error.Failed;
}

// Direct statement boundaries differ from deferred expressions: the trailing
// semicolon belongs to `defer`, not its nested `if`. Leave this one unbraced.
fn deferredConditional(flag: bool, value: *u8) void {
    defer if (flag) increment(value);
}

// Scope directives must not acquire a narrower block, even in an argument.
fn scopeDirective(comptime flag: bool) void {
    if (flag) @setRuntimeSafety(false);
}

fn nestedScopeDirective(comptime flag: bool) void {
    if (flag) consumeVoid(@setFloatMode(.strict));
}

fn consumeVoid(_: void) void {}

// Value-producing loop branches cannot become ordinary void blocks.
fn valueLoop() u8 {
    return while (false) identity(1) else identity(2);
}

// Comments at a brace boundary remain attached exactly where the author put them.
fn commentedBodies(flag: bool, value: *u8) void {
    if (flag) // Condition comment.
        increment(value);

    if (flag) increment(value); // Body comment.
}

// Multiline calls are deliberately outside the initial brace-fix policy.
fn multilineBody(flag: bool, value: *u8) void {
    if (flag) increment(
        value,
    );
}

fn commentedValue(flag: bool) u8 {
    return if (flag) identity(1) // Keep the then comment.
    else identity(2); // Keep the else comment.
}

fn nestedValue(first: bool, second: bool) u8 {
    return if (first) (if (second) identity(1) else identity(2)) else identity(3);
}

fn orderAndTry(value: *u8, fail: bool) error{Failed}!u8 {
    return step(value, 1) + try failingStep(value, 2, fail) + step(value, 3);
}

fn aggregateTry(fail: bool) error{Failed}!u8 {
    const result = .{ .value = try attempt(fail) };

    return result.value;
}

fn compactSwitch(flag: bool) u8 {
    switch (flag) {
        true => return 1,
        false => return 2,
    }
}

fn hiddenReturn() u8 { return 1; }
fn hiddenBreak() u8 { return result: { break :result 2; }; }
fn hiddenContinue() void { for (0..1) |_| { continue; } }
fn hiddenUnreachable() noreturn { unreachable; }

fn identity(value: u8) u8 {
    return value;
}

fn increment(value: *u8) void {
    value.* += 1;
}

fn attempt(fail: bool) error{Failed}!u8 {
    if (fail) {
        return error.Failed;
    }

    return 10;
}

fn recover(_: error{Failed}) u8 {
    return 11;
}

fn step(value: *u8, digit: u8) u8 {
    value.* = value.* * 10 + digit;

    return digit;
}

fn failingStep(value: *u8, digit: u8, fail: bool) error{Failed}!u8 {
    _ = step(value, digit);

    if (fail) {
        return error.Failed;
    }

    return digit;
}

test "control flow preserves values, effects, and cleanup" {
    scopeDirective(true);
    scopeDirective(false);
    nestedScopeDirective(true);
    nestedScopeDirective(false);
    try std.testing.expectEqual(@as(u8, 1), choose(true));
    try std.testing.expectEqual(@as(u8, 2), choose(false));
    try std.testing.expectEqual(@as(u8, 1), chooseChain(true, false));
    try std.testing.expectEqual(@as(u8, 2), chooseChain(false, true));
    try std.testing.expectEqual(@as(u8, 3), chooseChain(false, false));
    try std.testing.expectEqual(@as(u8, 4), chooseCaptured(null));
    try std.testing.expectEqual(@as(u8, 12), chooseCaptured(12));
    try std.testing.expectEqual(@as(u8, 5), groupedExit(true));
    try std.testing.expectEqual(@as(u8, 6), groupedExit(false));
    try std.testing.expectEqual(@as(u8, 7), optionalExit(null));
    try std.testing.expectEqual(@as(u8, 12), optionalExit(12));
    try std.testing.expectEqual(@as(u8, 8), fallback(null));
    try std.testing.expectEqual(@as(u8, 12), fallback(12));
    try std.testing.expectEqual(@as(u8, 11), errorFallback(true));
    try std.testing.expectEqual(@as(u8, 10), errorFallback(false));
    try std.testing.expectEqual(@as(u8, 9), errorExit(true));
    try std.testing.expectEqual(@as(u8, 10), errorExit(false));

    var value: u8 = 0;

    optionalWork(null, &value);
    optionalWork({}, &value);
    errorWork(error.Failed, &value);
    errorWork({}, &value);
    try std.testing.expectEqual(@as(u8, 2), value);
    try tryBody(false, &value);
    try std.testing.expectError(error.Failed, tryBody(true, &value));
    try std.testing.expectEqual(@as(u8, 3), value);

    value = 0;

    simpleBranches(true, &value);
    simpleBranches(false, &value);
    try std.testing.expectEqual(@as(u8, 3), value);

    value = 0;

    loopBodies(&value);
    try std.testing.expectEqual(@as(u8, 7), value);
    loopElse(&value);
    try std.testing.expectEqual(@as(u8, 9), value);
    loopExits(&value);
    try std.testing.expectEqual(@as(u8, 10), value);
    try std.testing.expectError(error.Failed, captureHandler(&value));
    try std.testing.expectEqual(@as(u8, 11), value);
    deferredConditional(true, &value);
    deferredConditional(false, &value);
    try std.testing.expectEqual(@as(u8, 12), value);
    try std.testing.expectEqual(@as(u8, 2), valueLoop());
    commentedBodies(true, &value);
    multilineBody(true, &value);
    try std.testing.expectEqual(@as(u8, 15), value);
    try std.testing.expectEqual(@as(u8, 1), commentedValue(true));
    try std.testing.expectEqual(@as(u8, 2), commentedValue(false));
    try std.testing.expectEqual(@as(u8, 1), nestedValue(true, true));
    try std.testing.expectEqual(@as(u8, 2), nestedValue(true, false));
    try std.testing.expectEqual(@as(u8, 3), nestedValue(false, false));

    value = 0;

    try std.testing.expectEqual(@as(u8, 6), try orderAndTry(&value, false));
    try std.testing.expectEqual(@as(u8, 123), value);

    value = 0;

    try std.testing.expectError(error.Failed, orderAndTry(&value, true));
    try std.testing.expectEqual(@as(u8, 12), value);
    try std.testing.expectEqual(@as(u8, 10), try aggregateTry(false));
    try std.testing.expectError(error.Failed, aggregateTry(true));
    try std.testing.expectEqual(@as(u8, 1), compactSwitch(true));
    try std.testing.expectEqual(@as(u8, 2), compactSwitch(false));
    try std.testing.expectEqual(@as(u8, 1), hiddenReturn());
    try std.testing.expectEqual(@as(u8, 2), hiddenBreak());
    hiddenContinue();
}
