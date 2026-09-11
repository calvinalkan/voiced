const std = @import("std");
const assert = std.debug.assert;

// This fixture also executes before and after fixing. Paragraph edits must
// preserve tokens, comment text, storage initialization, and cleanup behavior.

const Resource = struct {
    counter: *u8,

    fn deinit(resource: Resource) void {
        resource.counter.* += 1;
    }
};

fn cleanupGroup(counter: *u8) void {
    counter.* += 10;

    const resource: Resource = .{ .counter = counter }; // Setup comment.
    // Cleanup comment one.
    // Cleanup comment two. Keep both comments in order.
    defer resource.deinit(); // Registration comment.

    counter.* += 2;
}

fn cleanupChain(counter: *u8) error{Failed}!void {
    const resource: Resource = .{ .counter = counter };
    defer resource.deinit();
    errdefer resource.deinit();

    return error.Failed;
}

fn undefinedGroup() u8 {
    var first: u8 = undefined;
    // The second output slot.
    var second: u8 = undefined;
    // Initialize both before reading either.
    initialize(&first, &second);

    return first + second;
}

// Spacing can be repaired, but a missing reference must remain diagnostic-only.
fn missingFirstReference(counter: *u8) u8 {
    var value: u8 = undefined;
    increment(counter);

    value = 4;

    return value;
}

fn producerGuard(optional: ?u8) u8 {
    const value = optional;
    // A guard belongs to its producer even with an attached comment.
    if (value == null) {
        return 5;
    }

    return value orelse 0;
}

fn producerAssert() u8 {
    const value = compute();
    // The assertion belongs to the declaration.
    assert(value == 3);

    return value;
}

fn genericParagraphs(counter: *u8) u8 {
    increment(counter);

    const value = .{
        .first = @as(u8, 1),
        .second = @as(u8, 2),
    };

    consume(
        value.first,
        value.second,
    );

    counter.* += 1;

    {
        increment(counter);
    }

    if (counter.* != 0) {
        increment(counter);
    }

    var result: u8 = 0;

    result += value.first;
    result += value.second;

    // Optional separation within an assignment group stays untouched.
    result += 3;

    return result;
}

// A standalone terminator still belongs to its statement. Do not edit the
// empty line before it, or mistake that line for a following paragraph break.
fn standaloneSemicolon() u8 {
    const value = compute();

    consume(value, value);

    return value;
}

fn undefinedWithStandaloneSemicolon() u8 {
    var first: u8 = undefined;
    var second: u8 = undefined;
    initialize(&first, &second);

    return first + second;
}

fn switchComma(flag: bool, counter: *u8) void {
    switch (flag) {
        true => {
            increment(counter);
        },

        // Attached to the second arm, not the comma above.
        false => {
            counter.* += 2;
        },
    }
}

fn inlineCleanup(counter: *u8) void {
    const resource: Resource = .{ .counter = counter };
    defer resource.deinit();

    increment(counter);
}
fn inlineCategories(counter: *u8) u8 {
    increment(counter);

    const value: u8 = 3;
    return value;
}
fn inlineBlocks(flag: bool) void {
    switch (flag) {
        true => {},

        false => {},
    }
}

fn multilineString() []const u8 {
    const text =
        \\first
        \\
        \\// This is string content, not a comment or an empty physical line.
    ;

    consume(1, 2);

    return text;
}

fn initialize(first: *u8, second: *u8) void {
    first.* = 1;
    second.* = 2;
}

fn increment(counter: *u8) void {
    counter.* += 1;
}

fn compute() u8 {
    return 3;
}

fn consume(_: u8, _: u8) void {}

test "paragraph changes preserve execution and string contents" {
    var counter: u8 = 0;

    cleanupGroup(&counter);
    try std.testing.expectEqual(@as(u8, 13), counter);
    try std.testing.expectError(error.Failed, cleanupChain(&counter));
    try std.testing.expectEqual(@as(u8, 15), counter);
    try std.testing.expectEqual(@as(u8, 3), undefinedGroup());
    try std.testing.expectEqual(@as(u8, 4), missingFirstReference(&counter));
    try std.testing.expectEqual(@as(u8, 16), counter);
    try std.testing.expectEqual(@as(u8, 5), producerGuard(null));
    try std.testing.expectEqual(@as(u8, 6), producerGuard(6));
    try std.testing.expectEqual(@as(u8, 3), producerAssert());
    try std.testing.expectEqual(@as(u8, 6), genericParagraphs(&counter));
    try std.testing.expectEqual(@as(u8, 20), counter);
    try std.testing.expectEqual(@as(u8, 3), standaloneSemicolon());
    try std.testing.expectEqual(@as(u8, 3), undefinedWithStandaloneSemicolon());
    switchComma(true, &counter);
    switchComma(false, &counter);
    try std.testing.expectEqual(@as(u8, 23), counter);
    inlineCleanup(&counter);
    try std.testing.expectEqual(@as(u8, 25), counter);
    try std.testing.expectEqual(@as(u8, 3), inlineCategories(&counter));
    try std.testing.expectEqual(@as(u8, 26), counter);
    inlineBlocks(true);
    inlineBlocks(false);

    try std.testing.expectEqualStrings(
        "first\n\n// This is string content, not a comment or an empty physical line.",
        multilineString(),
    );
}
