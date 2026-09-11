const Fixes = @This();

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Ast = std.zig.Ast;
const Rule = @import("Rule.zig");

allocator: Allocator,
source_text: [:0]const u8 = "",
text_edits: std.ArrayList(Rule.Fix) = .empty,

comptime {
    assert(@sizeOf(Rule.Fix) <= 24);
}

pub fn init(allocator: Allocator) Fixes {
    return .{ .allocator = allocator };
}

pub fn deinit(fixes: *Fixes) void {
    fixes.text_edits.deinit(fixes.allocator);
}

pub fn reset(fixes: *Fixes, source_text: [:0]const u8) void {
    fixes.text_edits.clearRetainingCapacity();

    fixes.source_text = source_text;
}

pub fn count(fixes: *const Fixes) usize {
    return fixes.text_edits.items.len;
}

/// `add` borrows `proposed.replacement` until the plan is applied or reset.
pub fn add(fixes: *Fixes, proposed: Rule.Fix) Allocator.Error!void {
    if (proposed.range.start_offset <= proposed.range.end_offset and
        proposed.range.end_offset <= fixes.source_text.len and
        std.mem.eql(
            u8,
            fixes.source_text[proposed.range.start_offset..proposed.range.end_offset],
            proposed.replacement,
        ))
    {
        return;
    }

    if (fixes.text_edits.items.len == std.math.maxInt(u32)) {
        return error.OutOfMemory;
    }

    try fixes.text_edits.append(fixes.allocator, proposed);
}

pub const ApplyResult = struct {
    /// Null means unchanged. Otherwise the caller frees this slice with the
    /// allocator supplied to `init`.
    updated_source: ?[]u8 = null,

    applied_fix_count: usize = 0,
    skipped_fix_count: usize = 0,
};

/// `applyAndFormat` applies compatible edits and returns canonical Zig source
/// when it differs from `ast.source`. `ast` must be valid and must borrow the
/// source most recently supplied to `reset`.
pub fn applyAndFormat(
    fixes: *const Fixes,
    ast: Ast,
    file_size_max: u32,
) (Allocator.Error || error{ InvalidEdit, InvalidSource, FileTooLarge })!ApplyResult {
    assert(ast.errors.len == 0);
    assert(ast.source.ptr == fixes.source_text.ptr and ast.source.len == fixes.source_text.len);

    const applied = try fixes.applyEdits(file_size_max);
    defer if (applied.updated_source) |text| {
        fixes.allocator.free(text);
    };

    const formatted = if (applied.updated_source) |updated_source| formatted: {
        var updated_ast = try Ast.parse(fixes.allocator, updated_source, .zig);
        defer updated_ast.deinit(fixes.allocator);

        if (updated_ast.errors.len != 0) {
            return error.InvalidSource;
        }

        break :formatted try updated_ast.renderAlloc(fixes.allocator);
    } else try ast.renderAlloc(fixes.allocator);

    if (formatted.len > file_size_max) {
        fixes.allocator.free(formatted);

        return error.FileTooLarge;
    }

    if (std.mem.eql(u8, fixes.source_text, formatted)) {
        fixes.allocator.free(formatted);

        return .{
            .applied_fix_count = applied.applied_fix_count,
            .skipped_fix_count = applied.skipped_fix_count,
        };
    }

    return .{
        .updated_source = formatted,
        .applied_fix_count = applied.applied_fix_count,
        .skipped_fix_count = applied.skipped_fix_count,
    };
}

const EditResult = struct {
    updated_source: ?[:0]u8 = null,
    applied_fix_count: usize = 0,
    skipped_fix_count: usize = 0,
};

fn applyEdits(
    fixes: *const Fixes,
    file_size_max: u32,
) (Allocator.Error || error{ InvalidEdit, FileTooLarge })!EditResult {
    const source_text = fixes.source_text;
    const source_text_size = source_text.len;

    const text_edits: []const Rule.Fix = fixes.text_edits.items;
    const text_edits_count = text_edits.len;

    var result: EditResult = .{};

    if (source_text_size > file_size_max) {
        return error.FileTooLarge;
    }

    if (text_edits_count == 0) {
        return result;
    }

    // ── Validate And Order ──
    //
    // Sort indexes, not stored edits: emission order remains available as a
    // deterministic tie-breaker, and the same plan can be applied again.

    const edit_order = try fixes.allocator.alloc(u32, text_edits_count);
    defer fixes.allocator.free(edit_order);

    for (text_edits, 0..) |edit, edit_index| {
        if (edit.range.start_offset > edit.range.end_offset or
            edit.range.end_offset > source_text_size)
        {
            return error.InvalidEdit;
        }

        edit_order[edit_index] = @intCast(edit_index);
    }

    std.mem.sort(u32, edit_order, text_edits, applySortSourceOrder);

    // ── Select And Measure ──
    //
    // Accepted indexes compact into the front of edit_order. One previous
    // accepted range is enough because candidates arrive in source order.

    var previous_start_offset: usize = 0;
    var previous_end_offset: usize = 0;
    var output_size: u64 = source_text_size;

    for (edit_order) |edit_index| {
        const edit = text_edits[edit_index];
        const start_offset: usize = edit.range.start_offset;
        const end_offset: usize = edit.range.end_offset;

        if (result.applied_fix_count != 0 and
            (start_offset < previous_end_offset or start_offset == previous_start_offset))
        {
            result.skipped_fix_count += 1;

            continue;
        }

        edit_order[result.applied_fix_count] = edit_index;
        result.applied_fix_count += 1;
        previous_start_offset = start_offset;
        previous_end_offset = end_offset;
        output_size = output_size - (end_offset - start_offset) + edit.replacement.len;
    }

    if (output_size > file_size_max) {
        return error.FileTooLarge;
    }

    // ── Build Output ──
    //
    // Copy each unchanged source span once. The extra byte is the sentinel
    // required by `Ast.parse`'s [:0]const u8 source, not part of the output text.

    const allocation_size = std.math.add(usize, @intCast(output_size), 1) catch {
        return error.OutOfMemory;
    };

    const output_buffer = try fixes.allocator.alloc(u8, allocation_size);
    errdefer fixes.allocator.free(output_buffer);

    var source_offset: usize = 0;
    var output_offset: usize = 0;

    for (edit_order[0..result.applied_fix_count]) |edit_index| {
        const edit = text_edits[edit_index];
        const unchanged = source_text[source_offset..edit.range.start_offset];
        const replacement = edit.replacement;

        @memcpy(output_buffer[output_offset..][0..unchanged.len], unchanged);

        output_offset += unchanged.len;

        @memcpy(output_buffer[output_offset..][0..replacement.len], replacement);

        output_offset += replacement.len;
        source_offset = edit.range.end_offset;
    }

    const remaining_source = source_text[source_offset..];

    @memcpy(output_buffer[output_offset..][0..remaining_source.len], remaining_source);

    output_offset += remaining_source.len;

    assert(output_offset == output_size);

    output_buffer[output_offset] = 0;

    if (std.mem.eql(u8, source_text, output_buffer[0..output_offset])) {
        fixes.allocator.free(output_buffer);

        return result;
    }

    result.updated_source = output_buffer[0..output_offset :0];

    return result;
}

fn applySortSourceOrder(text_edits: []const Rule.Fix, left_index: u32, right_index: u32) bool {
    const left_offset = text_edits[left_index].range.start_offset;
    const right_offset = text_edits[right_index].range.start_offset;

    if (left_offset != right_offset) {
        return left_offset < right_offset;
    }

    return left_index < right_index;
}

// ─── Table Tests ───────────────────────────────────────────────────────────────────

test "raw fixes apply one insertion, deletion, or replacement" {
    const Case = struct {
        name: []const u8,
        source: [:0]const u8,
        edit: Rule.Fix,
        expected: []const u8,
    };

    const cases = [_]Case{
        .{
            .name = "insertion",
            .source = "ab",
            .edit = .{
                .range = .{ .start_offset = 1, .end_offset = 1 },
                .replacement = "_",
            },
            .expected = "a_b",
        },
        .{
            .name = "deletion",
            .source = "abc",
            .edit = .{
                .range = .{ .start_offset = 1, .end_offset = 2 },
                .replacement = "",
            },
            .expected = "ac",
        },
        .{
            .name = "replacement",
            .source = "abc",
            .edit = .{
                .range = .{ .start_offset = 1, .end_offset = 2 },
                .replacement = "B",
            },
            .expected = "aBc",
        },
    };

    for (cases) |case| {
        errdefer std.debug.print("raw fix case: {s}\n", .{case.name});

        try expectRawFixResult(case.source, &.{case.edit}, case.expected, 1, 0);
    }
}

test "raw fixes apply non-overlapping edits in source order" {
    // The later source edit is deliberately proposed first.
    try expectRawFixResult(
        "abcd",
        &.{
            .{
                .range = .{ .start_offset = 3, .end_offset = 4 },
                .replacement = "D",
            },
            .{
                .range = .{ .start_offset = 1, .end_offset = 2 },
                .replacement = "B",
            },
        },
        "aBcD",
        2,
        0,
    );
}

test "raw fix conflicts prefer the earliest range then emission order" {
    const Case = struct {
        name: []const u8,
        source: [:0]const u8,
        edits: []const Rule.Fix,
        expected: []const u8,
    };

    const cases = [_]Case{
        .{
            .name = "earliest overlapping range",
            .source = "abcdef",
            .edits = &.{
                .{
                    .range = .{ .start_offset = 2, .end_offset = 5 },
                    .replacement = "[later]",
                },
                .{
                    .range = .{ .start_offset = 1, .end_offset = 4 },
                    .replacement = "[earlier]",
                },
            },
            .expected = "a[earlier]ef",
        },
        .{
            .name = "first insertion at a shared offset",
            .source = "ab",
            .edits = &.{
                .{
                    .range = .{ .start_offset = 1, .end_offset = 1 },
                    .replacement = "X",
                },
                .{
                    .range = .{ .start_offset = 1, .end_offset = 1 },
                    .replacement = "Y",
                },
            },
            .expected = "aXb",
        },
    };

    for (cases) |case| {
        errdefer std.debug.print("raw fix conflict case: {s}\n", .{case.name});

        try expectRawFixResult(case.source, case.edits, case.expected, 1, 1);
    }
}

test "raw fixes reject invalid ranges and oversized output" {
    const allocator = std.testing.allocator;

    var fixes = Fixes.init(allocator);
    defer fixes.deinit();

    fixes.reset("abc");

    try fixes.add(.{
        .range = .{ .start_offset = 2, .end_offset = 6 },
        .replacement = "",
    });

    try std.testing.expectError(error.InvalidEdit, fixes.applyEdits(100));

    fixes.reset("abc");

    try fixes.add(.{
        .range = .{ .start_offset = 3, .end_offset = 3 },
        .replacement = "d",
    });

    try std.testing.expectError(error.FileTooLarge, fixes.applyEdits(3));
}

test "formatted apply combines targeted fixes with Zig fmt" {
    const allocator = std.testing.allocator;
    const source: [:0]const u8 = "fn value()u8{return 1;}\n";

    const expected =
        \\fn value() u8 {
        \\    return 2;
        \\}
        \\
    ;

    var fixes = Fixes.init(allocator);
    defer fixes.deinit();

    fixes.reset(source);

    const digit_offset = std.mem.indexOfScalar(u8, source, '1') orelse {
        unreachable;
    };

    try fixes.add(.{
        .range = .{
            .start_offset = @intCast(digit_offset),
            .end_offset = @intCast(digit_offset + 1),
        },
        .replacement = "2",
    });

    var ast = try Ast.parse(allocator, source, .zig);
    defer ast.deinit(allocator);

    const result = try fixes.applyAndFormat(ast, 100);
    defer if (result.updated_source) |text| {
        allocator.free(text);
    };

    try std.testing.expectEqualStrings(expected, result.updated_source orelse {
        return error.MissingFixedSource;
    });

    try std.testing.expectEqual(@as(usize, 1), result.applied_fix_count);
}

fn expectRawFixResult(
    source: [:0]const u8,
    edits: []const Rule.Fix,
    expected_source: []const u8,
    expected_applied_fix_count: usize,
    expected_skipped_fix_count: usize,
) !void {
    const allocator = std.testing.allocator;

    var fixes = Fixes.init(allocator);
    defer fixes.deinit();

    fixes.reset(source);

    for (edits) |edit| {
        try fixes.add(edit);
    }

    const result = try fixes.applyEdits(100);
    defer if (result.updated_source) |text| {
        allocator.free(text);
    };

    try std.testing.expectEqualStrings(expected_source, result.updated_source orelse {
        return error.MissingFixedSource;
    });

    try std.testing.expectEqual(expected_applied_fix_count, result.applied_fix_count);
    try std.testing.expectEqual(expected_skipped_fix_count, result.skipped_fix_count);
}
