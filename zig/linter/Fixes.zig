//! File-scoped text edits against one immutable source snapshot.
//!
//! rule -> ProposedEdit -> StoredEdit + shared replacement bytes
//! apply -> select non-overlapping edits -> allocate once -> copy spans forward
//!
//! Spike: one edit per fix. No multi-edit groups or filesystem writes yet.

const Fixes = @This();

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

allocator: Allocator,
source_text: [:0]const u8 = "",
text_edits: std.ArrayList(StoredEdit) = .empty,
replacement_bytes: std.ArrayList(u8) = .empty,

pub const ProposedEdit = struct {
    source_start_offset: u32,
    removal_size: u32,
    replacement: []const u8,
};

const StoredEdit = struct {
    source_start_offset: u32,
    removal_size: u32,
    replacement_start_offset: u32,
    replacement_size: u32,
};

comptime {
    assert(@sizeOf(StoredEdit) == 16);
}

pub const ApplyResult = struct {
    /// Null means unchanged. Otherwise the caller frees this sentinel slice
    /// with the output allocator, after re-linting or writing it.
    updated_source: ?[:0]u8 = null,

    applied_fix_count: usize = 0,
    skipped_fix_count: usize = 0,
};

pub fn init(allocator: Allocator) Fixes {
    return .{ .allocator = allocator };
}

pub fn deinit(fixes: *Fixes) void {
    fixes.text_edits.deinit(fixes.allocator);
    fixes.replacement_bytes.deinit(fixes.allocator);
}

/// Retain buffer capacities and borrow the new source. Keep that source alive
/// and immutable until applying or resetting this plan.
pub fn reset(fixes: *Fixes, source_text: [:0]const u8) void {
    fixes.text_edits.clearRetainingCapacity();
    fixes.replacement_bytes.clearRetainingCapacity();

    fixes.source_text = source_text;
}

pub fn count(fixes: *const Fixes) usize {
    return fixes.text_edits.items.len;
}

/// Copy the proposed replacement; the caller's slice need only survive this
/// call. Source ranges are validated by apply before any output is allocated.
pub fn add(fixes: *Fixes, proposed: ProposedEdit) Allocator.Error!void {
    const replacement_start_offset = fixes.replacement_bytes.items.len;
    if (fixes.text_edits.items.len == std.math.maxInt(u32) or
        proposed.replacement.len > std.math.maxInt(u32) - replacement_start_offset)
    {
        // Compact offsets must not truncate. Treat unrepresentable buffer sizes
        // like other allocation-size overflows rather than publishing a bad edit.
        return error.OutOfMemory;
    }

    // Reserve the edit slot first; publish it only after the replacement copy
    // succeeds. A failed allocation cannot leave a partially recorded edit.
    try fixes.text_edits.ensureUnusedCapacity(fixes.allocator, 1);
    try fixes.replacement_bytes.appendSlice(fixes.allocator, proposed.replacement);

    fixes.text_edits.appendAssumeCapacity(.{
        .source_start_offset = proposed.source_start_offset,
        .removal_size = proposed.removal_size,
        .replacement_start_offset = @intCast(replacement_start_offset),
        .replacement_size = @intCast(proposed.replacement.len),
    });
}

/// Apply only a plan from a successful lint pass. Selection is in source order;
/// emission order breaks ties. Later overlapping fixes are skipped, including
/// insertions at the same offset. Adjacent non-overlapping edits are allowed.
pub fn apply(
    fixes: *const Fixes,
    output_allocator: Allocator,
    file_size_max: u32,
) (Allocator.Error || error{ InvalidEdit, FileTooLarge })!ApplyResult {
    const source_text = fixes.source_text;
    const source_text_size = source_text.len;

    const text_edits: []const StoredEdit = fixes.text_edits.items;
    const text_edits_count = text_edits.len;

    var result: ApplyResult = .{};

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
        if (edit.source_start_offset > source_text_size or
            edit.removal_size > source_text_size - edit.source_start_offset)
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
        const source_start_offset: usize = edit.source_start_offset;

        if (result.applied_fix_count != 0 and
            (source_start_offset < previous_end_offset or source_start_offset == previous_start_offset))
        {
            result.skipped_fix_count += 1;

            continue;
        }

        edit_order[result.applied_fix_count] = edit_index;
        result.applied_fix_count += 1;
        previous_start_offset = source_start_offset;
        previous_end_offset = source_start_offset + edit.removal_size;
        output_size = output_size - edit.removal_size + edit.replacement_size;
    }

    if (output_size > file_size_max) {
        return error.FileTooLarge;
    }

    // ── Build Output ──
    //
    // Copy each unchanged source span once. The extra byte is the sentinel
    // required by lint_file's [:0]const u8 source, not part of the output text.

    const allocation_size = std.math.add(usize, @intCast(output_size), 1) catch {
        return error.OutOfMemory;
    };

    const output_buffer = try output_allocator.alloc(u8, allocation_size);
    errdefer output_allocator.free(output_buffer);

    var source_offset: usize = 0;
    var output_offset: usize = 0;

    for (edit_order[0..result.applied_fix_count]) |edit_index| {
        const edit = text_edits[edit_index];
        const unchanged = source_text[source_offset..edit.source_start_offset];
        const replacement = fixes.replacement_bytes.items[edit.replacement_start_offset..][0..edit.replacement_size];

        @memcpy(output_buffer[output_offset..][0..unchanged.len], unchanged);

        output_offset += unchanged.len;

        @memcpy(output_buffer[output_offset..][0..replacement.len], replacement);

        output_offset += replacement.len;
        source_offset = @as(usize, edit.source_start_offset) + edit.removal_size;
    }

    const remaining_source = source_text[source_offset..];

    @memcpy(output_buffer[output_offset..][0..remaining_source.len], remaining_source);

    output_offset += remaining_source.len;

    assert(output_offset == output_size);

    output_buffer[output_offset] = 0;

    if (std.mem.eql(u8, source_text, output_buffer[0..output_offset])) {
        output_allocator.free(output_buffer);

        return result;
    }

    result.updated_source = output_buffer[0..output_offset :0];

    return result;
}

fn applySortSourceOrder(text_edits: []const StoredEdit, left_index: u32, right_index: u32) bool {
    const left_offset = text_edits[left_index].source_start_offset;
    const right_offset = text_edits[right_index].source_start_offset;

    if (left_offset != right_offset) {
        return left_offset < right_offset;
    }

    return left_index < right_index;
}
