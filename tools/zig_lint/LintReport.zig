//! `Builder` reserves bounded diagnostic storage before lint work begins and
//! copies each retained finding into one fixed-width text slot. Fixed-file
//! output remains dynamically sized and is published only as a complete set.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Rule = @import("Rule.zig");

/// `Diagnostic` describes one finding in a source file.
///
/// Every slice points into its report's packed text storage. Keep the containing
/// `LintReport` alive and unchanged while inspecting a diagnostic.
pub const Diagnostic = struct {
    /// `path` uses the `ReportPathFormat` selected for the lint operation.
    path: []const u8,

    /// `rule_name` is the stable machine-readable identity of the rule.
    rule_name: []const u8,

    message: []const u8,

    /// Null means the finding has no concise corrective action.
    help: ?[]const u8 = null,

    /// Null means the finding needs no additional context.
    note: ?[]const u8 = null,

    location: Location,

    /// `source_line` excludes its line terminator. It is empty for a finding
    /// that has no source excerpt.
    source_line: []const u8 = &.{},

    /// `Location` identifies the diagnostic span in both display and byte
    /// coordinates.
    pub const Location = struct {
        /// `line` and `column` are one-based source coordinates.
        line: usize,
        column: usize,

        /// `range` uses zero-based byte offsets within the complete source.
        range: Rule.SourceRange,
    };
};

const diagnostic_storage_alignment = @max(@alignOf(Diagnostic), @alignOf(u32));

/// `LintReport` owns the immutable output of one lint operation.
///
/// Callers must not mutate diagnostics, fixed-file rows, or their backing
/// allocations. `deinit` releases all storage through the allocator supplied as
/// `Allocators.report`. Copying a report does not copy its allocations; only one
/// copy may own and deinitialize the storage. A caller using an arena may
/// instead release the entire arena after the last report use because reports
/// own no non-memory resources.
pub const LintReport = struct {
    /// `FixedFile` contains canonical text that differs from one input file.
    /// The linter never writes it; both slices remain valid for the report
    /// lifetime.
    pub const FixedFile = struct {
        /// `path` uses the same report-path format as `Diagnostic.path`.
        path: []const u8,

        /// `text` contains sentinel-terminated valid Zig source after compatible
        /// edits and canonical formatting.
        text: [:0]const u8,
    };

    /// `FixedFileList` stores paths and canonical texts as parallel columns.
    pub const FixedFileList = std.MultiArrayList(FixedFile).Slice;

    /// `diagnostics` contains the deterministic top-K retained findings in
    /// ascending report order.
    diagnostics: []const Diagnostic,

    /// `total_diagnostic_count` includes retained and omitted findings.
    total_diagnostic_count: usize,

    /// `oversized_diagnostic_count` counts findings whose combined copied text
    /// exceeded `LintOptions.diagnostic_text_size_max`.
    oversized_diagnostic_count: usize,

    /// `excess_diagnostic_count` counts fitting findings excluded by
    /// `LintOptions.diagnostics_count_max`.
    excess_diagnostic_count: usize,

    /// `fixed_files` is ordered by path and contains only changed files.
    fixed_files: FixedFileList,

    /// These totals cover every file in the operation.
    applied_fix_count: usize,
    skipped_fix_count: usize,

    diagnostic_storage: []align(diagnostic_storage_alignment) u8,
    fixed_text_storage: []const u8,

    /// `deinit` releases the report through the same allocator supplied as
    /// `Allocators.report` and invalidates every slice in the report.
    pub fn deinit(report: *LintReport, allocator: Allocator) void {
        report.fixed_files.deinit(allocator);
        allocator.free(report.fixed_text_storage);
        allocator.free(report.diagnostic_storage);

        report.* = undefined;
    }

    /// `renderHuman` returns deterministic terminal-safe text without ANSI
    /// escapes. The caller owns the returned slice.
    pub fn renderHuman(report: *const LintReport, output_allocator: Allocator) Allocator.Error![]u8 {
        var text: std.ArrayList(u8) = .empty;
        errdefer text.deinit(output_allocator);

        for (0..report.diagnostics.len) |diagnostic_index| {
            const diagnostic = report.diagnostics[diagnostic_index];

            if (diagnostic_index != 0) {
                try text.append(output_allocator, '\n');
            }

            try text.appendSlice(output_allocator, "error[");
            try appendInline(&text, output_allocator, diagnostic.rule_name);
            try text.appendSlice(output_allocator, "]: ");
            try appendInline(&text, output_allocator, diagnostic.message);
            try text.append(output_allocator, '\n');

            const margin_width = decimalWidth(diagnostic.location.line);

            try appendSpaces(&text, output_allocator, margin_width);
            try text.appendSlice(output_allocator, "--> ");
            try appendInline(&text, output_allocator, diagnostic.path);

            try text.print(output_allocator, ":{d}:{d}\n", .{
                diagnostic.location.line,
                diagnostic.location.column,
            });

            if (diagnostic.source_line.len != 0) {
                try appendGutter(&text, output_allocator, margin_width, "|");
                try text.append(output_allocator, '\n');
                try text.print(output_allocator, "{d} | ", .{diagnostic.location.line});
                try appendSource(&text, output_allocator, diagnostic.source_line);
                try text.append(output_allocator, '\n');

                try appendGutter(&text, output_allocator, margin_width, "| ");

                const byte_column = @min(diagnostic.location.column -| 1, diagnostic.source_line.len);
                const source_span_size = diagnostic.location.range.end_offset - diagnostic.location.range.start_offset;
                const span_end = @min(byte_column + @max(source_span_size, 1), diagnostic.source_line.len);
                const caret_column = sourceDisplayColumn(diagnostic.source_line[0..byte_column], 0);
                const caret_end = sourceDisplayColumn(diagnostic.source_line[byte_column..span_end], caret_column);

                try appendSpaces(&text, output_allocator, caret_column);
                try appendCarets(&text, output_allocator, @max(caret_end - caret_column, 1));
                try text.append(output_allocator, '\n');

                if (diagnostic.help != null or diagnostic.note != null) {
                    try appendGutter(&text, output_allocator, margin_width, "|");
                    try text.append(output_allocator, '\n');
                }
            }

            if (diagnostic.help) |help| {
                try appendGutter(&text, output_allocator, margin_width, "= help: ");
                try appendInline(&text, output_allocator, help);
                try text.append(output_allocator, '\n');
            }

            if (diagnostic.note) |note| {
                try appendGutter(&text, output_allocator, margin_width, "= note: ");
                try appendInline(&text, output_allocator, note);
                try text.append(output_allocator, '\n');
            }
        }

        if (report.oversized_diagnostic_count != 0 or report.excess_diagnostic_count != 0) {
            if (report.diagnostics.len != 0) {
                try text.append(output_allocator, '\n');
            }

            try text.print(output_allocator, "note: retained {d} of {d} diagnostics", .{
                report.diagnostics.len,
                report.total_diagnostic_count,
            });

            if (report.excess_diagnostic_count != 0) {
                try text.print(output_allocator, "; {d} exceeded `diagnostics_count_max`", .{
                    report.excess_diagnostic_count,
                });
            }

            if (report.oversized_diagnostic_count != 0) {
                try text.print(output_allocator, "; {d} exceeded `diagnostic_text_size_max`", .{
                    report.oversized_diagnostic_count,
                });
            }

            try text.append(output_allocator, '\n');
        }

        return text.toOwnedSlice(output_allocator);
    }

    /// `renderAgent` returns one self-contained line for each distinct problem,
    /// aggregating the adjacent locations ordered together in `diagnostics`.
    /// When findings were omitted, the final line reports every omission
    /// counter. The caller owns the returned slice.
    pub fn renderAgent(report: *const LintReport, output_allocator: Allocator) Allocator.Error![]u8 {
        var text: std.ArrayList(u8) = .empty;
        errdefer text.deinit(output_allocator);

        var group_start_index: usize = 0;

        while (group_start_index < report.diagnostics.len) {
            const description = report.diagnostics[group_start_index];
            var group_end_index = group_start_index + 1;

            while (group_end_index < report.diagnostics.len and
                sameAgentGroup(description, report.diagnostics[group_end_index]))
            {
                group_end_index += 1;
            }

            try appendInline(&text, output_allocator, description.path);
            try text.appendSlice(output_allocator, " [");
            try appendInline(&text, output_allocator, description.rule_name);
            try text.appendSlice(output_allocator, "] ");

            for (group_start_index..group_end_index, 0..) |diagnostic_index, location_index| {
                const diagnostic = report.diagnostics[diagnostic_index];

                if (location_index != 0) {
                    try text.append(output_allocator, ',');
                }

                try text.print(output_allocator, "{d}:{d}", .{
                    diagnostic.location.line,
                    diagnostic.location.column,
                });
            }

            try text.appendSlice(output_allocator, " | ");
            try appendInline(&text, output_allocator, description.message);

            if (description.help) |help| {
                try text.appendSlice(output_allocator, " | fix: ");
                try appendInline(&text, output_allocator, help);
            }

            if (description.note) |note| {
                try text.appendSlice(output_allocator, " | note: ");
                try appendInline(&text, output_allocator, note);
            }

            try text.append(output_allocator, '\n');

            group_start_index = group_end_index;
        }

        if (report.oversized_diagnostic_count != 0 or report.excess_diagnostic_count != 0) {
            try text.print(
                output_allocator,
                "[diagnostic-summary] retained={d} total={d} excess={d} oversized={d}\n",
                .{
                    report.diagnostics.len,
                    report.total_diagnostic_count,
                    report.excess_diagnostic_count,
                    report.oversized_diagnostic_count,
                },
            );
        }

        return text.toOwnedSlice(output_allocator);
    }
};

// ─── Report Construction ─────────────────────────────────────────────────────

/// `Builder` is the internal run-wide sink shared by per-file lint contexts.
/// Initialization reserves complete diagnostic capacity from the report
/// allocator. Diagnostic reporting allocates no memory afterward.
pub const Builder = struct {
    pub const Limits = struct {
        diagnostics_count_max: u32 = 500,
        diagnostic_text_size_max: u32 = 1024,
    };

    scratch_allocator: Allocator,
    report_allocator: Allocator,

    diagnostic_storage: []align(diagnostic_storage_alignment) u8,
    diagnostics: []Diagnostic,
    diagnostic_heap: []u32,
    diagnostic_text_slots: []u8,
    diagnostic_text_size_max: usize,
    retained_diagnostic_count: usize = 0,
    total_diagnostic_count: usize = 0,
    oversized_diagnostic_count: usize = 0,
    excess_diagnostic_count: usize = 0,

    fixed_files: std.MultiArrayList(PendingFixedFile) = .empty,
    fixed_text: std.ArrayList(u8) = .empty,
    last_fixed_file_path: ?TextRange = null,
    applied_fix_count: usize = 0,
    skipped_fix_count: usize = 0,

    pub fn init(
        scratch_allocator: Allocator,
        report_allocator: Allocator,
        limits: Limits,
    ) Allocator.Error!Builder {
        const diagnostic_capacity: usize = limits.diagnostics_count_max;

        const diagnostics_size = std.math.mul(
            usize,
            diagnostic_capacity,
            @sizeOf(Diagnostic),
        ) catch {
            return error.OutOfMemory;
        };

        const diagnostic_heap_offset = std.mem.alignForward(
            usize,
            diagnostics_size,
            @alignOf(u32),
        );

        const diagnostic_heap_size = std.math.mul(
            usize,
            diagnostic_capacity,
            @sizeOf(u32),
        ) catch {
            return error.OutOfMemory;
        };

        const diagnostic_heap_end = std.math.add(
            usize,
            diagnostic_heap_offset,
            diagnostic_heap_size,
        ) catch {
            return error.OutOfMemory;
        };

        const diagnostic_text_slots_size = std.math.mul(
            usize,
            diagnostic_capacity,
            limits.diagnostic_text_size_max,
        ) catch {
            return error.OutOfMemory;
        };

        const diagnostic_storage_size = std.math.add(
            usize,
            diagnostic_heap_end,
            diagnostic_text_slots_size,
        ) catch {
            return error.OutOfMemory;
        };

        const diagnostic_storage = try report_allocator.alignedAlloc(
            u8,
            .fromByteUnits(diagnostic_storage_alignment),
            diagnostic_storage_size,
        );

        return .{
            .scratch_allocator = scratch_allocator,
            .report_allocator = report_allocator,
            .diagnostic_storage = diagnostic_storage,
            .diagnostics = std.mem.bytesAsSlice(
                Diagnostic,
                diagnostic_storage[0..diagnostics_size],
            ),
            .diagnostic_heap = std.mem.bytesAsSlice(
                u32,
                @as(
                    []align(@alignOf(u32)) u8,
                    @alignCast(diagnostic_storage[diagnostic_heap_offset..diagnostic_heap_end]),
                ),
            ),
            .diagnostic_text_slots = diagnostic_storage[diagnostic_heap_end..],
            .diagnostic_text_size_max = limits.diagnostic_text_size_max,
        };
    }

    pub fn deinit(builder: *Builder) void {
        builder.fixed_files.deinit(builder.scratch_allocator);
        builder.fixed_text.deinit(builder.scratch_allocator);
        builder.report_allocator.free(builder.diagnostic_storage);

        builder.* = undefined;
    }

    pub fn totalDiagnosticCount(builder: *const Builder) usize {
        return builder.total_diagnostic_count;
    }

    /// `addDiagnostic` counts every finding and copies only retained findings
    /// into their permanent report slots. The caller may release every supplied
    /// slice when this function returns.
    pub fn addDiagnostic(builder: *Builder, diagnostic: Diagnostic) void {
        var candidate = diagnostic;

        candidate.source_line = std.mem.trimEnd(u8, candidate.source_line, "\r");

        builder.total_diagnostic_count += 1;

        const text_size = diagnosticTextSize(candidate) orelse {
            builder.oversized_diagnostic_count += 1;

            return;
        };

        if (text_size > builder.diagnostic_text_size_max) {
            builder.oversized_diagnostic_count += 1;

            return;
        }

        if (builder.retained_diagnostic_count < builder.diagnostics.len) {
            const diagnostic_index = builder.retained_diagnostic_count;

            builder.copyDiagnostic(diagnostic_index, candidate);

            builder.diagnostic_heap[diagnostic_index] = @intCast(diagnostic_index);
            builder.retained_diagnostic_count += 1;

            builder.diagnosticHeapSiftUp(diagnostic_index);

            return;
        }

        builder.excess_diagnostic_count += 1;

        if (builder.retained_diagnostic_count == 0) {
            return;
        }

        const worst_diagnostic_index: usize = builder.diagnostic_heap[0];
        if (!diagnosticLessThan({}, candidate, builder.diagnostics[worst_diagnostic_index])) {
            return;
        }

        builder.copyDiagnostic(worst_diagnostic_index, candidate);
        builder.diagnosticHeapSiftDown();
    }

    /// `addFixedFile` copies one changed canonical source into shared scratch
    /// storage and adds the terminator required by the final sentinel slice.
    pub fn addFixedFile(
        builder: *Builder,
        path: []const u8,
        text: []const u8,
        applied_fix_count: usize,
        skipped_fix_count: usize,
    ) Allocator.Error!void {
        const stored_path = try builder.appendFixedFilePath(path);
        const stored_text = try builder.appendFixedFileText(text);

        try builder.fixed_text.append(builder.scratch_allocator, 0);

        try builder.fixed_files.append(builder.scratch_allocator, .{
            .path = stored_path,
            .text = stored_text,
        });

        builder.applied_fix_count += applied_fix_count;
        builder.skipped_fix_count += skipped_fix_count;
    }

    pub fn addFixCounts(builder: *Builder, applied: usize, skipped: usize) void {
        builder.applied_fix_count += applied;
        builder.skipped_fix_count += skipped;
    }

    /// `finish` transfers diagnostic storage directly into the report and
    /// publishes fixed-file output without retaining scratch references.
    pub fn finish(builder: *Builder) Allocator.Error!LintReport {
        const diagnostics = builder.diagnostics[0..builder.retained_diagnostic_count];

        std.mem.sort(Diagnostic, diagnostics, {}, diagnosticLessThan);

        const fixed_text_storage = try builder.report_allocator.dupe(
            u8,
            builder.fixed_text.items,
        );
        errdefer builder.report_allocator.free(fixed_text_storage);

        var fixed_files = try std.MultiArrayList(LintReport.FixedFile).initCapacity(
            builder.report_allocator,
            builder.fixed_files.len,
        );
        errdefer fixed_files.deinit(builder.report_allocator);

        for (0..builder.fixed_files.len) |fixed_file_index| {
            const pending = builder.fixed_files.get(fixed_file_index);
            assert(fixed_text_storage[pending.text.end] == 0);

            fixed_files.appendAssumeCapacity(.{
                .path = pending.path.slice(fixed_text_storage),
                .text = fixed_text_storage[pending.text.start..pending.text.end :0],
            });
        }

        fixed_files.sort(FixedFileSort{ .rows = fixed_files.slice() });

        assert(builder.total_diagnostic_count ==
            builder.retained_diagnostic_count +
                builder.oversized_diagnostic_count +
                builder.excess_diagnostic_count);

        const report: LintReport = .{
            .diagnostics = diagnostics,
            .total_diagnostic_count = builder.total_diagnostic_count,
            .oversized_diagnostic_count = builder.oversized_diagnostic_count,
            .excess_diagnostic_count = builder.excess_diagnostic_count,
            .fixed_files = fixed_files.toOwnedSlice(),
            .applied_fix_count = builder.applied_fix_count,
            .skipped_fix_count = builder.skipped_fix_count,
            .diagnostic_storage = builder.diagnostic_storage,
            .fixed_text_storage = fixed_text_storage,
        };

        builder.diagnostic_storage = builder.diagnostic_storage[0..0];

        return report;
    }

    fn copyDiagnostic(builder: *Builder, diagnostic_index: usize, diagnostic: Diagnostic) void {
        const text_slot_start = diagnostic_index * builder.diagnostic_text_size_max;
        const text_slot = builder.diagnostic_text_slots[text_slot_start..][0..builder.diagnostic_text_size_max];

        var writer: DiagnosticTextSlotWriter = .{ .text = text_slot };

        builder.diagnostics[diagnostic_index] = .{
            .path = writer.write(diagnostic.path),
            .rule_name = writer.write(diagnostic.rule_name),
            .message = writer.write(diagnostic.message),
            .help = writer.writeOptional(diagnostic.help),
            .note = writer.writeOptional(diagnostic.note),
            .location = diagnostic.location,
            .source_line = writer.write(diagnostic.source_line),
        };
    }

    fn diagnosticHeapSiftUp(builder: *Builder, initial_heap_index: usize) void {
        var heap_index = initial_heap_index;

        while (heap_index != 0) {
            const parent_heap_index = (heap_index - 1) / 2;
            const diagnostic_index: usize = builder.diagnostic_heap[heap_index];
            const parent_diagnostic_index: usize = builder.diagnostic_heap[parent_heap_index];

            if (!diagnosticLessThan(
                {},
                builder.diagnostics[parent_diagnostic_index],
                builder.diagnostics[diagnostic_index],
            )) {
                break;
            }

            std.mem.swap(
                u32,
                &builder.diagnostic_heap[parent_heap_index],
                &builder.diagnostic_heap[heap_index],
            );

            heap_index = parent_heap_index;
        }
    }

    fn diagnosticHeapSiftDown(builder: *Builder) void {
        var heap_index: usize = 0;

        while (true) {
            const left_heap_index = heap_index * 2 + 1;
            if (left_heap_index >= builder.retained_diagnostic_count) {
                return;
            }

            var worse_child_heap_index = left_heap_index;
            const right_heap_index = left_heap_index + 1;

            if (right_heap_index < builder.retained_diagnostic_count) {
                const left_diagnostic_index: usize = builder.diagnostic_heap[left_heap_index];
                const right_diagnostic_index: usize = builder.diagnostic_heap[right_heap_index];

                if (diagnosticLessThan(
                    {},
                    builder.diagnostics[left_diagnostic_index],
                    builder.diagnostics[right_diagnostic_index],
                )) {
                    worse_child_heap_index = right_heap_index;
                }
            }

            const diagnostic_index: usize = builder.diagnostic_heap[heap_index];
            const worse_child_diagnostic_index: usize = builder.diagnostic_heap[worse_child_heap_index];

            if (!diagnosticLessThan(
                {},
                builder.diagnostics[diagnostic_index],
                builder.diagnostics[worse_child_diagnostic_index],
            )) {
                return;
            }

            std.mem.swap(
                u32,
                &builder.diagnostic_heap[heap_index],
                &builder.diagnostic_heap[worse_child_heap_index],
            );

            heap_index = worse_child_heap_index;
        }
    }

    fn appendFixedFilePath(builder: *Builder, path: []const u8) Allocator.Error!TextRange {
        if (builder.last_fixed_file_path) |last_path| {
            if (std.mem.eql(u8, last_path.slice(builder.fixed_text.items), path)) {
                return last_path;
            }
        }

        const stored = try builder.appendFixedFileText(path);

        builder.last_fixed_file_path = stored;

        return stored;
    }

    fn appendFixedFileText(builder: *Builder, value: []const u8) Allocator.Error!TextRange {
        const start = builder.fixed_text.items.len;

        try builder.fixed_text.appendSlice(builder.scratch_allocator, value);

        return .{ .start = start, .end = builder.fixed_text.items.len };
    }
};

const TextRange = struct {
    start: usize,
    end: usize,

    fn slice(range: TextRange, text: []const u8) []const u8 {
        return text[range.start..range.end];
    }
};

const PendingFixedFile = struct {
    path: TextRange,
    text: TextRange,
};

const DiagnosticTextSlotWriter = struct {
    text: []u8,
    offset: usize = 0,

    fn write(writer: *DiagnosticTextSlotWriter, value: []const u8) []const u8 {
        assert(value.len <= writer.text.len - writer.offset);

        const start = writer.offset;

        writer.offset += value.len;

        @memcpy(writer.text[start..writer.offset], value);

        return writer.text[start..writer.offset];
    }

    fn writeOptional(
        writer: *DiagnosticTextSlotWriter,
        value: ?[]const u8,
    ) ?[]const u8 {
        const text = value orelse {
            return null;
        };

        return writer.write(text);
    }
};

fn diagnosticTextSize(diagnostic: Diagnostic) ?usize {
    var size: usize = 0;

    for ([_][]const u8{
        diagnostic.path,
        diagnostic.rule_name,
        diagnostic.message,
        diagnostic.help orelse
            &.{},
        diagnostic.note orelse
            &.{},
        diagnostic.source_line,
    }) |text| {
        size = std.math.add(usize, size, text.len) catch {
            return null;
        };
    }

    return size;
}

fn diagnosticLessThan(_: void, left: Diagnostic, right: Diagnostic) bool {
    const path_order = std.mem.order(u8, left.path, right.path);
    if (path_order != .eq) {
        return path_order == .lt;
    }

    const rule_order = std.mem.order(u8, left.rule_name, right.rule_name);
    if (rule_order != .eq) {
        return rule_order == .lt;
    }

    const message_order = std.mem.order(u8, left.message, right.message);
    if (message_order != .eq) {
        return message_order == .lt;
    }

    const help_order = optionalTextOrder(left.help, right.help);
    if (help_order != .eq) {
        return help_order == .lt;
    }

    const note_order = optionalTextOrder(left.note, right.note);
    if (note_order != .eq) {
        return note_order == .lt;
    }

    if (left.location.line != right.location.line) {
        return left.location.line < right.location.line;
    }

    if (left.location.column != right.location.column) {
        return left.location.column < right.location.column;
    }

    if (left.location.range.start_offset != right.location.range.start_offset) {
        return left.location.range.start_offset < right.location.range.start_offset;
    }

    if (left.location.range.end_offset != right.location.range.end_offset) {
        return left.location.range.end_offset < right.location.range.end_offset;
    }

    return std.mem.lessThan(u8, left.source_line, right.source_line);
}

const FixedFileSort = struct {
    rows: std.MultiArrayList(LintReport.FixedFile).Slice,

    pub fn lessThan(sort: FixedFileSort, left_index: usize, right_index: usize) bool {
        return std.mem.lessThan(
            u8,
            sort.rows.get(left_index).path,
            sort.rows.get(right_index).path,
        );
    }
};

// ─── Rendering ───────────────────────────────────────────────────────────────

const tab_width = 4;
const hex = "0123456789abcdef";

fn sameAgentGroup(left: Diagnostic, right: Diagnostic) bool {
    return std.mem.eql(u8, left.path, right.path) and
        std.mem.eql(u8, left.rule_name, right.rule_name) and
        std.mem.eql(u8, left.message, right.message) and
        optionalTextOrder(left.help, right.help) == .eq and
        optionalTextOrder(left.note, right.note) == .eq;
}

fn optionalTextOrder(left: ?[]const u8, right: ?[]const u8) std.math.Order {
    if (left) |left_text| {
        const right_text = right orelse {
            return .gt;
        };

        return std.mem.order(u8, left_text, right_text);
    }

    return if (right == null) .eq else .lt;
}

fn appendInline(text: *std.ArrayList(u8), allocator: Allocator, bytes: []const u8) Allocator.Error!void {
    for (bytes) |byte| {
        if (byte >= ' ' and byte <= '~') {
            try text.append(allocator, byte);
        } else if (byte == '\t') {
            try text.appendSlice(allocator, "\\t");
        } else {
            try appendEscapedByte(text, allocator, byte);
        }
    }
}

fn appendSource(text: *std.ArrayList(u8), allocator: Allocator, bytes: []const u8) Allocator.Error!void {
    var column: usize = 0;

    for (bytes) |byte| {
        if (byte == '\t') {
            const spaces = tab_width - column % tab_width;

            try appendSpaces(text, allocator, spaces);

            column += spaces;
        } else if (byte >= ' ' and byte <= '~') {
            try text.append(allocator, byte);

            column += 1;
        } else {
            try appendEscapedByte(text, allocator, byte);

            column += 4;
        }
    }
}

fn sourceDisplayColumn(bytes: []const u8, initial_column: usize) usize {
    var column = initial_column;

    for (bytes) |byte| {
        if (byte == '\t') {
            column += tab_width - column % tab_width;
        } else if (byte >= ' ' and byte <= '~') {
            column += 1;
        } else {
            column += 4;
        }
    }

    return column;
}

fn appendEscapedByte(text: *std.ArrayList(u8), allocator: Allocator, byte: u8) Allocator.Error!void {
    try text.appendSlice(allocator, &.{ '\\', 'x', hex[byte >> 4], hex[byte & 0x0f] });
}

fn appendGutter(
    text: *std.ArrayList(u8),
    allocator: Allocator,
    margin_width: usize,
    suffix: []const u8,
) Allocator.Error!void {
    try appendSpaces(text, allocator, margin_width + 1);
    try text.appendSlice(allocator, suffix);
}

fn appendSpaces(text: *std.ArrayList(u8), allocator: Allocator, repeat_count: usize) Allocator.Error!void {
    for (0..repeat_count) |_| {
        try text.append(allocator, ' ');
    }
}

fn appendCarets(text: *std.ArrayList(u8), allocator: Allocator, repeat_count: usize) Allocator.Error!void {
    for (0..repeat_count) |_| {
        try text.append(allocator, '^');
    }
}

fn decimalWidth(value: usize) usize {
    var remaining = value;
    var width: usize = 1;

    while (remaining >= 10) {
        remaining /= 10;
        width += 1;
    }

    return width;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

test "human diagnostics are terminal-safe and column-stable" {
    const allocator = std.testing.allocator;

    var builder = try Builder.init(allocator, allocator, .{});
    defer builder.deinit();

    builder.addDiagnostic(.{
        .path = "dir/\x1b\t\xc3.zig",
        .rule_name = "no_help",
        .message = "standalone message",
        .location = .{
            .line = 1,
            .column = 1,
            .range = .{ .start_offset = 0, .end_offset = 0 },
        },
    });

    builder.addDiagnostic(.{
        .path = "dir/\x1b\t\xc3.zig",
        .rule_name = "terminal_safe",
        .message = "unsafe source byte",
        .help = "replace the byte",
        .note = "tabs use four-column stops",
        .location = .{
            .line = 12,
            .column = 3,
            .range = .{ .start_offset = 2, .end_offset = 4 },
        },
        .source_line = "\tA\x1b\xc3\r",
    });

    var report = try builder.finish();
    defer report.deinit(allocator);

    const rendered = try report.renderHuman(allocator);
    defer allocator.free(rendered);

    try std.testing.expectEqualStrings(
        \\error[no_help]: standalone message
        \\ --> dir/\x1b\t\xc3.zig:1:1
        \\
        \\error[terminal_safe]: unsafe source byte
        \\  --> dir/\x1b\t\xc3.zig:12:3
        \\   |
        \\12 |     A\x1b\xc3
        \\   |      ^^^^^^^^
        \\   |
        \\   = help: replace the byte
        \\   = note: tabs use four-column stops
        \\
    , rendered);
}

test "agent diagnostics emit self-contained groups" {
    const allocator = std.testing.allocator;

    var builder = try Builder.init(allocator, allocator, .{});
    defer builder.deinit();

    const diagnostics = [_]Diagnostic{
        .{
            .path = "b.zig",
            .rule_name = "alpha",
            .message = "call must start a new paragraph",
            .help = "insert a blank line before this call",
            .location = .{
                .line = 4,
                .column = 5,
                .range = .{ .start_offset = 0, .end_offset = 0 },
            },
        },
        .{
            .path = "a.zig",
            .rule_name = "alpha",
            .message = "`if` body requires braces",
            .help = "wrap the body in `{ ... }`",
            .location = .{
                .line = 8,
                .column = 1,
                .range = .{ .start_offset = 0, .end_offset = 0 },
            },
        },
        .{
            .path = "a.zig",
            .rule_name = "beta",
            .message = "standalone message",
            .location = .{
                .line = 1,
                .column = 1,
                .range = .{ .start_offset = 0, .end_offset = 0 },
            },
        },
        .{
            .path = "a.zig",
            .rule_name = "alpha",
            .message = "declaration must start a new paragraph",
            .help = "insert a blank line before this declaration",
            .location = .{
                .line = 5,
                .column = 2,
                .range = .{ .start_offset = 0, .end_offset = 0 },
            },
        },
        .{
            .path = "a.zig",
            .rule_name = "alpha",
            .message = "`if` body requires braces",
            .help = "wrap the body in `{ ... }`",
            .location = .{
                .line = 2,
                .column = 3,
                .range = .{ .start_offset = 0, .end_offset = 0 },
            },
        },
    };

    for (diagnostics) |diagnostic| {
        builder.addDiagnostic(diagnostic);
    }

    var report = try builder.finish();
    defer report.deinit(allocator);

    const rendered = try report.renderAgent(allocator);
    defer allocator.free(rendered);

    try std.testing.expectEqualStrings(
        \\a.zig [alpha] 2:3,8:1 | `if` body requires braces | fix: wrap the body in `{ ... }`
        \\a.zig [alpha] 5:2 | declaration must start a new paragraph | fix: insert a blank line before this declaration
        \\a.zig [beta] 1:1 | standalone message
        \\b.zig [alpha] 4:5 | call must start a new paragraph | fix: insert a blank line before this call
        \\
    , rendered);
}

test "diagnostics retain deterministic top K and count every omission" {
    const allocator = std.testing.allocator;

    var builder = try Builder.init(allocator, allocator, .{
        .diagnostics_count_max = 2,
        .diagnostic_text_size_max = 32,
    });
    defer builder.deinit();

    for ([_][]const u8{ "b.zig", "c.zig" }) |path| {
        builder.addDiagnostic(.{
            .path = path,
            .rule_name = "rule",
            .message = "message",
            .location = .{
                .line = 1,
                .column = 1,
                .range = .{ .start_offset = 0, .end_offset = 0 },
            },
        });
    }

    builder.addDiagnostic(.{
        .path = "oversized.zig",
        .rule_name = "rule",
        .message = "this diagnostic does not fit in one text slot",
        .location = .{
            .line = 1,
            .column = 1,
            .range = .{ .start_offset = 0, .end_offset = 0 },
        },
    });

    builder.addDiagnostic(.{
        .path = "a.zig",
        .rule_name = "rule",
        .message = "message",
        .location = .{
            .line = 1,
            .column = 1,
            .range = .{ .start_offset = 0, .end_offset = 0 },
        },
    });

    var report = try builder.finish();
    defer report.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 4), report.total_diagnostic_count);
    try std.testing.expectEqual(@as(usize, 1), report.oversized_diagnostic_count);
    try std.testing.expectEqual(@as(usize, 1), report.excess_diagnostic_count);
    try std.testing.expectEqual(@as(usize, 2), report.diagnostics.len);
    try std.testing.expectEqualStrings("a.zig", report.diagnostics[0].path);
    try std.testing.expectEqualStrings("b.zig", report.diagnostics[1].path);

    const rendered = try report.renderAgent(allocator);
    defer allocator.free(rendered);

    try std.testing.expectEqualStrings(
        \\a.zig [rule] 1:1 | message
        \\b.zig [rule] 1:1 | message
        \\[diagnostic-summary] retained=2 total=4 excess=1 oversized=1
        \\
    , rendered);
}
