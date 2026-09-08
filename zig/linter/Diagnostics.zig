const Diagnostics = @This();

const std = @import("std");
const FileContext = @import("FileContext.zig");

/// Every diagnostic is fatal; a lint run succeeds only when none are emitted.
pub const Diagnostic = struct {
    path: []const u8,
    rule_name: []const u8,
    message: []const u8,

    line: usize,
    column: usize,
    source_line: []const u8 = &.{},
    caret_len: usize = 0,
};

pub const empty: Diagnostics = .{};

items: std.ArrayList(Diagnostic) = .empty,

last_diagnostic_path: []const u8 = &.{},

/// Returns the number of fatal diagnostics emitted so far.
pub fn count(diagnostics: *const Diagnostics) usize {
    return diagnostics.items.items.len;
}

pub fn add(
    diagnostics: *Diagnostics,
    diagnostic_allocator: std.mem.Allocator,
    diagnostic: Diagnostic,
) std.mem.Allocator.Error!void {
    var path = diagnostics.last_diagnostic_path;
    if (!std.mem.eql(u8, path, diagnostic.path)) {
        // A path may borrow storage from the directory walker, which becomes
        // invalid when the walker advances. Diagnostics for one file arrive together,
        // so copy its path once and let consecutive diagnostics share that owned copy.
        path = try diagnostic_allocator.dupe(u8, diagnostic.path);
        diagnostics.last_diagnostic_path = path;
    }

    var source_line = std.mem.trimEnd(u8, diagnostic.source_line, "\r");
    if (source_line.len != 0) {
        // The source line usually points into the reusable per-file buffer. Keep a
        // copy with the diagnostic because that buffer is overwritten by the next file.
        // Dropping a CRLF line's carriage return keeps rendered output portable.
        source_line = try diagnostic_allocator.dupe(u8, source_line);
    }

    try diagnostics.items.append(diagnostic_allocator, .{
        .path = path,
        .line = diagnostic.line,
        .column = diagnostic.column,
        .rule_name = diagnostic.rule_name,
        .message = diagnostic.message,
        .source_line = source_line,
        .caret_len = diagnostic.caret_len,
    });
}

pub fn add_at_token(
    diagnostics: *Diagnostics,
    diagnostic_allocator: std.mem.Allocator,
    path: []const u8,
    file: *const FileContext,
    token: std.zig.Ast.TokenIndex,
    rule_name: []const u8,
    message: []const u8,
) std.mem.Allocator.Error!void {
    const ast = file.ast;
    const loc = file.tokenLocation(token);

    try diagnostics.add(diagnostic_allocator, .{
        .path = path,
        .line = loc.line + 1,
        .column = loc.column + 1,
        .rule_name = rule_name,
        .message = message,
        .source_line = ast.source[loc.line_start..loc.line_end],
        .caret_len = ast.tokenSlice(token).len,
    });
}

pub fn render(
    diagnostics: *Diagnostics,
    output_allocator: std.mem.Allocator,
) std.mem.Allocator.Error![]u8 {
    std.mem.sort(Diagnostic, diagnostics.items.items, {}, render_diagnostic_less_than);

    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(output_allocator);

    for (diagnostics.items.items, 0..) |diagnostic, index| {
        if (index != 0) {
            try text.append(output_allocator, '\n');
        }

        try text.print(output_allocator, "{s}:{d}:{d}: {s}: {s}\n", .{
            diagnostic.path,
            diagnostic.line,
            diagnostic.column,
            diagnostic.rule_name,
            diagnostic.message,
        });

        if (diagnostic.source_line.len == 0) {
            continue;
        }

        try text.print(output_allocator, "    {s}\n    ", .{diagnostic.source_line});

        const pad = diagnostic.column -| 1;
        for (0..pad) |offset| {
            // Reproduce tabs from the source prefix so the caret uses the same
            // terminal tab stops as the rendered source line. Other bytes become
            // spaces; columns are byte offsets, matching Zig's AST locations.
            const padding: u8 = if (offset < diagnostic.source_line.len and
                diagnostic.source_line[offset] == '\t')
                '\t'
            else
                ' ';
            try text.append(output_allocator, padding);
        }

        const caret_len = if (diagnostic.caret_len == 0) 1 else diagnostic.caret_len;
        for (0..caret_len) |_| {
            try text.append(output_allocator, '^');
        }

        try text.append(output_allocator, '\n');
    }

    return text.toOwnedSlice(output_allocator);
}

fn render_diagnostic_less_than(_: void, a: Diagnostic, b: Diagnostic) bool {
    const path_order = std.mem.order(u8, a.path, b.path);
    if (path_order != .eq) {
        return path_order == .lt;
    }

    if (a.line != b.line) {
        return a.line < b.line;
    }

    if (a.column != b.column) {
        return a.column < b.column;
    }

    return std.mem.lessThan(u8, a.rule_name, b.rule_name);
}
