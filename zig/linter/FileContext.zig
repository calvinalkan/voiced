const FileContext = @This();

const std = @import("std");
const Ast = std.zig.Ast;

ast: Ast,
line_starts: []const u32,

/// Borrow `ast` and its source; allocate `line_starts` with `allocator`.
/// The caller frees `line_starts` after all rules finish. Source size must
/// already satisfy the caller's `file_size_max` limit.
pub fn init(allocator: std.mem.Allocator, ast: Ast, file_size_max: u32) std.mem.Allocator.Error!FileContext {
    const source = ast.source;
    std.debug.assert(source.len <= file_size_max);

    var line_starts: std.ArrayList(u32) = .empty;
    errdefer line_starts.deinit(allocator);

    // Empty input still has line zero. A trailing newline adds an empty final
    // line whose start equals source.len, so EOF tokens have a valid location.
    try line_starts.append(allocator, 0);

    const width = 32;
    const newlines: @Vector(width, u8) = @splat('\n');
    var offset: usize = 0;

    while (source.len - offset >= width) : (offset += width) {
        const bytes: @Vector(width, u8) = source[offset..][0..width].*;
        var mask: u32 = @bitCast(bytes == newlines);

        // Each bit identifies a newline lane. Consume the lowest bit first
        // to preserve byte order without restarting a search for each match:
        //
        //   newline at offset + lane
        //   next line starts at offset + lane + 1
        while (mask != 0) {
            const lane = @ctz(mask);
            try line_starts.append(allocator, @intCast(offset + lane + 1));
            mask &= mask - 1;
        }
    }

    // Full vector loads never cross EOF; the final partial chunk is scalar.
    while (offset < source.len) : (offset += 1) {
        if (source[offset] == '\n') {
            try line_starts.append(allocator, @intCast(offset + 1));
        }
    }

    const owned_line_starts = try line_starts.toOwnedSlice(allocator);

    return .{ .ast = ast, .line_starts = owned_line_starts };
}

/// Return zero-based byte coordinates, matching `Ast.tokenLocation(0, token)`.
pub fn tokenLocation(file: *const FileContext, token: Ast.TokenIndex) Ast.Location {
    const offset = file.ast.tokenStart(token);
    const line = file.lineIndex(offset);
    const start = file.line_starts[line];

    return .{
        .line = line,
        .column = offset - start,
        .line_start = start,
        .line_end = start + file.lineText(line).len,
    };
}

fn lineIndex(file: *const FileContext, offset: u32) usize {
    // Find the last line start <= offset. Line zero guarantees a predecessor,
    // including for empty input and tokens at byte zero.
    var low: usize = 0;
    var high = file.line_starts.len;

    while (low < high) {
        const middle = low + (high - low) / 2;
        if (file.line_starts[middle] <= offset) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    return low - 1;
}

/// Return a physical line without its LF terminator. A CR in CRLF is retained,
/// matching the source-line slices used by AST diagnostics.
pub fn lineText(file: *const FileContext, line: usize) []const u8 {
    const end = if (line + 1 < file.line_starts.len)
        file.line_starts[line + 1] - 1
    else
        file.ast.source.len;

    return file.ast.source[file.line_starts[line]..end];
}

pub fn tokensOnSameLine(file: *const FileContext, left: Ast.TokenIndex, right: Ast.TokenIndex) bool {
    const left_line = file.lineIndex(file.ast.tokenStart(left));
    const right_offset = file.ast.tokenStart(right);

    return right_offset >= file.line_starts[left_line] and
        (left_line + 1 == file.line_starts.len or right_offset < file.line_starts[left_line + 1]);
}

/// Check only complete lines between the tokens. Spaces, tabs, and CR count as
/// whitespace; a comment alone does not provide an empty separator.
pub fn hasEmptyLineBetween(file: *const FileContext, left: Ast.TokenIndex, right: Ast.TokenIndex) bool {
    var line = file.lineIndex(file.ast.tokenStart(left)) + 1;
    const right_line = file.lineIndex(file.ast.tokenStart(right));

    while (line < right_line) : (line += 1) {
        if (std.mem.trim(u8, file.lineText(line), " \t\r").len == 0) {
            return true;
        }
    }

    return false;
}
