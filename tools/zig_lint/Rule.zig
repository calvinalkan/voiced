//! `Rule` defines the source-level contract shared by built-in and external
//! lint rules. A `Context` is local to the code invoking a rule; no value in
//! this file crosses a dynamic-library boundary without an explicit ABI adapter.

const std = @import("std");
const assert = std.debug.assert;
const Ast = std.zig.Ast;

pub const SourceRange = struct {
    /// `start_offset` is a zero-based inclusive byte offset in the source.
    start_offset: u32,

    /// `end_offset` is a zero-based exclusive byte offset in the source.
    end_offset: u32,
};

pub const Fix = struct {
    /// `range` must satisfy `0 <= start_offset <= end_offset <= source.len`.
    range: SourceRange,

    /// `replacement` remains borrowed only until `Context.report` returns.
    replacement: []const u8,
};

/// Finding text is borrowed only for the synchronous `Context.report` call.
pub const Finding = struct {
    /// `token` must select a token in `Context.ast`.
    token: Ast.TokenIndex,
    message: []const u8,
    help: []const u8 = "",
    note: []const u8 = "",
    fix: ?Fix = null,
};

pub const ReportError = error{
    /// The host rejected a finding's token or fix coordinates. The lint
    /// operation remains failed even when a rule catches this error.
    FindingRejected,

    /// The matching SDK received a response outside its host protocol. Rules
    /// propagate this error and must not return it directly.
    HostProtocolViolation,
};

pub const Error = std.mem.Allocator.Error || ReportError;

/// `Context` remains borrowed for one rule invocation. Rules may allocate
/// invocation-local data with `scratch_allocator` but must not retain the
/// context, its AST, or its metadata after returning.
pub const Context = struct {
    scratch_allocator: std.mem.Allocator,

    /// The caller-selected Zig release whose language and standard-library
    /// contracts version-sensitive diagnostics and fixes must target. This
    /// value does not identify the toolchain that produced `ast`.
    target_zig_version: std.SemanticVersion,

    /// `ast` and all storage reachable through it remain host-owned. Rules may
    /// inspect that storage but must not mutate or retain it.
    ast: Ast,

    /// `fixes_enabled` tells rules whether they may attach a non-null fix.
    fixes_enabled: bool,

    /// The start offset in the source of each line.
    line_start_offsets: []const u32,

    /// `token_line_indexes[n]` identifies the zero-based line containing token `n`.
    token_line_indexes: []const u32,

    /// The host and SDK initialize the reporting adapter. Rules call `report`
    /// rather than invoking or replacing these fields.
    report_state: *anyopaque,
    report_fn: *const fn (*anyopaque, Finding) Error!void,

    pub const LineLayout = struct {
        newline: []const u8,
        indent: []const u8,
        indent_unit: []const u8,
    };

    /// `report` synchronously copies every retained finding and fix slice.
    /// Rules must propagate its errors and must not manufacture reporting
    /// errors directly.
    pub fn report(context: *Context, finding: Finding) Error!void {
        return context.report_fn(context.report_state, finding);
    }

    pub fn tokenRange(context: *const Context, token: Ast.TokenIndex) SourceRange {
        const start_offset = context.ast.tokenStart(token);

        return .{
            .start_offset = start_offset,
            .end_offset = @intCast(start_offset + context.ast.tokenSlice(token).len),
        };
    }

    /// `rangeBetweenTokens` covers both endpoint tokens and every byte between.
    pub fn rangeBetweenTokens(
        context: *const Context,
        first_token: Ast.TokenIndex,
        last_token: Ast.TokenIndex,
    ) SourceRange {
        assert(first_token <= last_token);

        return .{
            .start_offset = context.ast.tokenStart(first_token),
            .end_offset = context.tokenRange(last_token).end_offset,
        };
    }

    /// `tokenReplacement` returns null when fixes are disabled or the token
    /// already has the requested spelling.
    pub fn tokenReplacement(
        context: *const Context,
        token: Ast.TokenIndex,
        replacement: []const u8,
    ) ?Fix {
        if (!context.fixes_enabled) {
            return null;
        }

        const range = context.tokenRange(token);
        if (std.mem.eql(u8, context.ast.source[range.start_offset..range.end_offset], replacement)) {
            return null;
        }

        return .{ .range = range, .replacement = replacement };
    }

    /// `tokenRemoval` returns null when fixes are disabled or the token is empty.
    pub fn tokenRemoval(context: *const Context, token: Ast.TokenIndex) ?Fix {
        if (!context.fixes_enabled) {
            return null;
        }

        const range = context.tokenRange(token);
        if (range.start_offset == range.end_offset) {
            return null;
        }

        return .{ .range = range, .replacement = "" };
    }

    /// `insertionBeforeToken` returns null when fixes are disabled or `text` is empty.
    pub fn insertionBeforeToken(
        context: *const Context,
        token: Ast.TokenIndex,
        text: []const u8,
    ) ?Fix {
        if (!context.fixes_enabled or text.len == 0) {
            return null;
        }

        const offset = context.ast.tokenStart(token);

        return .{
            .range = .{ .start_offset = offset, .end_offset = offset },
            .replacement = text,
        };
    }

    /// `insertionAfterToken` returns null when fixes are disabled or `text` is empty.
    pub fn insertionAfterToken(
        context: *const Context,
        token: Ast.TokenIndex,
        text: []const u8,
    ) ?Fix {
        if (!context.fixes_enabled or text.len == 0) {
            return null;
        }

        const offset = context.tokenRange(token).end_offset;

        return .{
            .range = .{ .start_offset = offset, .end_offset = offset },
            .replacement = text,
        };
    }

    /// Borrow the exact source spelling of a token already classified as an identifier.
    /// Quoted spelling is preserved, not decoded; no allocation or line index is needed.
    pub fn identifierText(ast: Ast, token: Ast.TokenIndex) []const u8 {
        assert(ast.tokenTag(token) == .identifier);

        const start = ast.tokenStart(token);
        if (ast.source[start] == '@') {
            // The AST already classified this identifier. Plain names need only their
            // end, not another tokenizer dispatch and keyword lookup. Quoted names keep
            // the standard tokenizer's escape handling and exact source spelling.
            return ast.tokenSlice(token);
        }

        var end = start + 1;

        while (end < ast.source.len) : (end += 1) {
            switch (ast.source[end]) {
                'a'...'z', 'A'...'Z', '0'...'9', '_' => {},
                else => break,
            }
        }

        return ast.source[start..end];
    }

    /// Return whether `node` directly declares a type without an alias or computation.
    pub fn nodeIsDirectType(ast: Ast, node: Ast.Node.Index) bool {
        var container_buffer: [2]Ast.Node.Index = undefined;
        if (ast.fullContainerDecl(&container_buffer, node) != null) {
            return true;
        }

        return switch (ast.nodeTag(node)) {
            .error_set_decl,
            .fn_proto_simple,
            .fn_proto_multi,
            .fn_proto_one,
            .fn_proto,
            => true,
            else => false,
        };
    }

    // These leaf classifiers run inside per-node rule loops. Keep them inline so
    // sharing their source does not add a call at every node.
    pub inline fn nodeIsBlock(ast: Ast, node: Ast.Node.Index) bool {
        return switch (ast.nodeTag(node)) {
            .block_two, .block_two_semicolon, .block, .block_semicolon => true,
            else => false,
        };
    }

    pub inline fn nodeTagIsCall(tag: Ast.Node.Tag) bool {
        return switch (tag) {
            .call_one,
            .call_one_comma,
            .call,
            .call_comma,
            .builtin_call_two,
            .builtin_call_two_comma,
            .builtin_call,
            .builtin_call_comma,
            => true,
            else => false,
        };
    }

    /// Return whether `tag` is a non-destructuring assignment whose RHS is
    /// `nodeData(node).node_and_node[1]`.
    pub inline fn nodeTagIsAssignment(tag: Ast.Node.Tag) bool {
        return switch (tag) {
            .assign_mul,
            .assign_div,
            .assign_mod,
            .assign_add,
            .assign_sub,
            .assign_shl,
            .assign_shl_sat,
            .assign_shr,
            .assign_bit_and,
            .assign_bit_xor,
            .assign_bit_or,
            .assign_mul_wrap,
            .assign_add_wrap,
            .assign_sub_wrap,
            .assign_mul_sat,
            .assign_add_sat,
            .assign_sub_sat,
            .assign,
            => true,
            else => false,
        };
    }

    pub inline fn nodeTagIsExit(tag: Ast.Node.Tag) bool {
        return switch (tag) {
            .@"return", .@"break", .@"continue", .unreachable_literal => true,
            else => false,
        };
    }

    /// Return zero-based byte coordinates, matching `Ast.tokenLocation(0, token)`.
    pub fn tokenLocation(context: *const Context, token: Ast.TokenIndex) Ast.Location {
        const offset = context.ast.tokenStart(token);
        const line_index = context.token_line_indexes[token];
        const start = context.line_start_offsets[line_index];

        return .{
            .line = line_index,
            .column = offset - start,
            .line_start = start,
            .line_end = start + context.lineText(line_index).len,
        };
    }

    /// Return a physical line without its LF terminator. A CR in CRLF is retained,
    /// matching the source-line slices used by AST diagnostics.
    pub fn lineText(context: *const Context, line_index: usize) []const u8 {
        const end = if (line_index + 1 < context.line_start_offsets.len)
            context.line_start_offsets[line_index + 1] - 1
        else
            context.ast.source.len;

        return context.ast.source[context.line_start_offsets[line_index]..end];
    }

    /// Return a line's newline spelling and indentation. At EOF, inherit the preceding
    /// newline when present; a tab-indented line continues with a tab, otherwise four spaces.
    pub fn lineLayout(context: *const Context, token: Ast.TokenIndex) LineLayout {
        const location = context.tokenLocation(token);
        const text = context.lineText(location.line);
        const indent_size = text.len - std.mem.trimStart(u8, text, " \t").len;
        const indent = text[0..indent_size];

        // Prefer this line's terminator. At EOF, inherit the preceding one rather
        // than introducing LF into a CRLF file with no final newline.
        const newline_offset = if (location.line + 1 < context.line_start_offsets.len)
            context.line_start_offsets[location.line + 1]
        else
            context.line_start_offsets[location.line];

        return .{
            .newline = if (newline_offset >= 2 and context.ast.source[newline_offset - 2] == '\r') "\r\n" else "\n",
            .indent = indent,
            .indent_unit = if (std.mem.indexOfScalar(u8, indent, '\t') != null) "\t" else "    ",
        };
    }

    pub fn areTokensOnSameLine(context: *const Context, left: Ast.TokenIndex, right: Ast.TokenIndex) bool {
        return context.token_line_indexes[left] == context.token_line_indexes[right];
    }

    /// Check only complete lines between the tokens. Spaces, tabs, and CR count as
    /// whitespace; a comment alone does not provide an empty separator.
    pub fn hasEmptyLineBetween(context: *const Context, left: Ast.TokenIndex, right: Ast.TokenIndex) bool {
        const right_line_index = context.token_line_indexes[right];

        var line_index: usize = @as(usize, context.token_line_indexes[left]) + 1;

        while (line_index < right_line_index) : (line_index += 1) {
            if (std.mem.trim(u8, context.lineText(line_index), " \t\r").len == 0) {
                return true;
            }
        }

        return false;
    }
};

/// `Definition` gives one rule a stable diagnostic identity and native entry
/// point. Plugin manifests translate this value rather than exporting its Zig
/// layout.
pub const Definition = struct {
    name: []const u8,
    lint: *const fn (*Context) Error!void,
};

/// `define` creates a compile-time rule descriptor and rejects an empty name.
pub fn define(
    comptime name: []const u8,
    comptime lint: *const fn (*Context) Error!void,
) Definition {
    if (name.len == 0) {
        @compileError("rule name must not be empty");
    }

    return .{ .name = name, .lint = lint };
}
