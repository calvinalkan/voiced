const std = @import("std");
const Ast = std.zig.Ast;
const Rule = @import("../Rule.zig");
const LintContext = Rule.Context;
const identifierText = LintContext.identifierText;
const nodeIsBlock = LintContext.nodeIsBlock;
const nodeTagIsAssignment = LintContext.nodeTagIsAssignment;
const nodeTagIsCall = LintContext.nodeTagIsCall;
const nodeTagIsExit = LintContext.nodeTagIsExit;

/// Classify each gap between sibling statements once, then enforce the one
/// strongest layout constraint assigned to it. Semantic setup groups override
/// generic statement-category boundaries, so one gap cannot receive competing
/// diagnostics from independent spacing rules.
pub fn lint(context: *LintContext) Rule.Error!void {
    const file_allocator = context.scratch_allocator;
    const ast = context.ast;

    var block_buffer: [2]Ast.Node.Index = undefined;
    for (ast.nodes.items(.tag), 0..) |tag, index_usize| {
        switch (tag) {
            .block_two, .block_two_semicolon, .block, .block_semicolon => {},
            else => continue,
        }

        const block: Ast.Node.Index = @enumFromInt(index_usize);

        const statements = ast.blockStatements(&block_buffer, block) orelse {
            continue;
        };

        if (statements.len == 0) {
            continue;
        }

        const gaps = try file_allocator.alloc(Gap, statements.len);
        defer file_allocator.free(gaps);

        @memset(gaps, .{});

        try markUndefinedStorageGroups(
            file_allocator,
            context,
            statements,
            gaps,
        );

        try markCleanupGroups(
            file_allocator,
            context,
            block,
            statements,
            gaps,
        );

        markProducerGuardAndAssertGroups(context, statements, gaps);
        markGenericParagraphs(context, statements, gaps);

        for (statements[1..], 1..) |statement, statement_index| {
            const gap = gaps[statement_index];
            if (gap.constraint == .unconstrained) {
                continue;
            }

            const right_token = ast.firstToken(statement);
            const left_token = terminatedToken(ast, ast.lastToken(statements[statement_index - 1]), right_token);

            const violates = switch (gap.constraint) {
                .unconstrained => false,
                .required_blank => !hasParagraphBreak(context, left_token, right_token),
                .forbidden_blank => context.hasEmptyLineBetween(left_token, right_token),
            };

            if (!violates) {
                continue;
            }

            const fix = if (context.fixes_enabled)
                try paragraphGapFix(file_allocator, context, ast.firstToken(block), left_token, right_token, gap.constraint)
            else
                null;

            const description = descriptionFor(context, statements[statement_index - 1], statement, gap.reason);

            try context.report(.{
                .token = right_token,
                .message = description.message,
                .help = description.help orelse "",
                .note = description.note orelse "",
                .fix = fix,
            });
        }
    }

    try lintSwitchArmParagraphs(file_allocator, context);
}

const Constraint = enum(u2) {
    unconstrained,
    required_blank,
    forbidden_blank,
};

const Reason = enum(u8) {
    none,
    exit_paragraph,
    control_paragraph,
    scope_paragraph,
    declaration_paragraph,
    multiline_declaration,
    call_paragraph,
    multiline_call,
    assignment_paragraph,
    cleanup_before,
    cleanup_internal,
    cleanup_after,
    undefined_before,
    undefined_internal,
    undefined_after,
    producer_guard,
    assertion_pair,
    assertion_after,
};

const Gap = struct {
    constraint: Constraint = .unconstrained,
    reason: Reason = .none,
};

fn requireBlank(gaps: []Gap, index: usize, reason: Reason) void {
    if (index == 0 or index >= gaps.len or gaps[index].constraint != .unconstrained) {
        return;
    }

    gaps[index] = .{ .constraint = .required_blank, .reason = reason };
}

fn forbidBlank(gaps: []Gap, index: usize, reason: Reason) void {
    if (index == 0 or index >= gaps.len) {
        return;
    }

    // Semantic attachment is an explicit exception to generic paragraph
    // boundaries and therefore owns the gap regardless of marking order.
    gaps[index] = .{ .constraint = .forbidden_blank, .reason = reason };
}

// ─── Undefined Storage ───────────────────────────────────────────────

fn markUndefinedStorageGroups(
    file_allocator: std.mem.Allocator,
    context: *LintContext,
    statements: []const Ast.Node.Index,
    gaps: []Gap,
) Rule.Error!void {
    const ast = context.ast;

    var undefined_names: std.StringHashMapUnmanaged(bool) = .empty;
    defer undefined_names.deinit(file_allocator);

    var index: usize = 0;

    while (index < statements.len) {
        if (undefinedVariableNameToken(ast, statements[index]) == null) {
            index += 1;

            continue;
        }

        const start = index;

        while (index < statements.len and undefinedVariableNameToken(ast, statements[index]) != null) {
            index += 1;
        }

        const declarations_end = index;

        requireBlank(gaps, start, .undefined_before);

        for (start + 1..declarations_end) |member| {
            forbidBlank(gaps, member, .undefined_internal);
        }

        if (declarations_end == statements.len) {
            for (statements[start..declarations_end]) |declaration| {
                try addMissingUndefinedUse(
                    context,
                    expectUndefinedVariableNameToken(ast, declaration),
                );
            }

            continue;
        }

        const consumer = statements[declarations_end];

        forbidBlank(gaps, declarations_end, .undefined_internal);
        undefined_names.clearRetainingCapacity();

        for (statements[start..declarations_end]) |declaration| {
            const name_token = expectUndefinedVariableNameToken(ast, declaration);

            try undefined_names.put(file_allocator, identifierText(ast, name_token), false);
        }

        markReferencedIdentifiers(ast, consumer, &undefined_names);

        for (statements[start..declarations_end]) |declaration| {
            const name_token = expectUndefinedVariableNameToken(ast, declaration);

            if (!(undefined_names.get(identifierText(ast, name_token)) orelse false)) {
                try addMissingUndefinedUse(context, name_token);
            }
        }

        // Cleanup registrations belong to the same setup paragraph. This also
        // prevents the generic resource grouping pass from assigning a second
        // boundary immediately after the first use.
        var group_end = declarations_end;

        while (group_end + 1 < statements.len and isCleanup(ast.nodeTag(statements[group_end + 1]))) {
            group_end += 1;

            forbidBlank(gaps, group_end, .cleanup_internal);
        }

        requireBlank(gaps, group_end + 1, .undefined_after);

        index = declarations_end + 1;
    }
}

fn undefinedVariableNameToken(ast: Ast, node: Ast.Node.Index) ?Ast.TokenIndex {
    switch (ast.nodeTag(node)) {
        .global_var_decl, .local_var_decl, .simple_var_decl, .aligned_var_decl => {},
        else => return null,
    }

    if (ast.tokenTag(ast.nodeMainToken(node)) != .keyword_var) {
        return null;
    }

    const variable = ast.fullVarDecl(node) orelse {
        unreachable;
    };

    const initializer = variable.ast.init_node.unwrap() orelse {
        return null;
    };

    if (ast.nodeTag(initializer) != .identifier or
        !std.mem.eql(u8, identifierText(ast, ast.nodeMainToken(initializer)), "undefined"))
    {
        return null;
    }

    const name_token = variable.ast.mut_token + 1;
    if (ast.tokenTag(name_token) != .identifier) {
        return null;
    }

    return name_token;
}

fn expectUndefinedVariableNameToken(ast: Ast, node: Ast.Node.Index) Ast.TokenIndex {
    return undefinedVariableNameToken(ast, node) orelse {
        unreachable;
    };
}

fn addMissingUndefinedUse(
    context: *LintContext,
    name_token: Ast.TokenIndex,
) Rule.Error!void {
    try context.report(.{
        .token = name_token,
        .message = "`var = undefined` storage is not used by the next statement",
        .help = "reference this variable in the statement immediately after the storage group",
        .note = "the first statement must reference every variable in the group",
    });
}

// ─── Resource Cleanup ────────────────────────────────────────────────

fn markCleanupGroups(
    file_allocator: std.mem.Allocator,
    context: *LintContext,
    block: Ast.Node.Index,
    statements: []const Ast.Node.Index,
    gaps: []Gap,
) Rule.Error!void {
    const file = context;
    const ast = file.ast;
    var index: usize = 0;

    while (index < statements.len) {
        if (!isCleanup(ast.nodeTag(statements[index])) or gaps[index].reason == .cleanup_internal) {
            index += 1;

            continue;
        }

        const start = index -| 1;
        var end = index;

        while (end + 1 < statements.len and isCleanup(ast.nodeTag(statements[end + 1]))) {
            end += 1;
        }

        requireBlank(gaps, start, .cleanup_before);

        for (start + 1..end + 1) |member| {
            forbidBlank(gaps, member, .cleanup_internal);
        }

        requireBlank(gaps, end + 1, .cleanup_after);

        index = end + 1;
    }

    for (statements) |statement| {
        if (!isCleanup(ast.nodeTag(statement))) {
            continue;
        }

        const token = ast.nodeMainToken(statement);

        if (token > 0 and file.areTokensOnSameLine(token - 1, token)) {
            const fix = if (context.fixes_enabled)
                try lineBreakFix(file_allocator, file, ast.firstToken(block), token - 1, token, false)
            else
                null;

            try context.report(.{
                .token = token,
                .message = "cleanup registration must start on its own line",
                .help = "move this cleanup registration below its setup statement",
                .fix = fix,
            });
        }
    }
}

fn isCleanup(tag: Ast.Node.Tag) bool {
    return tag == .@"defer" or tag == .@"errdefer";
}

// ─── Compact Semantic Pairs ─────────────────────────────────────────

fn markProducerGuardAndAssertGroups(
    file: *const LintContext,
    statements: []const Ast.Node.Index,
    gaps: []Gap,
) void {
    const ast = file.ast;

    for (statements[1..], 1..) |statement, index| {
        const producer = statements[index - 1];
        if (category(ast, producer) != .declaration or
            (index > 1 and category(ast, statements[index - 2]) == .declaration) or
            isMultiline(file, producer))
        {
            continue;
        }

        const name_token = declarationNameToken(ast, producer) orelse {
            continue;
        };

        const name = identifierText(ast, name_token);

        if (isEarlyExitGuard(ast, statement, name)) {
            forbidBlank(gaps, index, .producer_guard);
        } else if (!isMultiline(file, statement) and
            isAssertCall(ast, statement) and
            nodeReferencesIdentifier(ast, statement, name))
        {
            forbidBlank(gaps, index, .assertion_pair);
            requireBlank(gaps, index + 1, .assertion_after);
        }
    }
}

fn declarationNameToken(ast: Ast, node: Ast.Node.Index) ?Ast.TokenIndex {
    const variable = ast.fullVarDecl(node) orelse {
        return null;
    };

    const name_token = variable.ast.mut_token + 1;

    return if (ast.tokenTag(name_token) == .identifier) name_token else null;
}

fn isEarlyExitGuard(ast: Ast, node: Ast.Node.Index, producer_name: []const u8) bool {
    const conditional = ast.fullIf(node) orelse {
        return false;
    };

    if (conditional.ast.else_expr != .none or
        !nodeReferencesIdentifier(ast, conditional.ast.cond_expr, producer_name))
    {
        return false;
    }

    const body = conditional.ast.then_expr;

    var block_buffer: [2]Ast.Node.Index = undefined;
    if (ast.blockStatements(&block_buffer, body)) |statements| {
        return statements.len != 0 and nodeTagIsExit(ast.nodeTag(statements[statements.len - 1]));
    }

    return nodeTagIsExit(ast.nodeTag(body));
}

fn isAssertCall(ast: Ast, node: Ast.Node.Index) bool {
    var call_buffer: [1]Ast.Node.Index = undefined;
    const call = ast.fullCall(&call_buffer, unwrapStatement(ast, node)) orelse {
        return false;
    };

    return switch (ast.nodeTag(call.ast.fn_expr)) {
        .identifier => std.mem.eql(u8, identifierText(ast, ast.nodeMainToken(call.ast.fn_expr)), "assert"),
        .field_access => std.mem.eql(u8, identifierText(ast, ast.nodeData(call.ast.fn_expr).node_and_token[1]), "assert"),
        else => false,
    };
}

// ─── Generic Paragraph Boundaries ───────────────────────────────────

const Category = enum {
    declaration,
    control,
    scope,
    call,
    assignment,
    exit,
    other,
};

fn markGenericParagraphs(
    file: *const LintContext,
    statements: []const Ast.Node.Index,
    gaps: []Gap,
) void {
    const ast = file.ast;

    for (statements[1..], 1..) |right, index| {
        if (gaps[index].constraint != .unconstrained) {
            continue;
        }

        const left = statements[index - 1];
        const left_category = category(ast, left);
        const right_category = category(ast, right);

        if (right_category == .exit) {
            requireBlank(gaps, index, .exit_paragraph);
        } else if (left_category == .control or right_category == .control) {
            requireBlank(gaps, index, .control_paragraph);
        } else if (left_category == .scope or right_category == .scope) {
            requireBlank(gaps, index, .scope_paragraph);
        } else if ((left_category == .declaration and isMultiline(file, left)) or
            (right_category == .declaration and isMultiline(file, right)))
        {
            requireBlank(gaps, index, .multiline_declaration);
        } else if ((left_category == .call and isMultiline(file, left)) or
            (right_category == .call and isMultiline(file, right)))
        {
            requireBlank(gaps, index, .multiline_call);
        } else if (left_category != right_category and
            (left_category == .declaration or right_category == .declaration))
        {
            requireBlank(gaps, index, .declaration_paragraph);
        } else if (left_category != right_category and
            (left_category == .call or right_category == .call))
        {
            requireBlank(gaps, index, .call_paragraph);
        } else if (left_category != right_category and
            (left_category == .assignment or right_category == .assignment))
        {
            requireBlank(gaps, index, .assignment_paragraph);
        }
    }
}

fn category(ast: Ast, node: Ast.Node.Index) Category {
    const tag = ast.nodeTag(node);

    if (nodeIsBlock(ast, node)) {
        return .scope;
    }

    if (nodeTagIsCall(tag)) {
        return .call;
    }

    if (nodeTagIsAssignment(tag)) {
        return .assignment;
    }

    if (nodeTagIsExit(tag)) {
        return .exit;
    }

    return switch (tag) {
        .global_var_decl, .local_var_decl, .simple_var_decl, .aligned_var_decl => .declaration,
        .assign_destructure => categoryForDestructure(ast, node),

        .if_simple,
        .@"if",
        .while_simple,
        .while_cont,
        .@"while",
        .for_simple,
        .@"for",
        .@"switch",
        .switch_comma,
        => .control,

        .@"comptime", .@"try", .@"nosuspend" => category(ast, ast.nodeData(node).node),

        .@"catch",
        .@"orelse",
        .@"suspend",
        .@"resume",
        .@"asm",
        .asm_simple,
        => .call,

        else => .other,
    };
}

fn categoryForDestructure(ast: Ast, node: Ast.Node.Index) Category {
    const destructure = ast.assignDestructure(node);

    for (destructure.ast.variables) |variable| {
        if (ast.fullVarDecl(variable) != null) {
            return .declaration;
        }
    }

    return .assignment;
}

fn isMultiline(file: *const LintContext, node: Ast.Node.Index) bool {
    return !file.areTokensOnSameLine(file.ast.firstToken(node), file.ast.lastToken(node));
}

fn unwrapStatement(ast: Ast, node: Ast.Node.Index) Ast.Node.Index {
    return switch (ast.nodeTag(node)) {
        .@"try", .@"nosuspend", .@"comptime" => unwrapStatement(ast, ast.nodeData(node).node),
        else => node,
    };
}

fn markReferencedIdentifiers(
    ast: Ast,
    node: Ast.Node.Index,
    names: *std.StringHashMapUnmanaged(bool),
) void {
    var token = ast.firstToken(node);
    const last_token = ast.lastToken(node);

    while (token <= last_token) : (token += 1) {
        if (ast.tokenTag(token) != .identifier) {
            continue;
        }

        if (names.getPtr(identifierText(ast, token))) |referenced| {
            referenced.* = true;
        }
    }
}

fn nodeReferencesIdentifier(ast: Ast, node: Ast.Node.Index, name: []const u8) bool {
    var token = ast.firstToken(node);
    const last_token = ast.lastToken(node);

    while (token <= last_token) : (token += 1) {
        if (ast.tokenTag(token) == .identifier and std.mem.eql(u8, identifierText(ast, token), name)) {
            return true;
        }
    }

    return false;
}

// ─── Paragraph Fixes ─────────────────────────────────────────────────

fn paragraphGapFix(
    allocator: std.mem.Allocator,
    file: *const LintContext,
    owner_token: Ast.TokenIndex,
    left_token: Ast.TokenIndex,
    right_token: Ast.TokenIndex,
    constraint: Constraint,
) std.mem.Allocator.Error!?Rule.Fix {
    switch (constraint) {
        .unconstrained => return null,
        .required_blank => return try lineBreakFix(allocator, file, owner_token, left_token, right_token, true),

        .forbidden_blank => {
            const left_line = file.tokenLocation(left_token).line;
            const right_line = file.tokenLocation(right_token).line;

            if (right_line <= left_line + 1) {
                return null;
            }

            const source_start_offset = file.line_start_offsets[left_line + 1];
            const source_end_offset = file.line_start_offsets[right_line];

            var replacement: std.ArrayList(u8) = .empty;
            defer replacement.deinit(allocator);

            // Remove complete empty physical lines only. Comment lines retain
            // every byte, including indentation, trailing spaces, and CRLF.
            // Neither statement's own line is part of this replacement.
            for (left_line + 1..right_line) |line| {
                if (std.mem.trim(u8, file.lineText(line), " \t\r").len == 0) {
                    continue;
                }

                try replacement.appendSlice(
                    allocator,
                    file.ast.source[file.line_start_offsets[line]..file.line_start_offsets[line + 1]],
                );
            }

            if (replacement.items.len == source_end_offset - source_start_offset) {
                return null;
            }

            const owned_replacement = try replacement.toOwnedSlice(allocator);

            return .{
                .range = .{
                    .start_offset = source_start_offset,
                    .end_offset = source_end_offset,
                },
                .replacement = owned_replacement,
            };
        },
    }
}

fn lineBreakFix(
    allocator: std.mem.Allocator,
    file: *const LintContext,
    owner_token: Ast.TokenIndex,
    left_token: Ast.TokenIndex,
    right_token: Ast.TokenIndex,
    blank_line: bool,
) std.mem.Allocator.Error!?Rule.Fix {
    const ast = file.ast;
    const left_line = file.tokenLocation(left_token).line;
    const right_line = file.tokenLocation(right_token).line;
    const layout = file.lineLayout(left_token);

    if (left_line != right_line) {
        if (!blank_line) {
            return null;
        }

        // Insert after the preceding statement's complete line, not at the
        // next statement or arm's first token: attached comments stay attached.
        const replacement = try allocator.dupe(u8, layout.newline);

        const insertion_offset = file.line_start_offsets[left_line + 1];

        return .{
            .range = .{
                .start_offset = insertion_offset,
                .end_offset = insertion_offset,
            },
            .replacement = replacement,
        };
    }

    const source_start_offset = ast.tokenStart(left_token) + ast.tokenSlice(left_token).len;
    const source_end_offset = ast.tokenStart(right_token);

    if (std.mem.trim(u8, ast.source[source_start_offset..source_end_offset], " \t").len != 0) {
        return null;
    }

    // Splitting statements sharing one line also requires indentation choices.
    // Keep the existing statement indentation; add one level only when the
    // containing block or switch also opens on this physical line.
    const extra_indent = if (file.areTokensOnSameLine(owner_token, left_token)) layout.indent_unit else "";

    const replacement = try std.fmt.allocPrint(allocator, "{s}{s}{s}{s}", .{
        layout.newline,
        if (blank_line) layout.newline else "",
        layout.indent,
        extra_indent,
    });

    return .{
        .range = .{
            .start_offset = @intCast(source_start_offset),
            .end_offset = @intCast(source_end_offset),
        },
        .replacement = replacement,
    };
}

fn terminatedToken(ast: Ast, last_token: Ast.TokenIndex, right_token: Ast.TokenIndex) Ast.TokenIndex {
    // AST expression spans omit statement semicolons and switch-arm commas.
    // Those tokens may be on a separate line; that line still belongs to the
    // preceding statement, never to the editable paragraph gap.
    const next_token = last_token + 1;
    if (next_token < right_token and
        (ast.tokenTag(next_token) == .semicolon or ast.tokenTag(next_token) == .comma))
    {
        return next_token;
    }

    return last_token;
}

fn hasParagraphBreak(
    file: *const LintContext,
    left_token: Ast.TokenIndex,
    right_token: Ast.TokenIndex,
) bool {
    const left_line = file.tokenLocation(left_token).line;
    var paragraph_line = file.tokenLocation(right_token).line;

    // A contiguous line-comment block belongs to the following statement. The
    // empty separator must precede those comments, not split them from the code
    // they explain.
    while (paragraph_line > left_line + 1) {
        const previous = std.mem.trim(u8, file.lineText(paragraph_line - 1), " \t\r");
        if (!std.mem.startsWith(u8, previous, "//")) {
            break;
        }

        paragraph_line -= 1;
    }

    var line = left_line + 1;

    while (line < paragraph_line) : (line += 1) {
        if (std.mem.trim(u8, file.lineText(line), " \t\r").len == 0) {
            return true;
        }
    }

    return false;
}

// ─── Switch Arm Paragraphs ──────────────────────────────────────────

fn lintSwitchArmParagraphs(
    file_allocator: std.mem.Allocator,
    context: *LintContext,
) Rule.Error!void {
    const file = context;
    const ast = file.ast;

    for (ast.nodes.items(.tag), 0..) |tag, index_usize| {
        if (tag != .@"switch" and tag != .switch_comma) {
            continue;
        }

        const node: Ast.Node.Index = @enumFromInt(index_usize);
        const switch_node = ast.switchFull(node);
        const cases = switch_node.ast.cases;

        if (cases.len < 2) {
            continue;
        }

        for (cases[1..], 1..) |right, index| {
            const left = cases[index - 1];
            const left_case = switchCase(ast, left);
            const right_case = switchCase(ast, right);

            // Only block → block needs a separator; either simple arm ends the run.
            if (!nodeIsBlock(ast, left_case.ast.target_expr) or
                !nodeIsBlock(ast, right_case.ast.target_expr))
            {
                continue;
            }

            const right_token = ast.firstToken(right);
            const left_token = terminatedToken(ast, ast.lastToken(left), right_token);

            if (hasParagraphBreak(file, left_token, right_token)) {
                continue;
            }

            const fix = if (context.fixes_enabled)
                try lineBreakFix(file_allocator, file, ast.firstToken(node), left_token, right_token, true)
            else
                null;

            try context.report(.{
                .token = right_token,
                .message = "adjacent braced switch arms must be separated by a blank line",
                .help = "insert a blank line before this switch arm",
                .fix = fix,
            });
        }
    }
}

fn switchCase(ast: Ast, node: Ast.Node.Index) Ast.full.SwitchCase {
    return switch (ast.nodeTag(node)) {
        .switch_case_one, .switch_case_inline_one => ast.switchCaseOne(node),
        .switch_case, .switch_case_inline => ast.switchCase(node),
        else => unreachable,
    };
}

const DiagnosticText = struct {
    message: []const u8,
    help: ?[]const u8 = null,
    note: ?[]const u8 = null,
};

fn descriptionFor(
    file: *const LintContext,
    left: Ast.Node.Index,
    right: Ast.Node.Index,
    reason: Reason,
) DiagnosticText {
    const ast = file.ast;

    return switch (reason) {
        .none => unreachable,
        .exit_paragraph => exitDescription(ast, right),
        .control_paragraph => controlDescription(ast, left, right),
        .scope_paragraph => if (category(ast, right) == .scope)
            .{
                .message = "standalone block must start a new paragraph",
                .help = "insert a blank line before this block",
            }
        else
            .{
                .message = "standalone block must end its paragraph",
                .help = "insert a blank line after the preceding block",
            },
        .declaration_paragraph => if (category(ast, right) == .declaration)
            .{
                .message = "declaration must start a new paragraph",
                .help = "insert a blank line before this declaration",
            }
        else
            .{
                .message = "declaration must end its paragraph",
                .help = "insert a blank line after the preceding declaration",
            },
        .multiline_declaration => if (category(ast, right) == .declaration and isMultiline(file, right))
            .{
                .message = "multiline declaration must start a new paragraph",
                .help = "insert a blank line before this declaration",
            }
        else
            .{
                .message = "multiline declaration must end its paragraph",
                .help = "insert a blank line after the preceding declaration",
            },
        .call_paragraph => if (category(ast, right) == .call)
            .{
                .message = "call must start a new paragraph",
                .help = "insert a blank line before this call",
            }
        else
            .{
                .message = "call group must end its paragraph",
                .help = "insert a blank line after the preceding call",
            },
        .multiline_call => if (category(ast, right) == .call and isMultiline(file, right))
            .{
                .message = "multiline call must start a new paragraph",
                .help = "insert a blank line before this call",
            }
        else
            .{
                .message = "multiline call must end its paragraph",
                .help = "insert a blank line after the preceding call",
            },
        .assignment_paragraph => if (category(ast, right) == .assignment)
            .{
                .message = "assignment must start a new paragraph",
                .help = "insert a blank line before this assignment",
            }
        else
            .{
                .message = "assignment group must end its paragraph",
                .help = "insert a blank line after the preceding assignment",
            },
        .cleanup_before => .{
            .message = "resource setup must start a new paragraph",
            .help = "insert a blank line before this setup statement",
        },
        .cleanup_internal => if (ast.nodeTag(right) == .@"errdefer")
            .{
                .message = "`errdefer` must stay with its setup statement",
                .help = "remove the blank line before this `errdefer`",
            }
        else
            .{
                .message = "`defer` must stay with its setup statement",
                .help = "remove the blank line before this `defer`",
            },
        .cleanup_after => .{
            .message = "cleanup group must end its paragraph",
            .help = "insert a blank line after the preceding cleanup registration",
        },
        .undefined_before => .{
            .message = "`var = undefined` group must start a new paragraph",
            .help = "insert a blank line before this storage declaration",
        },
        .undefined_internal => if (undefinedVariableNameToken(ast, right) != null)
            .{
                .message = "`var = undefined` declarations must stay together",
                .help = "remove the blank line before this declaration",
            }
        else
            .{
                .message = "first use must stay with its `var = undefined` group",
                .help = "remove the blank line before this first-use statement",
            },
        .undefined_after => .{
            .message = "undefined-storage initialization group must end its paragraph",
            .help = "insert a blank line after the first-use statement",
        },
        .producer_guard => .{
            .message = "early-exit guard must stay with its producer declaration",
            .help = "remove the blank line before this guard",
        },
        .assertion_pair => .{
            .message = "assertion must stay with its producer declaration",
            .help = "remove the blank line before this assertion",
        },
        .assertion_after => .{
            .message = "producer-and-assertion pair must end its paragraph",
            .help = "insert a blank line after the preceding assertion",
        },
    };
}

fn controlDescription(ast: Ast, left: Ast.Node.Index, right: Ast.Node.Index) DiagnosticText {
    const starts_paragraph = category(ast, right) == .control;
    const control = unwrapStatement(ast, if (starts_paragraph) right else left);

    return switch (ast.nodeTag(control)) {
        .if_simple, .@"if" => if (starts_paragraph)
            .{
                .message = "`if` must start a new paragraph",
                .help = "insert a blank line before this `if`",
            }
        else
            .{
                .message = "`if` must end its paragraph",
                .help = "insert a blank line after the preceding `if`",
            },
        .while_simple, .while_cont, .@"while" => if (starts_paragraph)
            .{
                .message = "`while` must start a new paragraph",
                .help = "insert a blank line before this `while`",
            }
        else
            .{
                .message = "`while` must end its paragraph",
                .help = "insert a blank line after the preceding `while`",
            },
        .for_simple, .@"for" => if (starts_paragraph)
            .{
                .message = "`for` must start a new paragraph",
                .help = "insert a blank line before this `for`",
            }
        else
            .{
                .message = "`for` must end its paragraph",
                .help = "insert a blank line after the preceding `for`",
            },
        .@"switch", .switch_comma => if (starts_paragraph)
            .{
                .message = "`switch` must start a new paragraph",
                .help = "insert a blank line before this `switch`",
            }
        else
            .{
                .message = "`switch` must end its paragraph",
                .help = "insert a blank line after the preceding `switch`",
            },
        else => unreachable,
    };
}

fn exitDescription(ast: Ast, node: Ast.Node.Index) DiagnosticText {
    return switch (ast.nodeTag(unwrapStatement(ast, node))) {
        .@"return" => .{
            .message = "`return` must start a new paragraph",
            .help = "insert a blank line before this `return`",
            .note = "keep comments attached to the return below the blank line",
        },
        .@"break" => .{
            .message = "`break` must start a new paragraph",
            .help = "insert a blank line before this `break`",
            .note = "keep comments attached to the break below the blank line",
        },
        .@"continue" => .{
            .message = "`continue` must start a new paragraph",
            .help = "insert a blank line before this `continue`",
            .note = "keep comments attached to the continue below the blank line",
        },
        .unreachable_literal => .{
            .message = "`unreachable` must start a new paragraph",
            .help = "insert a blank line before this `unreachable`",
            .note = "keep comments attached to the exit below the blank line",
        },
        else => unreachable,
    };
}
