const std = @import("std");
const assert = std.debug.assert;
const Ast = std.zig.Ast;
const Diagnostics = @import("../Diagnostics.zig");
const FileContext = @import("../FileContext.zig");
const Fixes = @import("../Fixes.zig");

/// Keep statement control flow braced, complex value selection multiline, and
/// explicit or implicit exits visible. Direct switch arms remain compact.
pub fn lint(
    file_allocator: std.mem.Allocator,
    report: *Diagnostics.Report,
) std.mem.Allocator.Error!void {
    const file = report.file;
    const ast = file.ast;

    const node_flags = try file_allocator.alloc(u8, ast.nodes.len);
    defer file_allocator.free(node_flags);

    @memset(node_flags, 0);

    const pending_nodes = try file_allocator.alloc(Ast.Node.Index, ast.nodes.len);
    defer file_allocator.free(pending_nodes);

    const visible_try_tokens = try file_allocator.alloc(bool, ast.tokens.len);
    defer file_allocator.free(visible_try_tokens);

    @memset(visible_try_tokens, false);

    const violation_kinds = try file_allocator.alloc(Violation, ast.tokens.len);
    defer file_allocator.free(violation_kinds);

    @memset(violation_kinds, .none);

    // Check-only scans retain the one-byte classification. Fix collection also
    // remembers the owning construct, avoiding a parent search per diagnostic.
    const fix_nodes: []Ast.Node.Index = if (report.fixes != null)
        try file_allocator.alloc(Ast.Node.Index, ast.tokens.len)
    else
        &.{};
    defer file_allocator.free(fix_nodes);

    const violations: Violations = .{ .kinds = violation_kinds, .fix_nodes = fix_nodes };

    // ── Classify Context ──
    //
    // The AST has child indexes but no parent indexes. Seed every block
    // statement, then propagate statement position only through constructs
    // whose direct children execute as statements. The fixed worklist holds at
    // most one entry per node, avoiding a parent scan for every control node.
    classifyNodeContext(ast, node_flags, pending_nodes, visible_try_tokens);

    // ── Record Violations ──
    //
    // Structural checks run before token-level exit checks. A parent that
    // requires braces marks its direct body as covered, preventing a second
    // diagnostic for the same hidden `return` or `try`.
    recordStructuralViolations(file, node_flags, violations);
    recordHiddenFlowViolations(file, node_flags, visible_try_tokens, violations);

    // ── Emit In Source Order ──
    //
    // Recording one enum per token makes ordering a linear token scan rather
    // than a comparison sort over diagnostics.
    for (violations.kinds, 0..) |violation, token_usize| {
        if (violation == .none) {
            continue;
        }

        const description = descriptionFor(violation);
        const token: Ast.TokenIndex = @intCast(token_usize);

        const fix = if (report.fixes != null)
            try proposeFix(file_allocator, file, node_flags, violations.fix_nodes[token], token, violation)
        else
            null;
        defer if (fix) |edit| {
            file_allocator.free(edit.replacement);
        };

        try report.add(.{
            .token = token,
            .rule_name = "visible_control_flow",
            .message = description.message,
            .help = description.help,
            .note = description.note,
            .fix = fix,
        });
    }
}

const Violations = struct {
    kinds: []Violation,
    fix_nodes: []Ast.Node.Index,
};

const Violation = enum(u8) {
    none,
    unbraced_if_body,
    unbraced_while_body,
    unbraced_for_body,
    unbraced_if_exit_branch,
    inline_complex_if_value,
    unbraced_orelse_fallback,
    unbraced_catch_fallback,
    inline_complex_orelse_value,
    inline_complex_catch_value,
    inline_nested_try,
    unbraced_errdefer_capture,
    hidden_return,
    hidden_break,
    hidden_continue,
    hidden_unreachable,
};

const node_flag_statement: u8 = 1 << 0;
const node_flag_else_if: u8 = 1 << 1;
const node_flag_switch_arm: u8 = 1 << 2;
const node_flag_covered: u8 = 1 << 3;
const node_flag_block_child: u8 = 1 << 4;

fn classifyNodeContext(
    ast: Ast,
    node_flags: []u8,
    pending_nodes: []Ast.Node.Index,
    visible_try_tokens: []bool,
) void {
    var pending_count: usize = 0;

    for (ast.nodes.items(.tag), 0..) |tag, node_usize| {
        const node: Ast.Node.Index = @enumFromInt(node_usize);

        var block_buffer: [2]Ast.Node.Index = undefined;
        if (ast.blockStatements(&block_buffer, node)) |statements| {
            // Every direct block child is in statement position:
            //
            //   {
            //       const value = compute();  <- statement
            //       if (ready) { ... }         <- statement
            //   }
            for (statements) |statement| {
                node_flags[@intFromEnum(statement)] |= node_flag_block_child;

                markStatementPosition(node_flags, pending_nodes, &pending_count, statement);
            }
        }

        if (ast.fullIf(node)) |if_node| {
            if (if_node.ast.else_expr.unwrap()) |else_body| {
                if (ast.fullIf(else_body) != null) {
                    // Record the syntactic `else if` relationship independently
                    // of value/statement context. The chain root owns multiline
                    // validation so one compact chain produces one diagnostic:
                    //
                    //   if (a) .one else if (b) .two else .three
                    //                       ^^ `else_body`; marked
                    node_flags[@intFromEnum(else_body)] |= node_flag_else_if;
                }
            }
        }

        if (ast.fullSwitchCase(node)) |switch_case| {
            // Mark only the direct target so token-level `try` and exit checks
            // permit compact arms. Structural nodes still check their own
            // children, which are not marked:
            //
            //   .start => return,                    <- target; exempt exit
            //   .stop => if (ready) stopWorker(),    <- `if` target is visited
            //                ^^^^^^^^^^^ body; not marked and still checked
            const target = switch_case.ast.target_expr;

            node_flags[@intFromEnum(target)] |= node_flag_switch_arm;

            markLeadingTry(ast, visible_try_tokens, target);
        }

        if (directValue(ast, tag, node)) |value| {
            // A leading `try` remains the principal operation when its direct
            // value continues with `orelse` or another lower-precedence node:
            //
            //   const entry = try walker.next(io) orelse {
            //                 ^^^ first token of the whole initializer; visible
            //       break;
            //   };
            //
            // In contrast, an aggregate field or call argument does not begin
            // its containing value and remains subject to the nested-try rule.
            markLeadingTry(ast, visible_try_tokens, value);
        }
    }

    while (pending_count > 0) {
        pending_count -= 1;

        const node = pending_nodes[pending_count];
        const tag = ast.nodeTag(node);

        switch (tag) {
            .if_simple, .@"if" => {
                const if_node = ast.fullIf(node) orelse {
                    unreachable;
                };

                markStatementPosition(node_flags, pending_nodes, &pending_count, if_node.ast.then_expr);

                if (if_node.ast.else_expr.unwrap()) |else_body| {
                    markStatementPosition(node_flags, pending_nodes, &pending_count, else_body);
                }
            },

            .while_simple, .while_cont, .@"while" => {
                const while_node = ast.fullWhile(node) orelse {
                    unreachable;
                };

                markStatementPosition(node_flags, pending_nodes, &pending_count, while_node.ast.then_expr);

                if (while_node.ast.else_expr.unwrap()) |else_body| {
                    markStatementPosition(node_flags, pending_nodes, &pending_count, else_body);
                }
            },

            .for_simple, .@"for" => {
                const for_node = ast.fullFor(node) orelse {
                    unreachable;
                };

                markStatementPosition(node_flags, pending_nodes, &pending_count, for_node.ast.then_expr);

                if (for_node.ast.else_expr.unwrap()) |else_body| {
                    markStatementPosition(node_flags, pending_nodes, &pending_count, else_body);
                }
            },

            .@"switch", .switch_comma => {
                const switch_node = ast.fullSwitch(node) orelse {
                    unreachable;
                };

                for (switch_node.ast.cases) |case_node| {
                    const switch_case = ast.fullSwitchCase(case_node) orelse {
                        unreachable;
                    };

                    markStatementPosition(node_flags, pending_nodes, &pending_count, switch_case.ast.target_expr);
                }
            },

            .@"catch", .@"orelse" => {
                // A fallback inherits statement position from the complete
                // expression; its left side remains a value-producing attempt:
                //
                //   operation() catch logFailure();
                //                     ^^^^^^^^^^^^ right child; statement
                const fallback = ast.nodeData(node).node_and_node[1];

                markStatementPosition(node_flags, pending_nodes, &pending_count, fallback);
            },

            .@"defer" => {
                markStatementPosition(node_flags, pending_nodes, &pending_count, ast.nodeData(node).node);
            },

            .@"errdefer" => {
                markStatementPosition(node_flags, pending_nodes, &pending_count, ast.nodeData(node).opt_token_and_node[1]);
            },

            .@"comptime", .@"nosuspend", .@"suspend", .@"resume" => {
                markStatementPosition(node_flags, pending_nodes, &pending_count, ast.nodeData(node).node);
            },

            .grouped_expression => {
                markStatementPosition(node_flags, pending_nodes, &pending_count, ast.nodeData(node).node_and_token[0]);
            },

            else => {},
        }
    }
}

fn markStatementPosition(
    node_flags: []u8,
    pending_nodes: []Ast.Node.Index,
    pending_count: *usize,
    node: Ast.Node.Index,
) void {
    const node_usize = @intFromEnum(node);
    if (node_flags[node_usize] & node_flag_statement != 0) {
        return;
    }

    node_flags[node_usize] |= node_flag_statement;

    assert(pending_count.* < pending_nodes.len);

    pending_nodes[pending_count.*] = node;
    pending_count.* += 1;
}

fn recordStructuralViolations(
    file: *const FileContext,
    node_flags: []u8,
    violations: Violations,
) void {
    const ast = file.ast;

    for (ast.nodes.items(.tag), 0..) |tag, node_usize| {
        const node: Ast.Node.Index = @enumFromInt(node_usize);

        switch (tag) {
            .if_simple, .@"if" => recordIfViolations(file, node_flags, violations, node),
            .while_simple, .while_cont, .@"while" => recordWhileViolations(ast, node_flags, violations, node),
            .for_simple, .@"for" => recordForViolations(ast, node_flags, violations, node),
            .@"catch", .@"orelse" => recordFallbackViolation(file, node_flags, violations, node),
            .@"errdefer" => recordErrdeferViolation(ast, node_flags, violations, node),

            else => {},
        }
    }
}

fn recordIfViolations(
    file: *const FileContext,
    node_flags: []u8,
    violations: Violations,
    node: Ast.Node.Index,
) void {
    const ast = file.ast;

    const if_node = ast.fullIf(node) orelse {
        unreachable;
    };

    const else_body_optional = if_node.ast.else_expr.unwrap();

    const is_statement = node_flags[@intFromEnum(node)] & node_flag_statement != 0 or
        else_body_optional == null;

    if (is_statement) {
        // A physical line break still leaves the body structurally unbraced:
        //
        //   if (ready)
        //       startWorker();  <- rejected
        //
        //   if (ready) {
        //       startWorker();  <- accepted
        //   }
        recordRequiredBlock(ast, node_flags, violations, if_node.ast.then_expr, node, .unbraced_if_body);

        if (else_body_optional) |else_body| {
            // `else if` is the one unbraced statement branch: the nested `if`
            // is checked independently, and each of its own bodies needs braces.
            if (ast.fullIf(else_body) == null) {
                recordRequiredBlock(ast, node_flags, violations, else_body, node, .unbraced_if_body);
            }
        }

        return;
    }

    var exit_branch_reported = recordIfExitBranch(ast, node_flags, violations, if_node.ast.then_expr, node);

    if (else_body_optional) |else_body| {
        if (ast.fullIf(else_body) == null) {
            exit_branch_reported = recordIfExitBranch(ast, node_flags, violations, else_body, node) or exit_branch_reported;
        }
    }

    if (exit_branch_reported) {
        return;
    }

    if (node_flags[@intFromEnum(node)] & node_flag_else_if != 0 or
        !valueIfRequiresMultiline(ast, if_node))
    {
        return;
    }

    if (valueIfMultilineViolation(file, node)) |body| {
        // A complex value branch remains an expression; moving each branch
        // below its header exposes it without labeled value-producing blocks:
        //
        //   const worker = if (cached)
        //       getCachedWorker()
        //   else
        //       createWorker();
        recordViolationAtNode(ast, node_flags, violations, body, node, .inline_complex_if_value);
    }
}

fn recordIfExitBranch(
    ast: Ast,
    node_flags: []u8,
    violations: Violations,
    body: Ast.Node.Index,
    owner: Ast.Node.Index,
) bool {
    if (nodeIsBlock(ast, body)) {
        return false;
    }

    const effective_body = unwrapGroupedExpression(ast, body);
    if (!nodeIsExit(ast.nodeTag(effective_body))) {
        return false;
    }

    // Exits used as value branches require braces rather than line wrapping.
    // Transparent parentheses cannot turn the exit into an ordinary value:
    //
    //   const value = if (ready) return cached else fresh;     <- rejected
    //   const value = if (ready) { return cached; } else fresh; <- braced
    recordViolationAtNode(ast, node_flags, violations, effective_body, owner, .unbraced_if_exit_branch);

    return true;
}

fn recordWhileViolations(
    ast: Ast,
    node_flags: []u8,
    violations: Violations,
    node: Ast.Node.Index,
) void {
    const while_node = ast.fullWhile(node) orelse {
        unreachable;
    };

    recordRequiredBlock(ast, node_flags, violations, while_node.ast.then_expr, node, .unbraced_while_body);

    if (while_node.ast.else_expr.unwrap()) |else_body| {
        recordRequiredBlock(ast, node_flags, violations, else_body, node, .unbraced_while_body);
    }
}

fn recordForViolations(
    ast: Ast,
    node_flags: []u8,
    violations: Violations,
    node: Ast.Node.Index,
) void {
    const for_node = ast.fullFor(node) orelse {
        unreachable;
    };

    recordRequiredBlock(ast, node_flags, violations, for_node.ast.then_expr, node, .unbraced_for_body);

    if (for_node.ast.else_expr.unwrap()) |else_body| {
        recordRequiredBlock(ast, node_flags, violations, else_body, node, .unbraced_for_body);
    }
}

fn recordRequiredBlock(
    ast: Ast,
    node_flags: []u8,
    violations: Violations,
    body: Ast.Node.Index,
    owner: Ast.Node.Index,
    violation: Violation,
) void {
    if (nodeIsBlock(ast, body)) {
        return;
    }

    recordViolationAtNode(ast, node_flags, violations, body, owner, violation);
}

fn recordFallbackViolation(
    file: *const FileContext,
    node_flags: []u8,
    violations: Violations,
    node: Ast.Node.Index,
) void {
    const ast = file.ast;
    const fallback = ast.nodeData(node).node_and_node[1];
    const effective_fallback = unwrapGroupedExpression(ast, fallback);

    if (nodeIsBlock(ast, effective_fallback)) {
        return;
    }

    const tag = ast.nodeTag(node);
    const fallback_is_exit = nodeIsExit(ast.nodeTag(effective_fallback));
    const is_statement = node_flags[@intFromEnum(node)] & node_flag_statement != 0;

    if (is_statement or fallback_is_exit) {
        // Statement work and exits need a real scope boundary. A newline alone
        // does not make the fallback explicit:
        //
        //   operation() catch |err|
        //       logFailure(err);       <- rejected
        //
        //   operation() catch |err| {
        //       logFailure(err);       <- accepted
        //   };
        const violation: Violation = switch (tag) {
            .@"catch" => .unbraced_catch_fallback,
            .@"orelse" => .unbraced_orelse_fallback,
            else => unreachable,
        };

        recordViolationAtNode(
            ast,
            node_flags,
            violations,
            if (fallback_is_exit) effective_fallback else fallback,
            node,
            violation,
        );

        return;
    }

    if (!valueIsSimple(ast, fallback) and
        file.tokensOnSameLine(ast.nodeMainToken(node), ast.firstToken(fallback)))
    {
        // A compound value may stay unbraced, but it starts below the fallback
        // operator. Classifying the root as simple or compound also prevents
        // parentheses or arithmetic from hiding a nested call:
        //
        //   const worker = optional orelse
        //       createWorker();
        const violation: Violation = switch (tag) {
            .@"catch" => .inline_complex_catch_value,
            .@"orelse" => .inline_complex_orelse_value,
            else => unreachable,
        };

        recordViolationAtNode(ast, node_flags, violations, fallback, node, violation);
    }
}

fn recordErrdeferViolation(
    ast: Ast,
    node_flags: []u8,
    violations: Violations,
    node: Ast.Node.Index,
) void {
    const payload_token, const deferred = ast.nodeData(node).opt_token_and_node;

    if (payload_token.unwrap() == null or nodeIsBlock(ast, deferred)) {
        return;
    }

    // Plain `errdefer cleanup();` is already a direct registration. A capture
    // introduces a handler body and therefore requires its boundary:
    //
    //   errdefer |err| logFailure(err);     <- rejected
    //   errdefer |err| { logFailure(err); } <- braced
    recordViolationAtNode(ast, node_flags, violations, deferred, node, .unbraced_errdefer_capture);
}

fn valueIfRequiresMultiline(ast: Ast, if_node: Ast.full.If) bool {
    if (if_node.payload_token != null or if_node.error_token != null) {
        return true;
    }

    const else_body = if_node.ast.else_expr.unwrap() orelse {
        return false;
    };

    if (ast.fullIf(else_body) != null) {
        return true;
    }

    return !valueIsSimple(ast, if_node.ast.then_expr) or !valueIsSimple(ast, else_body);
}

fn unwrapGroupedExpression(ast: Ast, initial_node: Ast.Node.Index) Ast.Node.Index {
    var node = initial_node;

    while (ast.nodeTag(node) == .grouped_expression) {
        node = ast.nodeData(node).node_and_token[0];
    }

    return node;
}

fn valueIsSimple(ast: Ast, initial_node: Ast.Node.Index) bool {
    var node = initial_node;

    while (true) {
        switch (ast.nodeTag(node)) {
            .identifier,
            .enum_literal,
            .number_literal,
            .char_literal,
            .string_literal,
            .multiline_string_literal,
            .error_value,
            .anyframe_literal,
            => return true,

            .grouped_expression => {
                // Parentheses do not turn a leaf into a compound value, but
                // they also cannot disguise one: `(fallback)` remains simple,
                // while `(createFallback())` reaches the call and is rejected.
                node = ast.nodeData(node).node_and_token[0];
            },

            .field_access => {
                // Field chains remain one leaf-like value:
                //
                //   options.output.mode
                //   ^^^^^^^^^^^^^^^^^^^ accepted as a simple branch
                node = ast.nodeData(node).node_and_token[0];
            },

            else => return false,
        }
    }
}

fn valueIfMultilineViolation(file: *const FileContext, initial_if: Ast.Node.Index) ?Ast.Node.Index {
    const ast = file.ast;
    var node = initial_if;

    while (true) {
        const if_node = ast.fullIf(node) orelse {
            unreachable;
        };

        const then_body = if_node.ast.then_expr;

        if (!nodeIsBlock(ast, then_body) and bodySharesPrecedingTokenLine(file, then_body)) {
            // The token immediately before a then-body is the closing header
            // token, including a closing capture pipe:
            //
            //   if (optional) |value| value else fallback
            //                         ^^^^^ `then_body`; same line, rejected
            return then_body;
        }

        const else_body = if_node.ast.else_expr.unwrap() orelse {
            return null;
        };

        if (ast.fullIf(else_body) != null) {
            // Canonical chains retain `else if` on one line. Continue into the
            // nested conditional and check its then-body and eventual leaf:
            //
            //   else if (ready)
            //       .active
            //       ^^^^^^^ next iteration checks this body
            node = else_body;

            continue;
        }

        if (!nodeIsBlock(ast, else_body) and
            file.tokensOnSameLine(if_node.else_token, ast.firstToken(else_body)))
        {
            // A leaf else-body starts below `else`:
            //
            //   else
            //       createWorker();
            //       ^^^^^^^^^^^^ accepted on the following line
            return else_body;
        }

        return null;
    }
}

fn bodySharesPrecedingTokenLine(file: *const FileContext, body: Ast.Node.Index) bool {
    const first_token = file.ast.firstToken(body);
    assert(first_token > 0);

    return file.tokensOnSameLine(first_token - 1, first_token);
}

fn recordHiddenFlowViolations(
    file: *const FileContext,
    node_flags: []u8,
    visible_try_tokens: []const bool,
    violations: Violations,
) void {
    for (file.ast.nodes.items(.tag), 0..) |tag, node_usize| {
        const node: Ast.Node.Index = @enumFromInt(node_usize);

        switch (tag) {
            .@"try" => recordHiddenTryViolation(file, node_flags, visible_try_tokens, violations, node),
            .@"return", .@"break", .@"continue", .unreachable_literal => recordHiddenExitViolation(file, node_flags, violations, node),

            else => {},
        }
    }
}

fn recordHiddenTryViolation(
    file: *const FileContext,
    node_flags: []const u8,
    visible_try_tokens: []const bool,
    violations: Violations,
    node: Ast.Node.Index,
) void {
    const ast = file.ast;
    const flags = node_flags[@intFromEnum(node)];

    if (flags & (node_flag_covered | node_flag_switch_arm) != 0) {
        return;
    }

    const token = ast.nodeMainToken(node);
    if (visible_try_tokens[token]) {
        return;
    }

    if (token == 0 or !file.tokensOnSameLine(token - 1, token)) {
        return;
    }

    // Nested `try` hides an implicit error return among another operation's
    // arguments. Starting it on a physical line exposes that exit:
    //
    //   configure(try loadConfig());  <- rejected
    //
    //   configure(
    //       try loadConfig(),         <- accepted
    //   );
    recordViolation(violations, token, node, .inline_nested_try);
}

fn recordHiddenExitViolation(
    file: *const FileContext,
    node_flags: []const u8,
    violations: Violations,
    node: Ast.Node.Index,
) void {
    const ast = file.ast;
    const flags = node_flags[@intFromEnum(node)];

    if (flags & (node_flag_covered | node_flag_switch_arm) != 0) {
        return;
    }

    const token = ast.nodeMainToken(node);
    if (token == 0 or !file.tokensOnSameLine(token - 1, token)) {
        return;
    }

    const violation: Violation = switch (ast.nodeTag(node)) {
        .@"return" => .hidden_return,
        .@"break" => .hidden_break,
        .@"continue" => .hidden_continue,
        .unreachable_literal => .hidden_unreachable,
        else => unreachable,
    };

    recordViolation(violations, token, node, violation);
}

fn recordViolationAtNode(
    ast: Ast,
    node_flags: []u8,
    violations: Violations,
    node: Ast.Node.Index,
    fix_node: Ast.Node.Index,
    violation: Violation,
) void {
    node_flags[@intFromEnum(node)] |= node_flag_covered;

    recordViolation(violations, ast.firstToken(node), fix_node, violation);
}

fn recordViolation(
    violations: Violations,
    token: Ast.TokenIndex,
    fix_node: Ast.Node.Index,
    violation: Violation,
) void {
    if (violations.kinds[token] == .none) {
        violations.kinds[token] = violation;

        if (violations.fix_nodes.len != 0) {
            violations.fix_nodes[token] = fix_node;
        }
    }
}

// ─── Fix Proposals ───────────────────────────────────────────────────

fn proposeFix(
    allocator: std.mem.Allocator,
    file: *const FileContext,
    node_flags: []const u8,
    owner: Ast.Node.Index,
    token: Ast.TokenIndex,
    violation: Violation,
) std.mem.Allocator.Error!?Fixes.ProposedEdit {
    switch (violation) {
        .none => return null,
        .inline_complex_if_value => return try multilineIfFix(allocator, file, owner),

        .inline_complex_orelse_value,
        .inline_complex_catch_value,
        .inline_nested_try,
        .hidden_return,
        .hidden_break,
        .hidden_continue,
        .hidden_unreachable,
        => {
            const ast = file.ast;
            const layout = lineLayout(file, token);
            const token_offset = ast.tokenStart(token);
            var source_start_offset = token_offset;

            // Change only horizontal trivia immediately before the token.
            // No extraction, token changes, or movement across a line comment.
            while (source_start_offset > 0 and
                (ast.source[source_start_offset - 1] == ' ' or ast.source[source_start_offset - 1] == '\t'))
            {
                source_start_offset -= 1;
            }

            const after_statement = token > 0 and ast.tokenTag(token - 1) == .semicolon;
            const replacement = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
                layout.newline,
                layout.indent,
                if (after_statement) "" else layout.indent_unit,
            });

            return .{
                .source_start_offset = source_start_offset,
                .removal_size = token_offset - source_start_offset,
                .replacement = replacement,
            };
        },

        .unbraced_if_body,
        .unbraced_while_body,
        .unbraced_for_body,
        .unbraced_if_exit_branch,
        .unbraced_orelse_fallback,
        .unbraced_catch_fallback,
        .unbraced_errdefer_capture,
        => return try bracedBodyFix(allocator, file, node_flags, owner, token, violation),
    }
}

fn multilineIfFix(
    allocator: std.mem.Allocator,
    file: *const FileContext,
    owner: Ast.Node.Index,
) std.mem.Allocator.Error!Fixes.ProposedEdit {
    const ast = file.ast;
    const layout = lineLayout(file, ast.firstToken(owner));
    const source_start_offset = ast.tokenStart(ast.firstToken(owner));
    const last_token = ast.lastToken(owner);
    const source_end_offset = ast.tokenStart(last_token) + ast.tokenSlice(last_token).len;

    var replacement: std.ArrayList(u8) = .empty;
    defer replacement.deinit(allocator);

    var source_offset: usize = source_start_offset;
    var node = owner;

    // One replacement owns the complete chain. Copy every expression and
    // comment verbatim, replacing only same-line gaps at branch boundaries.
    while (true) {
        const if_node = ast.fullIf(node) orelse {
            unreachable;
        };

        if (!nodeIsBlock(ast, if_node.ast.then_expr)) {
            try appendBranchBreak(
                allocator,
                file,
                &replacement,
                &source_offset,
                ast.firstToken(if_node.ast.then_expr),
                layout,
                true,
            );
        }

        const else_body = if_node.ast.else_expr.unwrap() orelse {
            break;
        };

        try appendBranchBreak(
            allocator,
            file,
            &replacement,
            &source_offset,
            if_node.else_token,
            layout,
            false,
        );

        if (ast.fullIf(else_body) != null) {
            node = else_body;

            continue;
        }

        if (!nodeIsBlock(ast, else_body)) {
            try appendBranchBreak(
                allocator,
                file,
                &replacement,
                &source_offset,
                ast.firstToken(else_body),
                layout,
                true,
            );
        }

        break;
    }

    try replacement.appendSlice(allocator, ast.source[source_offset..source_end_offset]);

    return .{
        .source_start_offset = source_start_offset,
        .removal_size = @intCast(source_end_offset - source_start_offset),
        .replacement = try replacement.toOwnedSlice(allocator),
    };
}

const LineLayout = struct {
    newline: []const u8,
    indent: []const u8,
    indent_unit: []const u8,
};

fn appendBranchBreak(
    allocator: std.mem.Allocator,
    file: *const FileContext,
    replacement: *std.ArrayList(u8),
    source_offset: *usize,
    token: Ast.TokenIndex,
    layout: LineLayout,
    continuation: bool,
) std.mem.Allocator.Error!void {
    if (token == 0 or !file.tokensOnSameLine(token - 1, token)) {
        return;
    }

    const source = file.ast.source;
    const token_offset = file.ast.tokenStart(token);
    var gap_offset: usize = token_offset;

    while (gap_offset > source_offset.* and
        (source[gap_offset - 1] == ' ' or source[gap_offset - 1] == '\t'))
    {
        gap_offset -= 1;
    }

    try replacement.appendSlice(allocator, source[source_offset.*..gap_offset]);
    try replacement.appendSlice(allocator, layout.newline);
    try replacement.appendSlice(allocator, layout.indent);

    if (continuation) {
        try replacement.appendSlice(allocator, layout.indent_unit);
    }

    source_offset.* = token_offset;
}

fn bracedBodyFix(
    allocator: std.mem.Allocator,
    file: *const FileContext,
    node_flags: []const u8,
    owner: Ast.Node.Index,
    token: Ast.TokenIndex,
    violation: Violation,
) std.mem.Allocator.Error!?Fixes.ProposedEdit {
    const ast = file.ast;

    const owns_semicolon = switch (violation) {
        .unbraced_if_body, .unbraced_while_body, .unbraced_for_body, .unbraced_errdefer_capture => true,
        else => false,
    };

    // A block-child control statement owns its final semicolon; a control
    // expression nested in `defer`, a switch arm, or a value may not. Leave
    // those statement rewrites alone rather than guess which terminator to remove.
    if (owns_semicolon and node_flags[@intFromEnum(owner)] & node_flag_block_child == 0) {
        return null;
    }

    const body = switch (violation) {
        .unbraced_if_body, .unbraced_if_exit_branch => body: {
            const if_node = ast.fullIf(owner) orelse {
                unreachable;
            };

            if (ast.firstToken(unwrapGroupedExpression(ast, if_node.ast.then_expr)) == token or
                ast.firstToken(if_node.ast.then_expr) == token)
            {
                break :body if_node.ast.then_expr;
            }

            break :body if_node.ast.else_expr.unwrap() orelse {
                unreachable;
            };
        },
        .unbraced_while_body => body: {
            const while_node = ast.fullWhile(owner) orelse {
                unreachable;
            };

            if (ast.firstToken(while_node.ast.then_expr) == token) {
                break :body while_node.ast.then_expr;
            }

            break :body while_node.ast.else_expr.unwrap() orelse {
                unreachable;
            };
        },
        .unbraced_for_body => body: {
            const for_node = ast.fullFor(owner) orelse {
                unreachable;
            };

            if (ast.firstToken(for_node.ast.then_expr) == token) {
                break :body for_node.ast.then_expr;
            }

            break :body for_node.ast.else_expr.unwrap() orelse {
                unreachable;
            };
        },
        .unbraced_orelse_fallback, .unbraced_catch_fallback => ast.nodeData(owner).node_and_node[1],
        .unbraced_errdefer_capture => ast.nodeData(owner).opt_token_and_node[1],
        else => unreachable,
    };

    // Calls, assignments, and exits gain no declarations with a new lifetime.
    // Do not wrap `defer`, declarations, nested control expressions, or other
    // scope-sensitive forms. Multiline bodies need a separate indentation policy.
    if (!bodyCanGainScope(ast, body)) {
        return null;
    }

    const first_token = ast.firstToken(body);
    const last_token = ast.lastToken(body);

    if (first_token == 0 or !file.tokensOnSameLine(first_token, last_token)) {
        return null;
    }

    // Scope-setting builtins can occur inside a call argument or return value,
    // not just as the body's root. A new block must not narrow their effect.
    for (first_token..last_token + 1) |index_usize| {
        const index: Ast.TokenIndex = @intCast(index_usize);

        if (ast.tokenTag(index) != .builtin) {
            continue;
        }

        const name = ast.tokenSlice(index);

        if (std.mem.startsWith(u8, name, "@set") or std.mem.eql(u8, name, "@branchHint")) {
            return null;
        }
    }

    const source_start_offset = ast.tokenStart(first_token - 1) + ast.tokenSlice(first_token - 1).len;
    const body_start_offset = ast.tokenStart(first_token);
    const body_end_offset = ast.tokenStart(last_token) + ast.tokenSlice(last_token).len;

    // Keep comments in their original scope and attached to their original
    // statement. For now, a comment at either boundary makes this fix ineligible.
    if (std.mem.trim(u8, ast.source[source_start_offset..body_start_offset], " \t\r\n").len != 0) {
        return null;
    }

    var source_end_offset = body_end_offset;
    var following_token = last_token + 1;

    if (ast.tokenTag(following_token) == .semicolon) {
        if (owns_semicolon and ast.lastToken(owner) == last_token) {
            source_end_offset = ast.tokenStart(following_token) + ast.tokenSlice(following_token).len;
        }

        following_token += 1;
    }

    if (std.mem.indexOf(u8, ast.source[body_end_offset..ast.tokenStart(following_token)], "//") != null) {
        return null;
    }

    const layout = lineLayout(file, ast.firstToken(owner));
    const replacement = try std.fmt.allocPrint(allocator, " {{{s}{s}{s}{s};{s}{s}}}", .{
        layout.newline,
        layout.indent,
        layout.indent_unit,
        ast.source[body_start_offset..body_end_offset],
        layout.newline,
        layout.indent,
    });

    return .{
        .source_start_offset = @intCast(source_start_offset),
        .removal_size = @intCast(source_end_offset - source_start_offset),
        .replacement = replacement,
    };
}

fn bodyCanGainScope(ast: Ast, initial_node: Ast.Node.Index) bool {
    var node = unwrapGroupedExpression(ast, initial_node);

    if (ast.nodeTag(node) == .@"try") {
        node = ast.nodeData(node).node;
    }

    return switch (ast.nodeTag(node)) {
        .call_one,
        .call_one_comma,
        .call,
        .call_comma,
        .builtin_call_two,
        .builtin_call_two_comma,
        .builtin_call,
        .builtin_call_comma,
        .assign,
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
        .@"return",
        .@"break",
        .@"continue",
        .unreachable_literal,
        => true,

        else => false,
    };
}

fn lineLayout(file: *const FileContext, token: Ast.TokenIndex) LineLayout {
    const location = file.tokenLocation(token);
    const text = file.lineText(location.line);
    const indent_size = text.len - std.mem.trimStart(u8, text, " \t").len;
    const indent = text[0..indent_size];

    // Prefer this line's terminator. At EOF, inherit the preceding one rather
    // than introducing LF into a CRLF file with no final newline.
    const newline_offset = if (location.line + 1 < file.line_starts.len)
        file.line_starts[location.line + 1]
    else
        file.line_starts[location.line];

    return .{
        .newline = if (newline_offset >= 2 and file.ast.source[newline_offset - 2] == '\r') "\r\n" else "\n",
        .indent = indent,
        .indent_unit = if (std.mem.indexOfScalar(u8, indent, '\t') != null) "\t" else "    ",
    };
}

fn nodeIsBlock(ast: Ast, node: Ast.Node.Index) bool {
    return switch (ast.nodeTag(node)) {
        .block_two, .block_two_semicolon, .block, .block_semicolon => true,
        else => false,
    };
}

fn markLeadingTry(ast: Ast, visible_try_tokens: []bool, value: Ast.Node.Index) void {
    const first_token = ast.firstToken(value);

    if (ast.tokenTag(first_token) == .keyword_try) {
        visible_try_tokens[first_token] = true;
    }
}

fn directValue(ast: Ast, tag: Ast.Node.Tag, node: Ast.Node.Index) ?Ast.Node.Index {
    if (ast.fullVarDecl(node)) |declaration| {
        return declaration.ast.init_node.unwrap();
    }

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
        => ast.nodeData(node).node_and_node[1],
        .assign_destructure => ast.nodeData(node).extra_and_node[1],
        .@"return" => ast.nodeData(node).opt_node.unwrap(),
        .@"break", .@"continue" => ast.nodeData(node).opt_token_and_opt_node[1].unwrap(),
        else => null,
    };
}

fn nodeIsExit(tag: Ast.Node.Tag) bool {
    return switch (tag) {
        .@"return", .@"break", .@"continue", .unreachable_literal => true,
        else => false,
    };
}

fn descriptionFor(violation: Violation) Diagnostics.Description {
    return switch (violation) {
        .none => unreachable,
        .unbraced_if_body => .{
            .message = "`if` body requires braces",
            .help = "wrap the body in `{ ... }`",
        },
        .unbraced_while_body => .{
            .message = "`while` body requires braces",
            .help = "wrap the body in `{ ... }`",
        },
        .unbraced_for_body => .{
            .message = "`for` body requires braces",
            .help = "wrap the body in `{ ... }`",
            .note = "`inline for` follows the same rule",
        },
        .unbraced_if_exit_branch => .{
            .message = "`if` branch exits without braces",
            .help = "wrap the branch in `{ ... }`",
        },
        .inline_complex_if_value => .{
            .message = "complex `if` branches must start on separate lines",
            .help = "move each branch expression to the line after `if` or `else`",
            .note = "simple leaf values may remain inline",
        },
        .unbraced_orelse_fallback => .{
            .message = "`orelse` fallback requires braces for exits or statement work",
            .help = "wrap the fallback in `{ ... }`",
        },
        .unbraced_catch_fallback => .{
            .message = "`catch` fallback requires braces for exits or statement work",
            .help = "wrap the fallback in `{ ... }`",
        },
        .inline_complex_orelse_value => .{
            .message = "complex `orelse` fallback must start on a new line",
            .help = "move the fallback after `orelse` or wrap it in `{ ... }`",
            .note = "simple leaf values may remain inline",
        },
        .inline_complex_catch_value => .{
            .message = "complex `catch` fallback must start on a new line",
            .help = "move the fallback after `catch` or wrap it in `{ ... }`",
            .note = "simple leaf values may remain inline",
        },
        .inline_nested_try => .{
            .message = "`try` is hidden inside another same-line expression",
            .help = "start `try` on a new line or assign its result before the outer expression",
        },
        .unbraced_errdefer_capture => .{
            .message = "captured `errdefer` handler requires braces",
            .help = "wrap the handler in `{ ... }`",
            .note = "plain `errdefer cleanup();` may remain unbraced",
        },
        .hidden_return => .{
            .message = "`return` must start on its own line",
            .help = "move `return` to a new line or brace its containing control-flow body",
        },
        .hidden_break => .{
            .message = "`break` must start on its own line",
            .help = "move `break` to a new line or brace its containing control-flow body",
        },
        .hidden_continue => .{
            .message = "`continue` must start on its own line",
            .help = "move `continue` to a new line or brace its containing control-flow body",
        },
        .hidden_unreachable => .{
            .message = "`unreachable` must start on its own line",
            .help = "move `unreachable` to a new line or brace its containing control-flow body",
        },
    };
}
