const std = @import("std");
const assert = std.debug.assert;
const Ast = std.zig.Ast;
const Rule = @import("../Rule.zig");
const LintContext = Rule.Context;
const nodeIsBlock = LintContext.nodeIsBlock;
const nodeTagIsAssignment = LintContext.nodeTagIsAssignment;
const nodeTagIsCall = LintContext.nodeTagIsCall;
const nodeTagIsExit = LintContext.nodeTagIsExit;

/// Keep statement control flow braced, non-block compound value selection
/// multiline, and explicit exits visible. Direct switch arms remain compact.
pub fn lint(context: *LintContext) Rule.Error!void {
    const file_allocator = context.scratch_allocator;
    const file = context;
    const ast = file.ast;

    const node_flags = try file_allocator.alloc(u8, ast.nodes.len);
    defer file_allocator.free(node_flags);

    @memset(node_flags, 0);

    const pending_nodes = try file_allocator.alloc(Ast.Node.Index, ast.nodes.len);
    defer file_allocator.free(pending_nodes);

    const violation_kinds = try file_allocator.alloc(Violation, ast.tokens.len);
    defer file_allocator.free(violation_kinds);

    @memset(violation_kinds, .none);

    // Check-only scans retain the one-byte classification. Fix collection also
    // remembers the owning construct, avoiding a parent search per diagnostic.
    const fix_nodes: []Ast.Node.Index = if (context.fixes_enabled)
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
    classifyNodeContext(ast, node_flags, pending_nodes);

    // ── Record Violations ──
    //
    // Structural checks run before token-level exit checks. A parent that
    // requires braces marks its direct body as covered, preventing a second
    // diagnostic for the same hidden exit.
    recordStructuralViolations(file, node_flags, violations);
    recordHiddenFlowViolations(file, node_flags, violations);

    // ── Emit In Source Order ──
    //
    // Recording one enum per token makes ordering a linear token scan rather
    // than a comparison sort over diagnostics.
    for (violations.kinds, 0..) |violation, token_usize| {
        if (violation == .none) {
            continue;
        }

        const token: Ast.TokenIndex = @intCast(token_usize);

        const fix = if (context.fixes_enabled)
            try proposeFix(file_allocator, file, node_flags, violations.fix_nodes[token], token, violation)
        else
            null;

        const violation_index = @intFromEnum(violation);

        try context.report(.{
            .token = token,
            .message = violation_messages[violation_index],
            .help = violation_help[violation_index] orelse "",
            .note = violation_notes[violation_index] orelse "",
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
) void {
    var pending_count: usize = 0;

    for (ast.nodes.items(.tag), 0..) |tag, node_usize| {
        const node: Ast.Node.Index = @enumFromInt(node_usize);

        switch (tag) {
            .block_two, .block_two_semicolon, .block, .block_semicolon => {
                var block_buffer: [2]Ast.Node.Index = undefined;
                const statements = ast.blockStatements(&block_buffer, node) orelse {
                    unreachable;
                };

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
            },

            .if_simple, .@"if" => {
                const if_node = ast.fullIf(node) orelse {
                    unreachable;
                };

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
            },

            .switch_case_one, .switch_case_inline_one, .switch_case, .switch_case_inline => {
                const switch_case = ast.fullSwitchCase(node) orelse {
                    unreachable;
                };

                // Mark only the direct target so token-level exit checks permit
                // compact arms. Structural nodes still check their own
                // children, which are not marked:
                //
                //   .start => return,                    <- target; exempt exit
                //   .stop => if (ready) stopWorker(),    <- `if` target is visited
                //                ^^^^^^^^^^^ body; not marked and still checked
                const target = switch_case.ast.target_expr;

                node_flags[@intFromEnum(target)] |= node_flag_switch_arm;
            },

            else => {},
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
    file: *const LintContext,
    node_flags: []u8,
    violations: Violations,
) void {
    const ast = file.ast;

    for (ast.nodes.items(.tag), 0..) |tag, node_usize| {
        const node: Ast.Node.Index = @enumFromInt(node_usize);

        switch (tag) {
            .if_simple, .@"if" => recordIfViolations(file, node_flags, violations, node),

            .while_simple, .while_cont, .@"while" => {
                const while_node = ast.fullWhile(node) orelse {
                    unreachable;
                };

                recordRequiredBlock(ast, node_flags, violations, while_node.ast.then_expr, node, .unbraced_while_body);

                if (while_node.ast.else_expr.unwrap()) |else_body| {
                    recordRequiredBlock(ast, node_flags, violations, else_body, node, .unbraced_while_body);
                }
            },

            .for_simple, .@"for" => {
                const for_node = ast.fullFor(node) orelse {
                    unreachable;
                };

                recordRequiredBlock(ast, node_flags, violations, for_node.ast.then_expr, node, .unbraced_for_body);

                if (for_node.ast.else_expr.unwrap()) |else_body| {
                    recordRequiredBlock(ast, node_flags, violations, else_body, node, .unbraced_for_body);
                }
            },

            .@"catch", .@"orelse" => recordFallbackViolation(file, node_flags, violations, node),
            .@"errdefer" => recordErrdeferViolation(ast, node_flags, violations, node),

            else => {},
        }
    }
}

fn recordIfViolations(
    file: *const LintContext,
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
        !valueIfRequiresMultiline(ast, node))
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
    if (!nodeTagIsExit(ast.nodeTag(effective_body))) {
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
    file: *const LintContext,
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
    const fallback_is_exit = nodeTagIsExit(ast.nodeTag(effective_fallback));
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
        file.areTokensOnSameLine(ast.nodeMainToken(node), ast.firstToken(fallback)))
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

fn valueIfRequiresMultiline(ast: Ast, initial_if: Ast.Node.Index) bool {
    var node = initial_if;
    var requires_multiline = false;

    while (true) {
        const if_node = switch (ast.nodeTag(node)) {
            .if_simple => ast.ifSimple(node),
            .@"if" => ast.ifFull(node),
            else => unreachable,
        };

        // Zig fmt keeps `else` beside a closing brace. The block already exposes
        // the branch boundary, so leave the entire chain formatter-controlled.
        if (nodeIsBlock(ast, if_node.ast.then_expr)) {
            return false;
        }

        if (if_node.payload_token != null or
            if_node.error_token != null or
            !valueIsSimple(ast, if_node.ast.then_expr))
        {
            requires_multiline = true;
        }

        const else_body = if_node.ast.else_expr.unwrap() orelse {
            return false;
        };

        if (ast.fullIf(else_body) != null) {
            requires_multiline = true;
            node = else_body;

            continue;
        }

        if (nodeIsBlock(ast, else_body)) {
            return false;
        }

        return requires_multiline or !valueIsSimple(ast, else_body);
    }
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

fn valueIfMultilineViolation(file: *const LintContext, initial_if: Ast.Node.Index) ?Ast.Node.Index {
    const ast = file.ast;
    var node = initial_if;

    while (true) {
        const if_node = ast.fullIf(node) orelse {
            unreachable;
        };

        const then_body = if_node.ast.then_expr;

        if (!nodeIsBlock(ast, then_body)) {
            const first_token = ast.firstToken(then_body);
            assert(first_token > 0);

            if (file.areTokensOnSameLine(first_token - 1, first_token)) {
                // The token immediately before a then-body is the closing header
                // token, including a closing capture pipe:
                //
                //   if (optional) |value| value else fallback
                //                         ^^^^^ `then_body`; same line, rejected
                return then_body;
            }
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
            file.areTokensOnSameLine(if_node.else_token, ast.firstToken(else_body)))
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

fn recordHiddenFlowViolations(
    file: *const LintContext,
    node_flags: []const u8,
    violations: Violations,
) void {
    const ast = file.ast;

    for (ast.nodes.items(.tag), 0..) |tag, node_usize| {
        const violation: Violation = switch (tag) {
            .@"return" => .hidden_return,
            .@"break" => .hidden_break,
            .@"continue" => .hidden_continue,
            .unreachable_literal => .hidden_unreachable,
            else => continue,
        };

        const node: Ast.Node.Index = @enumFromInt(node_usize);
        const flags = node_flags[node_usize];

        if (flags & (node_flag_covered | node_flag_switch_arm) != 0) {
            continue;
        }

        const token = ast.nodeMainToken(node);
        if (token == 0 or !file.areTokensOnSameLine(token - 1, token)) {
            continue;
        }

        recordViolation(violations, token, node, violation);
    }
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
    file: *const LintContext,
    node_flags: []const u8,
    owner: Ast.Node.Index,
    token: Ast.TokenIndex,
    violation: Violation,
) std.mem.Allocator.Error!?Rule.Fix {
    switch (violation) {
        .none => return null,

        // This requires semantic extraction; adding braces inside the value
        // expression only exposes another violation after the first fix pass.
        .unbraced_if_exit_branch => return null,

        .inline_complex_if_value => return try multilineIfFix(allocator, file, owner),

        .inline_complex_orelse_value,
        .inline_complex_catch_value,
        .hidden_return,
        .hidden_break,
        .hidden_continue,
        .hidden_unreachable,
        => {
            const ast = file.ast;
            const layout = file.lineLayout(token);
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
                .range = .{
                    .start_offset = source_start_offset,
                    .end_offset = token_offset,
                },
                .replacement = replacement,
            };
        },

        .unbraced_if_body,
        .unbraced_while_body,
        .unbraced_for_body,
        .unbraced_orelse_fallback,
        .unbraced_catch_fallback,
        .unbraced_errdefer_capture,
        => return try bracedBodyFix(allocator, file, node_flags, owner, token, violation),
    }
}

fn multilineIfFix(
    allocator: std.mem.Allocator,
    file: *const LintContext,
    owner: Ast.Node.Index,
) std.mem.Allocator.Error!?Rule.Fix {
    const ast = file.ast;
    const layout = file.lineLayout(ast.firstToken(owner));
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

        // Zig fmt rejoins a block branch with `else`. Leave those conditionals
        // for a semantic rewrite rather than proposing an unstable line-only fix.
        if (nodeIsBlock(ast, if_node.ast.then_expr)) {
            return null;
        }

        try appendBranchBreak(
            allocator,
            file,
            &replacement,
            &source_offset,
            ast.firstToken(if_node.ast.then_expr),
            layout,
            true,
        );

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

        if (nodeIsBlock(ast, else_body)) {
            return null;
        }

        try appendBranchBreak(
            allocator,
            file,
            &replacement,
            &source_offset,
            ast.firstToken(else_body),
            layout,
            true,
        );

        break;
    }

    try replacement.appendSlice(allocator, ast.source[source_offset..source_end_offset]);

    const owned_replacement = try replacement.toOwnedSlice(allocator);

    return .{
        .range = .{
            .start_offset = source_start_offset,
            .end_offset = @intCast(source_end_offset),
        },
        .replacement = owned_replacement,
    };
}

fn appendBranchBreak(
    allocator: std.mem.Allocator,
    file: *const LintContext,
    replacement: *std.ArrayList(u8),
    source_offset: *usize,
    token: Ast.TokenIndex,
    layout: LintContext.LineLayout,
    continuation: bool,
) std.mem.Allocator.Error!void {
    if (token == 0 or !file.areTokensOnSameLine(token - 1, token)) {
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
    file: *const LintContext,
    node_flags: []const u8,
    owner: Ast.Node.Index,
    token: Ast.TokenIndex,
    violation: Violation,
) std.mem.Allocator.Error!?Rule.Fix {
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
        .unbraced_if_body => body: {
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

    if (first_token == 0 or !file.areTokensOnSameLine(first_token, last_token)) {
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

    const layout = file.lineLayout(ast.firstToken(owner));

    const replacement = try std.fmt.allocPrint(allocator, " {{{s}{s}{s}{s};{s}{s}}}", .{
        layout.newline,
        layout.indent,
        layout.indent_unit,
        ast.source[body_start_offset..body_end_offset],
        layout.newline,
        layout.indent,
    });

    return .{
        .range = .{
            .start_offset = @intCast(source_start_offset),
            .end_offset = @intCast(source_end_offset),
        },
        .replacement = replacement,
    };
}

fn bodyCanGainScope(ast: Ast, initial_node: Ast.Node.Index) bool {
    var node = unwrapGroupedExpression(ast, initial_node);

    if (ast.nodeTag(node) == .@"try") {
        node = ast.nodeData(node).node;
    }

    const tag = ast.nodeTag(node);

    return nodeTagIsCall(tag) or nodeTagIsAssignment(tag) or nodeTagIsExit(tag);
}

const violation_messages = [_][]const u8{
    "",
    "`if` body requires braces",
    "`while` body requires braces",
    "`for` body requires braces",
    "`if` value branch hides an exit",
    "complex `if` branches must start on separate lines",
    "`orelse` fallback requires braces for exits or statement work",
    "`catch` fallback requires braces for exits or statement work",
    "complex `orelse` fallback must start on a new line",
    "complex `catch` fallback must start on a new line",
    "captured `errdefer` handler requires braces",
    "`return` must start on its own line",
    "`break` must start on its own line",
    "`continue` must start on its own line",
    "`unreachable` must start on its own line",
};

const violation_help = [_]?[]const u8{
    null,
    "wrap the body in `{ ... }`",
    "wrap the body in `{ ... }`",
    "wrap the body in `{ ... }`",
    "move the exit into a preceding braced statement",
    "move each branch expression to the line after `if` or `else`",
    "wrap the fallback in `{ ... }`",
    "wrap the fallback in `{ ... }`",
    "move the fallback after `orelse` or wrap it in `{ ... }`",
    "move the fallback after `catch` or wrap it in `{ ... }`",
    "wrap the handler in `{ ... }`",
    "move `return` to a new line or brace its containing control-flow body",
    "move `break` to a new line or brace its containing control-flow body",
    "move `continue` to a new line or brace its containing control-flow body",
    "move `unreachable` to a new line or brace its containing control-flow body",
};

const violation_notes = [_]?[]const u8{
    null,
    null,
    null,
    "`inline for` follows the same rule",
    null,
    "simple leaf values may remain inline",
    null,
    null,
    "simple leaf values may remain inline",
    "simple leaf values may remain inline",
    "plain `errdefer cleanup();` may remain unbraced",
    null,
    null,
    null,
    null,
};

comptime {
    assert(violation_messages.len == @typeInfo(Violation).@"enum".fields.len);
    assert(violation_help.len == violation_messages.len);
    assert(violation_notes.len == violation_messages.len);
}
