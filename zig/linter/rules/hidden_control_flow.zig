const std = @import("std");
const Ast = std.zig.Ast;
const Diagnostics = @import("../Diagnostics.zig");
const FileContext = @import("../FileContext.zig");

/// Keep statement control flow braced, complex value selection multiline, and
/// explicit or implicit exits visible. Direct switch arms remain compact.
pub fn lint(
    file_allocator: std.mem.Allocator,
    diagnostic_allocator: std.mem.Allocator,
    path: []const u8,
    file: *const FileContext,
    diagnostics: *Diagnostics,
) std.mem.Allocator.Error!void {
    const ast = file.ast;

    const node_flags = try file_allocator.alloc(u8, ast.nodes.len);
    defer file_allocator.free(node_flags);

    @memset(node_flags, 0);

    const pending_nodes = try file_allocator.alloc(Ast.Node.Index, ast.nodes.len);
    defer file_allocator.free(pending_nodes);

    const visible_try_tokens = try file_allocator.alloc(bool, ast.tokens.len);
    defer file_allocator.free(visible_try_tokens);

    @memset(visible_try_tokens, false);

    const violations = try file_allocator.alloc(Violation, ast.tokens.len);
    defer file_allocator.free(violations);

    @memset(violations, .none);

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
    for (violations, 0..) |violation, token_usize| {
        if (violation == .none) {
            continue;
        }

        try diagnostics.add_at_token(
            diagnostic_allocator,
            path,
            file,
            @intCast(token_usize),
            "hidden_control_flow",
            messageFor(violation),
        );
    }
}

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

    std.debug.assert(pending_count.* < pending_nodes.len);

    pending_nodes[pending_count.*] = node;
    pending_count.* += 1;
}

fn recordStructuralViolations(
    file: *const FileContext,
    node_flags: []u8,
    violations: []Violation,
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
    violations: []Violation,
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
        recordRequiredBlock(ast, node_flags, violations, if_node.ast.then_expr, .unbraced_if_body);
        if (else_body_optional) |else_body| {
            // `else if` is the one unbraced statement branch: the nested `if`
            // is checked independently, and each of its own bodies needs braces.
            if (ast.fullIf(else_body) == null) {
                recordRequiredBlock(ast, node_flags, violations, else_body, .unbraced_if_body);
            }
        }

        return;
    }

    var exit_branch_reported = recordIfExitBranch(ast, node_flags, violations, if_node.ast.then_expr);
    if (else_body_optional) |else_body| {
        if (ast.fullIf(else_body) == null) {
            exit_branch_reported = recordIfExitBranch(ast, node_flags, violations, else_body) or exit_branch_reported;
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
        recordViolationAtNode(ast, node_flags, violations, body, .inline_complex_if_value);
    }
}

fn recordIfExitBranch(
    ast: Ast,
    node_flags: []u8,
    violations: []Violation,
    body: Ast.Node.Index,
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
    recordViolationAtNode(ast, node_flags, violations, effective_body, .unbraced_if_exit_branch);

    return true;
}

fn recordWhileViolations(
    ast: Ast,
    node_flags: []u8,
    violations: []Violation,
    node: Ast.Node.Index,
) void {
    const while_node = ast.fullWhile(node) orelse {
        unreachable;
    };
    recordRequiredBlock(ast, node_flags, violations, while_node.ast.then_expr, .unbraced_while_body);
    if (while_node.ast.else_expr.unwrap()) |else_body| {
        recordRequiredBlock(ast, node_flags, violations, else_body, .unbraced_while_body);
    }
}

fn recordForViolations(
    ast: Ast,
    node_flags: []u8,
    violations: []Violation,
    node: Ast.Node.Index,
) void {
    const for_node = ast.fullFor(node) orelse {
        unreachable;
    };
    recordRequiredBlock(ast, node_flags, violations, for_node.ast.then_expr, .unbraced_for_body);
    if (for_node.ast.else_expr.unwrap()) |else_body| {
        recordRequiredBlock(ast, node_flags, violations, else_body, .unbraced_for_body);
    }
}

fn recordRequiredBlock(
    ast: Ast,
    node_flags: []u8,
    violations: []Violation,
    body: Ast.Node.Index,
    violation: Violation,
) void {
    if (nodeIsBlock(ast, body)) {
        return;
    }

    recordViolationAtNode(ast, node_flags, violations, body, violation);
}

fn recordFallbackViolation(
    file: *const FileContext,
    node_flags: []u8,
    violations: []Violation,
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
        recordViolationAtNode(ast, node_flags, violations, fallback, violation);
    }
}

fn recordErrdeferViolation(
    ast: Ast,
    node_flags: []u8,
    violations: []Violation,
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
    recordViolationAtNode(ast, node_flags, violations, deferred, .unbraced_errdefer_capture);
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
    std.debug.assert(first_token > 0);

    return file.tokensOnSameLine(first_token - 1, first_token);
}

fn recordHiddenFlowViolations(
    file: *const FileContext,
    node_flags: []u8,
    visible_try_tokens: []const bool,
    violations: []Violation,
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
    violations: []Violation,
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
    recordViolation(violations, token, .inline_nested_try);
}

fn recordHiddenExitViolation(
    file: *const FileContext,
    node_flags: []const u8,
    violations: []Violation,
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
    recordViolation(violations, token, violation);
}

fn recordViolationAtNode(
    ast: Ast,
    node_flags: []u8,
    violations: []Violation,
    node: Ast.Node.Index,
    violation: Violation,
) void {
    node_flags[@intFromEnum(node)] |= node_flag_covered;
    recordViolation(violations, ast.firstToken(node), violation);
}

fn recordViolation(
    violations: []Violation,
    token: Ast.TokenIndex,
    violation: Violation,
) void {
    if (violations[token] == .none) {
        violations[token] = violation;
    }
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

fn messageFor(violation: Violation) []const u8 {
    return switch (violation) {
        .none => unreachable,
        .unbraced_if_body => "`if` has an unbraced statement body. Wrap the body in `{ ... }`; moving it to another line is not sufficient.",
        .unbraced_while_body => "`while` has an unbraced body. Wrap the body in `{ ... }`; moving it to another line is not sufficient.",
        .unbraced_for_body => "`for` has an unbraced body. Wrap the body in `{ ... }`; this also applies to `inline for`.",
        .unbraced_if_exit_branch => "An `if` value branch exits without braces. Wrap that branch in `{ ... }` so the exit is structurally visible.",
        .inline_complex_if_value => "A captured, chained, or compound `if` value is compressed onto one line. Put each branch expression on the line after its `if` or `else`; only simple leaf-value branches may remain inline.",
        .unbraced_orelse_fallback => "An `orelse` fallback performs statement work or exits without braces. Wrap the fallback in `{ ... }`; a line break alone is not sufficient.",
        .unbraced_catch_fallback => "A `catch` fallback performs statement work or exits without braces. Wrap the fallback in `{ ... }`; a line break alone is not sufficient.",
        .inline_complex_orelse_value => "A compound `orelse` fallback is inline. Start the fallback expression on the line after `orelse`, or use a braced fallback; only simple leaf values may remain inline.",
        .inline_complex_catch_value => "A compound `catch` fallback is inline. Start the fallback expression on the line after `catch`, or use a braced fallback; only simple leaf values may remain inline.",
        .inline_nested_try => "`try` is nested inside another expression on the same line, hiding its implicit error return. Start `try` on a new line or assign its result before the outer expression.",
        .unbraced_errdefer_capture => "An `errdefer` error capture has an unbraced handler. Wrap the handler in `{ ... }`; plain `errdefer cleanup();` remains allowed.",
        .hidden_return => "`return` is hidden behind another token on its line. Start it on a new line or put its containing control-flow body in braces. Direct switch arms remain compact.",
        .hidden_break => "`break` is hidden behind another token on its line. Start it on a new line or put its containing control-flow body in braces. Direct switch arms remain compact.",
        .hidden_continue => "`continue` is hidden behind another token on its line. Start it on a new line or put its containing control-flow body in braces. Direct switch arms remain compact.",
        .hidden_unreachable => "`unreachable` is hidden behind another token on its line. Start it on a new line or put its containing control-flow body in braces. Direct switch arms remain compact.",
    };
}
