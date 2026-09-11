const std = @import("std");
const assert = std.debug.assert;
const Ast = std.zig.Ast;
const Rule = @import("../Rule.zig");
const LintContext = Rule.Context;
const identifierText = LintContext.identifierText;

/// Order explicit signature vocabulary before each callable, keep peer
/// declaration batches from interrupting a public operation's local contract,
/// place owning aggregate types before private components, and put
/// unambiguous private implementations after their sole root caller. Ambiguous
/// relationships are left unconstrained rather than approximated as semantic
/// resolution.
pub fn lint(context: *LintContext) Rule.Error!void {
    try lintContainer(context.scratch_allocator, context, .root);
}

const NamedDeclaration = struct {
    name: []const u8,
    name_token: Ast.TokenIndex,
    member_index: usize,
    is_import: bool,
    is_public: bool,
};

const TypeDeclaration = struct {
    named_index: usize,
    initializer: Ast.Node.Index,
    owns_operations: bool,
    signature_consumers: union(enum) {
        none,
        sole: usize,
        multiple,
    } = .none,
};

const OwnedTypeEdge = struct {
    component_index: usize,
    next: ?usize,
};

const FunctionDeclaration = struct {
    name: []const u8,
    name_token: Ast.TokenIndex,
    member_index: usize,
    node: Ast.Node.Index,
    // Inclusive body boundaries; null for declarations without a body.
    body_tokens: ?struct {
        first: Ast.TokenIndex,
        last: Ast.TokenIndex,
    },
    is_externally_visible: bool,
    callers: union(enum) {
        none,
        sole: usize,
        ambiguous,
    } = .none,
};

comptime {
    assert(@sizeOf(FunctionDeclaration) <= 64);
}

const SignatureWalk = struct {
    context: *LintContext,
    named: []const NamedDeclaration,
    named_by_name: *const std.StringHashMapUnmanaged(usize),
    types: []TypeDeclaration,
    type_by_name: *const std.StringHashMapUnmanaged(usize),
    function: FunctionDeclaration,
    function_proto: Ast.full.FnProto,
    seen: []bool,
};

fn lintContainer(
    file_allocator: std.mem.Allocator,
    context: *LintContext,
    node: Ast.Node.Index,
) Rule.Error!void {
    const ast = context.ast;

    var container_buffer: [2]Ast.Node.Index = undefined;
    const container = ast.fullContainerDecl(&container_buffer, node) orelse {
        return;
    };

    const members = container.ast.members;

    const named_storage = try file_allocator.alloc(NamedDeclaration, members.len);
    defer file_allocator.free(named_storage);

    const type_storage = try file_allocator.alloc(TypeDeclaration, members.len);
    defer file_allocator.free(type_storage);

    const function_storage = try file_allocator.alloc(FunctionDeclaration, members.len);
    defer file_allocator.free(function_storage);

    var named_count: usize = 0;
    var type_count: usize = 0;
    var function_count: usize = 0;

    for (members, 0..) |member, member_index| {
        if (ast.fullVarDecl(member)) |variable| {
            const name_token = declarationNameToken(ast, variable) orelse {
                continue;
            };

            const initializer = variable.ast.init_node.unwrap() orelse {
                continue;
            };

            named_storage[named_count] = .{
                .name = identifierText(ast, name_token),
                .name_token = name_token,
                .member_index = member_index,
                .is_import = initializerIsImportDerived(ast, initializer),
                .is_public = variable.visib_token != null,
            };

            if (LintContext.nodeIsDirectType(ast, initializer)) {
                type_storage[type_count] = .{
                    .named_index = named_count,
                    .initializer = initializer,
                    .owns_operations = typeOwnsOperations(ast, initializer),
                };
                type_count += 1;
            }

            named_count += 1;

            continue;
        }

        var function_buffer: [1]Ast.Node.Index = undefined;
        const proto = ast.fullFnProto(&function_buffer, member) orelse {
            continue;
        };

        const name_token = proto.name_token orelse {
            continue;
        };

        // Resolve AST boundaries once, not for every reference's caller lookup.
        const body = functionBody(ast, member);

        function_storage[function_count] = .{
            .name = identifierText(ast, name_token),
            .name_token = name_token,
            .member_index = member_index,
            .node = member,
            .body_tokens = if (body) |body_node|
                .{
                    .first = ast.firstToken(body_node),
                    .last = ast.lastToken(body_node),
                }
            else
                null,
            .is_externally_visible = functionIsExternallyVisible(ast, proto),
        };
        function_count += 1;
    }

    const named = named_storage[0..named_count];
    const types = type_storage[0..type_count];
    const functions = function_storage[0..function_count];

    var named_by_name: std.StringHashMapUnmanaged(usize) = .empty;
    defer named_by_name.deinit(file_allocator);

    for (named, 0..) |declaration, index| {
        try named_by_name.put(file_allocator, declaration.name, index);
    }

    var type_by_name: std.StringHashMapUnmanaged(usize) = .empty;
    defer type_by_name.deinit(file_allocator);

    for (types, 0..) |declaration, index| {
        try type_by_name.put(file_allocator, named[declaration.named_index].name, index);
    }

    var function_by_name: std.StringHashMapUnmanaged(usize) = .empty;
    defer function_by_name.deinit(file_allocator);

    for (functions, 0..) |declaration, index| {
        try function_by_name.put(file_allocator, declaration.name, index);
    }

    try lintSignatureOrder(
        file_allocator,
        context,
        named,
        &named_by_name,
        types,
        &type_by_name,
        functions,
    );

    try lintTypeOrder(
        file_allocator,
        context,
        members,
        named,
        types,
        &type_by_name,
        functions,
    );

    try lintFunctionOrder(context, node, functions, &function_by_name);

    for (types) |declaration| {
        try lintContainer(file_allocator, context, declaration.initializer);
    }
}

fn lintSignatureOrder(
    file_allocator: std.mem.Allocator,
    context: *LintContext,
    named: []const NamedDeclaration,
    named_by_name: *const std.StringHashMapUnmanaged(usize),
    types: []TypeDeclaration,
    type_by_name: *const std.StringHashMapUnmanaged(usize),
    functions: []const FunctionDeclaration,
) Rule.Error!void {
    const ast = context.ast;

    const seen = try file_allocator.alloc(bool, named.len);
    defer file_allocator.free(seen);

    for (functions) |function| {
        @memset(seen, false);

        var function_buffer: [1]Ast.Node.Index = undefined;
        const proto = ast.fullFnProto(&function_buffer, function.node) orelse {
            continue;
        };

        const walk: SignatureWalk = .{
            .context = context,
            .named = named,
            .named_by_name = named_by_name,
            .types = types,
            .type_by_name = type_by_name,
            .function = function,
            .function_proto = proto,
            .seen = seen,
        };

        for (proto.ast.params) |type_expression| {
            try lintSignatureTypeExpression(&walk, null, type_expression);
        }

        const return_type = proto.ast.return_type.unwrap() orelse {
            continue;
        };

        try lintSignatureTypeExpression(&walk, null, return_type);
    }
}

fn lintSignatureTypeExpression(
    walk: *const SignatureWalk,
    nested_proto: ?Ast.full.FnProto,
    expression: Ast.Node.Index,
) Rule.Error!void {
    const ast = walk.context.ast;
    if (ast.fullArrayType(expression)) |array| {
        return lintSignatureTypeExpression(walk, nested_proto, array.ast.elem_type);
    }

    if (ast.fullPtrType(expression)) |pointer| {
        return lintSignatureTypeExpression(walk, nested_proto, pointer.ast.child_type);
    }

    var fn_buffer: [1]Ast.Node.Index = undefined;
    if (ast.fullFnProto(&fn_buffer, expression)) |inner_proto| {
        for (inner_proto.ast.params) |type_expression| {
            try lintSignatureTypeExpression(walk, inner_proto, type_expression);
        }

        const return_type = inner_proto.ast.return_type.unwrap() orelse {
            return;
        };

        return lintSignatureTypeExpression(walk, inner_proto, return_type);
    }

    switch (ast.nodeTag(expression)) {
        .identifier => try lintSignatureIdentifier(
            walk,
            nested_proto,
            ast.nodeMainToken(expression),
        ),
        .field_access => try lintSignatureTypeExpression(
            walk,
            nested_proto,
            ast.nodeData(expression).node_and_token[0],
        ),
        .optional_type => try lintSignatureTypeExpression(
            walk,
            nested_proto,
            ast.nodeData(expression).node,
        ),
        .grouped_expression => try lintSignatureTypeExpression(
            walk,
            nested_proto,
            ast.nodeData(expression).node_and_token[0],
        ),
        .error_union => {
            const error_set, const payload = ast.nodeData(expression).node_and_node;

            if (ast.nodeTag(error_set) != .error_set_decl) {
                try lintSignatureTypeExpression(walk, nested_proto, error_set);
            }

            try lintSignatureTypeExpression(walk, nested_proto, payload);
        },
        .anyframe_type => try lintSignatureTypeExpression(
            walk,
            nested_proto,
            ast.nodeData(expression).token_and_node[1],
        ),
        else => {},
    }
}

fn lintSignatureIdentifier(
    walk: *const SignatureWalk,
    nested_proto: ?Ast.full.FnProto,
    token: Ast.TokenIndex,
) Rule.Error!void {
    const ast = walk.context.ast;
    const name = identifierText(ast, token);

    // Every tracked type is also a named declaration. An absent name cannot
    // contribute an ordering constraint, regardless of parameter shadowing.
    const named_index = walk.named_by_name.get(name) orelse {
        return;
    };

    const nested_parameter_shadows = if (nested_proto) |proto|
        functionParameterHasName(ast, proto, name)
    else
        false;

    if (std.zig.isPrimitive(name) or
        std.mem.eql(u8, name, "Self") or
        functionParameterHasName(ast, walk.function_proto, name) or
        nested_parameter_shadows)
    {
        return;
    }

    if (walk.type_by_name.get(name)) |type_index| {
        const type_declaration = &walk.types[type_index];

        switch (type_declaration.signature_consumers) {
            .none => type_declaration.signature_consumers = .{ .sole = walk.function.member_index },

            .sole => |consumer_member| {
                if (consumer_member != walk.function.member_index) {
                    type_declaration.signature_consumers = .multiple;
                }
            },

            .multiple => {},
        }
    }

    const declaration = walk.named[named_index];
    if (declaration.is_import or
        declaration.member_index < walk.function.member_index or
        walk.seen[named_index])
    {
        return;
    }

    walk.seen[named_index] = true;

    const message = try std.fmt.allocPrint(
        walk.context.scratch_allocator,
        "`{s}` is used in `{s}`'s signature before its declaration",
        .{ name, walk.function.name },
    );
    defer walk.context.scratch_allocator.free(message);

    const help = try std.fmt.allocPrint(
        walk.context.scratch_allocator,
        "move `{s}` before `{s}`",
        .{ name, walk.function.name },
    );
    defer walk.context.scratch_allocator.free(help);

    try walk.context.report(.{
        .token = token,
        .message = message,
        .help = help,
    });
}

fn lintTypeOrder(
    file_allocator: std.mem.Allocator,
    context: *LintContext,
    members: []const Ast.Node.Index,
    named: []const NamedDeclaration,
    types: []const TypeDeclaration,
    type_by_name: *const std.StringHashMapUnmanaged(usize),
    functions: []const FunctionDeclaration,
) Rule.Error!void {
    if (types.len == 0) {
        return;
    }

    const incoming_count = try file_allocator.alloc(usize, types.len);
    defer file_allocator.free(incoming_count);

    const sole_parent = try file_allocator.alloc(?usize, types.len);
    defer file_allocator.free(sole_parent);

    const seen_generation = try file_allocator.alloc(usize, types.len);
    defer file_allocator.free(seen_generation);

    const first_owned_edge = try file_allocator.alloc(?usize, types.len);
    defer file_allocator.free(first_owned_edge);

    var owned_type_edges: std.ArrayList(OwnedTypeEdge) = .empty;
    defer owned_type_edges.deinit(file_allocator);

    @memset(incoming_count, 0);
    @memset(sole_parent, null);
    @memset(seen_generation, 0);
    @memset(first_owned_edge, null);

    // ── Build Ownership Graph ──

    const ast = context.ast;

    for (types, 0..) |owner, owner_index| {
        var container_buffer: [2]Ast.Node.Index = undefined;
        const container = ast.fullContainerDecl(&container_buffer, owner.initializer) orelse {
            continue;
        };

        for (container.ast.members) |member| {
            const field = ast.fullContainerField(member) orelse {
                continue;
            };

            const type_expression = field.ast.type_expr.unwrap() orelse {
                continue;
            };

            const token = simpleTypeReferenceToken(ast, type_expression) orelse {
                continue;
            };

            const component_index = type_by_name.get(identifierText(ast, token)) orelse {
                continue;
            };

            if (component_index == owner_index or
                seen_generation[component_index] == owner_index + 1)
            {
                continue;
            }

            seen_generation[component_index] = owner_index + 1;

            const edge_index = owned_type_edges.items.len;

            try owned_type_edges.append(file_allocator, .{
                .component_index = component_index,
                .next = first_owned_edge[owner_index],
            });

            first_owned_edge[owner_index] = edge_index;

            incoming_count[component_index] += 1;
            sole_parent[component_index] = if (incoming_count[component_index] == 1)
                owner_index
            else
                null;
        }
    }

    // ── Check Contract Locality ──

    try lintContractLocality(
        file_allocator,
        context,
        members,
        named,
        types,
        functions,
        incoming_count,
        first_owned_edge,
        owned_type_edges.items,
    );

    // ── Report Reversed Owners ──

    for (types, 0..) |component, component_index| {
        const has_signature_consumers = switch (component.signature_consumers) {
            .none => false,
            .sole, .multiple => true,
        };

        if (incoming_count[component_index] != 1 or
            named[component.named_index].is_public or
            has_signature_consumers or
            !ownershipChainHasRoot(component_index, incoming_count, sole_parent))
        {
            continue;
        }

        const owner_index = sole_parent[component_index] orelse {
            continue;
        };

        const owner = types[owner_index];
        const owner_named = named[owner.named_index];
        const component_named = named[component.named_index];

        if (!owner_named.is_public) {
            continue;
        }

        if (owner_named.member_index < component_named.member_index) {
            continue;
        }

        const message = try std.fmt.allocPrint(
            context.scratch_allocator,
            "owner type `{s}` is declared after its component `{s}`",
            .{ owner_named.name, component_named.name },
        );
        defer context.scratch_allocator.free(message);

        const help = try std.fmt.allocPrint(
            context.scratch_allocator,
            "move `{s}` before `{s}`",
            .{ owner_named.name, component_named.name },
        );
        defer context.scratch_allocator.free(help);

        try context.report(.{
            .token = owner_named.name_token,
            .message = message,
            .help = help,
        });
    }
}

fn lintContractLocality(
    file_allocator: std.mem.Allocator,
    context: *LintContext,
    members: []const Ast.Node.Index,
    named: []const NamedDeclaration,
    types: []const TypeDeclaration,
    functions: []const FunctionDeclaration,
    incoming_count: []const usize,
    first_owned_edge: []const ?usize,
    owned_type_edges: []const OwnedTypeEdge,
) Rule.Error!void {
    if (functions.len == 0 or containerHasStateFields(context.ast, members)) {
        return;
    }

    const type_by_member = try file_allocator.alloc(?usize, members.len);
    defer file_allocator.free(type_by_member);

    const contract_generation = try file_allocator.alloc(usize, types.len);
    defer file_allocator.free(contract_generation);

    const pending = try file_allocator.alloc(usize, types.len);
    defer file_allocator.free(pending);

    const remaining_parents = try file_allocator.alloc(usize, types.len);
    defer file_allocator.free(remaining_parents);

    @memset(type_by_member, null);
    @memset(contract_generation, 0);

    for (types, 0..) |declaration, type_index| {
        type_by_member[named[declaration.named_index].member_index] = type_index;
    }

    var generation: usize = 0;

    for (functions) |function| {
        if (!function.is_externally_visible) {
            continue;
        }

        generation += 1;

        @memcpy(remaining_parents, incoming_count);

        var pending_count: usize = 0;
        var earliest_member = function.member_index;
        var signature_type_follows_function = false;

        for (types, 0..) |declaration, type_index| {
            if (declaration.owns_operations) {
                continue;
            }

            const consumer_member = switch (declaration.signature_consumers) {
                .sole => |member_index| member_index,
                .none, .multiple => continue,
            };

            if (consumer_member != function.member_index) {
                continue;
            }

            contract_generation[type_index] = generation;
            pending[pending_count] = type_index;
            pending_count += 1;

            const member_index = named[declaration.named_index].member_index;

            earliest_member = @min(earliest_member, member_index);
            signature_type_follows_function = signature_type_follows_function or
                member_index > function.member_index;
        }

        if (pending_count == 0 or signature_type_follows_function) {
            continue;
        }

        while (pending_count != 0) {
            pending_count -= 1;

            const owner_index = pending[pending_count];
            var edge_index = first_owned_edge[owner_index];

            while (edge_index) |index| : (edge_index = owned_type_edges[index].next) {
                const child_index = owned_type_edges[index].component_index;
                const child = types[child_index];

                if (contract_generation[child_index] == generation) {
                    continue;
                }

                assert(remaining_parents[child_index] > 0);

                remaining_parents[child_index] -= 1;

                const has_peer_signature_consumer = switch (child.signature_consumers) {
                    .multiple => true,
                    .none, .sole => typeHasExternallyVisiblePeerConsumer(child, function, functions),
                };

                if (child.owns_operations or
                    remaining_parents[child_index] != 0 or
                    has_peer_signature_consumer)
                {
                    continue;
                }

                contract_generation[child_index] = generation;
                pending[pending_count] = child_index;
                pending_count += 1;
                earliest_member = @min(earliest_member, named[child.named_index].member_index);
            }
        }

        var late_component: ?usize = null;

        for (types, 0..) |declaration, type_index| {
            if (contract_generation[type_index] == generation and
                named[declaration.named_index].member_index > function.member_index)
            {
                late_component = type_index;

                break;
            }
        }

        if (late_component) |type_index| {
            const component = named[types[type_index].named_index];

            const message = try std.fmt.allocPrint(
                context.scratch_allocator,
                "`{s}` completes `{s}`'s signature contract after the function",
                .{ component.name, function.name },
            );
            defer context.scratch_allocator.free(message);

            const help = try std.fmt.allocPrint(
                context.scratch_allocator,
                "move `{s}` into the contract batch before `{s}`",
                .{ component.name, function.name },
            );
            defer context.scratch_allocator.free(help);

            try context.report(.{
                .token = component.name_token,
                .message = message,
                .help = help,
            });

            continue;
        }

        for (members[earliest_member..function.member_index], earliest_member..) |member, member_index| {
            if (type_by_member[member_index]) |type_index| {
                if (contract_generation[type_index] == generation) {
                    continue;
                }

                const declaration = types[type_index];

                if (!declaration.owns_operations) {
                    switch (declaration.signature_consumers) {
                        .multiple => continue,
                        .none, .sole => if (!typeHasExternallyVisiblePeerConsumer(declaration, function, functions)) {
                            continue;
                        },
                    }
                }

                try reportContractInterruption(context, function, member);

                break;
            }

            var function_buffer: [1]Ast.Node.Index = undefined;
            if (context.ast.fullFnProto(&function_buffer, member) != null or
                context.ast.nodeTag(member) == .test_decl)
            {
                try reportContractInterruption(context, function, member);

                break;
            }
        }
    }
}

fn containerHasStateFields(ast: Ast, members: []const Ast.Node.Index) bool {
    for (members) |member| {
        if (ast.fullContainerField(member) != null) {
            return true;
        }
    }

    return false;
}

fn reportContractInterruption(
    context: *LintContext,
    function: FunctionDeclaration,
    declaration: Ast.Node.Index,
) Rule.Error!void {
    const message = try std.fmt.allocPrint(
        context.scratch_allocator,
        "`{s}`'s signature contract is interrupted by a peer declaration batch",
        .{function.name},
    );
    defer context.scratch_allocator.free(message);

    const help = try std.fmt.allocPrint(
        context.scratch_allocator,
        "keep the local signature contract with `{s}`, ahead of peer types and operations",
        .{function.name},
    );
    defer context.scratch_allocator.free(help);

    try context.report(.{
        .token = context.ast.firstToken(declaration),
        .message = message,
        .help = help,
    });
}

fn typeHasExternallyVisiblePeerConsumer(
    declaration: TypeDeclaration,
    owner: FunctionDeclaration,
    functions: []const FunctionDeclaration,
) bool {
    const consumer_member = switch (declaration.signature_consumers) {
        .sole => |member_index| member_index,
        .none, .multiple => return false,
    };

    if (consumer_member == owner.member_index) {
        return false;
    }

    for (functions) |function| {
        if (function.member_index == consumer_member) {
            return function.is_externally_visible;
        }
    }

    unreachable;
}

// Follow only transparent wrappers around one identifier. Computed,
// qualified, callback, and anonymous forms remain deliberately unconstrained.
fn simpleTypeReferenceToken(ast: Ast, initial_expression: Ast.Node.Index) ?Ast.TokenIndex {
    var expression = initial_expression;

    while (true) {
        if (ast.fullArrayType(expression)) |array| {
            expression = array.ast.elem_type;

            continue;
        }

        if (ast.fullPtrType(expression)) |pointer| {
            expression = pointer.ast.child_type;

            continue;
        }

        switch (ast.nodeTag(expression)) {
            .identifier => return ast.nodeMainToken(expression),
            .optional_type => expression = ast.nodeData(expression).node,
            .grouped_expression => expression = ast.nodeData(expression).node_and_token[0],
            .error_union => expression = ast.nodeData(expression).node_and_node[1],
            else => return null,
        }
    }
}

fn ownershipChainHasRoot(
    initial_index: usize,
    incoming_count: []const usize,
    sole_parent: []const ?usize,
) bool {
    var current = initial_index;
    var traversed: usize = 0;

    while (incoming_count[current] == 1) {
        current = sole_parent[current] orelse {
            return false;
        };
        traversed += 1;

        if (traversed >= incoming_count.len) {
            return false;
        }
    }

    return incoming_count[current] == 0;
}

fn lintFunctionOrder(
    context: *LintContext,
    container_node: Ast.Node.Index,
    functions: []FunctionDeclaration,
    function_by_name: *const std.StringHashMapUnmanaged(usize),
) Rule.Error!void {
    // No function can acquire a caller or an ambiguous reference here.
    if (functions.len == 0) {
        return;
    }

    // A reversed-order diagnostic needs a private function before a later
    // caller body. Walk existing metadata backward to rule that out before
    // spelling or hashing source references. The private target itself need
    // not have a body; only the possible caller does.
    var has_later_body = false;
    var can_report_reversed_order = false;
    var reverse_function_index = functions.len;

    while (reverse_function_index != 0) {
        reverse_function_index -= 1;

        const function = functions[reverse_function_index];
        if (!function.is_externally_visible and has_later_body) {
            can_report_reversed_order = true;

            break;
        }

        has_later_body = has_later_body or function.body_tokens != null;
    }

    if (!can_report_reversed_order) {
        return;
    }

    // Reject impossible names before finding their full spelling or hashing
    // them. Collisions only admit extra exact lookups; no matching function
    // may be rejected. Quoted names retain their raw source spelling.
    var function_name_prefix_filter = std.StaticBitSet(256).initEmpty();
    var single_byte_function_names = std.StaticBitSet(256).initEmpty();

    for (functions) |function| {
        const name = function.name;
        if (name.len == 1) {
            single_byte_function_names.set(name[0]);

            continue;
        }

        function_name_prefix_filter.set(namePrefixIndex(name[0], name[1]));
    }

    const ast = context.ast;
    var token = ast.firstToken(container_node);
    const last_token = ast.lastToken(container_node);
    var caller_cursor: usize = 0;

    while (token <= last_token) : (token += 1) {
        if (ast.tokenTag(token) != .identifier) {
            continue;
        }

        const source_start_offset = ast.tokenStart(token);
        const first_byte = ast.source[source_start_offset];

        // A one-byte name remains a candidate regardless of its following
        // punctuation or whitespace. A token at EOF must not read past source.
        if (!single_byte_function_names.isSet(first_byte)) {
            const second_byte_offset = source_start_offset + 1;
            if (second_byte_offset >= ast.source.len) {
                continue;
            }

            const prefix_index = namePrefixIndex(first_byte, ast.source[second_byte_offset]);
            if (!function_name_prefix_filter.isSet(prefix_index)) {
                continue;
            }
        }

        // Prefix membership proves nothing about reference kind. Keep the
        // exact lookup and all non-call/outside-body ambiguity checks below.
        const function_index = function_by_name.get(identifierText(ast, token)) orelse {
            continue;
        };

        if (token == functions[function_index].name_token) {
            continue;
        }

        const caller_index = functionContainingToken(functions, token, &caller_cursor) orelse {
            functions[function_index].callers = .ambiguous;

            continue;
        };

        if (!identifierIsUnqualifiedCall(ast, token)) {
            functions[function_index].callers = .ambiguous;

            continue;
        }

        if (caller_index == function_index) {
            functions[function_index].callers = .ambiguous;

            continue;
        }

        switch (functions[function_index].callers) {
            .none => functions[function_index].callers = .{ .sole = caller_index },

            .sole => |existing_caller| {
                if (existing_caller != caller_index) {
                    functions[function_index].callers = .ambiguous;
                }
            },

            .ambiguous => {},
        }
    }

    for (functions) |function| {
        if (function.is_externally_visible) {
            continue;
        }

        const caller_index = switch (function.callers) {
            .sole => |index| index,
            .none, .ambiguous => continue,
        };

        const caller = functions[caller_index];

        // Enforce one Style-B layer beneath an otherwise uncalled operation.
        switch (caller.callers) {
            .sole, .ambiguous => continue,
            .none => {},
        }

        if (caller.member_index < function.member_index) {
            continue;
        }

        const message = try std.fmt.allocPrint(
            context.scratch_allocator,
            "`{s}` appears before its only caller `{s}`",
            .{ function.name, caller.name },
        );
        defer context.scratch_allocator.free(message);

        const help = try std.fmt.allocPrint(
            context.scratch_allocator,
            "move `{s}` below `{s}`, or inline it if it only implements this call-site phase",
            .{ function.name, caller.name },
        );
        defer context.scratch_allocator.free(help);

        try context.report(.{
            .token = function.name_token,
            .message = message,
            .help = help,
            .note = "keep a separate function when it owns an independent operation, invariant, failure policy, lifecycle, or substantial bounded phase",
        });
    }
}

// Build and query the same 256-bit filter. Only the low eight bits matter;
// widening before multiplication keeps every input byte defined in safe builds.
inline fn namePrefixIndex(first_byte: u8, second_byte: u8) u8 {
    return @truncate((@as(u16, first_byte) * 33) ^ second_byte);
}

fn functionContainingToken(
    functions: []const FunctionDeclaration,
    token: Ast.TokenIndex,
    caller_cursor: *usize,
) ?usize {
    // Container members and lookup tokens both follow source order. Once a
    // body ends, no later lookup can belong to it; declarations have no body.
    while (caller_cursor.* < functions.len) : (caller_cursor.* += 1) {
        const body_tokens = functions[caller_cursor.*].body_tokens orelse {
            continue;
        };

        if (token > body_tokens.last) {
            continue;
        }

        return if (token >= body_tokens.first)
            caller_cursor.*
        else
            null;
    }

    return null;
}

fn identifierIsUnqualifiedCall(ast: Ast, token: Ast.TokenIndex) bool {
    if (token + 1 >= ast.tokens.len or ast.tokenTag(token + 1) != .l_paren) {
        return false;
    }

    if (token > 0 and
        (ast.tokenTag(token - 1) == .period or ast.tokenTag(token - 1) == .keyword_fn))
    {
        return false;
    }

    return true;
}

fn functionParameterHasName(ast: Ast, proto: Ast.full.FnProto, name: []const u8) bool {
    var parameter_iterator = proto.iterate(&ast);

    while (parameter_iterator.next()) |parameter| {
        const name_token = parameter.name_token orelse {
            continue;
        };

        if (std.mem.eql(u8, identifierText(ast, name_token), name)) {
            return true;
        }
    }

    return false;
}

fn declarationNameToken(ast: Ast, variable: Ast.full.VarDecl) ?Ast.TokenIndex {
    const name_token = variable.ast.mut_token + 1;

    return if (ast.tokenTag(name_token) == .identifier) name_token else null;
}

fn functionBody(ast: Ast, node: Ast.Node.Index) ?Ast.Node.Index {
    return if (ast.nodeTag(node) == .fn_decl)
        ast.nodeData(node).node_and_node[1]
    else
        null;
}

fn functionIsExternallyVisible(ast: Ast, proto: Ast.full.FnProto) bool {
    if (proto.visib_token != null) {
        return true;
    }

    const extern_export_inline_token = proto.extern_export_inline_token orelse {
        return false;
    };

    return switch (ast.tokenTag(extern_export_inline_token)) {
        .keyword_extern, .keyword_export => true,
        else => false,
    };
}

fn initializerIsImportDerived(ast: Ast, initializer: Ast.Node.Index) bool {
    var expression = initializer;

    while (true) {
        switch (ast.nodeTag(expression)) {
            .field_access => expression = ast.nodeData(expression).node_and_token[0],
            .grouped_expression => expression = ast.nodeData(expression).node_and_token[0],
            else => return std.mem.eql(
                u8,
                ast.tokenSlice(ast.nodeMainToken(expression)),
                "@import",
            ),
        }
    }
}

fn typeOwnsOperations(ast: Ast, node: Ast.Node.Index) bool {
    var container_buffer: [2]Ast.Node.Index = undefined;
    const container = ast.fullContainerDecl(&container_buffer, node) orelse {
        return false;
    };

    for (container.ast.members) |member| {
        var function_buffer: [1]Ast.Node.Index = undefined;
        if (ast.fullFnProto(&function_buffer, member) != null) {
            return true;
        }
    }

    return false;
}
