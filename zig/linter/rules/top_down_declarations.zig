const std = @import("std");
const Ast = std.zig.Ast;
const Diagnostics = @import("../Diagnostics.zig");

/// Order explicit signature vocabulary before each callable, owning aggregate
/// types before uniquely referenced private components, and unambiguous private
/// implementations after their sole caller. Ambiguous relationships are left
/// unconstrained rather than approximated as semantic resolution.
pub fn lint(
    file_allocator: std.mem.Allocator,
    report: *Diagnostics.Report,
) std.mem.Allocator.Error!void {
    try lintContainer(file_allocator, report, .root, null);
}

const rule_name = "top_down_declarations";

const NamedDeclaration = struct {
    name: []const u8,
    name_token: Ast.TokenIndex,
    member_index: usize,
    initializer: Ast.Node.Index,
    is_import: bool,
    is_public: bool,
};

const TypeDeclaration = struct {
    named_index: usize,
    initializer: Ast.Node.Index,
};

const FunctionDeclaration = struct {
    name: []const u8,
    name_token: Ast.TokenIndex,
    member_index: usize,
    node: Ast.Node.Index,
    body: ?Ast.Node.Index,
    is_externally_visible: bool,
    direct_call_count: usize = 0,
    sole_caller: ?usize = null,
    has_ambiguous_reference: bool = false,
};

fn lintContainer(
    file_allocator: std.mem.Allocator,
    report: *Diagnostics.Report,
    node: Ast.Node.Index,
    container_name: ?[]const u8,
) std.mem.Allocator.Error!void {
    const ast = report.file.ast;

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
                .name = ast.tokenSlice(name_token),
                .name_token = name_token,
                .member_index = member_index,
                .initializer = initializer,
                .is_import = isImport(ast, initializer),
                .is_public = variable.visib_token != null,
            };

            if (isDirectType(ast, initializer)) {
                type_storage[type_count] = .{
                    .named_index = named_count,
                    .initializer = initializer,
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

        function_storage[function_count] = .{
            .name = ast.tokenSlice(name_token),
            .name_token = name_token,
            .member_index = member_index,
            .node = member,
            .body = functionBody(ast, member),
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

    try lintSignatureOrder(file_allocator, report, named, &named_by_name, functions);
    try lintTypeOwnerOrder(file_allocator, report, named, types, &type_by_name);
    try lintFunctionOrder(report, node, container_name, functions, &function_by_name);

    for (types) |declaration| {
        const name = named[declaration.named_index].name;

        if (ast.fullContainerDecl(&container_buffer, declaration.initializer) != null) {
            try lintContainer(file_allocator, report, declaration.initializer, name);
        }
    }
}

fn lintSignatureOrder(
    file_allocator: std.mem.Allocator,
    report: *Diagnostics.Report,
    named: []const NamedDeclaration,
    named_by_name: *const std.StringHashMapUnmanaged(usize),
    functions: []const FunctionDeclaration,
) std.mem.Allocator.Error!void {
    const ast = report.file.ast;

    const seen = try file_allocator.alloc(bool, named.len);
    defer file_allocator.free(seen);

    for (functions) |function| {
        @memset(seen, false);

        var function_buffer: [1]Ast.Node.Index = undefined;
        const proto = ast.fullFnProto(&function_buffer, function.node) orelse {
            continue;
        };

        var parameter_iterator = proto.iterate(&ast);

        while (parameter_iterator.next()) |parameter| {
            const type_expression = parameter.type_expr orelse {
                continue;
            };

            try lintSignatureTypeExpression(
                report,
                named,
                named_by_name,
                function,
                proto,
                type_expression,
                seen,
            );
        }

        const return_type = proto.ast.return_type.unwrap() orelse {
            continue;
        };

        try lintSignatureTypeExpression(
            report,
            named,
            named_by_name,
            function,
            proto,
            return_type,
            seen,
        );
    }
}

fn lintSignatureTypeExpression(
    report: *Diagnostics.Report,
    named: []const NamedDeclaration,
    named_by_name: *const std.StringHashMapUnmanaged(usize),
    function: FunctionDeclaration,
    proto: Ast.full.FnProto,
    expression: Ast.Node.Index,
    seen: []bool,
) std.mem.Allocator.Error!void {
    const ast = report.file.ast;
    var token = ast.firstToken(expression);
    const last_token = ast.lastToken(expression);

    while (token <= last_token) : (token += 1) {
        if (ast.tokenTag(token) != .identifier or
            identifierIsQualifiedMember(ast, token) or
            identifierIsFieldName(ast, token, last_token))
        {
            continue;
        }

        const name = ast.tokenSlice(token);
        if (std.zig.isPrimitive(name) or
            std.mem.eql(u8, name, "Self") or
            functionParameterHasName(ast, proto, name))
        {
            continue;
        }

        const named_index = named_by_name.get(name) orelse {
            continue;
        };

        const declaration = named[named_index];

        if (declaration.is_import or
            declaration.member_index < function.member_index or
            seen[named_index])
        {
            continue;
        }

        seen[named_index] = true;

        const message = try std.fmt.allocPrint(
            report.diagnostic_allocator,
            "`{s}` is used in `{s}`'s signature before its declaration",
            .{ name, function.name },
        );
        errdefer report.diagnostic_allocator.free(message);

        const help = try std.fmt.allocPrint(
            report.diagnostic_allocator,
            "move `{s}` before `{s}`",
            .{ name, function.name },
        );
        errdefer report.diagnostic_allocator.free(help);

        try report.add(.{
            .token = token,
            .rule_name = rule_name,
            .message = message,
            .help = help,
        });
    }
}

fn lintTypeOwnerOrder(
    file_allocator: std.mem.Allocator,
    report: *Diagnostics.Report,
    named: []const NamedDeclaration,
    types: []const TypeDeclaration,
    type_by_name: *const std.StringHashMapUnmanaged(usize),
) std.mem.Allocator.Error!void {
    if (types.len == 0) {
        return;
    }

    const incoming_count = try file_allocator.alloc(usize, types.len);
    defer file_allocator.free(incoming_count);

    const sole_parent = try file_allocator.alloc(?usize, types.len);
    defer file_allocator.free(sole_parent);

    const seen_generation = try file_allocator.alloc(usize, types.len);
    defer file_allocator.free(seen_generation);

    @memset(incoming_count, 0);
    @memset(sole_parent, null);
    @memset(seen_generation, 0);

    for (types, 0..) |owner, owner_index| {
        recordOwnedTypes(
            report.file.ast,
            owner.initializer,
            owner_index,
            owner_index + 1,
            type_by_name,
            incoming_count,
            sole_parent,
            seen_generation,
        );
    }

    for (types, 0..) |component, component_index| {
        if (incoming_count[component_index] != 1 or named[component.named_index].is_public) {
            continue;
        }

        const root_index = ownershipChainRoot(component_index, incoming_count, sole_parent) orelse {
            continue;
        };

        if (!named[types[root_index].named_index].is_public) {
            continue;
        }

        const owner_index = sole_parent[component_index] orelse {
            continue;
        };

        const owner = types[owner_index];
        const owner_named = named[owner.named_index];
        const component_named = named[component.named_index];

        if (owner_named.member_index < component_named.member_index) {
            continue;
        }

        const message = try std.fmt.allocPrint(
            report.diagnostic_allocator,
            "owner type `{s}` is declared after its component `{s}`",
            .{ owner_named.name, component_named.name },
        );
        errdefer report.diagnostic_allocator.free(message);

        const help = try std.fmt.allocPrint(
            report.diagnostic_allocator,
            "move `{s}` before `{s}`",
            .{ owner_named.name, component_named.name },
        );
        errdefer report.diagnostic_allocator.free(help);

        try report.add(.{
            .token = owner_named.name_token,
            .rule_name = rule_name,
            .message = message,
            .help = help,
        });
    }
}

fn recordOwnedTypes(
    ast: Ast,
    initializer: Ast.Node.Index,
    owner_index: usize,
    generation: usize,
    type_by_name: *const std.StringHashMapUnmanaged(usize),
    incoming_count: []usize,
    sole_parent: []?usize,
    seen_generation: []usize,
) void {
    var container_buffer: [2]Ast.Node.Index = undefined;
    const container = ast.fullContainerDecl(&container_buffer, initializer) orelse {
        return;
    };

    for (container.ast.members) |member| {
        const field = ast.fullContainerField(member) orelse {
            continue;
        };

        const type_expression = field.ast.type_expr.unwrap() orelse {
            continue;
        };

        var token = ast.firstToken(type_expression);
        const last_token = ast.lastToken(type_expression);

        while (token <= last_token) : (token += 1) {
            if (ast.tokenTag(token) != .identifier or
                identifierIsQualifiedMember(ast, token) or
                identifierIsFieldName(ast, token, last_token))
            {
                continue;
            }

            const name = ast.tokenSlice(token);
            if (std.zig.isPrimitive(name)) {
                continue;
            }

            const component_index = type_by_name.get(name) orelse {
                continue;
            };

            if (component_index == owner_index or seen_generation[component_index] == generation) {
                continue;
            }

            seen_generation[component_index] = generation;
            incoming_count[component_index] += 1;
            sole_parent[component_index] = if (incoming_count[component_index] == 1)
                owner_index
            else
                null;
        }
    }
}

fn ownershipChainRoot(
    initial_index: usize,
    incoming_count: []const usize,
    sole_parent: []const ?usize,
) ?usize {
    var current = initial_index;
    var traversed: usize = 0;

    while (incoming_count[current] == 1) {
        current = sole_parent[current] orelse {
            return null;
        };
        traversed += 1;

        if (traversed > incoming_count.len) {
            return null;
        }
    }

    return if (incoming_count[current] == 0) current else null;
}

fn lintFunctionOrder(
    report: *Diagnostics.Report,
    container_node: Ast.Node.Index,
    container_name: ?[]const u8,
    functions: []FunctionDeclaration,
    function_by_name: *const std.StringHashMapUnmanaged(usize),
) std.mem.Allocator.Error!void {
    const ast = report.file.ast;
    var token = ast.firstToken(container_node);
    const last_token = ast.lastToken(container_node);

    while (token <= last_token) : (token += 1) {
        if (ast.tokenTag(token) != .identifier) {
            continue;
        }

        const function_index = function_by_name.get(ast.tokenSlice(token)) orelse {
            continue;
        };

        if (token == functions[function_index].name_token) {
            continue;
        }

        const caller_index = functionContainingToken(ast, functions, token) orelse {
            functions[function_index].has_ambiguous_reference = true;

            continue;
        };

        if (!identifierIsDirectCall(ast, token, container_name)) {
            functions[function_index].has_ambiguous_reference = true;

            continue;
        }

        if (caller_index == function_index) {
            functions[function_index].has_ambiguous_reference = true;

            continue;
        }

        functions[function_index].direct_call_count += 1;
        functions[function_index].sole_caller = caller_index;
    }

    for (functions) |function| {
        if (function.is_externally_visible or
            function.has_ambiguous_reference or
            function.direct_call_count != 1)
        {
            continue;
        }

        const caller = functions[
            function.sole_caller orelse {
                continue;
            }
        ];

        if (caller.member_index < function.member_index) {
            continue;
        }

        const message = try std.fmt.allocPrint(
            report.diagnostic_allocator,
            "`{s}` appears before its only caller `{s}`",
            .{ function.name, caller.name },
        );
        errdefer report.diagnostic_allocator.free(message);

        const help = try std.fmt.allocPrint(
            report.diagnostic_allocator,
            "move `{s}` below `{s}`, or inline it if it only implements this call-site phase",
            .{ function.name, caller.name },
        );
        errdefer report.diagnostic_allocator.free(help);

        try report.add(.{
            .token = function.name_token,
            .rule_name = rule_name,
            .message = message,
            .help = help,
            .note = "keep a separate function when it owns an independent operation, invariant, failure policy, lifecycle, or substantial bounded phase",
        });
    }
}

fn functionContainingToken(
    ast: Ast,
    functions: []const FunctionDeclaration,
    token: Ast.TokenIndex,
) ?usize {
    for (functions, 0..) |function, index| {
        const body = function.body orelse {
            continue;
        };

        if (token >= ast.firstToken(body) and token <= ast.lastToken(body)) {
            return index;
        }
    }

    return null;
}

fn identifierIsDirectCall(ast: Ast, token: Ast.TokenIndex, container_name: ?[]const u8) bool {
    if (token + 1 >= ast.tokens.len or ast.tokenTag(token + 1) != .l_paren) {
        return false;
    }

    if (token == 0 or ast.tokenTag(token - 1) != .period) {
        return true;
    }

    if (token < 2 or ast.tokenTag(token - 2) != .identifier) {
        return false;
    }

    const qualifier = ast.tokenSlice(token - 2);
    if (std.mem.eql(u8, qualifier, "Self")) {
        return true;
    }

    const name = container_name orelse {
        return false;
    };

    return std.mem.eql(u8, qualifier, name);
}

fn functionParameterHasName(ast: Ast, proto: Ast.full.FnProto, name: []const u8) bool {
    var parameter_iterator = proto.iterate(&ast);

    while (parameter_iterator.next()) |parameter| {
        const name_token = parameter.name_token orelse {
            continue;
        };

        if (std.mem.eql(u8, ast.tokenSlice(name_token), name)) {
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

fn identifierIsQualifiedMember(ast: Ast, token: Ast.TokenIndex) bool {
    return token > 0 and ast.tokenTag(token - 1) == .period;
}

fn identifierIsFieldName(ast: Ast, token: Ast.TokenIndex, last_token: Ast.TokenIndex) bool {
    return token < last_token and ast.tokenTag(token + 1) == .colon;
}

fn isImport(ast: Ast, initializer: Ast.Node.Index) bool {
    return std.mem.eql(u8, ast.tokenSlice(ast.nodeMainToken(initializer)), "@import");
}

fn isDirectType(ast: Ast, node: Ast.Node.Index) bool {
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
