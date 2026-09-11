//! `LintContext` owns host-only per-file indexes and one source-level
//! `Rule.Context` projection. Each built-in rule receives an invocation-local
//! copy connected to report construction and fix collection.

const LintContext = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Ast = std.zig.Ast;
const Fixes = @import("Fixes.zig");
const LintReport = @import("LintReport.zig");
const Rule = @import("Rule.zig");

path: []const u8,
rule_context: Rule.Context,
report_builder: *LintReport.Builder,
fixes: ?*Fixes,

pub fn init(
    scratch_allocator: std.mem.Allocator,
    path: []const u8,
    ast: Ast,
    target_zig_version: std.SemanticVersion,
    report_builder: *LintReport.Builder,
    fixes: ?*Fixes,
) std.mem.Allocator.Error!LintContext {
    const source = ast.source;
    const source_size = source.len;

    assert(source_size <= std.math.maxInt(u32));

    // ── Allocate Index Storage ──
    //
    // Line zero starts at byte zero even for empty input. An LF at EOF later
    // appends `source.len`, preserving the empty final line for EOF tokens.

    var line_start_offsets: std.ArrayList(u32) = .empty;
    errdefer line_start_offsets.deinit(scratch_allocator);

    try line_start_offsets.append(scratch_allocator, 0);

    const token_line_indexes = try scratch_allocator.alloc(u32, ast.tokens.len);
    errdefer scratch_allocator.free(token_line_indexes);

    const token_start_offsets: []const Ast.ByteOffset = ast.tokens.items(.start);

    // ── Prepare Source Regions ──
    //
    // The prefix region borrows every complete source block. The tail region
    // is empty or contains one zero-padded block, so both feed the same SIMD
    // loop without a load crossing EOF.

    const source_bytes_per_block = 32;
    const SourceBlock = [source_bytes_per_block]u8;
    const SourceLanes = @Vector(source_bytes_per_block, u8);
    const SourceRegion = []const SourceBlock;

    const newline_lanes: SourceLanes = @splat('\n');

    const source_tail_start_offset = source_size - (source_size % source_bytes_per_block);
    const source_prefix_region = std.mem.bytesAsSlice(SourceBlock, source[0..source_tail_start_offset]);

    const source_tail_size = source_size - source_tail_start_offset;

    var source_tail_storage: [1]SourceBlock = undefined;
    const source_tail_region: SourceRegion = if (source_tail_size == 0)
        &.{}
    else tail: {
        @memset(&source_tail_storage[0], 0);
        @memcpy(source_tail_storage[0][0..source_tail_size], source[source_tail_start_offset..]);

        break :tail &source_tail_storage;
    };

    // ── Main SIMD Loop ──
    //
    // Two regions keep source-versus-tail selection outside the hot block loop;
    // padding the tail avoids a separate scalar scan.

    // Eight u32 token offsets occupy the same 32-byte vector as one source block.
    const token_offset_lane_count = 8;

    // This always selects the first token whose line assignment is not final.
    var first_unfinalized_token_index: usize = 0;

    for (
        [_]SourceRegion{
            source_prefix_region,
            source_tail_region,
        },
        [_]usize{ 0, source_tail_start_offset },
    ) |source_region, source_region_start_offset| {
        for (source_region, 0..) |source_block, source_block_index| {
            const source_block_start_offset = source_region_start_offset + source_block_index * source_bytes_per_block;
            const source_lanes: SourceLanes = source_block;

            var newline_lanes_mask: u32 = @bitCast(source_lanes == newline_lanes);

            while (newline_lanes_mask != 0) {
                // Mask bit `i` represents source lane `i`:
                //
                //   source lane:  0   1    2   3    4   5   6   7
                //   source byte: 'a' 'b' '\n' 'c' '\n' 'd' 'e' 'f'
                //   matches:      0   0    1   0    1   0   0   0
                //
                // Integer masks print the highest bit first:
                //
                //   mask bit:   7 6 5 4 3 2 1 0
                //   mask value: 0 0 0 1 0 1 0 0
                //                           ^
                //                  first set bit from bit zero
                //
                // `ctz` therefore returns lane 2, the earliest remaining newline.
                const newline_lane_index = @ctz(newline_lanes_mask);

                // The byte after this newline begins the next line.
                const next_line_start_offset: u32 = @intCast(source_block_start_offset + newline_lane_index + 1);

                // The next line is not appended yet, so `len - 1` is the
                // zero-based index of the line this newline closes.
                const current_line_index: u32 = @intCast(line_start_offsets.items.len - 1);

                const current_line_index_vector: @Vector(token_offset_lane_count, u32) = @splat(current_line_index);
                const next_line_start_offset_vector: @Vector(token_offset_lane_count, u32) = @splat(next_line_start_offset);

                finalize_current_line_tokens: {
                    while (token_start_offsets.len - first_unfinalized_token_index >= token_offset_lane_count) {
                        const token_start_offset_vector: @Vector(token_offset_lane_count, u32) =
                            token_start_offsets[first_unfinalized_token_index..][0..token_offset_lane_count].*;

                        const current_line_prefix_mask: u8 = @bitCast(token_start_offset_vector < next_line_start_offset_vector);
                        const current_line_prefix_token_count = @popCount(current_line_prefix_mask);

                        token_line_indexes[first_unfinalized_token_index..][0..token_offset_lane_count].* = current_line_index_vector;
                        first_unfinalized_token_index += current_line_prefix_token_count;

                        if (current_line_prefix_token_count < token_offset_lane_count) {
                            // A count below the lane count proves that at least one
                            // token offset is `>= next_line_start_offset`:
                            //
                            //   next line start: 24
                            //
                            //   token offsets: [4, 9, 12, 20 | 24, 30, 34, 40]
                            //                   ^  ^   ^   ^    ^
                            //                   four matches     first nonmatch
                            //
                            //   `< 24`: [1, 1, 1, 1 | 0, 0, 0, 0]
                            //   count:    4 < 8
                            //
                            // After `+= 4`, the cursor points at 24. The offsets are
                            // sorted, so this line has no tokens after that cursor.
                            break :finalize_current_line_tokens;
                        }
                    }

                    // Fewer than eight offsets remain in the token array, so a full
                    // vector load would cross its end. This is only the token tail;
                    // the source scan may still find more newlines.
                    while (true) {
                        if (first_unfinalized_token_index == token_start_offsets.len) {
                            // The cursor reached the end of the token array; no token
                            // remains to assign.
                            break;
                        }

                        if (token_start_offsets[first_unfinalized_token_index] >= next_line_start_offset) {
                            // The next line start is this line's exclusive end:
                            //
                            //   current line: [..., 24)
                            //   later lines:          [24, ...)
                            //                         ^ pending token
                            //
                            // Leave that token at the cursor. The outer source scan
                            // retries it at the next newline; if none remains, the
                            // final fill assigns it.
                            break;
                        }

                        token_line_indexes[first_unfinalized_token_index] = current_line_index;
                        first_unfinalized_token_index += 1;
                    }
                }

                // Before this append, `len - 1` indexes the line just finalized.
                // Appending advances it to the line opened by this newline.
                try line_start_offsets.append(scratch_allocator, next_line_start_offset);

                // Clear the lowest set bit that `ctz` selected.
                newline_lanes_mask &= newline_lanes_mask - 1;
            }
        }
    }

    // ── Finalize And Publish ──
    //
    // No newline closes the final line. Assign all remaining tokens to it,
    // including EOF on the empty line after a trailing LF. This also corrects
    // any provisional vector writes beyond the last finalized prefix.

    const final_line_index: u32 = @intCast(line_start_offsets.items.len - 1);

    @memset(token_line_indexes[first_unfinalized_token_index..], final_line_index);

    const owned_line_start_offsets = try line_start_offsets.toOwnedSlice(scratch_allocator);
    errdefer scratch_allocator.free(owned_line_start_offsets);

    return .{
        .path = path,

        // Each rule receives a copy with its stack-local reporting adapter.
        // Keeping that callback out of this stored projection avoids retaining
        // a self-pointer in a context returned by value.
        .rule_context = .{
            .scratch_allocator = scratch_allocator,
            .target_zig_version = target_zig_version,
            .ast = ast,
            .fixes_enabled = fixes != null,
            .line_start_offsets = owned_line_start_offsets,
            .token_line_indexes = token_line_indexes,
            .report_state = undefined,
            .report_fn = reportOutsideInvocation,
        },
        .report_builder = report_builder,
        .fixes = fixes,
    };
}

pub fn deinit(context: *LintContext) void {
    const allocator = context.rule_context.scratch_allocator;

    allocator.free(context.rule_context.token_line_indexes);
    allocator.free(context.rule_context.line_start_offsets);

    context.* = undefined;
}

const RuleInvocation = struct {
    host: *LintContext,
    definition: Rule.Definition,

    /// Reporting failures are sticky so a rule cannot catch one and let the
    /// operation continue after the host rejected a finding or failed to retain
    /// its fix.
    report_failure: ?Rule.Error = null,
};

pub fn runRule(context: *LintContext, definition: Rule.Definition) Rule.Error!void {
    var invocation: RuleInvocation = .{
        .host = context,
        .definition = definition,
    };

    var rule_context = context.rule_context;

    rule_context.report_state = &invocation;
    rule_context.report_fn = reportRuleFinding;

    definition.lint(&rule_context) catch |failure| {
        if (invocation.report_failure) |report_failure| {
            return report_failure;
        }

        return failure;
    };

    if (invocation.report_failure) |report_failure| {
        return report_failure;
    }
}

/// External and built-in findings share the same validation and synchronous
/// copy path; no plugin-owned slice escapes its report callback.
pub fn reportExternalFinding(
    context: *LintContext,
    rule_name: []const u8,
    finding: Rule.Finding,
) Rule.Error!void {
    return context.addFinding(rule_name, finding);
}

pub fn fixesEnabled(context: *const LintContext) bool {
    return context.fixes != null;
}

fn reportRuleFinding(state: *anyopaque, finding: Rule.Finding) Rule.Error!void {
    const invocation: *RuleInvocation = @ptrCast(@alignCast(state));
    if (invocation.report_failure) |report_failure| {
        return report_failure;
    }

    invocation.host.addFinding(invocation.definition.name, finding) catch |failure| {
        invocation.report_failure = failure;

        return failure;
    };
}

/// Resolve one transient finding into the run-wide report and retain its
/// optional fix. Validation happens before either output is mutated.
fn addFinding(
    context: *LintContext,
    rule_name: []const u8,
    finding: Rule.Finding,
) Rule.Error!void {
    const rule_context = &context.rule_context;
    if (finding.token >= rule_context.ast.tokens.len or finding.message.len == 0) {
        return error.FindingRejected;
    }

    if (finding.fix) |fix| {
        if (context.fixes == null or
            fix.range.start_offset > fix.range.end_offset or
            @as(usize, fix.range.end_offset) > rule_context.ast.source.len)
        {
            return error.FindingRejected;
        }
    }

    const location = rule_context.tokenLocation(finding.token);

    context.report_builder.addDiagnostic(.{
        .path = context.path,
        .rule_name = rule_name,
        .message = finding.message,
        .help = if (finding.help.len == 0) null else finding.help,
        .note = if (finding.note.len == 0) null else finding.note,
        .location = .{
            .line = location.line + 1,
            .column = location.column + 1,
            .range = rule_context.tokenRange(finding.token),
        },
        .source_line = rule_context.ast.source[location.line_start..location.line_end],
    });

    if (context.fixes) |fixes| {
        if (finding.fix) |fix| {
            if (!std.mem.eql(
                u8,
                rule_context.ast.source[fix.range.start_offset..fix.range.end_offset],
                fix.replacement,
            )) {
                const replacement = try rule_context.scratch_allocator.dupe(u8, fix.replacement);

                try fixes.add(.{ .range = fix.range, .replacement = replacement });
            }
        }
    }
}

fn reportOutsideInvocation(_: *anyopaque, _: Rule.Finding) Rule.Error!void {
    unreachable;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

test "identifier text matches AST token slices" {
    const allocator = std.testing.allocator;

    const sources = [_][:0]const u8{
        "",
        "fn name_0123() void { const U8 = u8; _ = U8; }",
        "first\r\nsecond\tlast_identifier",
        "@\"if\" @\"a b\" @\"\\x61\" @\"café\" @\"quote\\\"\"",
    };

    for (sources) |source| {
        var ast = try Ast.parse(allocator, source, .zig);
        defer ast.deinit(allocator);

        for (0..ast.tokens.len) |index| {
            const token: Ast.TokenIndex = @intCast(index);
            if (ast.tokenTag(token) != .identifier) {
                continue;
            }

            try std.testing.expectEqualStrings(ast.tokenSlice(token), Rule.Context.identifierText(ast, token));
        }
    }
}

test "rules receive the exact target Zig version" {
    const allocator = std.testing.allocator;
    const source: [:0]const u8 = "const value = 1;\n";

    const target: std.SemanticVersion = .{
        .major = 0,
        .minor = 16,
        .patch = 0,
        .pre = "dev.123",
        .build = "custom.456",
    };

    var ast = try Ast.parse(allocator, source, .zig);
    defer ast.deinit(allocator);

    var report_builder = try LintReport.Builder.init(allocator, allocator, .{});
    defer report_builder.deinit();

    var context = try LintContext.init(
        allocator,
        "target.zig",
        ast,
        target,
        &report_builder,
        null,
    );
    defer context.deinit();

    const Observer = struct {
        fn lint(rule_context: *Rule.Context) Rule.Error!void {
            const version = rule_context.target_zig_version;

            const prerelease = version.pre orelse {
                return;
            };

            const build = version.build orelse {
                return;
            };

            if (version.major != 0 or version.minor != 16 or version.patch != 0 or
                !std.mem.eql(u8, prerelease, "dev.123") or
                !std.mem.eql(u8, build, "custom.456"))
            {
                return;
            }

            try rule_context.report(.{
                .token = 0,
                .message = "observed target Zig version",
            });
        }
    };

    try context.runRule(Rule.define("target_observer", Observer.lint));
    try std.testing.expectEqual(@as(usize, 1), report_builder.totalDiagnosticCount());
}

test "generated line and token indexes match a scalar oracle" {
    const allocator = std.testing.allocator;
    const seed = 0x4c49_4e45_494e_4458;
    const alphabet = "abcXYZ_0123 \t\r\n(){}[]=;,.@\"\\/+*-";

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..512) |case_index| {
        errdefer std.debug.print("LintContext seed={d} case={d}\n", .{ seed, case_index });

        // The initial cases force every short vector boundary and tail length;
        // the remainder explore longer, irregular layouts reproducibly.
        const source_len = if (case_index < 130)
            case_index
        else
            random.uintLessThan(usize, 4097);

        const source = try allocator.allocSentinel(u8, source_len, 0);
        defer allocator.free(source);

        for (source) |*byte| {
            byte.* = alphabet[random.uintLessThan(usize, alphabet.len)];
        }

        var expected_line_starts: [4097]u32 = undefined;
        expected_line_starts[0] = 0;

        var expected_line_count: usize = 1;

        for (source, 0..) |byte, offset| {
            if (byte == '\n') {
                expected_line_starts[expected_line_count] = @intCast(offset + 1);
                expected_line_count += 1;
            }
        }

        var ast = try Ast.parse(allocator, source, .zig);
        defer ast.deinit(allocator);

        var report_builder = try LintReport.Builder.init(allocator, allocator, .{
            .diagnostics_count_max = 0,
        });
        defer report_builder.deinit();

        var context = try LintContext.init(
            allocator,
            "generated.zig",
            ast,
            builtin.zig_version,
            &report_builder,
            null,
        );
        defer context.deinit();

        try std.testing.expectEqualSlices(
            u32,
            expected_line_starts[0..expected_line_count],
            context.rule_context.line_start_offsets,
        );

        for (0..ast.tokens.len) |index| {
            const token: Ast.TokenIndex = @intCast(index);
            const expected_location = scalarTokenLocation(source, ast.tokenStart(token));

            try std.testing.expectEqualDeep(
                expected_location,
                context.rule_context.tokenLocation(token),
            );
        }
    }
}

fn scalarTokenLocation(source: []const u8, token_offset: Ast.ByteOffset) Ast.Location {
    const offset: usize = token_offset;

    var line: usize = 0;
    var line_start: usize = 0;

    for (source[0..offset], 0..) |byte, byte_offset| {
        if (byte == '\n') {
            line += 1;
            line_start = byte_offset + 1;
        }
    }

    const line_end = std.mem.indexOfScalarPos(u8, source, line_start, '\n') orelse source.len;

    return .{
        .line = line,
        .column = offset - line_start,
        .line_start = line_start,
        .line_end = line_end,
    };
}
