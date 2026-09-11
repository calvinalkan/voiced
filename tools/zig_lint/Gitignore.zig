//! `Gitignore` compiles and applies the `.gitignore` files active during one
//! directory scan.
//!
//! Rules remain active between matching `push` and `pop` calls. Matching checks
//! rules from newest to oldest, so later lines and `.gitignore` files in deeper
//! directories take precedence.
//! Blank and comment lines add no rule, malformed patterns never match, and
//! allocation failure is the only parsing failure.

const Gitignore = @This();

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

rules: std.ArrayList(Rule) = .empty,
scopes: std.ArrayList(Scope) = .empty,
source_bytes: std.ArrayList(u8) = .empty,

// ─── Scan Interface ──────────────────────────────────────────────────────────

pub fn deinit(gitignore: *Gitignore, allocator: Allocator) void {
    gitignore.source_bytes.deinit(allocator);
    gitignore.scopes.deinit(allocator);
    gitignore.rules.deinit(allocator);

    gitignore.* = undefined;
}

/// `push` activates one `.gitignore` relative to the shared matching root. It
/// copies `dir_path` and the compiled patterns, so the caller may release
/// `dir_path` and `contents` after return. `dir_path` is empty for the root and
/// otherwise has no leading or trailing slash. Use the same allocator for every
/// `push` and `deinit` call on one instance. Call `pop` once for each successful
/// `push`, in LIFO order.
pub fn push(
    gitignore: *Gitignore,
    allocator: Allocator,
    dir_path: []const u8,
    contents: []const u8,
) Allocator.Error!void {
    assert(dir_path.len == 0 or (dir_path[0] != '/' and dir_path[dir_path.len - 1] != '/'));

    // ── Record Scope Start ──

    const scope: Scope = .{
        .rules_count = gitignore.rules.items.len,
        .source_size = gitignore.source_bytes.items.len,
    };
    errdefer {
        gitignore.rules.items.len = scope.rules_count;
        gitignore.source_bytes.items.len = scope.source_size;
    }

    try gitignore.source_bytes.appendSlice(allocator, dir_path);

    const dir_path_range: ByteRange = .{
        .offset = scope.source_size,
        .size = dir_path.len,
    };

    // ── Compile Rules In File Order ──

    var lines = std.mem.splitScalar(u8, contents, '\n');

    while (lines.next()) |line| {
        const rule = try gitignore.parseRule(
            allocator,
            dir_path_range,
            line,
        ) orelse {
            continue;
        };

        try gitignore.rules.append(allocator, rule);
    }

    try gitignore.scopes.append(allocator, scope);
}

pub fn pop(gitignore: *Gitignore) void {
    const scope = gitignore.scopes.pop() orelse {
        unreachable;
    };

    assert(scope.rules_count <= gitignore.rules.items.len);
    assert(scope.source_size <= gitignore.source_bytes.items.len);

    gitignore.rules.items.len = scope.rules_count;
    gitignore.source_bytes.items.len = scope.source_size;
}

/// `pathIsIgnored` applies the active files to one nonempty, root-relative path
/// with `/` separators and no leading or trailing separator. A `true` result for
/// a directory is a prune boundary; callers must not inspect its descendants or
/// apply a descendant negation.
pub fn pathIsIgnored(gitignore: *const Gitignore, path: []const u8, path_is_dir: bool) bool {
    assert(path.len != 0 and path[0] != '/' and path[path.len - 1] != '/');

    const basename = std.fs.path.basename(path);
    var rule_index = gitignore.rules.items.len;

    while (rule_index != 0) {
        rule_index -= 1;

        const rule = gitignore.rules.items[rule_index];
        if (!gitignore.ruleMatches(rule, path, basename, path_is_dir)) {
            continue;
        }

        return rule.ignores;
    }

    return false;
}

const Scope = struct {
    rules_count: usize,
    source_size: usize,
};

// ─── Compiled Rules ──────────────────────────────────────────────────────────
//
// Rules store ranges into `source_bytes`, which owns copied directory paths and
// normalized patterns. Each scope records the rules count and source size so
// `pop` restores the previous rules without inspecting or moving their bytes.

const ByteRange = struct {
    offset: usize,
    size: usize,

    fn bytes(range: ByteRange, source: []const u8) []const u8 {
        return source[range.offset..][0..range.size];
    }
};

const Rule = struct {
    dir_path: ByteRange,
    pattern: ByteRange,
    ignores: bool,
    matches_dirs_only: bool,
    matcher: enum {
        basename_literal,
        basename_glob,
        path_literal,
        path_glob,
    },
};

fn parseRule(
    gitignore: *Gitignore,
    allocator: Allocator,
    dir_path: ByteRange,
    untrimmed_line: []const u8,
) Allocator.Error!?Rule {
    // ── Remove Line And Space Terminators ──

    var line_end_index = untrimmed_line.len;

    if (line_end_index != 0 and untrimmed_line[line_end_index - 1] == '\r') {
        line_end_index -= 1;
    }

    while (line_end_index != 0 and untrimmed_line[line_end_index - 1] == ' ') {
        const backslashes_count = countTrailingBackslashes(untrimmed_line[0 .. line_end_index - 1]);
        if (backslashes_count % 2 == 1) {
            // An odd backslash count escapes this final space, so both the
            // space and its escape remain part of the pattern.
            break;
        }

        line_end_index -= 1;
    }

    const line = untrimmed_line[0..line_end_index];
    if (line.len == 0 or line[0] == '#') {
        // Empty lines and lines beginning with an unescaped hash add no rule.
        return null;
    }

    // ── Decode Rule Controls ──

    const pattern_is_negated = line[0] == '!';
    var pattern_start_index: usize = @intFromBool(pattern_is_negated);

    const pattern_is_anchored = pattern_start_index < line.len and line[pattern_start_index] == '/';

    pattern_start_index += @intFromBool(pattern_is_anchored);

    var pattern_end_index = line.len;
    const pattern_targets_dir = pattern_end_index > pattern_start_index and line[pattern_end_index - 1] == '/';

    pattern_end_index -= @intFromBool(pattern_targets_dir);

    if (pattern_start_index == pattern_end_index) {
        // Bare `!`, `/`, and `!/` lines identify no candidate and therefore
        // contribute no rule.
        return null;
    }

    const pattern = line[pattern_start_index..pattern_end_index];
    const pattern_has_path_separator = std.mem.indexOfScalar(u8, pattern, '/') != null;
    const pattern_has_glob_syntax = std.mem.indexOfAny(u8, pattern, "*?[\\") != null;

    const matcher: @FieldType(Rule, "matcher") = if (pattern_is_anchored or pattern_has_path_separator)
        if (pattern_has_glob_syntax) .path_glob else .path_literal
    else if (pattern_has_glob_syntax)
        .basename_glob
    else
        .basename_literal;

    const pattern_offset = gitignore.source_bytes.items.len;

    try gitignore.appendNormalizedPattern(allocator, pattern);

    return .{
        .dir_path = dir_path,
        .pattern = .{
            .offset = pattern_offset,
            .size = gitignore.source_bytes.items.len - pattern_offset,
        },
        .ignores = !pattern_is_negated,
        .matches_dirs_only = pattern_targets_dir,
        .matcher = matcher,
    };
}

fn appendNormalizedPattern(gitignore: *Gitignore, allocator: Allocator, pattern: []const u8) Allocator.Error!void {
    try gitignore.source_bytes.ensureUnusedCapacity(allocator, pattern.len);

    var pattern_index: usize = 0;

    // `/` remains a path separator when escaped. The final backslash of an odd
    // run quotes the slash; preceding pairs still represent literal backslashes.

    while (pattern_index < pattern.len) {
        if (pattern[pattern_index] != '\\') {
            gitignore.source_bytes.appendAssumeCapacity(pattern[pattern_index]);

            pattern_index += 1;

            continue;
        }

        const backslashes_start_index = pattern_index;

        while (pattern_index < pattern.len and pattern[pattern_index] == '\\') {
            pattern_index += 1;
        }

        const backslashes_count = pattern_index - backslashes_start_index;
        const slash_follows = pattern_index < pattern.len and pattern[pattern_index] == '/';
        const retained_backslashes_count = backslashes_count - @intFromBool(slash_follows and backslashes_count % 2 == 1);

        gitignore.source_bytes.appendSliceAssumeCapacity(
            pattern[backslashes_start_index..][0..retained_backslashes_count],
        );

        if (slash_follows) {
            gitignore.source_bytes.appendAssumeCapacity('/');

            pattern_index += 1;
        }
    }
}

fn ruleMatches(
    gitignore: *const Gitignore,
    rule: Rule,
    path: []const u8,
    basename: []const u8,
    path_is_dir: bool,
) bool {
    if (rule.matches_dirs_only and !path_is_dir) {
        // A trailing slash gives the rule no authority over regular files with
        // the same basename.
        return false;
    }

    // ── Select Path Relative To The Gitignore Directory ──

    const gitignore_dir_path = rule.dir_path.bytes(gitignore.source_bytes.items);

    const relative_path = if (gitignore_dir_path.len == 0)
        path
    else relative: {
        if (path.len <= gitignore_dir_path.len or
            !std.mem.startsWith(u8, path, gitignore_dir_path) or
            path[gitignore_dir_path.len] != '/')
        {
            // The containing directory, sibling trees, and longer basenames
            // sharing its prefix are outside this `.gitignore` scope.
            return false;
        }

        break :relative path[gitignore_dir_path.len + 1 ..];
    };

    const pattern = rule.pattern.bytes(gitignore.source_bytes.items);

    return switch (rule.matcher) {
        .basename_literal => std.mem.eql(u8, pattern, basename),
        .basename_glob => segmentGlobMatches(pattern, basename),
        .path_literal => std.mem.eql(u8, pattern, relative_path),
        .path_glob => pathGlobMatches(pattern, relative_path),
    };
}

fn pathGlobMatches(pattern: []const u8, path: []const u8) bool {
    var pattern_index: usize = 0;
    var path_index: usize = 0;

    var double_star_retry: ?struct {
        pattern_index: usize,
        path_index: usize,
    } = null;

    // Each iteration attempts one pattern segment. A `**` records a retry point
    // that advances by complete path segments, so failed suffix matches never
    // trigger byte-by-byte recursive backtracking.
    while (true) {
        const pattern_segment_end_index = std.mem.indexOfScalarPos(
            u8,
            pattern,
            pattern_index,
            '/',
        ) orelse pattern.len;

        const next_pattern_index = pattern_segment_end_index + @intFromBool(pattern_segment_end_index != pattern.len);
        const pattern_segment = pattern[pattern_index..pattern_segment_end_index];

        if (pattern_segment.len >= 2 and std.mem.allEqual(u8, pattern_segment, '*')) {
            if (next_pattern_index == pattern.len) {
                // A trailing `/**` names contents beneath its prefix, not the
                // prefix directory itself.
                return path_index < path.len;
            }

            double_star_retry = .{
                .pattern_index = next_pattern_index,
                .path_index = path_index,
            };
            pattern_index = next_pattern_index;

            continue;
        }

        if (path_index < path.len) {
            const path_segment_end_index = std.mem.indexOfScalarPos(
                u8,
                path,
                path_index,
                '/',
            ) orelse path.len;

            if (segmentGlobMatches(pattern_segment, path[path_index..path_segment_end_index])) {
                pattern_index = next_pattern_index;
                path_index = path_segment_end_index + @intFromBool(path_segment_end_index != path.len);

                if (pattern_index == pattern.len and path_index == path.len) {
                    return true;
                }

                if (pattern_index != pattern.len and path_index != path.len) {
                    continue;
                }
            }
        }

        const retry = if (double_star_retry) |*value| value else {
            // No earlier `**` can absorb another path segment and retry this
            // unmatched suffix.
            return false;
        };

        if (retry.path_index == path.len) {
            // The recorded `**` already consumed every available segment.
            return false;
        }

        const consumed_segment_end_index = std.mem.indexOfScalarPos(
            u8,
            path,
            retry.path_index,
            '/',
        ) orelse path.len;

        retry.path_index = consumed_segment_end_index + @intFromBool(consumed_segment_end_index != path.len);
        pattern_index = retry.pattern_index;
        path_index = retry.path_index;
    }
}

fn segmentGlobMatches(pattern: []const u8, text: []const u8) bool {
    if (countTrailingBackslashes(pattern) % 2 == 1) {
        return false;
    }

    var pattern_index: usize = 0;
    var text_index: usize = 0;

    var star_retry: ?struct {
        pattern_index: usize,
        text_index: usize,
    } = null;

    // The most recent `*` is the only retry point required for a single path
    // segment. Every retry consumes one more byte, keeping matching iterative
    // and allocation-free.
    while (text_index < text.len) {
        if (pattern_index < pattern.len) {
            const pattern_byte = pattern[pattern_index];
            if (pattern_byte == '*') {
                while (pattern_index < pattern.len and pattern[pattern_index] == '*') {
                    pattern_index += 1;
                }

                if (pattern_index == pattern.len) {
                    // A trailing star consumes every byte that remains.
                    return true;
                }

                star_retry = .{
                    .pattern_index = pattern_index,
                    .text_index = text_index,
                };

                continue;
            }

            if (pattern_byte == '?') {
                pattern_index += 1;
                text_index += 1;

                continue;
            }

            if (pattern_byte == '[') {
                if (matchCharacterClass(pattern, pattern_index, text[text_index])) |pattern_end_index| {
                    pattern_index = pattern_end_index;
                    text_index += 1;

                    continue;
                }
            } else {
                const literal = consumePatternLiteral(pattern, &pattern_index);
                if (literal == text[text_index]) {
                    text_index += 1;

                    continue;
                }
            }
        }

        const retry = if (star_retry) |*value| value else {
            // No preceding star can consume the byte that failed to match.
            return false;
        };

        retry.text_index += 1;

        if (retry.text_index > text.len) {
            return false;
        }

        pattern_index = retry.pattern_index;
        text_index = retry.text_index;
    }

    while (pattern_index < pattern.len and pattern[pattern_index] == '*') {
        pattern_index += 1;
    }

    return pattern_index == pattern.len;
}

fn countTrailingBackslashes(bytes: []const u8) usize {
    var start_index = bytes.len;

    while (start_index != 0 and bytes[start_index - 1] == '\\') {
        start_index -= 1;
    }

    return bytes.len - start_index;
}

fn matchCharacterClass(
    pattern: []const u8,
    pattern_opening_index: usize,
    candidate: u8,
) ?usize {
    // ── Decode Class Prefix ──

    var index = pattern_opening_index + 1;
    if (index == pattern.len) {
        // An opening bracket at the end has no closing class delimiter.
        return null;
    }

    const is_negated = pattern[index] == '!' or pattern[index] == '^';

    index += @intFromBool(is_negated);

    // ── Evaluate Members And Ranges ──

    var has_member = false;
    var matches = false;

    while (index < pattern.len) {
        if (pattern[index] == ']' and has_member) {
            if (matches == is_negated) {
                return null;
            }

            // The caller resumes immediately after the matching class.
            return index + 1;
        }

        if (pattern[index] == '[' and index + 1 < pattern.len and pattern[index + 1] == ':') {
            const class_end_index = std.mem.indexOfPos(u8, pattern, index + 2, ":]") orelse {
                return null;
            };

            const class_matches = posixCharacterClassMatches(
                pattern[index + 2 .. class_end_index],
                candidate,
            ) orelse {
                return null;
            };

            has_member = true;
            matches = matches or class_matches;
            index = class_end_index + 2;

            continue;
        }

        const range_min = consumePatternLiteral(pattern, &index);

        has_member = true;

        if (index + 1 < pattern.len and pattern[index] == '-' and pattern[index + 1] != ']') {
            index += 1;

            const range_max = consumePatternLiteral(pattern, &index);

            matches = matches or (range_min <= candidate and candidate <= range_max);
        } else {
            matches = matches or candidate == range_min;
        }
    }

    // Reaching the pattern end before a closing bracket makes this rule
    // nonmatching rather than rejecting the complete ignore file.
    return null;
}

fn consumePatternLiteral(pattern: []const u8, pattern_index: *usize) u8 {
    assert(pattern_index.* < pattern.len);

    if (pattern[pattern_index.*] == '\\' and pattern_index.* + 1 < pattern.len) {
        pattern_index.* += 1;
    }

    const literal = pattern[pattern_index.*];

    pattern_index.* += 1;

    return literal;
}

fn posixCharacterClassMatches(name: []const u8, candidate: u8) ?bool {
    const Class = enum {
        alnum,
        alpha,
        blank,
        cntrl,
        digit,
        graph,
        lower,
        print,
        punct,
        space,
        upper,
        xdigit,
    };

    const class_by_name = std.StaticStringMap(Class).initComptime(.{
        .{ "alnum", .alnum },
        .{ "alpha", .alpha },
        .{ "blank", .blank },
        .{ "cntrl", .cntrl },
        .{ "digit", .digit },
        .{ "graph", .graph },
        .{ "lower", .lower },
        .{ "print", .print },
        .{ "punct", .punct },
        .{ "space", .space },
        .{ "upper", .upper },
        .{ "xdigit", .xdigit },
    });

    const class = class_by_name.get(name) orelse {
        return null;
    };

    const is_lower = candidate >= 'a' and candidate <= 'z';
    const is_upper = candidate >= 'A' and candidate <= 'Z';
    const is_alpha = is_lower or is_upper;
    const is_digit = candidate >= '0' and candidate <= '9';
    const is_alphanumeric = is_alpha or is_digit;
    const is_graphical = candidate >= 0x21 and candidate <= 0x7e;

    return switch (class) {
        .alnum => is_alphanumeric,
        .alpha => is_alpha,
        .blank => candidate == ' ' or candidate == '\t',
        .cntrl => candidate <= 0x1f or candidate == 0x7f,
        .digit => is_digit,
        .graph => is_graphical,
        .lower => is_lower,
        .print => candidate >= 0x20 and candidate <= 0x7e,
        .punct => is_graphical and !is_alphanumeric,
        .space => std.ascii.isWhitespace(candidate),
        .upper => is_upper,
        .xdigit => is_digit or
            (candidate >= 'a' and candidate <= 'f') or
            (candidate >= 'A' and candidate <= 'F'),
    };
}

// ─── Gitignore Syntax ────────────────────────────────────────────────────────

test "gitignore lines handle empty text, comments, escapes, spaces, and malformed globs" {
    const cases = [_]TestCase{
        // ── Empty Lines And Comments ──

        // No rule decides, so the entry remains included.
        patternIncluded(
            .{
                .name = "empty line adds no rule",
                .pattern = "",
                .path = "anything",
            },
        ),

        // Unescaped trailing spaces are removed before parsing.
        patternIncluded(
            .{
                .name = "spaces become an empty line",
                .pattern = "   ",
                .path = "anything",
            },
        ),

        // A comment contributes no matching rule.
        patternIncluded(
            .{
                .name = "line beginning with hash is a comment",
                .pattern = "# generated files",
                .path = "generated files",
            },
        ),

        // Splitting on newline leaves a removable carriage return.
        patternIncluded(
            .{
                .name = "carriage return from an empty CRLF line is removed",
                .pattern = "\r",
                .path = "anything",
            },
        ),

        // ── Escaped Prefix Characters ──

        // Escaping prevents the hash from starting a comment.
        patternIgnored(
            .{
                .name = "escaped hash is a literal basename",
                .pattern = "\\#notes",
                .path = "docs/#notes",
            },
        ),

        // Escaping prevents the exclamation mark from negating.
        patternIgnored(
            .{
                .name = "escaped exclamation mark is a literal basename",
                .pattern = "\\!important",
                .path = "docs/!important",
            },
        ),

        // Gitignore has no trailing inline-comment syntax.
        patternIgnored(
            .{
                .name = "hash after the first byte remains literal",
                .pattern = "name#part",
                .path = "name#part",
            },
        ),

        // ── Trailing Whitespace And CRLF ──

        // The normalized pattern is `name`.
        patternIgnored(
            .{
                .name = "unescaped trailing spaces are removed",
                .pattern = "name   ",
                .path = "name",
            },
        ),

        // The backslash protects the final space.
        patternIgnored(
            .{
                .name = "escaped trailing space is retained",
                .pattern = "name\\ ",
                .path = "name ",
            },
        ),

        // The normalized pattern excludes the CR terminator.
        patternIgnored(
            .{
                .name = "carriage return is removed from a CRLF rule",
                .pattern = "name\r",
                .path = "name",
            },
        ),

        // ── Permissive Malformed Syntax ──

        // Git leaves malformed class patterns nonmatching.
        patternIncluded(
            .{
                .name = "unclosed character class does not match literal text",
                .pattern = "file[.zig",
                .path = "file[.zig",
            },
        ),

        // Malformed syntax does not reject the remaining file.
        patternIncluded(
            .{
                .name = "unclosed character class does not act as a glob",
                .pattern = "file[.zig",
                .path = "filex.zig",
            },
        ),

        // Git leaves a terminal unpaired escape nonmatching.
        patternIncluded(
            .{
                .name = "dangling escape does not match a literal backslash",
                .pattern = "file\\",
                .path = "file\\",
            },
        ),
    };

    for (cases) |case| {
        try runCase(case);
    }
}

// ─── Literal Paths And Directory Targets ─────────────────────────────────────

test "gitignore literals distinguish basenames, anchored paths, files, and directories" {
    const cases = [_]TestCase{
        // ── Unanchored Basenames ──

        // A pattern without an internal slash matches a basename.
        patternIgnored(
            .{
                .name = "basename matches at the root",
                .pattern = ".venv/",
                .path = ".venv",
                .path_is_dir = true,
            },
        ),

        // An unanchored basename is not limited to the root.
        patternIgnored(
            .{
                .name = "basename matches at any descendant depth",
                .pattern = ".venv/",
                .path = "tools/.venv",
                .path_is_dir = true,
            },
        ),

        // The trailing slash restricts the rule to directories.
        patternIncluded(
            .{
                .name = "directory-only basename rejects a regular file",
                .pattern = ".venv/",
                .path = ".venv",
            },
        ),

        // A target-neutral basename can match a regular file.
        patternIgnored(
            .{
                .name = "literal without trailing slash matches a file",
                .pattern = "NOTICE",
                .path = "docs/NOTICE",
            },
        ),

        // A target-neutral basename can also match a directory.
        patternIgnored(
            .{
                .name = "literal without trailing slash matches a directory",
                .pattern = "NOTICE",
                .path = "docs/NOTICE",
                .path_is_dir = true,
            },
        ),

        // ── Root-Anchored Paths ──

        // The candidate occupies the anchored root-relative path.
        patternIgnored(
            .{
                .name = "leading slash matches the root entry",
                .pattern = "/.zig-cache/",
                .path = ".zig-cache",
                .path_is_dir = true,
            },
        ),

        // The leading slash anchors the match to the `.gitignore` directory.
        patternIncluded(
            .{
                .name = "leading slash rejects the same nested basename",
                .pattern = "/.zig-cache/",
                .path = "src/.zig-cache",
                .path_is_dir = true,
            },
        ),

        // A slash makes the complete rule-relative path significant.
        patternIgnored(
            .{
                .name = "internal slash matches one exact relative path",
                .pattern = "docs/output.zig",
                .path = "docs/output.zig",
            },
        ),

        // Git accepts an escaped slash without making it literal data.
        patternIgnored(
            .{
                .name = "escaped separator remains a path separator",
                .pattern = "docs\\/output.zig",
                .path = "docs/output.zig",
            },
        ),

        // Path rules do not search for matching suffixes.
        patternIncluded(
            .{
                .name = "internal slash rejects an additional prefix",
                .pattern = "docs/output.zig",
                .path = "archive/docs/output.zig",
            },
        ),

        // Literal path rules match the complete relative path.
        patternIncluded(
            .{
                .name = "internal slash rejects an additional suffix",
                .pattern = "docs/output.zig",
                .path = "docs/output.zig/child",
            },
        ),

        // ── Nested Gitignore Directories ──

        // The rule-relative candidate is exactly `generated`.
        patternIgnored(
            .{
                .name = "nested leading slash anchors to its own scope",
                .gitignore_dir_path = "src",
                .pattern = "/generated/",
                .path = "src/generated",
                .path_is_dir = true,
            },
        ),

        // `lib/generated` is not anchored at the scope root.
        patternIncluded(
            .{
                .name = "nested anchor rejects a deeper directory",
                .gitignore_dir_path = "src",
                .pattern = "/generated/",
                .path = "src/lib/generated",
                .path_is_dir = true,
            },
        ),

        // The `.gitignore` directory prefix is removed before matching.
        patternIgnored(
            .{
                .name = "nested relative path matches beneath its Gitignore directory",
                .gitignore_dir_path = "src",
                .pattern = "generated/output.zig",
                .path = "src/generated/output.zig",
            },
        ),

        // The candidate is outside the `.gitignore` directory tree.
        patternIncluded(
            .{
                .name = "nested rule cannot match a sibling tree",
                .gitignore_dir_path = "src",
                .pattern = "generated/output.zig",
                .path = "tools/generated/output.zig",
            },
        ),
    };

    for (cases) |case| {
        try runCase(case);
    }
}

// ─── Single-Segment Globs ────────────────────────────────────────────────────

test "gitignore globs keep star, question mark, and character classes inside one segment" {
    const cases = [_]TestCase{
        // ── Star ──

        // Basename matching gives `*` the `main` prefix.
        patternIgnored(
            .{
                .name = "star matches a nonempty prefix",
                .pattern = "*.zig",
                .path = "src/main.zig",
            },
        ),

        // `*` may consume zero bytes.
        patternIgnored(
            .{
                .name = "star matches an empty prefix",
                .pattern = "*.zig",
                .path = ".zig",
            },
        ),

        // Gitignore globs do not apply shell dotfile suppression.
        patternIgnored(
            .{
                .name = "star matches a leading dot",
                .pattern = "*",
                .path = ".hidden",
            },
        ),

        // The complete basename must end in `.zig`.
        patternIncluded(
            .{
                .name = "star does not discard a required suffix",
                .pattern = "*.zig",
                .path = "main.zig.txt",
            },
        ),

        // `*` may consume the empty region between literals.
        patternIgnored(
            .{
                .name = "middle star matches no bytes",
                .pattern = "foo*bar",
                .path = "foobar",
            },
        ),

        // `*` consumes `-a-b-` within the basename.
        patternIgnored(
            .{
                .name = "middle star matches several bytes",
                .pattern = "foo*bar",
                .path = "foo-a-b-bar",
            },
        ),

        // `*` consumes bytes after the one required slash.
        patternIgnored(
            .{
                .name = "path star matches one direct segment",
                .pattern = "src/*.zig",
                .path = "src/main.zig",
            },
        ),

        // Ordinary `*` cannot consume the `lib/` segment.
        patternIncluded(
            .{
                .name = "path star cannot cross a slash",
                .pattern = "src/*.zig",
                .path = "src/lib/main.zig",
            },
        ),

        // ── Question Mark ──

        // `?` consumes exactly `1`.
        patternIgnored(
            .{
                .name = "question mark matches one byte",
                .pattern = "test?.zig",
                .path = "test1.zig",
            },
        ),

        // No byte is available for `?`.
        patternIncluded(
            .{
                .name = "question mark does not match zero bytes",
                .pattern = "test?.zig",
                .path = "test.zig",
            },
        ),

        // One `?` cannot consume both `1` and `2`.
        patternIncluded(
            .{
                .name = "question mark does not match two bytes",
                .pattern = "test?.zig",
                .path = "test12.zig",
            },
        ),

        // ── Character Classes ──

        // The escape prevents `[` from opening a class.
        patternIgnored(
            .{
                .name = "escaped opening bracket is literal text",
                .pattern = "file\\[.zig",
                .path = "file[.zig",
            },
        ),

        // Git character classes use their ASCII definitions.
        patternIgnored(
            .{
                .name = "POSIX character class accepts a member",
                .pattern = "file[[:digit:]].zig",
                .path = "file7.zig",
            },
        ),

        // `x` is not an ASCII digit.
        patternIncluded(
            .{
                .name = "POSIX character class rejects a nonmember",
                .pattern = "file[[:digit:]].zig",
                .path = "filex.zig",
            },
        ),

        // `7` lies in the inclusive `0-9` range.
        patternIgnored(
            .{
                .name = "character range accepts a member",
                .pattern = "file[0-9].zig",
                .path = "file7.zig",
            },
        ),

        // `x` lies outside the numeric range.
        patternIncluded(
            .{
                .name = "character range rejects a nonmember",
                .pattern = "file[0-9].zig",
                .path = "filex.zig",
            },
        ),

        // `b` is one of the listed class members.
        patternIgnored(
            .{
                .name = "character set accepts a listed byte",
                .pattern = "file[abc].zig",
                .path = "fileb.zig",
            },
        ),

        // `x` satisfies the negated numeric class.
        patternIgnored(
            .{
                .name = "negated class accepts an unlisted byte",
                .pattern = "file[!0-9].zig",
                .path = "filex.zig",
            },
        ),

        // `7` is excluded by the class negation.
        patternIncluded(
            .{
                .name = "negated class rejects a listed byte",
                .pattern = "file[!0-9].zig",
                .path = "file7.zig",
            },
        ),

        // ── Escaped Metacharacters ──

        // The backslash prevents wildcard interpretation.
        patternIgnored(
            .{
                .name = "escaped star is literal",
                .pattern = "literal\\*.zig",
                .path = "literal*.zig",
            },
        ),

        // The candidate has no literal `*` byte.
        patternIncluded(
            .{
                .name = "escaped star rejects substituted text",
                .pattern = "literal\\*.zig",
                .path = "literal-any.zig",
            },
        ),

        // The backslash prevents single-byte wildcard matching.
        patternIgnored(
            .{
                .name = "escaped question mark is literal",
                .pattern = "why\\?",
                .path = "why?",
            },
        ),
    };

    for (cases) |case| {
        try runCase(case);
    }
}

// ─── Cross-Segment Globs ─────────────────────────────────────────────────────

test "gitignore double-star globs consume zero or more complete path segments" {
    const cases = [_]TestCase{
        // ── Middle Double Star ──

        // `**/` may consume no intermediate segment.
        patternIgnored(
            .{
                .name = "middle double star consumes zero directories",
                .pattern = "src/**/generated.zig",
                .path = "src/generated.zig",
            },
        ),

        // `**/` consumes `a/`.
        patternIgnored(
            .{
                .name = "middle double star consumes one directory",
                .pattern = "src/**/generated.zig",
                .path = "src/a/generated.zig",
            },
        ),

        // `**/` consumes `a/b/`.
        patternIgnored(
            .{
                .name = "middle double star consumes several directories",
                .pattern = "src/**/generated.zig",
                .path = "src/a/b/generated.zig",
            },
        ),

        // Consuming directories cannot replace `generated.zig`.
        patternIncluded(
            .{
                .name = "middle double star preserves its following literal",
                .pattern = "src/**/generated.zig",
                .path = "src/a/other.zig",
            },
        ),

        // Git gives a complete run of stars double-star semantics.
        patternIgnored(
            .{
                .name = "longer all-star segment is recursive",
                .pattern = "src/***/generated.zig",
                .path = "src/a/b/generated.zig",
            },
        ),

        // ── Leading Double Star ──

        // `**/` consumes zero leading segments.
        patternIgnored(
            .{
                .name = "leading double star matches a root directory",
                .pattern = "**/cache/",
                .path = "cache",
                .path_is_dir = true,
            },
        ),

        // `**/` consumes `a/b/`.
        patternIgnored(
            .{
                .name = "leading double star matches a nested directory",
                .pattern = "**/cache/",
                .path = "a/b/cache",
                .path_is_dir = true,
            },
        ),

        // The basename matches, but the candidate is a file.
        patternIncluded(
            .{
                .name = "leading double star retains directory-only targeting",
                .pattern = "**/cache/",
                .path = "a/cache",
            },
        ),

        // ── Trailing Double Star ──

        // The suffix consumes the child path.
        patternIgnored(
            .{
                .name = "trailing double star matches a direct child",
                .pattern = "assets/**",
                .path = "assets/logo.svg",
            },
        ),

        // The suffix crosses every remaining slash.
        patternIgnored(
            .{
                .name = "trailing double star matches a deep child",
                .pattern = "assets/**",
                .path = "assets/icons/logo.svg",
            },
        ),

        // The pattern addresses entries inside `assets`.
        patternIncluded(
            .{
                .name = "trailing double star does not match its parent directory",
                .pattern = "assets/**",
                .path = "assets",
                .path_is_dir = true,
            },
        ),

        // ── Consecutive Stars Inside A Segment ──

        // Only a complete `**` path segment crosses slashes.
        patternIgnored(
            .{
                .name = "double star inside a segment behaves as ordinary star",
                .pattern = "ab**cd",
                .path = "abXYZcd",
            },
        ),

        // `**b` is an ordinary single-segment glob.
        patternIncluded(
            .{
                .name = "partial double star cannot cross a slash",
                .pattern = "a/**b/c",
                .path = "a/x/b/c",
            },
        ),
    };

    for (cases) |case| {
        try runCase(case);
    }
}

// ─── Ordering, Sources, And Scope Lifetime ───────────────────────────────────

test "gitignore decisions apply file order, nested scopes, and pruning" {
    const cases = [_]TestCase{
        // ── Default And File Order ──

        // Inclusion is the default when no rule matches.
        .{
            .name = "no matching rule includes",
            .pattern = "",
            .path = "src/main.zig",
            .path_is_ignored = false,
        },

        // The only matching rule has ignore action.
        .{
            .name = "one matching repository rule ignores",
            .pattern = "*.gen.zig\n",
            .path = "src/a.gen.zig",
            .path_is_ignored = true,
        },

        // The later matching negation wins.
        .{
            .name = "later negation includes",
            .pattern = "*.gen.zig\n!keep.gen.zig\n",
            .path = "src/keep.gen.zig",
            .path_is_ignored = false,
        },

        // The final matching rule has ignore action.
        .{
            .name = "later ignore overrides an earlier negation",
            .pattern = "!keep.gen.zig\n*.gen.zig\n",
            .path = "src/keep.gen.zig",
            .path_is_ignored = true,
        },

        // The negation does not match this basename.
        .{
            .name = "nonmatching negation leaves earlier ignore active",
            .pattern = "*.gen.zig\n!other.gen.zig\n",
            .path = "src/keep.gen.zig",
            .path_is_ignored = true,
        },

        // ── Nested Repository Scopes ──

        // A deeper matching rule occurs later than parent rules.
        .{
            .name = "child scope overrides its parent",
            .root_pattern = "*.gen.zig\n",
            .gitignore_dir_path = "src",
            .pattern = "!keep.gen.zig\n",
            .path = "src/keep.gen.zig",
            .path_is_ignored = false,
        },

        // Only the inherited parent rule matches.
        .{
            .name = "child nonmatch falls back to its parent",
            .root_pattern = "*.gen.zig\n",
            .gitignore_dir_path = "src",
            .pattern = "!keep.gen.zig\n",
            .path = "src/other.gen.zig",
            .path_is_ignored = true,
        },

        // The child rule's `src` directory excludes this sibling path.
        .{
            .name = "child scope cannot affect a sibling tree",
            .root_pattern = "*.gen.zig\n",
            .gitignore_dir_path = "src",
            .pattern = "!keep.gen.zig\n",
            .path = "tools/keep.gen.zig",
            .path_is_ignored = true,
        },

        // ── Directory Pruning And Re-Inclusion ──

        // Traversal stops before the child negation is considered.
        .{
            .name = "ignored parent directory establishes a prune boundary",
            .pattern = "build/\n!build/keep.zig\n",
            .path = "build",
            .path_is_dir = true,
            .path_is_ignored = true,
        },

        // `build` remains traversable, so the child negation wins.
        .{
            .name = "ignored children permit one child to be re-included",
            .pattern = "build/*\n!build/keep.zig\n",
            .path = "build/keep.zig",
            .path_is_ignored = false,
        },
    };

    for (cases) |case| {
        try runCase(case);
    }
}

test "gitignore pop restores the parent before entering a sibling" {
    var gitignore: Gitignore = .{};
    defer gitignore.deinit(std.testing.allocator);

    // ── Root And Child Active ──

    try gitignore.push(std.testing.allocator, "", "*.gen.zig\n");

    const root_source_size = gitignore.source_bytes.items.len;

    try gitignore.push(std.testing.allocator, "src", "!keep.gen.zig\n");

    try std.testing.expect(gitignore.source_bytes.items.len > root_source_size);
    try std.testing.expect(!gitignore.pathIsIgnored("src/keep.gen.zig", false));

    // ── Child Removed, Parent Retained ──

    gitignore.pop();

    try std.testing.expectEqual(root_source_size, gitignore.source_bytes.items.len);
    try std.testing.expect(gitignore.pathIsIgnored("src/keep.gen.zig", false));

    // ── Sibling Gets Its Own Rules, Not The Removed Child Rules ──

    try gitignore.push(std.testing.allocator, "tools", "!tool.gen.zig\n");

    try std.testing.expect(!gitignore.pathIsIgnored("tools/tool.gen.zig", false));
    try std.testing.expect(gitignore.pathIsIgnored("tools/keep.gen.zig", false));
    gitignore.pop();

    try std.testing.expectEqual(root_source_size, gitignore.source_bytes.items.len);

    // ── All Repository Rules Removed ──

    gitignore.pop();

    try std.testing.expectEqual(@as(usize, 0), gitignore.source_bytes.items.len);
    try std.testing.expect(!gitignore.pathIsIgnored("src/keep.gen.zig", false));
}

test "gitignore releases partially constructed state after every allocation failure" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        struct {
            fn exercise(allocator: Allocator) !void {
                var gitignore: Gitignore = .{};
                defer gitignore.deinit(allocator);

                try gitignore.push(allocator, "", "*.tmp\ncache/\n");
                defer gitignore.pop();

                try gitignore.push(allocator, "src", "!keep.tmp\ngenerated/**\n");
                defer gitignore.pop();

                try std.testing.expect(!gitignore.pathIsIgnored("src/keep.tmp", false));
            }
        }.exercise,
        .{},
    );
}

// ─── Test Support ────────────────────────────────────────────────────────────

const TestCase = struct {
    name: []const u8,
    root_pattern: []const u8 = "",
    gitignore_dir_path: []const u8 = "",
    pattern: []const u8,
    path: []const u8,
    path_is_dir: bool = false,
    path_is_ignored: bool = false,
};

fn patternIgnored(case: TestCase) TestCase {
    var ignored_case = case;

    ignored_case.path_is_ignored = true;

    return ignored_case;
}

fn patternIncluded(case: TestCase) TestCase {
    return case;
}

fn runCase(case: TestCase) !void {
    errdefer std.debug.print("test case failed: {s}\n", .{case.name});

    var gitignore: Gitignore = .{};
    defer gitignore.deinit(std.testing.allocator);

    try gitignore.push(std.testing.allocator, "", case.root_pattern);
    defer gitignore.pop();

    try gitignore.push(std.testing.allocator, case.gitignore_dir_path, case.pattern);
    defer gitignore.pop();

    try std.testing.expectEqual(
        case.path_is_ignored,
        gitignore.pathIsIgnored(case.path, case.path_is_dir),
    );
}
