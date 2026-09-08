const std = @import("std");

pub const Linter = @import("Linter.zig");
pub const Diagnostics = @import("Diagnostics.zig");
pub const Diagnostic = Diagnostics.Diagnostic;
pub const File = Linter.File;

// ─── Tests ───────────────────────────────────────────────────────────

test "hidden control flow" {
    try expect_lint_snapshot(
        "hidden_control_flow.zig",
        @embedFile("__fixtures__/hidden_control_flow.zig"),
        @embedFile("__fixtures__/hidden_control_flow.expected"),
    );
}

test "explicit optional unwrap" {
    try expect_lint_snapshot(
        "explicit_optional_unwrap.zig",
        @embedFile("__fixtures__/explicit_optional_unwrap.zig"),
        @embedFile("__fixtures__/explicit_optional_unwrap.expected"),
    );
}

test "blank line before control flow exit" {
    try expect_lint_snapshot(
        "blank_line_before_control_flow_exit.zig",
        @embedFile("__fixtures__/blank_line_before_control_flow_exit.zig"),
        @embedFile("__fixtures__/blank_line_before_control_flow_exit.expected"),
    );
}

test "visible resource lifetime" {
    try expect_lint_snapshot(
        "visible_resource_lifetime.zig",
        @embedFile("__fixtures__/visible_resource_lifetime.zig"),
        @embedFile("__fixtures__/visible_resource_lifetime.expected"),
    );
}

test "file size limit is inclusive and checked before parsing" {
    const gpa = std.testing.allocator;

    const default_limit = (Linter.LintDirectoryOptions{}).file_size_max;

    for ([_]u32{ 0, 64, default_limit, default_limit + 32 }) |file_size_max| {
        const source = try gpa.allocSentinel(u8, file_size_max + 1, 0);
        defer gpa.free(source);

        @memset(source, ' ');
        var linter: Linter = .empty;
        try std.testing.expectError(error.FileTooLarge, linter.lint_file(
            std.testing.failing_allocator,
            std.testing.failing_allocator,
            .{ .path = "oversized.zig", .text = source },
            file_size_max,
        ));
        try std.testing.expectEqual(@as(usize, 0), linter.diagnostics.count());

        source[file_size_max] = 0;
        try linter.lint_file(
            gpa,
            std.testing.failing_allocator,
            .{ .path = "at_limit.zig", .text = source[0..file_size_max :0] },
            file_size_max,
        );
        try std.testing.expectEqual(@as(usize, 0), linter.diagnostics.count());
    }
}

test "SIMD line index matches scalar offsets and AST locations" {
    const FileContext = @import("FileContext.zig");
    const gpa = std.testing.allocator;

    for (0..130) |length| {
        const source = try gpa.allocSentinel(u8, length, 0);
        defer gpa.free(source);

        // Shift CRLF and identifier tokens through every SIMD lane and tail
        // length. Include a dense-newline pass to exercise every mask bit.
        for (0..8) |pattern| {
            var expected: [131]u32 = undefined;
            expected[0] = 0;
            var count: usize = 1;

            for (source, 0..) |*byte, offset| {
                byte.* = if (pattern == 7 or (offset + pattern) % 7 == 0)
                    '\n'
                else if ((offset + pattern) % 7 == 6)
                    '\r'
                else
                    'x';

                if (byte.* == '\n') {
                    expected[count] = @intCast(offset + 1);
                    count += 1;
                }
            }

            var ast = try std.zig.Ast.parse(gpa, source, .zig);
            defer ast.deinit(gpa);

            const file = try FileContext.init(gpa, ast, @intCast(length));
            defer gpa.free(file.line_starts);

            try std.testing.expectEqualSlices(u32, expected[0..count], file.line_starts);

            for (0..ast.tokens.len) |index| {
                const token: std.zig.Ast.TokenIndex = @intCast(index);
                try std.testing.expectEqualDeep(ast.tokenLocation(0, token), file.tokenLocation(token));
            }
        }
    }
}

test "directory clean cache, bypass, identity, and metadata invalidation" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;

    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();

    try temporary.dir.createDirPath(io, "source");

    const root = try temporary.dir.realPathFileAlloc(io, "source", gpa);
    defer gpa.free(root);

    const cache_base = try std.fmt.allocPrint(gpa, "{s}/../cache", .{root});
    defer gpa.free(cache_base);

    const base = try std.fs.path.join(gpa, &.{ cache_base, "v1" });
    defer gpa.free(base);

    const options: Linter.LintDirectoryOptions = .{ .cache = .{
        .base_dir = base,
    } };
    const clean = "const value = 1;\n";
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = clean });
    try expect_directory_count(root, options, 0);

    // Deliberately preserve all three metadata fields while replacing valid
    // source with invalid bytes. A cache hit skips parsing; an uncached scan
    // still reports the syntax error. This also documents the chosen heuristic.
    const before = try temporary.dir.statFile(io, "source/a.zig", .{});
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = "?" ** clean.len });
    {
        var file = try temporary.dir.openFile(io, "source/a.zig", .{ .mode = .read_write });
        defer file.close(io);

        try file.setTimestamps(io, .{ .modify_timestamp = .{ .new = before.mtime } });
    }
    const unchanged = try temporary.dir.statFile(io, "source/a.zig", .{});
    try std.testing.expectEqual(before.inode, unchanged.inode);
    try std.testing.expectEqual(before.mtime.nanoseconds, unchanged.mtime.nanoseconds);
    try expect_directory_count(root, options, 0);
    try expect_directory_count(root, .{}, 1);

    const next_version_base = try std.fs.path.join(gpa, &.{ cache_base, "v2" });
    defer gpa.free(next_version_base);

    try expect_directory_count(root, .{ .cache = .{ .base_dir = next_version_base } }, 1);

    // The same target metadata beneath another root cannot borrow v1's hit.
    try temporary.dir.createDirPath(io, "other");
    try temporary.dir.symLink(io, "../source/a.zig", "other/a.zig", .{});

    const other_root = try temporary.dir.realPathFileAlloc(io, "other", gpa);
    defer gpa.free(other_root);

    try expect_directory_count(other_root, options, 1);

    // A size change invalidates v1 even when the inode stays unchanged.
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = "?" });
    try expect_directory_count(root, options, 1);

    // Dirty results are never retained; fixing the file repopulates its entry.
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = clean });
    try expect_directory_count(root, options, 0);
    try temporary.dir.deleteFile(io, "source/a.zig");
    try expect_directory_count(root, options, 0);

    // All three indexes now contain zero clean entries: deleted paths are
    // removed, not carried forward indefinitely from an earlier walk.
    {
        var cache_dir = try std.Io.Dir.cwd().openDir(io, cache_base, .{ .iterate = true });
        defer cache_dir.close(io);

        var walker = try cache_dir.walk(gpa);
        defer walker.deinit();

        while (true) {
            const entry = try walker.next(io) orelse {
                break;
            };

            if (entry.kind != .file) {
                continue;
            }

            var header: [16]u8 = undefined;
            const bytes = try entry.dir.readFile(io, entry.basename, &header);
            try std.testing.expectEqual(@as(usize, 16), bytes.len);
            try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, header[12..16], .little));
        }
    }

    // Cache I/O failure is best-effort: a regular file cannot be a base dir.
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = "?" });

    const blocked_base = try std.fs.path.join(gpa, &.{ root, "a.zig" });
    defer gpa.free(blocked_base);

    try expect_directory_count(root, .{ .cache = .{ .base_dir = blocked_base } }, 1);

    // Lowering the limit drops only the oversized entry. Preserve invalid
    // bytes behind a fitting entry's metadata to prove it remains a cache hit.
    const higher_limit: Linter.LintDirectoryOptions = .{
        .cache = options.cache,
        .file_size_max = clean.len + 1,
    };
    const lower_limit: Linter.LintDirectoryOptions = .{
        .cache = options.cache,
        .file_size_max = clean.len,
    };

    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = clean });
    try temporary.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = clean ++ "\n" });
    try expect_directory_count(root, higher_limit, 0);

    const fitting_metadata = try temporary.dir.statFile(io, "source/a.zig", .{});
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = "?" ** clean.len });
    {
        var file = try temporary.dir.openFile(io, "source/a.zig", .{ .mode = .read_write });
        defer file.close(io);

        try file.setTimestamps(io, .{ .modify_timestamp = .{ .new = fitting_metadata.mtime } });
    }

    {
        var diagnostics = std.heap.ArenaAllocator.init(gpa);
        defer diagnostics.deinit();

        var linter: Linter = .empty;
        try linter.lintDirectory(gpa, diagnostics.allocator(), io, root, lower_limit);
        try std.testing.expectEqual(@as(usize, 1), linter.diagnostics.count());
        try std.testing.expectEqualStrings("b.zig", linter.diagnostics.items.items[0].path);
        try std.testing.expectEqualStrings("file is larger than 17 bytes; split it.", linter.diagnostics.items.items[0].message);
    }

    // Both scanned roots have blobs beneath this base. Select this root's
    // index explicitly rather than depending on directory iteration order.
    var root_digest: [std.crypto.hash.Blake3.digest_length]u8 = undefined;
    std.crypto.hash.Blake3.hash(root, &root_digest, .{});

    const cache_name = try std.fmt.allocPrint(gpa, "{s}.bin", .{std.fmt.bytesToHex(root_digest, .lower)});
    defer gpa.free(cache_name);

    // The saved blob retains a.zig but no longer contains b.zig.
    {
        var cache_dir = try std.Io.Dir.cwd().openDir(io, base, .{});
        defer cache_dir.close(io);

        var buffer: [256]u8 = undefined;
        const encoded = try cache_dir.readFile(io, cache_name, &buffer);
        try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, encoded[12..16], .little));
        try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, encoded[8..12], .little));
        try std.testing.expectEqualStrings("a.zig", encoded[48..]);
    }

    // Raising the limit admits b.zig again. If it then shrinks below the old
    // limit, its stale oversized record must not prevent a normal clean scan.
    try expect_directory_count(root, higher_limit, 0);
    try temporary.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = clean });
    try expect_directory_count(root, lower_limit, 0);

    // Even the largest encoded u32 size invalidates only that record. The
    // fitting a.zig entry must remain a hit while b.zig is checked and replaced.
    {
        var cache_dir = try std.Io.Dir.cwd().openDir(io, base, .{});
        defer cache_dir.close(io);

        const encoded = try cache_dir.readFileAlloc(io, cache_name, gpa, .limited(4096));
        defer gpa.free(encoded);

        var offset: usize = 16;
        var replaced = false;

        while (offset < encoded.len) {
            const fields = encoded[offset..][0..32];
            const path_size = std.mem.readInt(u32, fields[0..4], .little);
            const path = encoded[offset + 32 ..][0..path_size];

            if (std.mem.eql(u8, path, "b.zig")) {
                std.mem.writeInt(u32, fields[12..16], std.math.maxInt(u32), .little);
                replaced = true;
            }

            offset += 32 + path_size;
        }

        try std.testing.expect(replaced);
        try cache_dir.writeFile(io, .{ .sub_path = cache_name, .data = encoded });
        try expect_directory_count(root, lower_limit, 0);
    }

    // A zero-byte directory limit accepts empty input, including cache hits,
    // but still reports nonempty files rather than treating zero as unlimited.
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = "" });
    try temporary.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = "" });
    const empty_only: Linter.LintDirectoryOptions = .{ .cache = options.cache, .file_size_max = 0 };
    try expect_directory_count(root, empty_only, 0);
    try expect_directory_count(root, empty_only, 0);
    try temporary.dir.writeFile(io, .{ .sub_path = "source/b.zig", .data = " " });
    try expect_directory_count(root, empty_only, 1);
}

test "malformed cache records never become partial cache hits" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;

    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();

    try temporary.dir.createDirPath(io, "source");

    const root = try temporary.dir.realPathFileAlloc(io, "source", gpa);
    defer gpa.free(root);

    const base = try std.fmt.allocPrint(gpa, "{s}/../cache", .{root});
    defer gpa.free(base);

    const options: Linter.LintDirectoryOptions = .{ .cache = .{ .base_dir = base } };
    const clean = "const value = 1;\n";
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = clean });
    try expect_directory_count(root, options, 0);

    var cache_dir = try std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true });
    defer cache_dir.close(io);

    var walker = try cache_dir.walk(gpa);
    defer walker.deinit();

    const cache_path = path: {
        while (true) {
            const entry = try walker.next(io) orelse {
                break;
            };

            if (entry.kind == .file) {
                break :path try gpa.dupe(u8, entry.path);
            }
        }

        return error.MissingCacheFile;
    };
    defer gpa.free(cache_path);

    const encoded = try cache_dir.readFileAlloc(io, cache_path, gpa, .limited(4096));
    defer gpa.free(encoded);

    const before = try temporary.dir.statFile(io, "source/a.zig", .{});
    try temporary.dir.writeFile(io, .{ .sub_path = "source/a.zig", .data = "?" ** clean.len });
    {
        var file = try temporary.dir.openFile(io, "source/a.zig", .{ .mode = .read_write });
        defer file.close(io);

        try file.setTimestamps(io, .{ .modify_timestamp = .{ .new = before.mtime } });
    }

    // Corrupt magic/version/count/length, truncate fields, append junk, or
    // duplicate a record. Even corruption after a valid entry rejects all hits.
    for (0..8) |case| {
        var malformed: std.ArrayList(u8) = .empty;
        defer malformed.deinit(gpa);

        try malformed.appendSlice(gpa, encoded);
        switch (case) {
            0 => malformed.items[0] ^= 1,
            1 => std.mem.writeInt(u32, malformed.items[8..12], 99, .little),
            2 => std.mem.writeInt(u32, malformed.items[12..16], std.math.maxInt(u32), .little),
            3 => std.mem.writeInt(u32, malformed.items[16..20], std.math.maxInt(u32), .little),
            4 => malformed.shrinkRetainingCapacity(17),
            5 => try malformed.append(gpa, 0),
            6 => {
                try malformed.appendSlice(gpa, encoded[16..]);
                std.mem.writeInt(u32, malformed.items[12..16], 2, .little);
            },
            7 => {
                // A valid version 1 record used a u64 size. It must rebuild,
                // not be decoded using version 2's narrower field offsets.
                std.mem.writeInt(u32, malformed.items[8..12], 1, .little);
                try malformed.insertSlice(gpa, 32, &.{ 0, 0, 0, 0 });
            },
            else => unreachable,
        }
        try cache_dir.writeFile(io, .{ .sub_path = cache_path, .data = malformed.items });
        try expect_directory_count(root, options, 1);
    }
}

fn expect_directory_count(path: []const u8, options: Linter.LintDirectoryOptions, expected: usize) !void {
    var diagnostics = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer diagnostics.deinit();

    var linter: Linter = .empty;
    try linter.lintDirectory(std.testing.allocator, diagnostics.allocator(), std.testing.io, path, options);
    try std.testing.expectEqual(expected, linter.diagnostics.count());
}

fn expect_lint_snapshot(path: []const u8, source: [:0]const u8, expected: []const u8) !void {
    const gpa = std.testing.allocator;

    var file_arena = std.heap.ArenaAllocator.init(gpa);
    defer file_arena.deinit();

    var diagnostic_arena = std.heap.ArenaAllocator.init(gpa);
    defer diagnostic_arena.deinit();

    var linter: Linter = .empty;

    try linter.lint_file(
        file_arena.allocator(),
        diagnostic_arena.allocator(),
        .{ .path = path, .text = source },
        (Linter.LintDirectoryOptions{}).file_size_max,
    );

    const rendered = try linter.diagnostics.render(gpa);
    defer gpa.free(rendered);

    try std.testing.expectEqualStrings(expected, rendered);
}
