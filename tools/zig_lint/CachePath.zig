//! Private cache namespace path components shared by cache layers.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;

/// Return an allocator-owned, path-component-safe shard for one exact semantic
/// version. Stable releases retain their readable `major.minor.patch` spelling;
/// prerelease and build metadata use a digest to keep arbitrary caller-provided
/// bytes and lengths out of filesystem path components.
pub fn allocVersionShard(
    allocator: Allocator,
    version: std.SemanticVersion,
) Allocator.Error![]u8 {
    if (version.pre == null and version.build == null) {
        return std.fmt.allocPrint(
            allocator,
            "{d}.{d}.{d}",
            .{ version.major, version.minor, version.patch },
        );
    }

    var hasher = Blake3.init(.{});

    var encoded_component: [8]u8 = undefined;
    for ([_]usize{ version.major, version.minor, version.patch }) |component| {
        std.mem.writeInt(u64, &encoded_component, @intCast(component), .little);
        hasher.update(&encoded_component);
    }

    hashOptionalBytes(&hasher, version.pre);
    hashOptionalBytes(&hasher, version.build);

    var digest: [Blake3.digest_length]u8 = undefined;
    hasher.final(&digest);

    return std.fmt.allocPrint(
        allocator,
        "{d}.{d}.{d}-{s}",
        .{
            version.major,
            version.minor,
            version.patch,
            std.fmt.bytesToHex(digest, .lower),
        },
    );
}

fn hashOptionalBytes(hasher: *Blake3, value: ?[]const u8) void {
    hasher.update(if (value == null)
        &.{0}
    else
        &.{1});

    if (value) |bytes| {
        var encoded_length: [8]u8 = undefined;
        std.mem.writeInt(u64, &encoded_length, @intCast(bytes.len), .little);

        hasher.update(&encoded_length);
        hasher.update(bytes);
    }
}
