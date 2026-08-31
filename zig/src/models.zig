//! Names the runtime models that Voiced installs and may load. The model name
//! preserves both its publisher and repository variant so configuration never
//! turns an arbitrary directory into an implicit model identity.

const std = @import("std");
const assert = std.debug.assert;

pub const Vendor = enum(u8) {
    systran = 1,
};

pub const SystranVariant = enum(u8) {
    base_en = 1,
    small_en = 2,
};

pub const Model = union(Vendor) {
    systran: SystranVariant,

    /// `parse` accepts the canonical publisher/repository name used by setup,
    /// configuration, and diagnostics. It accepts no filesystem path aliases.
    pub fn parse(text: []const u8) ?Model {
        if (std.mem.eql(u8, text, "Systran/faster-whisper-base.en")) {
            return .{ .systran = .base_en };
        }
        if (std.mem.eql(u8, text, "Systran/faster-whisper-small.en")) {
            return .{ .systran = .small_en };
        }
        return null;
    }

    /// `name` returns the stable publisher/repository identity shown in
    /// configuration and used as the model's path below Voiced's model root.
    pub fn name(model: Model) []const u8 {
        return switch (model) {
            .systran => |variant| switch (variant) {
                .base_en => "Systran/faster-whisper-base.en",
                .small_en => "Systran/faster-whisper-small.en",
            },
        };
    }

    pub fn vendorCode(model: Model) u8 {
        return @intFromEnum(std.meta.activeTag(model));
    }

    pub fn variantCode(model: Model) u8 {
        return switch (model) {
            .systran => |variant| @intFromEnum(variant),
        };
    }

    pub fn fromCodes(vendor_code: u8, variant_code: u8) ?Model {
        if (vendor_code != @intFromEnum(Vendor.systran)) return null;

        const variant: SystranVariant = switch (variant_code) {
            @intFromEnum(SystranVariant.base_en) => .base_en,
            @intFromEnum(SystranVariant.small_en) => .small_en,
            else => return null,
        };
        return .{ .systran = variant };
    }
};

pub const default: Model = .{ .systran = .small_en };

/// `allocInstalledDirectoryPath` resolves one named model under the user's XDG
/// data directory. The caller owns the returned sentinel-terminated path and
/// must free it with `init.gpa`.
pub fn allocInstalledDirectoryPath(
    init: std.process.Init,
    model: Model,
) ![:0]u8 {
    const data_home_path = if (init.environ_map.get("XDG_DATA_HOME")) |path|
        path
    else
        init.environ_map.get("HOME") orelse return error.HomeNotSet;
    if (!std.fs.path.isAbsolute(data_home_path)) {
        return error.DataHomeNotAbsolute;
    }

    const path_parts: []const []const u8 = if (init.environ_map.get("XDG_DATA_HOME") != null)
        &.{ data_home_path, "voiced/models", model.name() }
    else
        &.{ data_home_path, ".local/share/voiced/models", model.name() };

    const path = try std.fs.path.join(init.gpa, path_parts);
    defer init.gpa.free(path);

    const sentinel_path = try init.gpa.dupeSentinel(u8, path, 0);
    assert(std.mem.eql(u8, sentinel_path, path));

    return sentinel_path;
}
