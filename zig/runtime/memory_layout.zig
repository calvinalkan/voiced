//! Typed descriptors define aligned regions before caller-owned memory is bound.
//! Layout owners use the same descriptors for size measurement and slice
//! construction, so those operations cannot acquire separate region sequences.

const std = @import("std");
const assert = std.debug.assert;

pub const alignment: usize = 64;

pub const Builder = struct {
    size: usize = 0,

    pub fn add(builder: *Builder, comptime Element: type, elements_count: usize) Region(Element) {
        assert(elements_count > 0);

        const offset = std.mem.alignForward(usize, builder.size, alignment);
        const region_size = std.math.mul(usize, elements_count, @sizeOf(Element)) catch unreachable;
        builder.size = std.math.add(usize, offset, region_size) catch unreachable;

        return .{ .offset = offset, .elements_count = elements_count };
    }
};

pub fn Region(comptime Element: type) type {
    return struct {
        offset: usize,
        elements_count: usize,

        pub fn bind(region: @This(), memory: []align(alignment) u8) []align(alignment) Element {
            assert(region.offset % alignment == 0);
            const region_size = std.math.mul(usize, region.elements_count, @sizeOf(Element)) catch unreachable;
            assert(region.offset <= memory.len);
            assert(region_size <= memory.len - region.offset);

            const values: [*]align(alignment) Element = @ptrCast(@alignCast(memory.ptr + region.offset));
            return values[0..region.elements_count];
        }
    };
}
