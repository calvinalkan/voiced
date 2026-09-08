//! Reads and writes Voiced's packed models files.

const reader = @import("reader.zig");
const writer = @import("writer.zig");

pub const load = reader.load;
pub const writeFromCTranslate2 = writer.writeFromCTranslate2;
