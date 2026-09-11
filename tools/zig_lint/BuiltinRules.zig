//! Built-in rules use the same source-level descriptors exposed to Zig plugin
//! authors. Rule identity is declared once here rather than repeated by every
//! finding a rule emits.

const Rule = @import("Rule.zig");

pub const all = [_]Rule.Definition{
    Rule.define("visible_control_flow", @import("rules/visible_control_flow.zig").lint),
    Rule.define("explicit_optional_unwrap", @import("rules/explicit_optional_unwrap.zig").lint),
    Rule.define("visible_statement_paragraphs", @import("rules/visible_statement_paragraphs.zig").lint),
    Rule.define("visible_type_declarations", @import("rules/visible_type_declarations.zig").lint),
    Rule.define("top_down_declarations", @import("rules/top_down_declarations.zig").lint),
    Rule.define("assert_header_snapshot", @import("rules/assert_header_snapshot.zig").lint),
};
