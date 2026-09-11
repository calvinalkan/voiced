const linter = @import("Linter.zig");
const report = @import("LintReport.zig");

pub const Plugin = @import("Plugin.zig");
pub const Rule = @import("Rule.zig");

pub const Allocators = linter.Allocators;
pub const LintOptions = linter.LintOptions;
pub const ReportPathFormat = linter.ReportPathFormat;
pub const Diagnostic = report.Diagnostic;
pub const LintReport = report.LintReport;
pub const SourceRange = Rule.SourceRange;
pub const Fix = Rule.Fix;
pub const Finding = Rule.Finding;
pub const lint = linter.lint;

// ─── Tests ───────────────────────────────────────────────────────────────────

test {
    _ = @import("root_tests.zig");
    _ = @import("LintReport.zig");
    _ = @import("LintContext.zig");
    _ = @import("Fixes.zig");
    _ = @import("BuiltinRules.zig");
    _ = @import("Plugin.zig");
    _ = @import("plugin/abi.zig");
    _ = @import("plugin/sdk.zig");
    _ = @import("LintCache.zig");
    _ = @import("AstTokenCache.zig");
    _ = @import("Gitignore.zig");
}
