const std = @import("std");

fn acceptsLateInput(value: LateInput) void {
    _ = value;
}

const LateInput = struct {};

fn returnsLateResult() LateResult {
    return .{};
}

const LateResult = struct {};

fn usesLateAlias(value: LateAlias) void {
    _ = value;
}

const LateAlias = std.mem.Allocator;

const ReadyInput = struct {};

const ReadyResult = struct {};

fn ready(input: ReadyInput) ReadyResult {
    _ = input;

    return .{};
}

fn imported(input: std.mem.Allocator) void {
    _ = input;
}

fn generic(comptime T: type, value: T) T {
    return value;
}

const Service = struct {
    fn start(options: LateOptions) LateServiceResult {
        _ = options;

        return .{};
    }

    const LateOptions = struct {};

    const LateServiceResult = struct {};

    const ReadyOptions = struct {};

    fn readyStart(options: ReadyOptions) void {
        _ = options;
    }
};
