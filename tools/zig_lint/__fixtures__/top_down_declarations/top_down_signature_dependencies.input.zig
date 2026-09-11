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

fn usesNamedErrorSet() LateErrors!void {
    return error.Failed;
}

const LateErrors = error{Failed};

fn usesAnonymousPayload(value: struct { payload: LateAnonymousPayload }) void {
    _ = value;
}

const LateAnonymousPayload = struct {};

fn usesCallbackPayload(callback: *const fn (LateCallbackPayload) void) void {
    _ = callback;
}

const LateCallbackPayload = struct {};

fn usesLocalErrorMember() error{LateErrorName}!void {
    return error.LateErrorName;
}

const LateErrorName = struct {};

fn usesComputedArrayLength(value: [LateLength]u8) void {
    _ = value;
}

const LateLength = 4;

fn usesComputedPointerAlignment(value: *align(LateAlignment) u8) void {
    _ = value;
}

const LateAlignment = 8;

fn usesComputedSentinel(value: [:LateSentinel]const u8) void {
    _ = value;
}

const LateSentinel = 0;

fn usesNestedGeneric(callback: fn (comptime LateShadow: type, LateShadow) void) void {
    _ = callback;
}

const LateShadow = struct {};

fn usesComputedType(value: LateFactory(LateArgument)) void {
    _ = value;
}

const LateFactory = struct {};

const LateArgument = 1;

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

fn usesAnonymousShadow(value: struct {
    const ShadowedPayload = u8;
    payload: ShadowedPayload,
}) void {
    _ = value;
}

const ShadowedPayload = struct {};

fn usesSelectedImport(value: late_mem.Allocator) void {
    _ = value;
}

const late_mem = @import("std").mem;

const ExactSignatureDependency = struct {
    fn entry(value: Later) void {
        _ = value;
    }

    const Later = struct {};
};

const UnknownSignatureDependency = struct {
    fn entry(value: Unknown) void {
        _ = value;
    }
};

const ShadowedSignatureDependency = struct {
    fn entry(comptime T: type, value: T) void {
        _ = value;
    }

    const T = struct {};
};

const ShadowedCallbackDependency = struct {
    fn entry(callback: *const fn (T: type, value: T) void) void {
        _ = callback;
    }

    const T = struct {};
};

const CallbackSignatureDependency = struct {
    fn entry(callback: *const fn (value: Later) void) void {
        _ = callback;
    }

    const Later = struct {};
};

const SelfSignatureDependency = struct {
    fn entry(value: Self) void {
        _ = value;
    }

    const Self = @This();
};
