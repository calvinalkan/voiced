// `invalid...` functions specify rejected forms; `valid...` functions specify
// accepted alternatives. Section headings state the policy relating them.

// ─── Explicit Exits Must Not Be Hidden ─────────────────────────────────────

fn invalidInlineIfReturn(condition: bool) void {
    if (condition) {
        return;
    }
}

fn invalidInlineIfErrorReturn(condition: bool) !void {
    if (condition) {
        return error.NotReady;
    }
}

fn invalidInlineIfBreak(condition: bool) void {
    while (true) {
        if (condition) {
            break;
        }
    }
}

fn invalidInlineIfContinue(condition: bool) void {
    while (true) {
        if (condition) {
            continue;
        }
    }
}

fn invalidInlineCapturedIfReturn(optional: ?usize) usize {
    if (optional) |value| {
        return value;
    }

    return 0;
}

fn invalidInlineIfElseReturn(condition: bool) !usize {
    const value = if (condition) 1 else return error.NotReady;

    return value;
}

fn invalidInlineIfThenReturn(condition: bool) usize {
    const value = if (condition) return 1 else 0;

    return value;
}

fn invalidGroupedIfThenReturn(condition: bool) usize {
    return if (condition)
        (return 1)
    else
        0;
}

fn invalidInlineOrelseReturn(optional: ?usize) usize {
    const value = optional orelse {
        return 0;
    };

    return value;
}

fn invalidInlineCatchReturn() void {
    operation() catch {
        return;
    };
}

fn invalidInlineCapturedCatchReturn() !void {
    operation() catch |err| {
        return err;
    };
}

fn invalidInlineWhileBreak() void {
    while (true) {
        break;
    }
}

fn invalidInlineForContinue(items: []const u8) void {
    for (items) |_| {
        continue;
    }
}

fn invalidNestedInlineIfInSwitchArm(items: []const u8) void {
    for (items) |item| {
        switch (item) {
            0 => if (item == 0) continue,

            else => {},
        }
    }
}

// ─── Statement Bodies Require Braces ───────────────────────────────────────

fn invalidInlineIfCall(condition: bool) void {
    if (condition) {
        startWorker();
    }
}

fn invalidMultilineIfCall(condition: bool) void {
    if (condition) {
        startWorker();
    }
}

fn validBracedIfCall(condition: bool) void {
    if (condition) {
        startWorker();
    }
}

fn invalidUnbracedElseCall(condition: bool) void {
    if (condition) {
        startWorker();
    } else {
        queueWorker();
    }
}

fn validBracedIfElseCalls(condition: bool) void {
    if (condition) {
        startWorker();
    } else {
        queueWorker();
    }
}

fn validBracedElseIfChain(first: bool, second: bool) void {
    if (first) {
        startWorker();
    } else if (second) {
        queueWorker();
    } else {
        stopWorker();
    }
}

fn invalidMultilineIfReturn(condition: bool) void {
    if (condition) {
        return;
    }
}

fn validBracedIfReturn(condition: bool) void {
    if (condition) {
        return;
    }
}

fn invalidMultilineWhileBody(queue: *Queue) void {
    while (queue.pop()) |item| {
        process(item);
    }
}

fn validBracedWhileBody(queue: *Queue) void {
    while (queue.pop()) |item| {
        process(item);
    }
}

fn invalidMultilineForBody(workers: []Worker) void {
    for (workers) |worker| {
        worker.stop();
    }
}

fn validBracedForBody(workers: []Worker) void {
    for (workers) |worker| {
        worker.stop();
    }
}

fn invalidUnbracedInlineForBody(fields: []Field) void {
    inline for (fields) |field| {
        registerField(field);
    }
}

fn validBracedInlineForBody(fields: []Field) void {
    inline for (fields) |field| {
        registerField(field);
    }
}

fn invalidMultilineComptimeIfBody() void {
    if (comptime feature_enabled) {
        startWorker();
    }
}

fn validBracedComptimeIfBody() void {
    if (comptime feature_enabled) {
        startWorker();
    }
}

fn invalidConditionalDeferWithoutBraces(opened: bool) void {
    defer if (opened) closeWorker();
}

fn validConditionalDeferWithBraces(opened: bool) void {
    defer if (opened) {
        closeWorker();
    };
}

fn validNestedBracedIfInSwitchArm(kind: Kind, ready: bool) void {
    switch (kind) {
        .start => if (ready) {
            startWorker();
        } else {
            queueWorker();
        },
        .stop => stopWorker(),
    }
}

// ─── Complex Value Conditionals Require Multiple Lines ─────────────────────

fn validInlineSimpleIfValue(ready: bool) State {
    return if (ready) .active else .idle;
}

fn validInlineConditionCallIfValue(worker: Worker) State {
    return if (worker.ready()) .active else .idle;
}

fn invalidInlineCallIfValue(cached: bool) Worker {
    return if (cached)
        getCachedWorker()
    else
        createWorker();
}

fn validMultilineCallIfValue(cached: bool) Worker {
    return if (cached)
        getCachedWorker()
    else
        createWorker();
}

fn invalidInlineBuiltinIfValue(limited: bool, requested: usize, maximum: usize) usize {
    return if (limited)
        @min(requested, maximum)
    else
        requested;
}

fn validMultilineBuiltinIfValue(limited: bool, requested: usize, maximum: usize) usize {
    return if (limited)
        @min(requested, maximum)
    else
        requested;
}

fn invalidInlineBinaryIfValue(forward: bool, base: usize, distance: usize) usize {
    return if (forward)
        base + distance
    else
        base - distance;
}

fn validMultilineBinaryIfValue(forward: bool, base: usize, distance: usize) usize {
    return if (forward)
        base + distance
    else
        base - distance;
}

fn invalidInlineCapturedIfValue(optional: ?usize, fallback: usize) usize {
    return if (optional) |value|
        value
    else
        fallback;
}

fn validMultilineCapturedIfValue(optional: ?usize, fallback: usize) usize {
    return if (optional) |value|
        value
    else
        fallback;
}

fn invalidInlineChainedIfValue(failed: bool, ready: bool) State {
    return if (failed)
        .failed
    else if (ready)
        .active
    else
        .idle;
}

fn validMultilineChainedIfValue(failed: bool, ready: bool) State {
    return if (failed)
        .failed
    else if (ready)
        .active
    else
        .idle;
}

// ─── Optional And Error Fallbacks ──────────────────────────────────────────

fn validInlineSimpleOrelse(optional: ?usize, fallback: usize) usize {
    return optional orelse fallback;
}

fn validMultilineSimpleOrelse(optional: ?usize, fallback: usize) usize {
    return optional orelse
        fallback;
}

fn invalidInlineOrelseCall(optional: ?Worker) Worker {
    return optional orelse
        createWorker();
}

fn validMultilineOrelseCall(optional: ?Worker) Worker {
    return optional orelse
        createWorker();
}

fn invalidInlineCompoundOrelseValue(optional: ?usize, offset: usize) usize {
    return optional orelse
        createValue() + offset;
}

fn validMultilineCompoundOrelseValue(optional: ?usize, offset: usize) usize {
    return optional orelse
        createValue() + offset;
}

fn invalidMultilineOrelseReturn(optional: ?Worker) !Worker {
    const worker = optional orelse {
        return error.MissingWorker;
    };

    return worker;
}

fn invalidGroupedOrelseReturn(optional: ?Worker) !Worker {
    const worker = optional orelse {
        (return error.MissingWorker);
    };

    return worker;
}

fn validBracedOrelseReturn(optional: ?Worker) !Worker {
    const worker = optional orelse {
        return error.MissingWorker;
    };

    return worker;
}

fn invalidMultilineOrelseUnreachable(optional: ?usize) usize {
    return optional orelse {
        unreachable;
    };
}

fn validBracedOrelseUnreachable(optional: ?usize) usize {
    return optional orelse {
        unreachable;
    };
}

fn invalidMultilineSideEffectOrelse(maybe_notification: ?void) void {
    maybe_notification orelse {
        reportUnavailable();
    };
}

fn validBracedSideEffectOrelse(maybe_notification: ?void) void {
    maybe_notification orelse {
        reportUnavailable();
    };
}

fn validInlineSimpleCatch() usize {
    return parseValue() catch 3;
}

fn validMultilineSimpleCatch() usize {
    return parseValue() catch
        3;
}

fn invalidInlineCatchCall() Worker {
    return loadWorker() catch
        recoverWorker();
}

fn validMultilineCatchCall() Worker {
    return loadWorker() catch
        recoverWorker();
}

fn invalidInlineCapturedCatchCall() Worker {
    return loadWorker() catch |err|
        recoverWorkerFrom(err);
}

fn validMultilineCapturedCatchCall() Worker {
    return loadWorker() catch |err|
        recoverWorkerFrom(err);
}

fn invalidMultilineCatchReturn() !Worker {
    const worker = loadWorker() catch |err| {
        return err;
    };

    return worker;
}

fn validBracedCatchReturn() !Worker {
    const worker = loadWorker() catch |err| {
        return err;
    };

    return worker;
}

fn invalidMultilineCatchUnreachable() Worker {
    return loadWorker() catch {
        unreachable;
    };
}

fn validBracedCatchUnreachable() Worker {
    return loadWorker() catch {
        unreachable;
    };
}

fn invalidMultilineSideEffectCatch() void {
    operation() catch |err| {
        logFailure(err);
    };
}

fn validBracedSideEffectCatch() void {
    operation() catch |err| {
        logFailure(err);
    };
}

// ─── Nested Try Follows Zig Fmt ────────────────────────────────────────────

fn validInlineNestedTry() !void {
    configure(try loadConfig());
}

fn validMultilineNestedTry() !void {
    configure(
        try loadConfig(),
    );
}

fn validExtractedTry() !void {
    const configuration = try loadConfig();

    configure(configuration);
}

fn validLeadingTryBeforeOrelse() !Worker {
    const worker = try loadOptionalWorker() orelse {
        return error.MissingWorker;
    };

    return worker;
}

fn validInlineNestedTryInAggregate() !void {
    configure(.{ .value = try loadValue() });
}

fn validExtractedTryForAggregate() !void {
    const value = try loadValue();

    configure(.{ .value = value });
}

// ─── Defer Forms ───────────────────────────────────────────────────────────

fn validInlineDefer() void {
    defer finishOperation();
}

fn validBracedDefer() void {
    defer {
        finishOperation();
    }
}

fn validInlineErrdefer() !void {
    errdefer failOperation();
}

fn validBracedErrdefer() !void {
    errdefer {
        failOperation();
    }
}

fn invalidCapturedErrdeferWithoutBraces() !void {
    errdefer |err| {
        logFailure(err);
    }
}

fn validCapturedErrdeferWithBraces() !void {
    errdefer |err| {
        logFailure(err);
    }
}

// ─── Compact Switch Arms Are Valid ─────────────────────────────────────────

fn validCompactSwitchActions(kind: Kind) !void {
    switch (kind) {
        .start => startWorker(),
        .stop => try stopWorker(),
    }
}

fn validCompactSwitchReturn(kind: Kind) void {
    switch (kind) {
        .start => return,
        .stop => stopWorker(),
    }
}

fn validCompactCapturedSwitchReturn(err: error{NotReady}) error{NotReady} {
    switch (err) {
        error.NotReady => |captured| return captured,
    }
}

fn validCompactLabeledSwitchBreak(items: []const u8) u8 {
    return selected: {
        for (items) |item| {
            switch (item) {
                0 => break :selected item,

                else => {},
            }
        }

        break :selected 0;
    };
}

fn validBracedSwitchArm(kind: Kind) void {
    switch (kind) {
        .start => {
            startWorker();
        },

        .stop => stopWorker(),
    }
}

fn validCompactValueSwitch(kind: Kind) !Worker {
    return switch (kind) {
        .start => getCachedWorker(),
        .stop => try createWorker(),
    };
}

// ─── Block Value Branches Follow Zig Fmt ───────────────────────────────────

fn validThenBlockIfValue(ready: bool) usize {
    return if (ready) selected: {
        break :selected 1;
    } else 0;
}

fn validElseBlockIfValue(ready: bool) usize {
    return if (ready) 1 else selected: {
        break :selected 0;
    };
}
