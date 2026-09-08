fn missingAfter() void {
    var client: Client = .{};
    defer client.deinit();
    doWork();
}

fn detachedCleanup() void {
    var client: Client = .{};

    defer client.deinit();

    doWork();
}

fn missingAfterStateRestore() void {
    state.busy = true;
    defer state.busy = false;
    doWork();
}

fn commentIsNotSeparator() void {
    var client: Client = .{};
    defer client.deinit();
    // Start requests.
    doWork();
}

fn detachedRegistration() void {
    const output = try createOutput();
    defer output.close();

    errdefer deleteOutput();

    doWork();
}

fn missingBefore() void {
    doOtherWork();
    var client: Client = .{};
    defer client.deinit();

    doWork();
}

fn unrelatedSetup() void {
    var resource: Resource = undefined;
    configureSomethingElse();
    defer resource.deinit();

    doWork();
}

// Keep this intentionally invalid same-line registration as executable policy.
// zig fmt: off
fn compactCleanup() void {
    acquire(); defer release();
}
// zig fmt: on

fn missingBetweenPairs() void {
    const first = acquire();
    defer first.deinit();
    const second = acquire();
    defer second.deinit();
}

fn detachedInitialization() void {
    var resource: Resource = undefined;

    try resource.init();
    defer resource.deinit();
}

fn missingBeforeTwoPhaseSetup() void {
    doOtherWork();
    var resource: Resource = undefined;
    try resource.init();
    defer resource.deinit();
}

fn wrongReceiver() void {
    var resource: Resource = undefined;
    try other.init();
    defer resource.deinit();
}

fn constIsNotTwoPhaseSetup() void {
    const resource: Resource = undefined;
    resource.init();
    defer resource.deinit();
}

fn definedIsNotTwoPhaseSetup() void {
    var resource: Resource = .{};
    resource.init();
    defer resource.deinit();
}

fn tightPair() void {
    var client: Client = .{};
    defer client.deinit();

    doWork();
}

fn consecutiveRegistrations() void {
    const output = try createOutput();
    defer output.close();
    errdefer deleteOutput();

    doWork();
}

fn blockEnd() void {
    var client: Client = .{};
    defer client.deinit();
}

fn attachedComment() void {
    var client: Client = .{};
    // Keep the client alive until all requests finish.
    defer client.deinit();

    doWork();
}

fn nextWorkComment() void {
    var client: Client = .{};
    defer client.deinit();

    // Start requests.
    doWork();
}

fn multilineCleanup() void {
    const output = try createOutput();
    defer {
        output.flush();
        output.close();
    }

    doWork();
}

fn standaloneRegistration() void {
    defer notifyFinished();

    doWork();
}

fn standaloneAtEnd() void {
    errdefer notifyFailed();
}

fn multilineSetup() void {
    doOtherWork();

    const output = try createOutput(.{
        .path = path,
        .mode = .exclusive,
    });
    defer output.close();

    doWork();
}

fn twoPhaseSetup() void {
    doOtherWork();

    var resource: Resource = undefined;
    try resource.init();
    defer resource.deinit();

    doWork();
}

fn twoPhaseSetupWithoutTry() void {
    var resource: Resource = undefined;
    resource.init();
    defer resource.deinit();
}

fn nestedCleanup() void {
    const outer = acquire();
    defer {
        const inner = acquire();
        defer inner.deinit();

        doWork();
    }

    doWork();
}

fn errdeferCapture() void {
    const output = try createOutput();
    errdefer |err| {
        output.report(err);
        output.close();
    }

    doWork();
}
