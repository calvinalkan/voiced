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
    acquire();
    defer release();
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

// ─── Invalid Statement Paragraphs ───────────────────────────────────

fn invalidIfStartsParagraph(condition: bool) void {
    prepare();

    if (condition) {
        stop();
    }
}

fn invalidIfEndsParagraph(condition: bool) void {
    if (condition) {
        start();
    }

    publish();
}

fn invalidLoopStartsParagraph(items: []const Item) void {
    prepare();

    for (items) |item| {
        consume(item);
    }
}

fn invalidLoopEndsParagraph(items: []const Item) void {
    for (items) |item| {
        consume(item);
    }

    publish();
}

fn invalidConsecutiveGuards(first: bool, second: bool) void {
    if (first) {
        return;
    }

    if (second) {
        return;
    }
}

fn invalidDetachedProducerGuard() !void {
    const record = try parseRecord();
    if (!record.is_valid) {
        return error.InvalidRecord;
    }
}

fn invalidClassifierDispatch(record: Record) void {
    const kind = classify(record);

    switch (kind) {
        .ready => start(),
        .failed => stop(),
    }
}

fn validBlockThenSimpleArm(state: State) void {
    switch (state) {
        .ready => {
            start();
        },
        .failed => stop(),
    }
}

fn invalidDeclarationRunEndsParagraph() void {
    const first = acquireFirst();
    const second = acquireSecond();

    consumeBoth(first, second);
}

fn invalidDeclarationBeforeCall() void {
    const request = buildRequest();

    send(request);
}

fn invalidCallBeforeDeclaration() void {
    prepare();

    const request = buildRequest();
}

fn invalidCallBeforeAssignment() void {
    refresh();

    state.ready = true;
}

fn invalidAssignmentBeforeDeclaration() void {
    state.ready = true;

    const request = buildRequest();
}

fn invalidDeclarationBeforeAssignment() void {
    const deadline = calculateDeadline();

    state.deadline = deadline;
}

fn invalidMultilineDeclarationStartsParagraph() void {
    prepare();

    const request: Request = .{
        .timeout = timeout,
    };
}

fn invalidMultilineDeclarationEndsParagraph() void {
    const request: Request = .{
        .timeout = timeout,
    };

    send(request);
}

fn invalidMultilineCallStartsParagraph() void {
    prepare();

    send(.{
        .timeout = timeout,
    });
}

fn invalidMultilineCallEndsParagraph() void {
    send(.{
        .timeout = timeout,
    });

    publish();
}

fn invalidBareBlockStartsParagraph() void {
    prepare();

    {
        useScratch();
    }
}

fn invalidBareBlockEndsParagraph() void {
    {
        useScratch();
    }

    publish();
}

fn invalidComptimeBlockStartsParagraph() void {
    prepare();

    comptime {
        validateLayout();
    }
}

fn invalidUndefinedStartsParagraph() void {
    prepare();

    var buffer: [64]u8 = undefined;
    fill(&buffer);
}

fn invalidDetachedUndefinedFirstUse() void {
    var buffer: [64]u8 = undefined;
    fill(&buffer);
}

fn invalidUndefinedFirstUseMissing() void {
    var buffer: [64]u8 = undefined;
    fillSomethingElse();
}

fn invalidUndefinedGroupMissesOneUse() void {
    var header: Header = undefined;
    var payload: Payload = undefined;
    fillHeader(&header);
}

fn invalidUndefinedGroupEndsParagraph() void {
    var buffer: [64]u8 = undefined;
    fill(&buffer);

    publish();
}

fn invalidDetachedUndefinedLoop() void {
    var frame: [4]Event = undefined;
    for (&frame) |*event| {
        event.* = makeEvent();
    }
}

fn invalidDetachedAssertion() void {
    const header = parseHeader();
    assert(header.is_valid);
}

fn invalidAssertionPairEndsParagraph() void {
    const header = parseHeader();
    assert(header.is_valid);

    consume(header);
}

fn invalidCommentIsNotParagraph() void {
    prepare();

    // Build only after preparation succeeds.
    const request = buildRequest();
}

fn invalidSwitchEndsParagraph(state: State) void {
    switch (state) {
        .ready => start(),
        .failed => stop(),
    }

    publish();
}

fn invalidTwoProducersBeforeGuard() !void {
    const record = try parseRecord();
    const policy = loadPolicy();

    if (!record.is_valid or !policy.accepts(record)) {
        return error.InvalidRecord;
    }
}

fn invalidAttachedGuardEndsParagraph() !void {
    const record = try parseRecord();
    if (!record.is_valid) {
        return error.InvalidRecord;
    }

    publishRecord(record);
}

fn invalidDeclarationRunBeforeCompoundAssignment() void {
    const header_size = measureHeader();
    const payload_size = measurePayload();

    total_size += header_size + payload_size;
}

fn invalidAssignmentBeforeCall() void {
    state.ready = true;

    publish();
}

fn invalidComptimeBlockEndsParagraph() void {
    comptime {
        validateLayout();
    }

    publish();
}

// ─── Valid Statement Paragraphs ─────────────────────────────────────

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

fn validControlParagraphs(condition: bool, items: []const Item, state: State) void {
    prepare();

    if (condition) {
        start();
    }

    while (condition) {
        break;
    }

    for (items) |item| {
        consume(item);
    }

    switch (state) {
        .ready => start(),
        .failed => stop(),
    }
}

fn validAttachedProducerGuard() !void {
    const record = try parseRecord();
    if (!record.is_valid) {
        return error.InvalidRecord;
    }

    publishRecord(record);
}

fn validSeparatedGuards(first: bool, second: bool) void {
    if (first) {
        return;
    }

    if (second) {
        return;
    }
}

fn validTwoProducersBeforeGuard() !void {
    const record = try parseRecord();
    const policy = loadPolicy();

    if (!record.is_valid or !policy.accepts(record)) {
        return error.InvalidRecord;
    }
}

fn validClassifierDispatch(record: Record) void {
    const kind = classify(record);

    switch (kind) {
        .ready => start(),
        .failed => stop(),
    }
}

fn validOptionalSwitchArmSeparator(state: State) void {
    switch (state) {
        .ready => {
            start();
        },

        .failed => stop(),
    }
}

fn validDeclarationRun() void {
    const first = acquireFirst();
    var second = acquireSecond();

    consumeBoth(first, second);
}

fn validDeclarationBeforeCall() void {
    const request = buildRequest();

    send(request);
}

fn validCallGroup() void {
    prepare();
    validate();
    publish();
}

fn validCallBeforeAssignment() void {
    refresh();

    state.ready = true;
}

fn validAssignmentBeforeDeclaration() void {
    state.ready = true;

    const request = buildRequest();

    send(request);
}

fn validDeclarationBeforeAssignment() void {
    const deadline = calculateDeadline();

    state.deadline = deadline;
}

fn validAssignmentGroup() void {
    state.offset = offset;
    state.remaining = remaining;
    state.ready = true;
}

fn validSeparatedAssignments() void {
    state.offset = offset;

    state.remaining = remaining;
}

fn validTightMultilineAssignment() void {
    state.configuration = .{
        .timeout = timeout,
        .mode = mode,
    };
    state.ready = true;
}

fn validSeparatedMultilineAssignment() void {
    state.configuration = .{
        .timeout = timeout,
        .mode = mode,
    };

    state.ready = true;
}

fn validMultilineDeclarationParagraph() void {
    prepare();

    const request: Request = .{
        .timeout = timeout,
    };

    send(request);
}

fn validMultilineDeclarationCleanupException() void {
    const request: Request = .{
        .timeout = timeout,
    };
    defer request.deinit();

    send(request);
}

fn validMultilineCallParagraph() void {
    prepare();

    send(.{
        .timeout = timeout,
    });

    publish();
}

fn validBareBlockParagraph() void {
    prepare();

    {
        useScratch();
    }

    publish();
}

fn validComptimeBlockParagraph() void {
    prepare();

    comptime {
        validateLayout();
    }

    publish();
}

fn validUndefinedSingle() void {
    var buffer: [64]u8 = undefined;
    const count = fill(&buffer);

    consumeCount(count);
}

fn validUndefinedGroup() void {
    var header: Header = undefined;
    var payload: Payload = undefined;
    fillMessage(&header, &payload);

    publishMessage(header, payload);
}

fn validUndefinedLoop() void {
    var frame: [4]Event = undefined;
    for (&frame) |*event| {
        event.* = makeEvent();
    }

    publishFrame(frame);
}

fn validUndefinedComment() void {
    var buffer: [64]u8 = undefined;
    // The syscall initializes the borrowed output buffer.
    readInto(&buffer);

    publishBuffer(buffer);
}

fn validAssertionPairs() void {
    const header = parseHeader();
    assert(header.is_valid);

    const payload = parsePayload(header);
    assert(payload.len != 0);
}

fn validCommentStartsNextParagraph() void {
    prepare();

    // Build only after preparation succeeds.
    const request = buildRequest();

    send(request);
}
