//! A shared scheduler executes synchronous collectives from independent callers.
//! Each operation retains its own barrier, tile cursor, and borrowed context.
//! Only whole groups yield, after all lanes finish a resumable compute step.

const Scheduler = @This();
const std = @import("std");
const abi = @import("abi.zig");
const linux = std.os.linux;
const assert = std.debug.assert;

const Word = std.atomic.Value(usize);
const word_bits = @bitSizeOf(usize);

/// `Lane` identifies one active worker in the current operation. All active
/// lanes use the same count and barrier; indices remain stable during a step.
pub const Lane = struct {
    index: usize,
    count: usize,
    // The barrier and tile cursor belong to one operation and share its lifetime.
    operation: *Operation,

    pub fn isLeader(lane: Lane) bool {
        assert(lane.index < lane.count);

        return lane.index == 0;
    }

    pub fn range(lane: Lane, items_count: usize) Range {
        assert(lane.index < lane.count);

        const items_per_lane = items_count / lane.count;
        const leftover_items_count = items_count % lane.count;
        const lane_has_leftover = lane.index < leftover_items_count;
        const leftovers_before_lane = @min(lane.index, leftover_items_count);
        const start_index = lane.index * items_per_lane + leftovers_before_lane;
        const end_index = start_index + items_per_lane + @intFromBool(lane_has_leftover);

        assert(start_index <= end_index);
        assert(end_index <= items_count);

        return .{ .start_index = start_index, .end_index = end_index };
    }

    /// `tiles` starts a collective traversal of disjoint tile IDs. All lanes
    /// must finish the previous traversal and synchronize before calling it;
    /// the cursor belongs to this operation. Lanes synchronize here to publish
    /// the reset, then claim chunks as they finish.
    pub fn tiles(lane: Lane, tiles_count: usize, claim_count_max: usize) TileIterator {
        assert(claim_count_max > 0);
        const cursor = &lane.operation.tile_cursor;
        // Large inputs amortize the atomic claim across several tiles.
        // Short inputs need several reservations per lane so a slow core
        // cannot hold one oversized final chunk after fast lanes run dry.
        const claim_count = @min(claim_count_max, @max(1, tiles_count / (lane.count * 4)));
        if (lane.isLeader()) cursor.store(0, .monotonic);
        lane.sync();
        return .{ .cursor = cursor, .tiles_count = tiles_count, .claim_count = claim_count };
    }

    pub fn sync(lane: Lane) void {
        assert(lane.index < lane.count);

        lane.operation.barrier.wait(lane.count);
    }
};

pub const Range = struct {
    start_index: usize,
    end_index: usize,
};

pub const TileIterator = struct {
    cursor: *std.atomic.Value(usize),
    tiles_count: usize,
    claim_count: usize,
    next_index: usize = 0,
    end_index: usize = 0,

    pub fn next(tiles: *TileIterator) ?usize {
        if (tiles.next_index == tiles.end_index) {
            const begin = tiles.cursor.fetchAdd(tiles.claim_count, .monotonic);
            if (begin >= tiles.tiles_count) {
                return null;
            }
            tiles.next_index = begin;
            tiles.end_index = begin + @min(tiles.claim_count, tiles.tiles_count - begin);
        }
        const index = tiles.next_index;
        tiles.next_index += 1;
        return index;
    }
};

pub const Step = enum { more, done };
/// Every lane executes the same collective step; only lane zero's return is
/// authoritative. Returning yields no references to scratch or lane identity.
/// Publish shared output through lane.sync when another lane reads it inside
/// this step. The scheduler supplies the final barrier and may resize the next group.
pub const StepFunction = *const fn (*anyopaque, Lane) Step;

/// The caller owns this stable scheduler and starts exactly workersCount() instances
/// of workerMain. Stop only after every synchronous run has returned.
mutex: Mutex = .{},
stopping: bool = false,
// A circular FIFO needs only its tail: tail.next is the head. Unlocked
// readers only test for null; traversing or changing links needs the mutex.
queue_tail: std.atomic.Value(?*Operation) = .init(null),
// The mutex protects assignment and bit changes. Checkpoints only scan these
// atomic words for an availability hint; they never claim workers unlocked.
idle_words: []Word,
wake_generation: std.atomic.Value(u32) = .init(0),
assignments: []Assignment,

pub fn requiredMemory(count: usize) InitError!usize {
    return (try Layout.init(count)).size;
}

pub fn init(scheduler: *Scheduler, memory: []align(abi.memory_alignment) u8, count: usize) InitError!void {
    const layout = try Layout.init(count);
    if (memory.len < layout.size) return error.MemoryTooSmall;
    assert(@intFromPtr(scheduler) == @intFromPtr(memory.ptr));
    const words: [*]Word = @ptrCast(@alignCast(memory.ptr + layout.words_offset));
    const assignments: [*]Assignment = @ptrCast(@alignCast(memory.ptr + layout.assignments_offset));
    scheduler.* = .{
        .idle_words = words[0..layout.words_count],
        .assignments = assignments[0..count],
    };
    for (scheduler.idle_words, 0..) |*word, index| {
        // The final word may be partial. Padding bits must never name workers.
        const valid_bits = @min(word_bits, count - index * word_bits);
        word.* = .init(@as(usize, std.math.maxInt(usize)) >> @intCast(word_bits - valid_bits));
    }
}

pub const InitError = error{ InvalidConfig, MemorySizeOverflow, MemoryTooSmall };

pub fn workersCount(scheduler: *const Scheduler) usize {
    return scheduler.assignments.len;
}

pub fn requestStop(scheduler: *Scheduler) void {
    scheduler.mutex.lock();
    defer scheduler.mutex.unlock();
    assert(scheduler.queue_tail.load(.monotonic) == null);
    for (scheduler.idle_words, 0..) |*word, index| {
        const valid_bits = @min(word_bits, scheduler.workersCount() - index * word_bits);
        assert(@popCount(word.load(.monotonic)) == valid_bits);
    }
    scheduler.stopping = true;
    scheduler.wake_generation.store(scheduler.wake_generation.load(.monotonic) +% 1, .release);
    futexWake(&scheduler.wake_generation.raw, scheduler.workersCount());
}

/// Multiple callers may run concurrently. Each call borrows context and
/// cancellation until every assigned worker has released its reference.
/// Callbacks may not recursively submit work to this scheduler.
pub fn run(scheduler: *Scheduler, count_max: usize, context: *anyopaque, step: StepFunction, cancellation: ?*const std.atomic.Value(bool)) error{Cancelled}!void {
    assert(count_max > 0 and count_max <= scheduler.workersCount());
    if (cancellation) |flag| if (flag.load(.acquire)) return error.Cancelled;
    var operation: Operation = undefined;
    operation = .{
        .context = context,
        .step = step,
        .workers_count_max = count_max,
        .cancellation = cancellation,
        .state = .{ .queued = &operation },
    };
    scheduler.mutex.lock();
    assert(!scheduler.stopping);
    scheduler.enqueue(&operation);
    scheduler.schedule();
    while (operation.state != .complete) {
        scheduler.mutex.unlock();
        futexWait(&operation.wake.raw, 0);
        scheduler.mutex.lock();
    }
    // Reacquiring the mutex also joins the final worker's wake syscall.
    // The worker cannot still dereference this stack record after return.
    const completion = operation.state.complete;
    scheduler.mutex.unlock();
    if (completion == .cancelled) return error.Cancelled;
}

fn enqueue(scheduler: *Scheduler, operation: *Operation) void {
    const tail = scheduler.queue_tail.load(.monotonic);
    operation.state = .{ .queued = if (tail) |last| last.state.queued else operation };
    if (tail) |last| last.state.queued = operation;
    scheduler.queue_tail.store(operation, .release);
}

fn schedule(scheduler: *Scheduler) void {
    var assigned = false;
    while (scheduler.queue_tail.load(.monotonic)) |tail| {
        const operation = tail.state.queued;
        // Cancellation before admission needs no compute lane. Keeping the
        // record linked until this point avoids concurrent queue removal.
        const cancelled = if (operation.cancellation) |flag| flag.load(.acquire) else false;
        var count: usize = 0;
        if (!cancelled) {
            for (scheduler.idle_words) |*word| {
                count += @min(@as(usize, @popCount(word.load(.monotonic))), operation.workers_count_max - count);
                if (count == operation.workers_count_max) break;
            }
            if (count == 0) break;
        }
        if (operation == tail) scheduler.queue_tail.store(null, .release) else tail.state.queued = operation.state.queued;
        if (cancelled) {
            operation.state = .{ .complete = .cancelled };
            operation.wake.store(1, .release);
            futexWake(&operation.wake.raw, 1);
            continue;
        }
        operation.state = .{ .running = .{ .count = count, .remaining = count } };
        var lane_index: usize = 0;
        for (scheduler.idle_words, 0..) |*word, word_index| {
            var idle = word.load(.monotonic);
            while (idle != 0 and lane_index < count) : (lane_index += 1) {
                const bit_index = @ctz(idle);
                idle &= idle - 1;
                const worker_index = word_index * word_bits + bit_index;
                scheduler.assignments[worker_index] = .{ .operation = operation, .lane_index = lane_index };
            }
            word.store(idle, .release);
            if (lane_index == count) break;
        }
        assigned = true;
    }
    if (assigned) {
        scheduler.wake_generation.store(scheduler.wake_generation.load(.monotonic) +% 1, .release);
        futexWake(&scheduler.wake_generation.raw, scheduler.workersCount());
    }
}

pub fn workerMain(scheduler: *Scheduler, worker_index: usize) void {
    assert(worker_index < scheduler.workersCount());
    const word = &scheduler.idle_words[worker_index / word_bits];
    const worker_bit = @as(usize, 1) << @intCast(worker_index % word_bits);
    scheduler.mutex.lock();
    defer scheduler.mutex.unlock();
    while (true) {
        while (word.load(.monotonic) & worker_bit != 0) {
            if (scheduler.stopping) return;
            const generation = scheduler.wake_generation.load(.monotonic);
            scheduler.mutex.unlock();
            futexWait(&scheduler.wake_generation.raw, generation);
            scheduler.mutex.lock();
        }
        const assignment = scheduler.assignments[worker_index];
        const operation = assignment.operation;
        const lane: Lane = .{
            .index = assignment.lane_index,
            .count = operation.state.running.count,
            .operation = operation,
        };
        scheduler.mutex.unlock();

        const action = while (true) {
            const step = operation.step(operation.context, lane);
            // Only the leader's result is authoritative. The checkpoint's
            // acq_rel arrival chain publishes it to whichever lane arrives last.
            if (lane.isLeader()) operation.step_result = step;
            const action = operation.checkpoint(scheduler, lane.count);
            if (action != .continue_running) break action;
        };

        scheduler.mutex.lock();
        word.store(word.load(.monotonic) | worker_bit, .release);
        operation.state.running.remaining -= 1;
        if (operation.state.running.remaining == 0) {
            // Arriving at the barrier alone does not release a reference:
            // peers still read its generation. Only the final departure can
            // requeue the record or let its caller reclaim stack storage.
            if (action == .yield) {
                scheduler.enqueue(operation);
            } else {
                operation.state = .{ .complete = if (action == .cancelled) .cancelled else .done };
                operation.wake.store(1, .release);
                futexWake(&operation.wake.raw, 1);
            }
        }
        // No dereference of operation after schedule: it may complete or be
        // reassigned, and its caller may destroy it as soon as we unlock.
        scheduler.schedule();
    }
}

fn hasIdleWorkers(scheduler: *const Scheduler) bool {
    for (scheduler.idle_words) |*word| {
        if (word.load(.acquire) != 0) return true;
    }
    return false;
}

const Assignment = struct { operation: *Operation, lane_index: usize };
const Action = enum(u2) { continue_running, yield, done, cancelled };

const Operation = struct {
    context: *anyopaque,
    step: StepFunction,
    workers_count_max: usize,
    cancellation: ?*const std.atomic.Value(bool),
    state: union(enum) {
        queued: *Operation,
        running: struct { count: usize, remaining: usize },
        complete: enum { done, cancelled },
    },
    wake: std.atomic.Value(u32) = .init(0),
    step_result: Step = undefined,
    barrier: Barrier align(std.atomic.cache_line) = .{},
    tile_cursor: std.atomic.Value(usize) align(std.atomic.cache_line) = .init(0),

    fn checkpoint(operation: *Operation, scheduler: *Scheduler, count: usize) Action {
        const barrier = &operation.barrier;
        const generation = barrier.generation.load(.monotonic);
        const arrivals = barrier.arrivals_count.fetchAdd(1, .acq_rel);
        assert(arrivals < count);
        if (arrivals + 1 == count) {
            const cancelled = if (operation.cancellation) |flag| flag.load(.acquire) else false;
            const action: Action = if (cancelled) .cancelled else if (operation.step_result == .done) .done else if (scheduler.queue_tail.load(.acquire) != null or
                (count < operation.workers_count_max and scheduler.hasIdleWorkers())) .yield else .continue_running;
            barrier.arrivals_count.store(0, .monotonic);
            // Publish the decision WITH the barrier generation. A separate
            // mutable decision field could be overwritten by a fast next step
            // before a delayed peer reads this step's decision.
            barrier.generation.store(((generation & ~@as(usize, 3)) +% 4) | @intFromEnum(action), .release);
            return action;
        }
        var published = barrier.generation.load(.acquire);
        while (published == generation) {
            std.atomic.spinLoopHint();
            published = barrier.generation.load(.acquire);
        }
        return @enumFromInt(published & 3);
    }
};

const Barrier = struct {
    arrivals_count: std.atomic.Value(usize) = .init(0),
    generation: std.atomic.Value(usize) = .init(0),

    fn wait(barrier: *Barrier, count: usize) void {
        if (count == 1) return;
        const generation = barrier.generation.load(.monotonic);
        const arrivals = barrier.arrivals_count.fetchAdd(1, .acq_rel);
        assert(arrivals < count);
        if (arrivals + 1 == count) {
            barrier.arrivals_count.store(0, .monotonic);
            barrier.generation.store((generation & ~@as(usize, 3)) +% 4, .release);
            return;
        }
        while (barrier.generation.load(.acquire) == generation) std.atomic.spinLoopHint();
    }
};

const Mutex = struct {
    state: std.atomic.Value(u32) = .init(0),

    fn lock(mutex: *Mutex) void {
        if (mutex.state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) return;
        while (mutex.state.swap(2, .acquire) != 0) futexWait(&mutex.state.raw, 2);
    }

    fn unlock(mutex: *Mutex) void {
        if (mutex.state.swap(0, .release) == 2) futexWake(&mutex.state.raw, 1);
    }
};

fn futexWait(address: *const u32, expected: u32) void {
    switch (linux.errno(linux.futex_4arg(address, .{ .cmd = .WAIT, .private = true }, expected, null))) {
        .SUCCESS, .INTR, .AGAIN => {},
        else => @panic("inference futex wait failed"),
    }
}

fn futexWake(address: *const u32, count: usize) void {
    // Linux interprets the wake limit as a signed int; INT_MAX wakes all
    // waiters without narrowing a caller-selected scheduler size to a negative value.
    switch (linux.errno(linux.futex_3arg(address, .{ .cmd = .WAKE, .private = true }, @intCast(@min(count, std.math.maxInt(i32)))))) {
        .SUCCESS => {},
        else => @panic("inference futex wake failed"),
    }
}

const Layout = struct {
    words_offset: usize,
    words_count: usize,
    assignments_offset: usize,
    size: usize,

    fn init(count: usize) InitError!Layout {
        if (count == 0) return error.InvalidConfig;
        const words_count = count / word_bits + @intFromBool(count % word_bits != 0);
        const words_offset = std.mem.alignForward(usize, @sizeOf(Scheduler), abi.memory_alignment);
        const words_size = std.math.mul(usize, words_count, @sizeOf(Word)) catch return error.MemorySizeOverflow;
        const assignments_offset = std.math.add(usize, words_offset, words_size) catch return error.MemorySizeOverflow;
        const assignments_size = std.math.mul(usize, count, @sizeOf(Assignment)) catch return error.MemorySizeOverflow;
        const end = std.math.add(usize, assignments_offset, assignments_size) catch return error.MemorySizeOverflow;
        const rounded = std.math.add(usize, end, abi.memory_alignment - 1) catch return error.MemorySizeOverflow;
        return .{
            .words_offset = words_offset,
            .words_count = words_count,
            .assignments_offset = assignments_offset,
            .size = rounded & ~(abi.memory_alignment - 1),
        };
    }
};

test "worker bitset handles word boundaries, sparse availability and queued cancellation" {
    const allocator = std.testing.allocator;
    const NoWork = struct {
        fn step(_: *anyopaque, _: Lane) Step {
            unreachable; // This test drives admission without starting workers.
        }
    };
    try std.testing.expectError(error.InvalidConfig, requiredMemory(0));
    try std.testing.expectError(error.MemorySizeOverflow, requiredMemory(std.math.maxInt(usize)));
    for ([_]usize{ 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1025 }) |count| {
        const size = try requiredMemory(count);
        const memory = try allocator.alignedAlloc(u8, .fromByteUnits(abi.memory_alignment), size + 64);
        defer allocator.free(memory);
        @memset(memory[size..], 0xa5);
        const scheduler: *Scheduler = @ptrCast(memory.ptr);
        try std.testing.expectError(error.MemoryTooSmall, scheduler.init(memory[0 .. size - 1], count));
        try scheduler.init(memory[0..size], count);
        try std.testing.expectEqual(count, scheduler.workersCount());
        var available: usize = 0;
        for (scheduler.idle_words) |*word| available += @popCount(word.load(.monotonic));
        try std.testing.expectEqual(count, available);

        var first: Operation = undefined;
        first = .{ .context = scheduler, .step = NoWork.step, .workers_count_max = count, .cancellation = null, .state = .{ .queued = &first } };
        scheduler.enqueue(&first);
        scheduler.schedule();
        try std.testing.expectEqual(count, first.state.running.count);
        try std.testing.expect(!scheduler.hasIdleWorkers());
        for (scheduler.assignments, 0..) |assignment, index| {
            try std.testing.expectEqual(&first, assignment.operation);
            try std.testing.expectEqual(index, assignment.lane_index);
        }

        // Release only the highest valid worker: every preceding word is empty.
        // Its assignment must still get lane zero in the newly admitted group.
        const last = count - 1;
        scheduler.idle_words[last / word_bits].store(@as(usize, 1) << @intCast(last % word_bits), .release);
        var second: Operation = undefined;
        second = .{ .context = scheduler, .step = NoWork.step, .workers_count_max = count, .cancellation = null, .state = .{ .queued = &second } };
        scheduler.enqueue(&second);
        scheduler.schedule();
        try std.testing.expectEqual(@as(usize, 1), second.state.running.count);
        try std.testing.expectEqual(&second, scheduler.assignments[last].operation);
        try std.testing.expectEqual(@as(usize, 0), scheduler.assignments[last].lane_index);
        try std.testing.expect(!scheduler.hasIdleWorkers());

        // A queued cancellation must complete even with no idle word set.
        var cancellation: std.atomic.Value(bool) = .init(true);
        var cancelled: Operation = undefined;
        cancelled = .{ .context = scheduler, .step = NoWork.step, .workers_count_max = count, .cancellation = &cancellation, .state = .{ .queued = &cancelled } };
        scheduler.enqueue(&cancelled);
        scheduler.schedule();
        try std.testing.expectEqual(.cancelled, cancelled.state.complete);
        try std.testing.expect(scheduler.queue_tail.load(.monotonic) == null);
        for (memory[size..]) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
    }
}

test "worker bitset dispatches across words with real threads" {
    const allocator = std.testing.allocator;
    const count = 65;
    const memory = try allocator.alignedAlloc(u8, .fromByteUnits(abi.memory_alignment), try requiredMemory(count));
    defer allocator.free(memory);
    const scheduler: *Scheduler = @ptrCast(memory.ptr);
    try scheduler.init(memory, count);
    var threads: [count]std.Thread = undefined;
    var started: usize = 0;
    defer {
        scheduler.requestStop();
        for (threads[0..started]) |thread| thread.join();
    }
    while (started < threads.len) : (started += 1) {
        threads[started] = try std.Thread.spawn(.{}, workerMain, .{ scheduler, started });
    }
    const Context = struct {
        seen: [count]bool = @splat(false),
        admitted: ?*std.atomic.Value(u32) = null,
        fn step(raw: *anyopaque, lane: Lane) Step {
            const context: *@This() = @ptrCast(@alignCast(raw));
            if (context.admitted) |admitted| {
                if (lane.isLeader() and admitted.fetchAdd(1, .acq_rel) == 1) futexWake(&admitted.raw, count);
                var value = admitted.load(.acquire);
                while (value < 2) : (value = admitted.load(.acquire)) futexWait(&admitted.raw, value);
            }
            context.seen[lane.index] = true;
            return .done;
        }

        fn run(pool_arg: *Scheduler, context: *@This(), width: usize) void {
            pool_arg.run(width, context, step, null) catch unreachable;
        }
    };
    // One collective checkpoint per call; avoid repeatedly spinning dozens of
    // oversubscribed threads just to test the second bitset word.
    for ([_]usize{ 1, 32, 33, 64, 65 }) |width| {
        var context: Context = .{};
        try scheduler.run(width, &context, Context.step, null);
        for (context.seen, 0..) |seen, index| try std.testing.expectEqual(index < width, seen);
    }
    // Park the first admitted group until both groups exist. Together they own
    // all 65 slots, so concurrent assignments necessarily cross the word edge.
    var admitted: std.atomic.Value(u32) = .init(0);
    var first_context: Context = .{ .admitted = &admitted };
    var second_context: Context = .{ .admitted = &admitted };
    const first = try std.Thread.spawn(.{}, Context.run, .{ scheduler, &first_context, 33 });
    const second = std.Thread.spawn(.{}, Context.run, .{ scheduler, &second_context, 32 }) catch |err| {
        admitted.store(2, .release);
        futexWake(&admitted.raw, count);
        first.join();
        return err;
    };
    first.join();
    second.join();
    for (first_context.seen, 0..) |seen, index| try std.testing.expectEqual(index < 33, seen);
    for (second_context.seen, 0..) |seen, index| try std.testing.expectEqual(index < 32, seen);
}

comptime {
    // The packed mask words end at an address suitable for assignment records.
    assert(@alignOf(Word) >= @alignOf(Assignment));
    assert(abi.memory_alignment >= @alignOf(Scheduler));
}

// Kernel checks own a local scheduler so callback functions and their callers
// are compiled together. Application inference uses the public WorkerPool API.
pub const TestPool = struct {
    scheduler: *Scheduler,
    memory: []align(abi.memory_alignment) u8,
    threads: []std.Thread,

    pub fn init(pool: *TestPool, count: usize) !void {
        const memory = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(abi.memory_alignment), try Scheduler.requiredMemory(count));
        errdefer std.testing.allocator.free(memory);
        const threads = try std.testing.allocator.alloc(std.Thread, count);
        errdefer std.testing.allocator.free(threads);
        const scheduler: *Scheduler = @ptrCast(memory.ptr);
        try scheduler.init(memory, count);
        pool.* = .{ .scheduler = scheduler, .memory = memory, .threads = threads };
        var started: usize = 0;
        errdefer {
            pool.scheduler.requestStop();
            for (pool.threads[0..started]) |thread| thread.join();
        }
        while (started < count) : (started += 1) {
            pool.threads[started] = try std.Thread.spawn(.{}, Scheduler.workerMain, .{ pool.scheduler, started });
        }
    }

    pub fn deinit(pool: *TestPool) void {
        pool.scheduler.requestStop();
        for (pool.threads) |thread| thread.join();
        std.testing.allocator.free(pool.threads);
        std.testing.allocator.free(pool.memory);
    }

    /// Runs the real collective path, including for single-worker references.
    /// Arguments are borrowed until all lanes finish; the lane is appended last.
    pub fn run(pool: *TestPool, comptime function: anytype, arguments: anytype) void {
        var context = arguments;
        const Callback = struct {
            fn step(raw: *anyopaque, lane: Lane) Step {
                const args: *@TypeOf(arguments) = @ptrCast(@alignCast(raw));
                @call(.auto, function, args.* ++ .{lane});
                return .done;
            }
        };
        pool.scheduler.run(pool.threads.len, &context, Callback.step, null) catch unreachable;
    }
};
