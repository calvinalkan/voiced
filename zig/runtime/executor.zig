//! A synchronous executor owns one persistent group of CPU workers. Operations
//! use the whole pool or a prefix with stable lane indices; inactive members
//! park while the selected group cooperates through its own barrier and tiles.

const std = @import("std");
const assert = std.debug.assert;
const Operation = *const fn (*anyopaque, Lane) void;

pub const workers_count_max: usize = 32;

pub const ExecutorError = error{
    InvalidWorkersCount,
    ThreadSpawnFailed,
};

/// `Lane` identifies one active worker in the current operation. All active
/// lanes use the same count and barrier; indices remain stable for a prefix.
pub const Lane = struct {
    index: usize,
    count: usize,
    barrier: *Barrier,
    tile_cursor: ?*std.atomic.Value(usize) = null,

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
    /// the cursor is shared between operations. Executor lanes synchronize here
    /// to publish the reset, then claim chunks as they finish. Standalone lanes
    /// without a cursor retain static partitioning and do not synchronize.
    pub fn tiles(lane: Lane, tiles_count: usize, claim_count_max: usize) TileIterator {
        assert(claim_count_max > 0);
        if (lane.tile_cursor) |cursor| {
            // Large inputs amortize the atomic claim across several tiles.
            // Short inputs need several reservations per lane so a slow core
            // cannot hold one oversized final chunk after fast lanes run dry.
            const claim_count = @min(claim_count_max, @max(1, tiles_count / (lane.count * 4)));
            if (lane.isLeader()) cursor.store(0, .monotonic);
            lane.sync();
            return .{ .cursor = cursor, .tiles_count = tiles_count, .claim_count = claim_count };
        }
        const indices = lane.range(tiles_count);
        return .{ .next_index = indices.start_index, .end_index = indices.end_index };
    }

    pub fn sync(lane: Lane) void {
        assert(lane.index < lane.count);

        lane.barrier.wait(lane.count);
    }
};

pub const Range = struct {
    start_index: usize,
    end_index: usize,
};

pub const TileIterator = struct {
    cursor: ?*std.atomic.Value(usize) = null,
    tiles_count: usize = 0,
    claim_count: usize = 0,
    next_index: usize = 0,
    end_index: usize = 0,

    pub fn next(tiles: *TileIterator) ?usize {
        if (tiles.next_index == tiles.end_index) {
            const cursor = tiles.cursor orelse {
                return null;
            };
            const begin = cursor.fetchAdd(tiles.claim_count, .monotonic);
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

/// `Executor` starts `workers_count` threads during `init` and joins them during
/// `deinit`. The supplied `io` must remain valid for that lifetime. Do not move
/// an initialized executor because its workers retain its address. `run`
/// accepts one operation at a time and returns after every lane exits it.
pub const Executor = struct {
    io: std.Io = undefined,
    mutex: std.Io.Mutex = .init,
    work_available: std.Io.Condition = .init,
    work_completed: std.Io.Condition = .init,
    threads: [workers_count_max]std.Thread = undefined,
    workers_count: usize = 0,
    operation: ?Operation = null,
    operation_context: ?*anyopaque = null,
    operation_generation: u64 = 0,
    operation_workers_count: usize = 0,
    workers_completed_count: usize = 0,
    is_stopping: bool = false,
    barrier: Barrier = .{},
    tile_cursor: std.atomic.Value(usize) align(64) = .init(0),

    pub fn init(executor: *Executor, io: std.Io, workers_count: usize) ExecutorError!void {
        if (workers_count == 0 or workers_count > workers_count_max) {
            return error.InvalidWorkersCount;
        }

        executor.* = .{ .io = io };
        executor.workers_count = workers_count;

        var workers_started_count: usize = 0;

        errdefer {
            executor.mutex.lockUncancelable(executor.io);
            executor.is_stopping = true;
            executor.work_available.broadcast(executor.io);
            executor.mutex.unlock(executor.io);

            for (executor.threads[0..workers_started_count]) |thread| {
                thread.join();
            }

            executor.* = undefined;
        }

        while (workers_started_count < workers_count) : (workers_started_count += 1) {
            executor.threads[workers_started_count] = std.Thread.spawn(.{}, workerMain, .{ executor, workers_started_count }) catch {
                return error.ThreadSpawnFailed;
            };
        }
    }

    pub fn deinit(executor: *Executor) void {
        assert(executor.workers_count > 0);
        assert(executor.workers_count <= workers_count_max);

        executor.mutex.lockUncancelable(executor.io);
        assert(executor.operation == null);

        executor.is_stopping = true;
        executor.work_available.broadcast(executor.io);
        executor.mutex.unlock(executor.io);

        for (executor.threads[0..executor.workers_count]) |thread| {
            thread.join();
        }

        executor.* = undefined;
    }

    pub fn run(executor: *Executor, context: *anyopaque, operation: Operation) void {
        executor.runWithWorkers(executor.workers_count, context, operation);
    }

    /// `runWithWorkers` runs on the first `workers_count` members of the pool.
    /// The count must be positive and no greater than `executor.workers_count`.
    /// Other members park instead of joining its barriers. Calls cannot overlap.
    pub fn runWithWorkers(executor: *Executor, workers_count: usize, context: *anyopaque, operation: Operation) void {
        assert(workers_count > 0 and workers_count <= executor.workers_count);

        executor.mutex.lockUncancelable(executor.io);
        defer executor.mutex.unlock(executor.io);

        assert(!executor.is_stopping);
        assert(executor.operation == null);
        assert(executor.workers_completed_count == 0);

        // The preceding run joined every active lane, so no worker retains a
        // reference to an in-progress barrier generation when its width changes.
        executor.operation_workers_count = workers_count;
        executor.operation = operation;
        executor.operation_context = context;
        executor.operation_generation +%= 1;
        executor.work_available.broadcast(executor.io);

        while (executor.workers_completed_count != workers_count) {
            executor.work_completed.waitUncancelable(executor.io, &executor.mutex);
        }

        executor.workers_completed_count = 0;
        executor.operation_workers_count = 0;
        executor.operation = null;
        executor.operation_context = null;
    }
};

fn workerMain(executor: *Executor, worker_index: usize) void {
    assert(worker_index < executor.workers_count);

    var observed_operation_generation: u64 = 0;
    executor.mutex.lockUncancelable(executor.io);
    defer executor.mutex.unlock(executor.io);

    while (true) {
        while (!executor.is_stopping and observed_operation_generation == executor.operation_generation) {
            executor.work_available.waitUncancelable(executor.io, &executor.mutex);
        }
        if (executor.is_stopping) {
            return;
        }

        observed_operation_generation = executor.operation_generation;
        const workers_count = executor.operation_workers_count;
        if (worker_index >= workers_count) {
            // A parked worker can wake after a narrow run has already finished
            // and cleared its operation. A zero active count covers that case.
            continue;
        }
        const operation = executor.operation.?;
        const operation_context = executor.operation_context.?;
        executor.mutex.unlock(executor.io);

        const lane: Lane = .{
            .index = worker_index,
            .count = workers_count,
            .barrier = &executor.barrier,
            .tile_cursor = &executor.tile_cursor,
        };
        operation(operation_context, lane);

        executor.mutex.lockUncancelable(executor.io);
        executor.workers_completed_count += 1;
        assert(executor.workers_completed_count <= workers_count);
        if (executor.workers_completed_count == workers_count) {
            executor.work_completed.signal(executor.io);
        }
    }
}

const Barrier = struct {
    arrivals_count: std.atomic.Value(usize) = .init(0),
    generation: std.atomic.Value(usize) = .init(0),

    fn wait(barrier: *Barrier, lanes_count: usize) void {
        assert(lanes_count > 0);

        if (lanes_count == 1) {
            return;
        }

        const generation = barrier.generation.load(.monotonic);
        const previous_arrivals_count = barrier.arrivals_count.fetchAdd(1, .acq_rel);
        assert(previous_arrivals_count < lanes_count);

        if (previous_arrivals_count + 1 == lanes_count) {
            barrier.arrivals_count.store(0, .monotonic);
            barrier.generation.store(generation +% 1, .release);
            return;
        }

        while (barrier.generation.load(.acquire) == generation) {
            std.atomic.spinLoopHint();
        }
    }
};

const TileTraversalCheck = struct {
    workers_count: usize,
    tiles_count: usize,
    claim_count: usize,
    visits: [257]std.atomic.Value(usize) = @splat(.init(0)),
    worker_mask: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
};

fn checkTileTraversal(raw_context: *anyopaque, lane: Lane) void {
    const context: *TileTraversalCheck = @ptrCast(@alignCast(raw_context));

    if (lane.count != context.workers_count) context.failed.store(true, .release);
    _ = context.worker_mask.fetchOr(@as(u32, 1) << @intCast(lane.index), .monotonic);

    for (0..3) |_| {
        var tiles = lane.tiles(context.tiles_count, context.claim_count);
        while (tiles.next()) |index| {
            _ = context.visits[index].fetchAdd(1, .monotonic);
        }
        lane.sync();
    }
}

test "narrow runs and collective tiles reuse the pool without omissions" {
    var executor: Executor = undefined;
    try executor.init(std.testing.io, 16);
    defer executor.deinit();

    const widths = [_]usize{ 1, 16, 3, 8, 1, 4, 16, 2 };
    const sizes = [_]usize{ 0, 1, 2, 31, 257 };

    for (0..64) |iteration| {
        var context: TileTraversalCheck = .{
            .workers_count = widths[iteration % widths.len],
            .tiles_count = sizes[iteration % sizes.len],
            .claim_count = 1 + iteration % 7,
        };

        executor.runWithWorkers(context.workers_count, &context, checkTileTraversal);

        try std.testing.expect(!context.failed.load(.acquire));
        try std.testing.expectEqual((@as(u32, 1) << @intCast(context.workers_count)) - 1, context.worker_mask.load(.acquire));

        for (&context.visits, 0..) |*visits, index| {
            try std.testing.expectEqual(@as(usize, if (index < context.tiles_count) 3 else 0), visits.load(.acquire));
        }
    }
}
