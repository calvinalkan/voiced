//! A synchronous executor owns one persistent group of CPU workers. Every
//! submitted operation runs on all workers with a stable lane index; operations
//! go narrow explicitly instead of creating nested thread pools.

const std = @import("std");
const assert = std.debug.assert;
const Operation = *const fn (*anyopaque, Lane) void;

pub const workers_count_max: usize = 32;

pub const ExecutorError = error{
    InvalidWorkersCount,
    ThreadSpawnFailed,
};

/// `Lane` identifies one worker within the executor's permanent group. All
/// lanes executing one operation share its barrier.
pub const Lane = struct {
    index: usize,
    count: usize,
    barrier: *Barrier,

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

    pub fn sync(lane: Lane) void {
        assert(lane.index < lane.count);
        assert(lane.count == lane.barrier.lanes_count);

        lane.barrier.wait();
    }
};

pub const Range = struct {
    start_index: usize,
    end_index: usize,
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
    workers_completed_count: usize = 0,
    is_running_operation: bool = false,
    is_stopping: bool = false,
    barrier: Barrier = .{},

    pub fn init(executor: *Executor, io: std.Io, workers_count: usize) ExecutorError!void {
        if (workers_count == 0 or workers_count > workers_count_max) {
            return error.InvalidWorkersCount;
        }

        executor.* = .{ .io = io };
        executor.workers_count = workers_count;
        executor.barrier.init(workers_count);

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
        assert(!executor.is_running_operation);

        executor.is_stopping = true;
        executor.work_available.broadcast(executor.io);
        executor.mutex.unlock(executor.io);

        for (executor.threads[0..executor.workers_count]) |thread| {
            thread.join();
        }

        executor.* = undefined;
    }

    pub fn run(executor: *Executor, context: *anyopaque, operation: Operation) void {
        assert(executor.workers_count > 0);

        executor.mutex.lockUncancelable(executor.io);
        defer executor.mutex.unlock(executor.io);

        assert(!executor.is_stopping);
        assert(!executor.is_running_operation);
        assert(executor.workers_completed_count == 0);

        executor.operation = operation;
        executor.operation_context = context;
        executor.is_running_operation = true;
        executor.operation_generation +%= 1;
        executor.work_available.broadcast(executor.io);

        while (executor.workers_completed_count != executor.workers_count) {
            executor.work_completed.waitUncancelable(executor.io, &executor.mutex);
        }

        executor.workers_completed_count = 0;
        executor.operation = null;
        executor.operation_context = null;
        executor.is_running_operation = false;
    }
};

fn workerMain(executor: *Executor, worker_index: usize) void {
    assert(worker_index < executor.workers_count);

    var observed_operation_generation: u64 = 0;

    while (true) {
        executor.mutex.lockUncancelable(executor.io);
        while (!executor.is_stopping and observed_operation_generation == executor.operation_generation) {
            executor.work_available.waitUncancelable(executor.io, &executor.mutex);
        }
        if (executor.is_stopping) {
            executor.mutex.unlock(executor.io);
            return;
        }

        const operation = executor.operation.?;
        const operation_context = executor.operation_context.?;
        observed_operation_generation = executor.operation_generation;
        executor.mutex.unlock(executor.io);

        const lane: Lane = .{
            .index = worker_index,
            .count = executor.workers_count,
            .barrier = &executor.barrier,
        };
        operation(operation_context, lane);

        executor.mutex.lockUncancelable(executor.io);
        executor.workers_completed_count += 1;
        assert(executor.workers_completed_count <= executor.workers_count);
        if (executor.workers_completed_count == executor.workers_count) {
            executor.work_completed.signal(executor.io);
        }
        executor.mutex.unlock(executor.io);
    }
}

const Barrier = struct {
    lanes_count: usize = 0,
    arrivals_count: std.atomic.Value(usize) = .init(0),
    generation: std.atomic.Value(usize) = .init(0),

    fn init(barrier: *Barrier, lanes_count: usize) void {
        assert(lanes_count > 0);

        barrier.* = .{
            .lanes_count = lanes_count,
            .arrivals_count = .init(0),
            .generation = .init(0),
        };
    }

    fn wait(barrier: *Barrier) void {
        assert(barrier.lanes_count > 0);

        if (barrier.lanes_count == 1) {
            return;
        }

        const generation = barrier.generation.load(.monotonic);
        const previous_arrivals_count = barrier.arrivals_count.fetchAdd(1, .acq_rel);
        assert(previous_arrivals_count < barrier.lanes_count);

        if (previous_arrivals_count + 1 == barrier.lanes_count) {
            barrier.arrivals_count.store(0, .monotonic);
            barrier.generation.store(generation +% 1, .release);
            return;
        }

        while (barrier.generation.load(.acquire) == generation) {
            std.atomic.spinLoopHint();
        }
    }
};
