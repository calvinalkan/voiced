//! Defines the production timing policy that turns observed audio activity into
//! Whisper chunk and automatic-stop candidates. The activity detector owns only
//! signal classification; capture and supervisor code consume these constants
//! when they publish slots or end a `listen` recording.

const std = @import("std");
const audio_exchange = @import("audio_exchange.zig");
const assert = std.debug.assert;

/// Selects whether quiet may end capture. Both modes still publish internal
/// chunks at natural pauses and stop at the caller's absolute duration limit.
pub const AutomaticStop = enum(u8) {
    disabled,
    after_quiet,
};

pub const internal_chunk_duration_seconds_min: u32 = 20;
pub const natural_boundary_quiet_duration_ms: u32 = 300;
pub const automatic_stop_quiet_duration_ms: u32 = 800;
pub const internal_chunk_samples_count_min: u32 = internal_chunk_duration_seconds_min * audio_exchange.sample_rate_hz;
pub const natural_boundary_quiet_samples_count: u32 =
    natural_boundary_quiet_duration_ms * audio_exchange.sample_rate_hz /
    std.time.ms_per_s;
pub const automatic_stop_quiet_samples_count: u32 =
    automatic_stop_quiet_duration_ms * audio_exchange.sample_rate_hz /
    std.time.ms_per_s;

comptime {
    // Whisper consumes at most one 30-second feature window. The selected
    // minimum leaves ten seconds in which to find a natural boundary before
    // capture must force publication at the physical slot capacity.
    assert(audio_exchange.slot_duration_seconds_max == 30);
    assert(internal_chunk_duration_seconds_min > 0);
    assert(internal_chunk_duration_seconds_min < audio_exchange.slot_duration_seconds_max);
    assert(internal_chunk_samples_count_min < audio_exchange.slot_samples_capacity);

    // Internal chunking reacts before automatic stopping so a model can begin
    // work while `listen` continues confirming the same quiet interval.
    assert(natural_boundary_quiet_duration_ms > 0);
    assert(natural_boundary_quiet_duration_ms < automatic_stop_quiet_duration_ms);
    assert(natural_boundary_quiet_samples_count > 0);
    assert(natural_boundary_quiet_samples_count < automatic_stop_quiet_samples_count);
}
