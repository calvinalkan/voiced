//! Defines the production timing policy that turns observed audio activity into
//! Whisper chunk boundaries. The activity detector owns signal classification;
//! capture owns publication policy.

const std = @import("std");
const AudioExchange = @import("../audio_exchange.zig");
const assert = std.debug.assert;

pub const internal_chunk_duration_seconds_min: u32 = 20;
pub const natural_boundary_quiet_duration_ms: u32 = 300;
pub const internal_chunk_samples_count_min: u32 = internal_chunk_duration_seconds_min * AudioExchange.sample_rate_hz;
pub const natural_boundary_quiet_samples_count: u32 =
    natural_boundary_quiet_duration_ms * AudioExchange.sample_rate_hz /
    std.time.ms_per_s;

comptime {
    // Whisper consumes at most one 30-second feature window. The selected
    // minimum leaves ten seconds in which to find a natural boundary before
    // capture must force publication at the physical slot capacity.
    assert(AudioExchange.slot_duration_seconds_max == 30);
    assert(internal_chunk_duration_seconds_min > 0);
    assert(internal_chunk_duration_seconds_min < AudioExchange.slot_duration_seconds_max);
    assert(internal_chunk_samples_count_min < AudioExchange.slot_samples_capacity);

    assert(natural_boundary_quiet_duration_ms > 0);
    assert(natural_boundary_quiet_samples_count > 0);
}
