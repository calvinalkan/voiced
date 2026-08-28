//! Defines the fixed audio memory shared by the supervisor, audio process, and
//! Whisper process. Audio owns a slot while writing it, release-publishes the
//! complete slot, and never touches it again until the supervisor has consumed
//! and released it. No shared value contains a process-local pointer.
//!
//! The exchange is a trusted internal contract: the supervisor creates and
//! initializes it, then passes it only to Voiced processes running the same
//! executable. Impossible state is therefore asserted as a programming defect.
//! External PipeWire formats and buffers are validated before they enter here.

const std = @import("std");
const assert = std.debug.assert;

pub const sample_rate_hz: u32 = 16_000;
pub const channels_count: u32 = 1;
pub const slots_count: u32 = 3;
pub const slot_duration_seconds_max: u32 = 30;
pub const slot_samples_capacity: u32 = sample_rate_hz * slot_duration_seconds_max;

// One callback may copy at most 100 ms of mono float32 audio. This is not a
// PipeWire format requirement: it is Voiced's explicit bound on work performed
// by the realtime thread. A larger block is rejected instead of turning an
// external buffer size into an unbounded realtime copy.
pub const callback_samples_count_max: u32 = sample_rate_hz / 10;

pub const format_version: u32 = 4;

pub const AudioExchange = extern struct {
    version: u32,
    reserved: u32,
    generation: u64,
    audio_callbacks_count: u64,
    audio_samples_count: u32,
    reserved_2: u32,
    slots: [slots_count]AudioSlot,
};

pub const AudioSlot = extern struct {
    state: u32,
    samples_count: u32,
    generation: u64,
    publication_ordinal: u32,
    reserved: u32,
    samples: [slot_samples_capacity]f32,
};

pub const PublishedSlot = struct {
    generation: u64,
    publication_ordinal: u32,
    samples_count: u32,
};

const SlotState = enum(u32) {
    available,
    audio_writing,
    published,
};

pub const slot_state_available = @intFromEnum(SlotState.available);

comptime {
    assert(sample_rate_hz > 0);
    assert(channels_count == 1);
    assert(slots_count > 1);
    assert(callback_samples_count_max > 0);
    assert(callback_samples_count_max <= slot_samples_capacity);

    assert(@offsetOf(AudioExchange, "audio_callbacks_count") % @alignOf(u64) == 0);
    assert(@offsetOf(AudioExchange, "audio_samples_count") % @alignOf(u32) == 0);
    assert(@offsetOf(AudioExchange, "slots") % @alignOf(AudioSlot) == 0);
    assert(@offsetOf(AudioSlot, "state") % @alignOf(u32) == 0);
    assert(@offsetOf(AudioSlot, "samples") % @alignOf(f32) == 0);
    assert(@sizeOf(AudioSlot) == @offsetOf(AudioSlot, "samples") + slot_samples_capacity * @sizeOf(f32));
}

/// Clears the exchange and assigns a new recording generation. The supervisor
/// must call this only while no audio or model process can access the mapping.
pub fn initialize(exchange: *AudioExchange, generation: u64) void {
    assert(generation > 0);

    @memset(std.mem.asBytes(exchange), 0);

    exchange.version = format_version;
    exchange.generation = generation;

    assert(exchange.version == format_version);
    assert(exchange.reserved == 0);
    assert(exchange.generation == generation);
    assert(exchange.audio_callbacks_count == 0);
    assert(exchange.audio_samples_count == 0);
    assert(exchange.reserved_2 == 0);

    for (&exchange.slots) |*slot| {
        assert(@atomicLoad(u32, &slot.state, .acquire) == @intFromEnum(SlotState.available));
        assert(slot.samples_count == 0);
        assert(slot.generation == 0);
        assert(slot.publication_ordinal == 0);
        assert(slot.reserved == 0);
    }
}

/// Publishes callback progress independently of slot boundaries. The supervisor
/// uses this heartbeat to distinguish a healthy long slot from a stream that
/// connected but never produced data. The count is diagnostic only: slot state,
/// not this value, authorizes access to samples.
pub fn publishAudioCallbacksCount(exchange: *AudioExchange, callbacks_count: u64) void {
    assert(exchange.version == format_version);
    assert(exchange.reserved == 0);
    assert(exchange.generation > 0);
    assert(exchange.reserved_2 == 0);
    assert(callbacks_count > 0);
    assert(callbacks_count == @atomicLoad(u64, &exchange.audio_callbacks_count, .monotonic) + 1);

    @atomicStore(u64, &exchange.audio_callbacks_count, callbacks_count, .release);
}

/// Acquires the latest callback heartbeat. A zero value means the audio process
/// has not entered its process callback during this exchange generation.
pub fn acquireAudioCallbacksCount(exchange: *const AudioExchange) u64 {
    assert(exchange.version == format_version);
    assert(exchange.reserved == 0);
    assert(exchange.generation > 0);
    assert(exchange.reserved_2 == 0);

    return @atomicLoad(u64, &exchange.audio_callbacks_count, .acquire);
}

/// Publishes actual sample progress after a complete callback block has entered
/// a producer-owned slot. Unlike the callback heartbeat, this count does not
/// advance when PipeWire schedules the process function without a buffer.
pub fn publishAudioSamplesCount(exchange: *AudioExchange, samples_count: u32) void {
    assert(exchange.version == format_version);
    assert(exchange.reserved == 0);
    assert(exchange.generation > 0);
    assert(exchange.reserved_2 == 0);

    const samples_count_previous =
        @atomicLoad(u32, &exchange.audio_samples_count, .monotonic);
    assert(samples_count > samples_count_previous);
    assert(samples_count - samples_count_previous <= callback_samples_count_max);

    @atomicStore(u32, &exchange.audio_samples_count, samples_count, .release);
}

/// Acquires the total samples copied during this generation. This is a liveness
/// signal only; consumers still need a `published` slot before reading samples.
pub fn acquireAudioSamplesCount(exchange: *const AudioExchange) u32 {
    assert(exchange.version == format_version);
    assert(exchange.reserved == 0);
    assert(exchange.generation > 0);
    assert(exchange.reserved_2 == 0);

    return @atomicLoad(u32, &exchange.audio_samples_count, .acquire);
}

/// Transfers one available slot to the audio producer. Returning `false` means
/// the consumer still owns every physical slot, so audio must apply its pipeline
/// pressure policy rather than overwrite unread samples.
pub fn tryBeginWrite(
    slot: *AudioSlot,
    generation: u64,
    publication_ordinal: u32,
) bool {
    assert(generation > 0);

    const state = @atomicLoad(u32, &slot.state, .monotonic);

    switch (state) {
        @intFromEnum(SlotState.available),
        @intFromEnum(SlotState.audio_writing),
        @intFromEnum(SlotState.published),
        => {},
        else => unreachable,
    }

    assert(slot.reserved == 0);

    if (@cmpxchgStrong(
        u32,
        &slot.state,
        @intFromEnum(SlotState.available),
        @intFromEnum(SlotState.audio_writing),
        .acquire,
        .monotonic,
    ) != null) {
        return false;
    }

    slot.samples_count = 0;
    slot.generation = generation;
    slot.publication_ordinal = publication_ordinal;

    assert(@atomicLoad(u32, &slot.state, .monotonic) == @intFromEnum(SlotState.audio_writing));
    assert(slot.samples_count == 0);
    assert(slot.generation == generation);
    assert(slot.publication_ordinal == publication_ordinal);

    return true;
}

/// Publishes the complete prefix written by audio. The release-store makes the
/// metadata and every preceding sample write visible to the consumer that
/// acquire-loads the `published` state.
pub fn publishWrittenSlot(slot: *AudioSlot, samples_count: u32) void {
    assert(samples_count > 0);
    assert(samples_count <= slot_samples_capacity);
    assert(@atomicLoad(u32, &slot.state, .monotonic) == @intFromEnum(SlotState.audio_writing));
    assert(slot.samples_count == 0);
    assert(slot.generation > 0);
    assert(slot.reserved == 0);

    slot.samples_count = samples_count;
    @atomicStore(
        u32,
        &slot.state,
        @intFromEnum(SlotState.published),
        .release,
    );
}

/// Returns metadata for a complete published prefix, or `null` while audio owns
/// the slot or the slot is available. The slot is written only by trusted
/// Voiced processes, so impossible state or metadata is an assertion failure.
/// The caller may read the returned prefix until `releaseConsumedSlot`.
pub fn acquirePublishedSlot(slot: *const AudioSlot) ?PublishedSlot {
    const state = @atomicLoad(u32, &slot.state, .acquire);
    switch (state) {
        @intFromEnum(SlotState.available),
        @intFromEnum(SlotState.audio_writing),
        @intFromEnum(SlotState.published),
        => {},
        else => unreachable,
    }
    assert(slot.reserved == 0);

    if (state != @intFromEnum(SlotState.published)) return null;

    assert(slot.samples_count > 0);
    assert(slot.samples_count <= slot_samples_capacity);
    assert(slot.generation > 0);

    return .{
        .generation = slot.generation,
        .publication_ordinal = slot.publication_ordinal,
        .samples_count = slot.samples_count,
    };
}

/// Returns a consumed slot to audio. The producer's publication and the
/// consumer's release form the two sides of the ownership contract: only a
/// published slot may become available again.
pub fn releaseConsumedSlot(slot: *AudioSlot) void {
    assert(@atomicLoad(u32, &slot.state, .monotonic) ==
        @intFromEnum(SlotState.published));
    assert(slot.samples_count > 0);
    assert(slot.samples_count <= slot_samples_capacity);
    assert(slot.generation > 0);
    assert(slot.reserved == 0);

    @atomicStore(
        u32,
        &slot.state,
        @intFromEnum(SlotState.available),
        .release,
    );
}

/// Returns an empty producer-owned slot without publishing it. This is used
/// when capture ends before the first sample reaches a newly claimed slot.
pub fn abandonEmptyWrite(slot: *AudioSlot) void {
    assert(@atomicLoad(u32, &slot.state, .monotonic) ==
        @intFromEnum(SlotState.audio_writing));
    assert(slot.samples_count == 0);
    assert(slot.generation > 0);
    assert(slot.reserved == 0);

    @atomicStore(
        u32,
        &slot.state,
        @intFromEnum(SlotState.available),
        .release,
    );
}
