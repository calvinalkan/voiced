//! Owns the fixed PCM exchange shared by the supervisor, capture thread, and
//! transcription thread. Audio privately fills the next physical slot,
//! release-publishes its sample count, and never touches it again until the
//! supervisor restores that count to zero. The transcription thread only reads
//! the supervisor-selected slot and never changes ownership.
//!
//! The ring is strictly ordered. Publication N always occupies N modulo three,
//! so neither slots nor messages store a second copy of the ordinal. A slot
//! needs no shared writing or transcribing state: zero samples means private or
//! empty, and a positive count exposes one complete immutable sample prefix.
const AudioExchange = @This();

const std = @import("std");
const assert = std.debug.assert;

pub const sample_rate_hz: u32 = 16_000;
pub const channels_count: u32 = 1;
pub const slots_count: u32 = 3;
pub const slot_duration_seconds_max: u32 = 30;
pub const slot_samples_capacity: u32 = sample_rate_hz * slot_duration_seconds_max;

// One callback may publish at most 100 ms of mono audio. This is not a
// PipeWire format requirement: it bounds realtime validation and conversion.
pub const callback_samples_count_max: u32 = sample_rate_hz / 10;

// Callback liveness is separate from slot publication because one slot may
// remain private for twenty seconds.
audio_callbacks_count: std.atomic.Value(u64),
slots: [slots_count]AudioSlot,

pub const AudioSlot = struct {
    // Audio writes metadata and the sample prefix before release-storing a
    // positive count. The supervisor restores zero only after transcription.
    published_samples_count: std.atomic.Value(u32),
    contains_activity: bool,
    samples: [slot_samples_capacity]f32,
};

pub const SlotIndex = enum(u8) {
    slot_0,
    slot_1,
    slot_2,

    pub fn arrayIndex(index: SlotIndex) usize {
        return @intFromEnum(index);
    }

    pub fn fromPublicationOrdinal(ordinal: u32) SlotIndex {
        return @enumFromInt(ordinal % slots_count);
    }
};

pub const SlotWriter = struct { slot: *AudioSlot };

pub const SlotPublication = struct {
    samples_count: u32,
    contains_activity: bool,
};

pub const PublishedSlot = struct {
    samples_count: u32,
    contains_activity: bool,
};

comptime {
    assert(sample_rate_hz > 0);
    assert(channels_count == 1);
    assert(slots_count > 1);
    assert(callback_samples_count_max > 0);
    assert(callback_samples_count_max <= slot_samples_capacity);
    assert(@offsetOf(AudioExchange, "audio_callbacks_count") % @alignOf(std.atomic.Value(u64)) == 0);
    assert(@offsetOf(AudioExchange, "slots") % @alignOf(AudioSlot) == 0);
    assert(@offsetOf(AudioSlot, "published_samples_count") % @alignOf(std.atomic.Value(u32)) == 0);
    assert(@offsetOf(AudioSlot, "samples") % @alignOf(f32) == 0);
}

/// Reset only ownership metadata after both workers acknowledge completion.
/// Clearing 5.49 MiB of PCM would expose no useful data; capture overwrites
/// every prefix before publishing its count.
pub fn initialize(exchange: *AudioExchange) void {
    exchange.audio_callbacks_count = .init(0);
    for (&exchange.slots) |*slot| slot.published_samples_count = .init(0);
}

/// Publish callback progress separately from slot boundaries. The count is
/// diagnostic and authorizes no sample read.
pub fn publishAudioCallbacksCount(exchange: *AudioExchange, callbacks_count: u64) void {
    assert(callbacks_count > 0);
    assert(callbacks_count == exchange.audio_callbacks_count.load(.monotonic) + 1);
    exchange.audio_callbacks_count.store(callbacks_count, .release);
}

pub fn acquireAudioCallbacksCount(exchange: *const AudioExchange) u64 {
    return exchange.audio_callbacks_count.load(.acquire);
}

/// Claim an empty slot. Only the capture thread calls this; the acquire pairs
/// with the supervisor's release after the previous transcription completes.
pub fn tryAcquireWriter(slot: *AudioSlot) ?SlotWriter {
    if (slot.published_samples_count.load(.acquire) != 0) return null;
    return .{ .slot = slot };
}

/// Make the complete metadata and sample prefix visible. A positive sample
/// count is the publication flag; zero always means empty or privately owned.
pub fn publishWrittenSlot(writer: SlotWriter, publication: SlotPublication) void {
    assert(publication.samples_count > 0);
    assert(publication.samples_count <= slot_samples_capacity);
    assert(writer.slot.published_samples_count.load(.monotonic) == 0);
    writer.slot.contains_activity = publication.contains_activity;
    writer.slot.published_samples_count.store(publication.samples_count, .release);
}

/// Return immutable publication metadata. The positive count keeps the sample
/// prefix borrowed until the supervisor calls `releaseConsumedSlot`.
pub fn acquireSlot(slot: *const AudioSlot) ?PublishedSlot {
    const samples_count = slot.published_samples_count.load(.acquire);
    if (samples_count == 0) return null;
    assert(samples_count <= slot_samples_capacity);
    return .{
        .samples_count = samples_count,
        .contains_activity = slot.contains_activity,
    };
}

/// Return a published slot to capture after neither consumer can read its PCM.
/// Only the supervisor performs this transition.
pub fn releaseConsumedSlot(slot: *AudioSlot) void {
    const samples_count = slot.published_samples_count.load(.monotonic);
    assert(samples_count > 0);
    assert(samples_count <= slot_samples_capacity);
    slot.published_samples_count.store(0, .release);
}

/// Verify that capture exposed none of its private partial bytes.
pub fn abandonEmptyWrite(writer: SlotWriter) void {
    assert(writer.slot.published_samples_count.load(.acquire) == 0);
}

test "slot publication exposes metadata until release" {
    const slot = try std.testing.allocator.create(AudioSlot);
    defer std.testing.allocator.destroy(slot);

    slot.published_samples_count = .init(0);
    try std.testing.expectEqual(null, acquireSlot(slot));

    const writer = tryAcquireWriter(slot).?;
    publishWrittenSlot(writer, .{ .samples_count = 1, .contains_activity = true });

    const publication = acquireSlot(slot).?;

    try std.testing.expectEqual(@as(u32, 1), publication.samples_count);
    try std.testing.expect(publication.contains_activity);

    releaseConsumedSlot(slot);

    try std.testing.expectEqual(null, acquireSlot(slot));
}
