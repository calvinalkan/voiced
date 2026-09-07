//! Owns the fixed PCM exchange shared by the supervisor, audio process, and
//! transcription process. Audio privately fills one slot, release-publishes its
//! sample count, and never touches the slot again until the supervisor restores
//! that count to zero. The transcription process only reads supervisor-selected
//! published slots; it never changes audio-slot ownership.
//!
//! A slot needs no shared `writing` or `transcribing` state. The audio process
//! privately owns one `SlotWriter`, while the supervisor privately owns any
//! in-flight transcription. The shared count therefore has one meaning:
//! zero exposes no complete payload, and a positive value exposes that complete
//! immutable sample prefix.

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

pub const format_version: u32 = 8;

pub const TimelineValidation = enum(u32) {
    header_only = 1,
    full = 2,
};

pub const AudioExchange = extern struct {
    version: u32,
    timeline_validation_atomic: u32,
    session_id: u64,

    // These counters report callback liveness; slot publication alone may be
    // twenty or more seconds apart and cannot enforce the progress deadline.
    audio_callbacks_count_atomic: u64,
    audio_samples_count_atomic: u32,
    reserved_2: u32,

    slots: [slots_count]AudioSlot,
};

pub const AudioSlot = extern struct {
    // Audio writes every metadata field and the sample prefix, then
    // release-stores a positive count. The supervisor acquire-loads that count
    // before reading the immutable publication and restores zero only after its
    // transcript is retained.
    published_samples_count_atomic: u32,
    publication_ordinal: u32,
    contains_activity: u32,
    reserved: u32,

    // The extractor borrows normalized PCM directly. Keep the published prefix
    // immutable until result acceptance: a replacement model worker needs the
    // same audio even after the first worker has finished feature extraction.
    samples: [slot_samples_capacity]f32,
};

pub const SlotIndex = enum(u8) {
    slot_0,
    slot_1,
    slot_2,

    pub fn arrayIndex(index: SlotIndex) usize {
        return @intFromEnum(index);
    }

    pub fn fromArrayIndex(index: usize) SlotIndex {
        assert(index < slots_count);
        return @enumFromInt(index);
    }
};

pub const SlotWriter = struct {
    slot: *AudioSlot,
    publication_ordinal: u32,
};

pub const SlotPublication = struct {
    samples_count: u32,
    contains_activity: bool,
};

pub const PublishedSlot = struct {
    publication_ordinal: u32,
    samples_count: u32,
    contains_activity: bool,
};

pub const PublishedSlotRef = struct {
    index: SlotIndex,
    publication: PublishedSlot,
};

/// Shared-slot contents the other process published. `empty` is normal;
/// `corrupt` is a protocol failure, never a clamped prefix.
pub const AcquiredSlot = union(enum) {
    empty,
    published: PublishedSlot,
    corrupt,
};

pub const AcquiredTimelineValidation = union(enum) {
    unpublished,
    published: TimelineValidation,
    corrupt,
};

comptime {
    assert(sample_rate_hz > 0);
    assert(channels_count == 1);
    assert(slots_count > 1);
    assert(callback_samples_count_max > 0);
    assert(callback_samples_count_max <= slot_samples_capacity);

    assert(@offsetOf(AudioExchange, "timeline_validation_atomic") % @alignOf(u32) == 0);
    assert(@offsetOf(AudioExchange, "audio_callbacks_count_atomic") % @alignOf(u64) == 0);
    assert(@offsetOf(AudioExchange, "audio_samples_count_atomic") % @alignOf(u32) == 0);
    assert(@offsetOf(AudioExchange, "slots") % @alignOf(AudioSlot) == 0);
    assert(@offsetOf(AudioSlot, "published_samples_count_atomic") % @alignOf(u32) == 0);
    assert(@offsetOf(AudioSlot, "samples") % @alignOf(f32) == 0);
    assert(@sizeOf(AudioSlot) == 16 + slot_samples_capacity * @sizeOf(f32));
}

/// `initialize` assigns a fresh session while no capture process exists. A
/// retained model worker must have acknowledged its last result and await its
/// next command without touching the exchange. After cancellation, every old
/// worker must be reaped before this call.
pub fn initialize(exchange: *AudioExchange, session_id: u64) void {
    assert(session_id > 0);

    // Only published prefixes are readable. Clearing 5.49 MiB of PCM here
    // would touch every page on each start without exposing any useful data;
    // the next capture overwrites each prefix before publishing its count.
    exchange.version = format_version;
    exchange.timeline_validation_atomic = 0;
    exchange.session_id = session_id;
    exchange.audio_callbacks_count_atomic = 0;
    exchange.audio_samples_count_atomic = 0;
    exchange.reserved_2 = 0;
    for (&exchange.slots) |*slot| {
        slot.published_samples_count_atomic = 0;
        slot.publication_ordinal = 0;
        slot.contains_activity = 0;
        slot.reserved = 0;
    }

    assert(exchange.version == format_version);
    assert(exchange.timeline_validation_atomic == 0);
    assert(exchange.session_id == session_id);
    assert(exchange.audio_callbacks_count_atomic == 0);
    assert(exchange.audio_samples_count_atomic == 0);
    assert(exchange.reserved_2 == 0);

    for (&exchange.slots) |*slot| {
        assert(@atomicLoad(
            u32,
            &slot.published_samples_count_atomic,
            .acquire,
        ) == 0);
        assert(slot.publication_ordinal == 0);
        assert(slot.contains_activity == 0);
        assert(slot.reserved == 0);
    }
}

/// `publishTimelineValidation` announces the timeline validation performed by
/// the capture worker before its first callback heartbeat.
/// The supervisor reads it when capture first makes sample progress and emits
/// any degraded-mode warning once per worker.
pub fn publishTimelineValidation(
    exchange: *AudioExchange,
    validation: TimelineValidation,
) void {
    assert(exchange.version == format_version);
    assert(exchange.session_id > 0);
    assert(@atomicLoad(
        u32,
        &exchange.timeline_validation_atomic,
        .monotonic,
    ) == 0);

    @atomicStore(
        u32,
        &exchange.timeline_validation_atomic,
        @intFromEnum(validation),
        .release,
    );
}

pub fn acquireTimelineValidation(
    exchange: *const AudioExchange,
) AcquiredTimelineValidation {
    if (exchange.version != format_version or exchange.session_id == 0) {
        return .corrupt;
    }

    const encoded = @atomicLoad(
        u32,
        &exchange.timeline_validation_atomic,
        .acquire,
    );
    if (encoded == 0) {
        return .unpublished;
    }
    if (encoded == @intFromEnum(TimelineValidation.header_only)) {
        return .{ .published = .header_only };
    }
    if (encoded == @intFromEnum(TimelineValidation.full)) {
        return .{ .published = .full };
    }
    return .corrupt;
}

/// `publishAudioCallbacksCount` release-publishes callback progress separately
/// from slot boundaries. The count is diagnostic and authorizes no sample read.
pub fn publishAudioCallbacksCount(exchange: *AudioExchange, callbacks_count: u64) void {
    assert(exchange.version == format_version);
    assert(exchange.session_id > 0);
    assert(callbacks_count > 0);
    assert(callbacks_count == @atomicLoad(
        u64,
        &exchange.audio_callbacks_count_atomic,
        .monotonic,
    ) + 1);

    @atomicStore(
        u64,
        &exchange.audio_callbacks_count_atomic,
        callbacks_count,
        .release,
    );
}

pub fn acquireAudioCallbacksCount(exchange: *const AudioExchange) u64 {
    assert(exchange.version == format_version);
    assert(exchange.session_id > 0);

    return @atomicLoad(
        u64,
        &exchange.audio_callbacks_count_atomic,
        .acquire,
    );
}

/// `publishAudioSamplesCount` records complete callback blocks after audio has
/// copied them into its private active slot. Consumers still require a positive
/// slot publication count before reading PCM.
pub fn publishAudioSamplesCount(exchange: *AudioExchange, samples_count: u32) void {
    assert(exchange.version == format_version);
    assert(exchange.session_id > 0);

    const samples_count_previous = @atomicLoad(
        u32,
        &exchange.audio_samples_count_atomic,
        .monotonic,
    );
    assert(samples_count > samples_count_previous);
    assert(samples_count - samples_count_previous <= callback_samples_count_max);

    @atomicStore(
        u32,
        &exchange.audio_samples_count_atomic,
        samples_count,
        .release,
    );
}

pub fn acquireAudioSamplesCount(exchange: *const AudioExchange) u32 {
    assert(exchange.version == format_version);
    assert(exchange.session_id > 0);

    return @atomicLoad(
        u32,
        &exchange.audio_samples_count_atomic,
        .acquire,
    );
}

/// `tryAcquireWriter` returns the only value accepted by publication and
/// abandonment. This couples the selected physical slot and ordinal so ordinary
/// producer code cannot publish a different slot than the one it acquired.
pub fn tryAcquireWriter(
    exchange: *AudioExchange,
    index: SlotIndex,
    publication_ordinal: u32,
) ?SlotWriter {
    const slot = &exchange.slots[index.arrayIndex()];
    if (@atomicLoad(
        u32,
        &slot.published_samples_count_atomic,
        .acquire,
    ) != 0) {
        return null;
    }

    slot.publication_ordinal = publication_ordinal;

    assert(@atomicLoad(
        u32,
        &slot.published_samples_count_atomic,
        .monotonic,
    ) == 0);
    assert(slot.publication_ordinal == publication_ordinal);
    return .{
        .slot = slot,
        .publication_ordinal = publication_ordinal,
    };
}

/// `publishWrittenSlot` makes the complete metadata and sample prefix visible
/// to consumers. `samples_count` must be positive because zero means empty.
pub fn publishWrittenSlot(writer: SlotWriter, publication: SlotPublication) void {
    assert(writer.slot.publication_ordinal == writer.publication_ordinal);
    assert(publication.samples_count > 0);
    assert(publication.samples_count <= slot_samples_capacity);
    assert(@atomicLoad(
        u32,
        &writer.slot.published_samples_count_atomic,
        .monotonic,
    ) == 0);

    writer.slot.contains_activity = @intFromBool(publication.contains_activity);
    writer.slot.reserved = 0;

    @atomicStore(
        u32,
        &writer.slot.published_samples_count_atomic,
        publication.samples_count,
        .release,
    );
}

/// `acquireSlot` returns immutable publication metadata while the shared count
/// remains positive and in range. The caller may read that sample prefix until
/// the supervisor calls `releaseConsumedSlot` after retaining its transcript.
/// Out-of-range fields are `corrupt`; do not clamp them into a readable prefix.
pub fn acquireSlot(slot: *const AudioSlot) AcquiredSlot {
    const samples_count = @atomicLoad(
        u32,
        &slot.published_samples_count_atomic,
        .acquire,
    );
    if (samples_count == 0) {
        return .empty;
    }
    if (samples_count > slot_samples_capacity or
        slot.contains_activity > 1 or
        slot.reserved != 0)
    {
        return .corrupt;
    }
    return .{ .published = .{
        .publication_ordinal = slot.publication_ordinal,
        .samples_count = samples_count,
        .contains_activity = slot.contains_activity == 1,
    } };
}

/// `releaseConsumedSlot` returns a published slot to audio after no process can
/// read its old PCM. Only the supervisor performs this transition.
pub fn releaseConsumedSlot(slot: *AudioSlot) void {
    const samples_count = @atomicLoad(
        u32,
        &slot.published_samples_count_atomic,
        .monotonic,
    );
    assert(samples_count > 0);
    assert(samples_count <= slot_samples_capacity);
    assert(slot.contains_activity <= 1);
    assert(slot.reserved == 0);

    @atomicStore(
        u32,
        &slot.published_samples_count_atomic,
        0,
        .release,
    );
}

/// `abandonEmptyWrite` verifies that audio has exposed none of its private
/// partial bytes. No shared write is needed because zero already means that no
/// complete payload exists.
pub fn abandonEmptyWrite(writer: SlotWriter) void {
    assert(writer.slot.publication_ordinal == writer.publication_ordinal);
    assert(@atomicLoad(
        u32,
        &writer.slot.published_samples_count_atomic,
        .acquire,
    ) == 0);
}

test "acquireSlot rejects out-of-range publication fields" {
    const slot = try std.testing.allocator.create(AudioSlot);
    defer std.testing.allocator.destroy(slot);
    slot.* = std.mem.zeroes(AudioSlot);
    try std.testing.expectEqual(AcquiredSlot.empty, acquireSlot(slot));

    slot.published_samples_count_atomic = 1;
    slot.contains_activity = 1;
    switch (acquireSlot(slot)) {
        .published => |publication| {
            try std.testing.expectEqual(@as(u32, 1), publication.samples_count);
            try std.testing.expect(publication.contains_activity);
        },
        .empty, .corrupt => return error.UnexpectedSlotState,
    }

    slot.published_samples_count_atomic = slot_samples_capacity + 1;
    try std.testing.expectEqual(AcquiredSlot.corrupt, acquireSlot(slot));
    slot.published_samples_count_atomic = std.math.maxInt(u32);
    try std.testing.expectEqual(AcquiredSlot.corrupt, acquireSlot(slot));
    slot.published_samples_count_atomic = 1;
    slot.contains_activity = 2;
    try std.testing.expectEqual(AcquiredSlot.corrupt, acquireSlot(slot));
    slot.contains_activity = 1;
    slot.reserved = 1;
    try std.testing.expectEqual(AcquiredSlot.corrupt, acquireSlot(slot));
}

test "acquireTimelineValidation rejects unknown tags" {
    const exchange = try std.testing.allocator.create(AudioExchange);
    defer std.testing.allocator.destroy(exchange);
    initialize(exchange, 1);
    try std.testing.expectEqual(AcquiredTimelineValidation.unpublished, acquireTimelineValidation(exchange));

    publishTimelineValidation(exchange, .full);
    switch (acquireTimelineValidation(exchange)) {
        .published => |value| try std.testing.expectEqual(TimelineValidation.full, value),
        .unpublished, .corrupt => return error.UnexpectedTimelineState,
    }

    exchange.timeline_validation_atomic = 3;
    try std.testing.expectEqual(AcquiredTimelineValidation.corrupt, acquireTimelineValidation(exchange));
    exchange.timeline_validation_atomic = 1;
    exchange.version = format_version + 1;
    try std.testing.expectEqual(AcquiredTimelineValidation.corrupt, acquireTimelineValidation(exchange));
}
