//! Clipboard backends retain each publication identity while its source is
//! offered or a transfer survives replacement. The storage index controls
//! transcript-buffer borrowing; the recording ID preserves later log attribution.

pub const StorageIndex = u1;

pub const PublicationId = struct {
    storage_index: StorageIndex,
    recording_id: u64,

    pub fn eql(first: PublicationId, second: PublicationId) bool {
        return first.storage_index == second.storage_index and first.recording_id == second.recording_id;
    }
};

pub const TextTransfer = struct {
    publication: PublicationId,
    text_size: usize,
    started_monotonic_ns: u64,
    completed_monotonic_ns: u64,
};
