//! The result of one inference call, shared by the application facade and LLVM object.

/// `TranscriptionResult` describes one completed transcription call.
/// Callers handle success, limits, cancellation, and errors with `switch`.
///
/// Returning a result allocates no memory. All worker references to the
/// call's storage have been released before `transcribe` returns.
pub const TranscriptionResult = union(enum) {
    /// `ok` means the model emitted end-of-text and produced valid UTF-8.
    /// The text can be empty.
    ok: Output,

    /// `token_limit` means generation reached the configured token capacity.
    /// Text is valid UTF-8, but can end mid-word or mid-sentence.
    ///
    /// If generation stopped inside a UTF-8 character, `text` excludes
    /// that incomplete trailing character. `tokens` retains every
    /// generated token, including those contributing the omitted bytes.
    token_limit: Output,

    /// `invalid_transcript_encoding` preserves output containing malformed
    /// UTF-8. An incomplete trailing character at the token limit is handled
    /// by `.token_limit`; other encoding errors arrive here.
    ///
    /// `text` preserves the original assembled bytes. Treat them as raw
    /// bytes until validated or repaired.
    ///
    /// Generation reached capacity if `tokens.len` equals the runtime's
    /// configured `transcript_tokens_count_max`; otherwise the model
    /// emitted end-of-text.
    invalid_transcript_encoding: Output,

    /// `cancelled` means the runtime observed the cancellation flag.
    /// Timings retain work already performed, including elapsed time
    /// in an interrupted phase. No transcript is published.
    cancelled: Timings,

    /// `audio_too_short` means the input contains fewer than 201 samples.
    /// The runtime accepts mono audio sampled at 16,000 Hz.
    audio_too_short,

    /// `audio_duration_exceeds_limit` means the supplied sample count
    /// exceeds `Config.audio_samples_count_max`, excluding added padding.
    audio_duration_exceeds_limit,

    /// `encoder_padding_exceeds_limit` means the requested padding
    /// exceeds `Config.encoder_padding_max`.
    encoder_padding_exceeds_limit,

    /// `invalid_worker_count` means a resolved encoder or decoder worker
    /// maximum is zero or exceeds the bound pool's capacity.
    /// Hardware core count does not impose this limit.
    invalid_worker_count,

    /// `invalid_samples` means a sample is NaN, infinite, or outside
    /// the inclusive normalized range [-1, 1]. Samples are not clipped.
    invalid_samples,

    /// `Output` contains generated text, tokens, and decoding diagnostics.
    ///
    /// Its slices borrow runtime storage until the next transcription
    /// begins or the runtime is deinitialized. Copy their contents before
    /// then if they must survive. Copying `Output` alone copies slice
    /// descriptors and scalar metadata, not the underlying bytes or tokens.
    pub const Output = struct {
        /// `text` contains assembled token bytes, without a terminating NUL.
        /// Leading spaces are preserved; special tokens contribute no text.
        ///
        /// `.ok` and `.token_limit` guarantee valid UTF-8.
        /// `.invalid_transcript_encoding` preserves malformed bytes.
        text: []const u8,

        /// `tokens` contains every generated model-vocabulary ID in order.
        /// It excludes the initial prompt and terminating end-of-text token.
        /// Its length does not exceed `Config.transcript_tokens_count_max`.
        ///
        /// Tokens remain intact when an incomplete trailing character is
        /// omitted from `text`. Together with the same model vocabulary,
        /// they allow reconstruction of the original assembled bytes.
        tokens: []const u32,

        /// `encoder_positions_count` is the actual sequence length processed
        /// by encoder attention, including trailing padding and alignment.
        /// It counts encoder positions, not audio samples or Mel frames.
        encoder_positions_count: usize,

        /// `no_speech_probability` is the model's no-speech-token probability
        /// after processing the initial start token, in the range [0, 1].
        /// The runtime applies no speech-rejection threshold to this value.
        no_speech_probability: f32,

        /// `average_log_probability` is the mean natural-log probability
        /// of selected generated tokens after token suppression.
        /// It includes end-of-text when selected, but excludes the prompt.
        /// Values are nonpositive; values nearer zero indicate greater
        /// model confidence in its selections.
        average_log_probability: f32,

        /// `timings` preserves compute-phase durations, including when
        /// generation reaches capacity or text encoding fails.
        timings: Timings,
    };

    /// `Timings` contains monotonic wall-clock durations in nanoseconds,
    /// including scheduling and waiting within each phase.
    /// These values are not summed CPU time across workers.
    ///
    /// Unstarted phases remain zero. Cancellation retains elapsed time
    /// in the interrupted phase. Validation, model loading, and conversion
    /// of tokens to text are excluded; the sum is not total call latency.
    pub const Timings = struct {
        /// `log_mel_ns` measures feature extraction, including construction
        /// of the padded log-Mel input.
        log_mel_ns: u64,

        /// `encoder_ns` measures audio encoding, including worker dispatch
        /// and waiting for completion.
        encoder_ns: u64,

        /// `cross_key_values_ns` measures preparation of decoder
        /// cross-attention keys and values from the encoded audio,
        /// including scheduling and waiting within that phase.
        cross_key_values_ns: u64,

        /// `decoder_ns` measures prompt processing and token generation,
        /// including scheduling and synchronization. It excludes token
        /// conversion to text and UTF-8 validation.
        decoder_ns: u64,
    };
};
