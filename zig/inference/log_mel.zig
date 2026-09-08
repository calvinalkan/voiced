//! Whisper log-Mel extraction borrows immutable DFT/Mel coefficient tables and
//! a separate workspace supplied for each calculation. It accepts normalized 16 kHz
//! mono samples and returns a row-major `[80, frames_count]` feature matrix
//! whose logical length includes the per-transcription trailing-silence policy.

const std = @import("std");
const memory_layout = @import("memory_layout.zig");
const assert = std.debug.assert;

const fft_samples_count: usize = 400;

pub const sample_rate_hz: usize = 16_000;
pub const mel_bins_count: usize = 80;
pub const encoder_frames_count_max: usize = 3000;
pub const samples_count_min: usize = fft_samples_count / 2 + 1;
pub const samples_count_max: usize = 30 * sample_rate_hz;

const fft_bins_count: usize = fft_samples_count / 2 + 1;
const hop_samples_count: usize = 160;
const frames_per_second: usize = sample_rate_hz / hop_samples_count;
const center_padding_samples_count: usize = fft_samples_count / 2;
const simd_lanes_count: usize = 8;
const encoder_features_offset: usize = std.mem.alignForward(usize, fft_bins_count, memory_layout.alignment / @sizeOf(f32));
// A nonempty triangular filter contributes one block, then one more at each
// crossed block boundary. At most two filters cross any internal boundary.
const mel_filter_coefficients_capacity: usize = (mel_bins_count + 2 * ((fft_bins_count - 1) / simd_lanes_count - 1)) * simd_lanes_count;

/// `workspace_values_count` covers one spectrum and the largest feature matrix.
/// The caller may reuse this workspace after consuming the returned features.
pub const workspace_values_count: usize = encoder_features_offset + mel_bins_count * encoder_frames_count_max;

const F32x8 = @Vector(simd_lanes_count, f32);

comptime {
    assert(sample_rate_hz % hop_samples_count == 0);
    assert(encoder_frames_count_max == 30 * frames_per_second);
    assert(encoder_frames_count_max % simd_lanes_count == 0);
    assert(fft_samples_count % simd_lanes_count == 0);
    assert((fft_bins_count - 1) % simd_lanes_count == 0);
}

pub const Error = error{
    AudioTooShort,
    AudioDurationExceedsLimit,
    MemoryTooSmall,
};

/// `EncoderTrailingPadding` selects the normalized silence appended after the
/// recorded content. The complete encoder input remains capped at Whisper's
/// 30-second context. A 30-second tail therefore produces the standard fixed
/// 3,000-frame input for every accepted recording.
pub const EncoderTrailingPadding = enum {
    seconds_5,
    seconds_10,
    seconds_30,

    fn framesCount(padding: EncoderTrailingPadding) usize {
        return switch (padding) {
            .seconds_5 => 5 * frames_per_second,
            .seconds_10 => 10 * frames_per_second,
            .seconds_30 => 30 * frames_per_second,
        };
    }
};

/// `Features.values` is row-major `[mel_bins_count, frames_count]`; every Mel
/// row uses the returned logical frame count as its stride. The slice borrows
/// the calculation's workspace and remains valid until the caller reuses it,
/// including for another calculation.
pub const Features = struct {
    values: []const f32,
    frames_count: usize,

    pub fn encoderPositionsCount(features: Features) usize {
        assert(features.frames_count > 0);
        assert(features.frames_count % 2 == 0);

        return @divExact(features.frames_count, 2);
    }
};

/// `Extractor` borrows immutable coefficient storage for its complete lifetime.
/// `calculate` writes a separate caller-owned workspace and performs no allocation.
pub const Extractor = struct {
    samples_count_max: usize,
    hann_window: []const f32,
    dft_cosines: []const f32,
    dft_sines: []const f32,
    mel_filters: []const MelFilter,
    mel_filter_coefficients: []const f32,

    pub fn requiredMemorySize(configured_samples_count_max: usize) usize {
        return MemoryLayout.init(configured_samples_count_max).size;
    }

    pub fn init(extractor: *Extractor, memory: []align(memory_layout.alignment) u8, configured_samples_count_max: usize) Error!void {
        const layout = MemoryLayout.init(configured_samples_count_max);
        if (memory.len < layout.size) {
            return error.MemoryTooSmall;
        }

        const hann_window = layout.hann_window.bind(memory);
        const dft_cosines = layout.dft_cosines.bind(memory);
        const dft_sines = layout.dft_sines.bind(memory);
        const mel_filters = layout.mel_filters.bind(memory);
        const mel_filter_coefficients = layout.mel_filter_coefficients.bind(memory);

        for (hann_window, 0..) |*coefficient, sample_index| {
            const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt(sample_index)) / fft_samples_count;
            coefficient.* = @floatCast(0.5 - 0.5 * @cos(angle));
        }

        for (0..fft_bins_count) |fft_bin_index| {
            for (0..fft_samples_count) |sample_index| {
                const coefficient_index = fft_bin_index * fft_samples_count + sample_index;
                const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt(fft_bin_index * sample_index)) / fft_samples_count;
                dft_cosines[coefficient_index] = @floatCast(@cos(angle));
                dft_sines[coefficient_index] = @floatCast(@sin(angle));
            }
        }

        calculateMelFilters(mel_filters, mel_filter_coefficients);

        extractor.* = .{
            .samples_count_max = configured_samples_count_max,
            .hann_window = hann_window,
            .dft_cosines = dft_cosines,
            .dft_sines = dft_sines,
            .mel_filters = mel_filters,
            .mel_filter_coefficients = mel_filter_coefficients,
        };
    }

    /// `calculate` normalizes the recorded content before appending zero-valued
    /// Mel frames selected by `trailing_padding`. Padding is capped at the model's
    /// 30-second context. `workspace` must contain at least
    /// `workspace_values_count` floats and may alias neither the samples nor the
    /// extractor's coefficient storage. Returned features borrow this workspace.
    pub fn calculate(extractor: *const Extractor, samples: []const f32, trailing_padding: EncoderTrailingPadding, workspace: []f32) Error!Features {
        if (samples.len < samples_count_min) {
            return error.AudioTooShort;
        }
        if (samples.len > extractor.samples_count_max) {
            return error.AudioDurationExceedsLimit;
        }
        if (workspace.len < workspace_values_count) {
            return error.MemoryTooSmall;
        }

        const content_frames_count = samples.len / hop_samples_count;
        assert(content_frames_count > 0);
        assert(content_frames_count <= encoder_frames_count_max);

        // ── Stream Spectra Into Final Mel Storage ──
        //
        // Only a single 201-value spectrum is live at a time. Interior FFT
        // windows borrow caller samples directly; the few boundary windows use
        // a small local buffer for Whisper's reflected prefix and zero suffix.

        const encoder_frames_count = paddedEncoderFramesCount(content_frames_count, trailing_padding);
        const power_spectrum = workspace[0..fft_bins_count];
        const encoder_features = workspace[encoder_features_offset..][0 .. mel_bins_count * encoder_frames_count];

        var boundary_frame_samples: [fft_samples_count]f32 = undefined;
        for (0..content_frames_count) |frame_index| {
            const frame_origin = @as(isize, @intCast(frame_index * hop_samples_count)) - center_padding_samples_count;
            const frame_end = frame_origin + fft_samples_count;
            const frame_samples: []const f32 = if (frame_origin >= 0 and frame_end <= samples.len) blk: {
                const origin: usize = @intCast(frame_origin);
                break :blk samples[origin..][0..fft_samples_count];
            } else blk: {
                for (&boundary_frame_samples, 0..) |*sample, frame_sample_index| {
                    const sample_index = frame_origin + @as(isize, @intCast(frame_sample_index));
                    if (sample_index < 0) {
                        sample.* = samples[@intCast(-sample_index)];
                    } else if (sample_index < samples.len) {
                        sample.* = samples[@intCast(sample_index)];
                    } else {
                        sample.* = 0.0;
                    }
                }
                break :blk &boundary_frame_samples;
            };

            calculatePowerSpectrum(power_spectrum, frame_samples, extractor.hann_window, extractor.dft_cosines, extractor.dft_sines);
            calculateMelFrame(encoder_features, encoder_frames_count, frame_index, power_spectrum, extractor.mel_filters, extractor.mel_filter_coefficients);
        }

        // ── Normalize Content And Preserve Zero Tail ──
        //
        // The maximum is visited in the same Mel-major order as the canonical
        // implementation. Appended silence remains normalized zero and does not
        // participate in content normalization.

        var feature_maximum: f32 = -std.math.inf(f32);
        for (0..mel_bins_count) |mel_bin_index| {
            const content_row = encoder_features[mel_bin_index * encoder_frames_count ..][0..content_frames_count];
            for (content_row) |*feature| {
                feature.* = @log10(@max(feature.*, 1.0e-10));
                feature_maximum = @max(feature_maximum, feature.*);
            }
        }

        for (0..mel_bins_count) |mel_bin_index| {
            const row = encoder_features[mel_bin_index * encoder_frames_count ..][0..encoder_frames_count];
            normalizeFeatures(row[0..content_frames_count], feature_maximum - 8.0);
            @memset(row[content_frames_count..], 0.0);
        }

        return .{ .values = encoder_features, .frames_count = encoder_frames_count };
    }
};

fn paddedEncoderFramesCount(content_frames_count: usize, trailing_padding: EncoderTrailingPadding) usize {
    assert(content_frames_count > 0);
    assert(content_frames_count <= encoder_frames_count_max);

    const requested_frames_count = content_frames_count + trailing_padding.framesCount();
    const capped_frames_count = @min(requested_frames_count, encoder_frames_count_max);
    const aligned_frames_count = std.mem.alignForward(usize, capped_frames_count, simd_lanes_count);

    assert(aligned_frames_count >= content_frames_count);
    assert(aligned_frames_count <= encoder_frames_count_max);
    assert(aligned_frames_count % simd_lanes_count == 0);

    return aligned_frames_count;
}

fn calculatePowerSpectrum(power_spectrum: []f32, frame_samples: []const f32, hann_window: []const f32, dft_cosines: []const f32, dft_sines: []const f32) void {
    assert(power_spectrum.len == fft_bins_count);
    assert(frame_samples.len == fft_samples_count);

    for (0..fft_bins_count) |fft_bin_index| {
        const coefficient_offset = fft_bin_index * fft_samples_count;
        const cosines = dft_cosines[coefficient_offset..][0..fft_samples_count];
        const sines = dft_sines[coefficient_offset..][0..fft_samples_count];
        var real_parts: F32x8 = @splat(0.0);
        var imaginary_parts: F32x8 = @splat(0.0);

        var sample_index: usize = 0;
        while (sample_index < fft_samples_count) : (sample_index += simd_lanes_count) {
            const samples_vector: F32x8 = frame_samples[sample_index..][0..simd_lanes_count].*;
            const window_vector: F32x8 = hann_window[sample_index..][0..simd_lanes_count].*;
            const windowed_samples = samples_vector * window_vector;
            real_parts += windowed_samples * cosines[sample_index..][0..simd_lanes_count].*;
            imaginary_parts -= windowed_samples * sines[sample_index..][0..simd_lanes_count].*;
        }

        const real_part = @reduce(.Add, real_parts);
        const imaginary_part = @reduce(.Add, imaginary_parts);
        power_spectrum[fft_bin_index] = real_part * real_part + imaginary_part * imaginary_part;
    }
}

fn calculateMelFrame(features: []f32, frames_count: usize, frame_index: usize, power_spectrum: []const f32, mel_filters: []const MelFilter, mel_filter_coefficients: []const f32) void {
    assert(features.len == mel_bins_count * frames_count);
    assert(frame_index < frames_count);
    assert(power_spectrum.len == fft_bins_count);
    assert(mel_filters.len == mel_bins_count);

    // Support begins on the original eight-bin boundary, preserving each
    // product's SIMD lane and accumulation order. Omitted coefficients are zero;
    // normalized input gives finite, nonnegative power, so those adds are inert.
    var coefficient_offset: usize = 0;
    for (mel_filters, 0..) |mel_filter, mel_bin_index| {
        const coefficients_count = mel_filter.fft_bin_end - mel_filter.fft_bin_begin;
        const coefficients = mel_filter_coefficients[coefficient_offset..][0..coefficients_count];
        var sums: F32x8 = @splat(0.0);

        var fft_bin_index: usize = mel_filter.fft_bin_begin;
        while (fft_bin_index < mel_filter.fft_bin_end) : (fft_bin_index += simd_lanes_count) {
            const spectrum_values: F32x8 = power_spectrum[fft_bin_index..][0..simd_lanes_count].*;
            const filter_values: F32x8 = coefficients[fft_bin_index - mel_filter.fft_bin_begin ..][0..simd_lanes_count].*;
            sums += spectrum_values * filter_values;
        }

        var sum = @reduce(.Add, sums);
        sum += power_spectrum[fft_bins_count - 1] * mel_filter.nyquist_coefficient;

        features[mel_bin_index * frames_count + frame_index] = sum;
        coefficient_offset += coefficients_count;
    }
}

fn normalizeFeatures(features: []f32, feature_minimum: f32) void {
    assert(features.len > 0);
    assert(std.math.isFinite(feature_minimum));

    const minimums: F32x8 = @splat(feature_minimum);
    const additions: F32x8 = @splat(4.0);
    const scales: F32x8 = @splat(0.25);
    var feature_index: usize = 0;

    while (feature_index + simd_lanes_count <= features.len) : (feature_index += simd_lanes_count) {
        const values: F32x8 = features[feature_index..][0..simd_lanes_count].*;
        features[feature_index..][0..simd_lanes_count].* = (@max(values, minimums) + additions) * scales;
    }
    while (feature_index < features.len) : (feature_index += 1) {
        features[feature_index] = (@max(features[feature_index], feature_minimum) + 4.0) * 0.25;
    }
}

const MelFilter = struct {
    fft_bin_begin: u8,
    fft_bin_end: u8,
    nyquist_coefficient: f32,
};

fn calculateMelFilters(filters: []MelFilter, coefficients: []f32) void {
    const mel_points_count = mel_bins_count + 2;
    var frequencies_hz: [mel_points_count]f64 = undefined;

    assert(filters.len == mel_bins_count);
    assert(coefficients.len == mel_filter_coefficients_capacity);
    comptime assert(fft_bins_count - 1 <= std.math.maxInt(u8));

    for (&frequencies_hz, 0..) |*frequency_hz, mel_point_index| {
        const mel = 45.245640471924965 * @as(f64, @floatFromInt(mel_point_index)) / (mel_points_count - 1);
        const linear_frequency_hz = (200.0 / 3.0) * mel;
        const logarithmic_start = 1000.0 / (200.0 / 3.0);
        frequency_hz.* = if (mel >= logarithmic_start) 1000.0 * @exp(@log(6.4) / 27.0 * (mel - logarithmic_start)) else linear_frequency_hz;
    }

    var coefficient_offset: usize = 0;
    for (filters, 0..) |*filter, mel_bin_index| {
        const lower_frequency_hz = frequencies_hz[mel_bin_index];
        const center_frequency_hz = frequencies_hz[mel_bin_index + 1];
        const upper_frequency_hz = frequencies_hz[mel_bin_index + 2];
        const energy_scale = 2.0 / (upper_frequency_hz - lower_frequency_hz);

        var dense_filter: [fft_bins_count]f32 = undefined;
        for (&dense_filter, 0..) |*coefficient, fft_bin_index| {
            const frequency_hz = @as(f64, @floatFromInt(fft_bin_index)) * sample_rate_hz / fft_samples_count;
            const lower_weight = (frequency_hz - lower_frequency_hz) / (center_frequency_hz - lower_frequency_hz);
            const upper_weight = (upper_frequency_hz - frequency_hz) / (upper_frequency_hz - center_frequency_hz);
            coefficient.* = @floatCast(@max(0.0, @min(lower_weight, upper_weight)) * energy_scale);
        }

        // Find support after float32 rounding. The final FFT bin remains the
        // original scalar term even when its generated coefficient is tiny.
        var fft_bin_begin: usize = fft_bins_count - 1;
        var fft_bin_end: usize = 0;
        for (dense_filter[0 .. fft_bins_count - 1], 0..) |coefficient, fft_bin_index| {
            if (coefficient == 0.0) continue;
            fft_bin_begin = @min(fft_bin_begin, fft_bin_index);
            fft_bin_end = fft_bin_index + 1;
        }
        if (fft_bin_end == 0) fft_bin_end = fft_bin_begin;
        fft_bin_begin = std.mem.alignBackward(usize, fft_bin_begin, simd_lanes_count);
        fft_bin_end = std.mem.alignForward(usize, fft_bin_end, simd_lanes_count);
        assert(fft_bin_end <= fft_bins_count - 1);
        const coefficients_count = fft_bin_end - fft_bin_begin;
        assert(coefficients_count <= coefficients.len - coefficient_offset);
        @memcpy(coefficients[coefficient_offset..][0..coefficients_count], dense_filter[fft_bin_begin..fft_bin_end]);
        filter.* = .{
            .fft_bin_begin = @intCast(fft_bin_begin),
            .fft_bin_end = @intCast(fft_bin_end),
            .nyquist_coefficient = dense_filter[fft_bins_count - 1],
        };
        coefficient_offset += coefficients_count;
    }
}

const MemoryLayout = struct {
    hann_window: memory_layout.Region(f32),
    dft_cosines: memory_layout.Region(f32),
    dft_sines: memory_layout.Region(f32),
    mel_filters: memory_layout.Region(MelFilter),
    mel_filter_coefficients: memory_layout.Region(f32),
    size: usize,

    fn init(configured_samples_count_max: usize) MemoryLayout {
        assert(configured_samples_count_max >= samples_count_min);
        assert(configured_samples_count_max <= samples_count_max);

        var builder: memory_layout.Builder = .{};
        const hann_window = builder.add(f32, fft_samples_count);
        const dft_cosines = builder.add(f32, fft_bins_count * fft_samples_count);
        const dft_sines = builder.add(f32, fft_bins_count * fft_samples_count);
        const mel_filters = builder.add(MelFilter, mel_bins_count);
        const mel_filter_coefficients = builder.add(f32, mel_filter_coefficients_capacity);

        return .{
            .hann_window = hann_window,
            .dft_cosines = dft_cosines,
            .dft_sines = dft_sines,
            .mel_filters = mel_filters,
            .mel_filter_coefficients = mel_filter_coefficients,
            .size = builder.size,
        };
    }
};

test "compact Mel filters preserve dense SIMD accumulation" {
    var filters: [mel_bins_count]MelFilter = undefined;
    var coefficients: [mel_filter_coefficients_capacity]f32 = @splat(std.math.nan(f32));
    calculateMelFilters(&filters, &coefficients);

    // Keep the original dense construction and reduction as an independent
    // reference. Impulses expose omitted or misplaced bins; mixed spectra also
    // exercise the SIMD lane sums. Unused compact capacity stays poisoned.
    var dense_filters: [mel_bins_count * fft_bins_count]f32 = undefined;
    var frequencies_hz: [mel_bins_count + 2]f64 = undefined;
    for (&frequencies_hz, 0..) |*frequency_hz, mel_point_index| {
        const mel = 45.245640471924965 * @as(f64, @floatFromInt(mel_point_index)) / (mel_bins_count + 1);
        const linear_frequency_hz = (200.0 / 3.0) * mel;
        const logarithmic_start = 1000.0 / (200.0 / 3.0);
        frequency_hz.* = if (mel >= logarithmic_start) 1000.0 * @exp(@log(6.4) / 27.0 * (mel - logarithmic_start)) else linear_frequency_hz;
    }
    for (0..mel_bins_count) |mel_bin_index| {
        const lower_frequency_hz = frequencies_hz[mel_bin_index];
        const center_frequency_hz = frequencies_hz[mel_bin_index + 1];
        const upper_frequency_hz = frequencies_hz[mel_bin_index + 2];
        const energy_scale = 2.0 / (upper_frequency_hz - lower_frequency_hz);
        for (0..fft_bins_count) |fft_bin_index| {
            const frequency_hz = @as(f64, @floatFromInt(fft_bin_index)) * sample_rate_hz / fft_samples_count;
            const lower_weight = (frequency_hz - lower_frequency_hz) / (center_frequency_hz - lower_frequency_hz);
            const upper_weight = (upper_frequency_hz - frequency_hz) / (upper_frequency_hz - center_frequency_hz);
            dense_filters[mel_bin_index * fft_bins_count + fft_bin_index] = @floatCast(@max(0.0, @min(lower_weight, upper_weight)) * energy_scale);
        }
    }

    var random = std.Random.DefaultPrng.init(42);
    for (0..fft_bins_count + 5) |case_index| {
        var power_spectrum: [fft_bins_count]f32 = @splat(0.0);
        if (case_index < fft_bins_count) {
            power_spectrum[case_index] = 1.0;
        } else if (case_index > fft_bins_count) {
            for (&power_spectrum) |*power| power.* = random.random().float(f32) * 160_000.0;
        }
        var actual: [mel_bins_count]f32 = @splat(std.math.nan(f32));
        calculateMelFrame(&actual, 1, 0, &power_spectrum, &filters, &coefficients);

        var expected: [mel_bins_count]f32 = undefined;
        for (&expected, 0..) |*feature, mel_bin_index| {
            const filter = dense_filters[mel_bin_index * fft_bins_count ..][0..fft_bins_count];
            var sums: F32x8 = @splat(0.0);
            var fft_bin_index: usize = 0;
            while (fft_bin_index < fft_bins_count - 1) : (fft_bin_index += simd_lanes_count) {
                const spectrum_values: F32x8 = power_spectrum[fft_bin_index..][0..simd_lanes_count].*;
                const filter_values: F32x8 = filter[fft_bin_index..][0..simd_lanes_count].*;
                sums += spectrum_values * filter_values;
            }
            var sum = @reduce(.Add, sums);
            sum += power_spectrum[fft_bins_count - 1] * filter[fft_bins_count - 1];
            feature.* = sum;
        }
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&actual));
    }

    // Keep bin 200 independent of vector support even on toolchains whose
    // generated final coefficient rounds to zero.
    filters[mel_bins_count - 1].nyquist_coefficient = 1.0e-20;
    var nyquist_power: [fft_bins_count]f32 = @splat(0.0);
    nyquist_power[fft_bins_count - 1] = 1.0;
    var nyquist_features: [mel_bins_count]f32 = undefined;
    calculateMelFrame(&nyquist_features, 1, 0, &nyquist_power, &filters, &coefficients);
    try std.testing.expectEqual(@as(f32, 1.0e-20), nyquist_features[mel_bins_count - 1]);
}
