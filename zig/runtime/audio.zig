//! Whisper audio preparation owns immutable DFT/Mel coefficient tables and a
//! fixed working set supplied by `Runtime`. Extraction accepts normalized 16 kHz
//! mono samples and returns the model's fixed `[80, 3000]` feature matrix.

const std = @import("std");
const assert = std.debug.assert;

const sample_rate_hz: usize = 16_000;
const fft_samples_count: usize = 400;
const fft_bins_count: usize = fft_samples_count / 2 + 1;
const hop_samples_count: usize = 160;
const mel_bins_count: usize = 80;
const encoder_frames_count: usize = 3000;
const center_padding_samples_count: usize = fft_samples_count / 2;
const tail_padding_samples_count: usize = hop_samples_count;
const simd_lanes_count: usize = 8;
const memory_alignment: usize = 64;

const F32x8 = @Vector(simd_lanes_count, f32);

comptime {
    assert(fft_samples_count % simd_lanes_count == 0);
    assert((fft_bins_count - 1) % simd_lanes_count == 0);
}

pub const Error = error{
    AudioTooShort,
    AudioDurationExceedsLimit,
    MemoryTooSmall,
};

pub const Features = struct {
    values: []const f32,
    frames_count: usize,
};

/// `Extractor` borrows one caller-owned memory region for its complete
/// lifetime. `calculate` overwrites that region and performs no allocation.
pub const Extractor = struct {
    samples_count_max: usize,
    hann_window: []f32,
    dft_cosines: []f32,
    dft_sines: []f32,
    mel_filters: []f32,
    centered_samples: []f32,
    power_spectra: []f32,
    content_features: []f32,
    encoder_features: []f32,

    pub fn requiredMemorySize(samples_count_max: usize) usize {
        assert(samples_count_max >= fft_samples_count / 2 + 1);
        assert(samples_count_max <= 30 * sample_rate_hz);

        var size: usize = memory_alignment - 1;
        size = addValuesSize(size, fft_samples_count);
        size = addValuesSize(size, fft_bins_count * fft_samples_count);
        size = addValuesSize(size, fft_bins_count * fft_samples_count);
        size = addValuesSize(size, mel_bins_count * fft_bins_count);
        size = addValuesSize(size, samples_count_max + 2 * center_padding_samples_count + tail_padding_samples_count);
        size = addValuesSize(size, encoder_frames_count * fft_bins_count);
        size = addValuesSize(size, mel_bins_count * encoder_frames_count);
        size = addValuesSize(size, mel_bins_count * encoder_frames_count);

        return size;
    }

    pub fn init(extractor: *Extractor, memory: []u8, samples_count_max: usize) Error!void {
        const required_memory_size = requiredMemorySize(samples_count_max);
        if (memory.len < required_memory_size) {
            return error.MemoryTooSmall;
        }

        var cursor: MemoryCursor = .{ .memory = memory };
        const hann_window = cursor.takeF32(fft_samples_count);
        const dft_cosines = cursor.takeF32(fft_bins_count * fft_samples_count);
        const dft_sines = cursor.takeF32(fft_bins_count * fft_samples_count);
        const mel_filters = cursor.takeF32(mel_bins_count * fft_bins_count);
        const centered_samples = cursor.takeF32(samples_count_max + 2 * center_padding_samples_count + tail_padding_samples_count);
        const power_spectra = cursor.takeF32(encoder_frames_count * fft_bins_count);
        const content_features = cursor.takeF32(mel_bins_count * encoder_frames_count);
        const encoder_features = cursor.takeF32(mel_bins_count * encoder_frames_count);
        assert(cursor.offset <= memory.len);

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

        calculateMelFilters(mel_filters);

        extractor.* = .{
            .samples_count_max = samples_count_max,
            .hann_window = hann_window,
            .dft_cosines = dft_cosines,
            .dft_sines = dft_sines,
            .mel_filters = mel_filters,
            .centered_samples = centered_samples,
            .power_spectra = power_spectra,
            .content_features = content_features,
            .encoder_features = encoder_features,
        };
    }

    /// `calculate` returns features borrowed from the extractor. The next call
    /// overwrites the returned values.
    pub fn calculate(extractor: *Extractor, samples: []const f32) Error!Features {
        if (samples.len < fft_samples_count / 2 + 1) {
            return error.AudioTooShort;
        }
        if (samples.len > extractor.samples_count_max) {
            return error.AudioDurationExceedsLimit;
        }

        const content_frames_count = samples.len / hop_samples_count;
        assert(content_frames_count > 0);
        assert(content_frames_count <= encoder_frames_count);

        // ── Center And Pad ──

        const centered_samples_count = samples.len + 2 * center_padding_samples_count + tail_padding_samples_count;
        const centered_samples = extractor.centered_samples[0..centered_samples_count];
        const content_samples = centered_samples[center_padding_samples_count..][0..samples.len];
        @memcpy(content_samples, samples);

        for (0..center_padding_samples_count) |padding_index| {
            centered_samples[padding_index] = content_samples[center_padding_samples_count - padding_index];
        }

        const tail_padding_offset = center_padding_samples_count + samples.len;
        @memset(centered_samples[tail_padding_offset..][0..tail_padding_samples_count], 0.0);

        for (0..center_padding_samples_count) |padding_index| {
            const source_index = samples.len + tail_padding_samples_count - 2 - padding_index;
            const target_index = center_padding_samples_count + samples.len + tail_padding_samples_count + padding_index;
            centered_samples[target_index] = if (source_index < samples.len) content_samples[source_index] else 0.0;
        }

        // ── Project Spectrum To Mel Bands ──

        const power_spectra = extractor.power_spectra[0 .. content_frames_count * fft_bins_count];
        calculatePowerSpectra(power_spectra, centered_samples, content_frames_count, extractor.hann_window, extractor.dft_cosines, extractor.dft_sines);

        const content_features = extractor.content_features[0 .. mel_bins_count * content_frames_count];
        calculateMelFeatures(content_features, power_spectra, content_frames_count, extractor.mel_filters);

        var feature_maximum: f32 = -std.math.inf(f32);
        for (content_features) |*feature| {
            feature.* = @log10(@max(feature.*, 1.0e-10));
            feature_maximum = @max(feature_maximum, feature.*);
        }

        normalizeFeatures(content_features, feature_maximum - 8.0);

        // ── Pad Normalized Features ──
        //
        // Padding waveform samples would change the maximum used by Whisper's
        // logarithmic normalization. Pad the completed Mel rows instead.

        @memset(extractor.encoder_features, 0.0);
        for (0..mel_bins_count) |mel_bin_index| {
            const content_row = content_features[mel_bin_index * content_frames_count ..][0..content_frames_count];
            const encoder_row = extractor.encoder_features[mel_bin_index * encoder_frames_count ..][0..content_frames_count];
            @memcpy(encoder_row, content_row);
        }

        return .{ .values = extractor.encoder_features, .frames_count = encoder_frames_count };
    }
};

fn calculatePowerSpectra(power_spectra: []f32, centered_samples: []const f32, frames_count: usize, hann_window: []const f32, dft_cosines: []const f32, dft_sines: []const f32) void {
    assert(power_spectra.len == frames_count * fft_bins_count);
    assert(centered_samples.len >= (frames_count - 1) * hop_samples_count + fft_samples_count);

    for (0..frames_count) |frame_index| {
        const frame_samples = centered_samples[frame_index * hop_samples_count ..][0..fft_samples_count];

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
            power_spectra[frame_index * fft_bins_count + fft_bin_index] = real_part * real_part + imaginary_part * imaginary_part;
        }
    }
}

fn calculateMelFeatures(features: []f32, power_spectra: []const f32, frames_count: usize, mel_filters: []const f32) void {
    assert(features.len == mel_bins_count * frames_count);
    assert(power_spectra.len == frames_count * fft_bins_count);

    for (0..mel_bins_count) |mel_bin_index| {
        const mel_filter = mel_filters[mel_bin_index * fft_bins_count ..][0..fft_bins_count];

        for (0..frames_count) |frame_index| {
            const spectrum = power_spectra[frame_index * fft_bins_count ..][0..fft_bins_count];
            var sums: F32x8 = @splat(0.0);

            var fft_bin_index: usize = 0;
            while (fft_bin_index < fft_bins_count - 1) : (fft_bin_index += simd_lanes_count) {
                const spectrum_values: F32x8 = spectrum[fft_bin_index..][0..simd_lanes_count].*;
                const filter_values: F32x8 = mel_filter[fft_bin_index..][0..simd_lanes_count].*;
                sums += spectrum_values * filter_values;
            }

            var sum = @reduce(.Add, sums);
            while (fft_bin_index < fft_bins_count) : (fft_bin_index += 1) {
                sum += spectrum[fft_bin_index] * mel_filter[fft_bin_index];
            }

            features[mel_bin_index * frames_count + frame_index] = sum;
        }
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

fn calculateMelFilters(filters: []f32) void {
    const mel_points_count = mel_bins_count + 2;
    var frequencies_hz: [mel_points_count]f64 = undefined;

    assert(filters.len == mel_bins_count * fft_bins_count);

    for (&frequencies_hz, 0..) |*frequency_hz, mel_point_index| {
        const mel = 45.245640471924965 * @as(f64, @floatFromInt(mel_point_index)) / (mel_points_count - 1);
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
            filters[mel_bin_index * fft_bins_count + fft_bin_index] = @floatCast(@max(0.0, @min(lower_weight, upper_weight)) * energy_scale);
        }
    }
}

fn addValuesSize(size: usize, values_count: usize) usize {
    const aligned_size = std.mem.alignForward(usize, size, memory_alignment);

    return aligned_size + values_count * @sizeOf(f32);
}

const MemoryCursor = struct {
    memory: []u8,
    offset: usize = 0,

    fn takeF32(cursor: *MemoryCursor, values_count: usize) []f32 {
        cursor.offset = std.mem.alignForward(usize, cursor.offset, memory_alignment);
        assert(cursor.offset + values_count * @sizeOf(f32) <= cursor.memory.len);

        const values: [*]f32 = @ptrCast(@alignCast(cursor.memory.ptr + cursor.offset));
        cursor.offset += values_count * @sizeOf(f32);

        return values[0..values_count];
    }
};
