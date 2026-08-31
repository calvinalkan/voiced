//! Decodes Whisper vocabulary strings from GPT-2's reversible byte-to-Unicode
//! representation. CTranslate2 returns concatenated token strings in that
//! representation; callers must reverse it before treating the bytes as UTF-8.

const std = @import("std");

/// `decodeInto` writes the complete UTF-8 transcript into `output`. The returned
/// slice aliases `output`; encoded input and output must not overlap.
pub fn decodeInto(output: []u8, encoded_text: []const u8) ![]const u8 {
    const unmapped: u16 = 256;
    var byte_by_codepoint: [512]u16 = @splat(unmapped);
    var next_mapped_codepoint: u16 = 0;

    // GPT-2 leaves visible Latin-1 bytes at their Unicode code points and maps
    // every remaining byte, in byte order, to consecutive code points at 256.
    // Reconstructing the inverse table once per transcript is bounded by 512
    // entries and keeps tokenizer files out of this byte-level operation.
    for (0..256) |byte| {
        const is_visible = (byte >= 33 and byte <= 126) or
            (byte >= 161 and byte <= 172) or
            (byte >= 174 and byte <= 255);

        if (is_visible) {
            byte_by_codepoint[byte] = @intCast(byte);
        } else {
            while (byte_by_codepoint[next_mapped_codepoint] != unmapped) {
                next_mapped_codepoint += 1;
            }
            byte_by_codepoint[256 + next_mapped_codepoint] = @intCast(byte);
            next_mapped_codepoint += 1;
        }
    }

    var output_size: usize = 0;
    var encoded_codepoints = (try std.unicode.Utf8View.init(encoded_text)).iterator();
    while (encoded_codepoints.nextCodepoint()) |encoded_codepoint| {
        if (encoded_codepoint >= byte_by_codepoint.len) {
            return error.InvalidGpt2TokenCodepoint;
        }

        const byte = byte_by_codepoint[encoded_codepoint];
        if (byte == unmapped) return error.InvalidGpt2TokenCodepoint;
        if (output_size == output.len) return error.TranscriptExceedsLimit;

        output[output_size] = @intCast(byte);
        output_size += 1;
    }

    const transcript = output[0..output_size];
    if (!std.unicode.utf8ValidateSlice(transcript)) {
        return error.InvalidTranscriptUtf8;
    }
    return transcript;
}
