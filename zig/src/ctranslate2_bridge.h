#ifndef VOICED_CTRANSLATE2_BRIDGE_H
#define VOICED_CTRANSLATE2_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    log_mel_bins_count = 80,
    log_mel_frames_count = 3000,
    log_mel_values_count = log_mel_bins_count * log_mel_frames_count,
    error_message_capacity = 1024,
};

/*
 * `Error` carries one bounded, null-terminated diagnostic. Operations that
 * accept an error output require a non-null pointer, clear the message on
 * entry, and truncate longer diagnostics.
 */
typedef struct Error {
    char message[error_message_capacity];
} Error;

/*
 * `Gpt2EncodedText` describes one Zig-owned output buffer. `bytes` must point
 * to at least `capacity` writable bytes, and `capacity` must be nonzero.
 * Transcription clears `size` on entry. On success, `[0, size)` contains the
 * complete encoded output without a null terminator.
 */
typedef struct Gpt2EncodedText {
    char *bytes;
    uint32_t capacity;
    uint32_t size;
} Gpt2EncodedText;

/*
 * `ModelHandle` owns one resident English Whisper model. Its C++ representation
 * and native-library allocations remain private to the bridge.
 */
typedef struct ModelHandle ModelHandle;

/*
 * `model_create` loads one English Whisper model for int8 CPU inference.
 *
 * `model_directory_path` must be a null-terminated CTranslate2 model directory.
 *
 * `inference_threads_count` must be nonzero. The caller owns the upper policy
 * because useful concurrency depends on the host CPU and worker configuration.
 *
 * `error_out` must be non-null. The operation returns null and writes a
 * diagnostic on invalid arguments or a model-loading failure. The caller owns
 * a successful handle and must destroy it exactly once.
 */
ModelHandle *model_create(
    const char *model_directory_path,
    uint32_t inference_threads_count,
    Error *error_out
);

/*
 * `model_destroy` releases a handle returned by `model_create`. A null handle
 * is accepted and has no effect.
 */
void model_destroy(ModelHandle *model);

/*
 * `model_transcribe` synchronously produces one deterministic English
 * transcript without timestamps. `log_mel_values` must contain exactly
 * `log_mel_values_count` row-major float32 values with shape
 * `[1, log_mel_bins_count, log_mel_frames_count]`. The bridge reads them only
 * during this call.
 *
 * `gpt2_encoded_text_out` and `error_out` must be non-null.
 * `gpt2_encoded_text_out` receives concatenated Whisper vocabulary strings in
 * GPT-2's reversible byte-to-Unicode encoding. The operation returns false on
 * invalid arguments or a transcription failure; the encoded `size` remains
 * zero, its bytes are unspecified, and `error_out` describes the failure. Zig
 * must decode the GPT-2 mapping before treating the output as a UTF-8
 * transcript.
 */
bool model_transcribe(
    ModelHandle *model,
    const float *log_mel_values,
    Gpt2EncodedText *gpt2_encoded_text_out,
    Error *error_out
);

#ifdef __cplusplus
}
#endif

#endif
