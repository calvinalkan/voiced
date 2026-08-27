/*
 * This file is the boundary between Zig and CTranslate2's C++ API. Zig
 * owns every input and output buffer. C++ owns the model handle and catches all
 * exceptions because they must never unwind through Zig stack frames.
 */

#include "ctranslate2_bridge.h"

#include <ctranslate2/models/whisper.h>

#include <cassert>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using ctranslate2::ComputeType;
using ctranslate2::Device;
using ctranslate2::ReplicaPoolConfig;
using ctranslate2::StorageView;
using ctranslate2::dim_t;
using ctranslate2::models::ModelLoader;
using ctranslate2::models::Whisper;
using ctranslate2::models::WhisperGenerationResult;
using ctranslate2::models::WhisperOptions;

// `<|endoftext|>` is the first special token after the original GPT-2 text
// vocabulary. Output filtering discards this ID and every greater ID.
static constexpr std::size_t whisper_end_of_text_token_id = 50256;

// `<|startoftranscript|>` begins every Whisper decoder prompt.
static constexpr std::size_t whisper_start_of_transcript_token_id = 50257;

// `<|notimestamps|>` asks Whisper to emit text without timestamp tokens.
static constexpr std::size_t whisper_no_timestamps_token_id = 50362;

// Whisper's text decoder has 448 token positions shared by the prompt and
// generated continuation. CTranslate2 uses this as its total decoder-length
// ceiling.
static constexpr std::size_t whisper_decoder_context_tokens_count = 448;

struct ModelHandle {
    std::unique_ptr<Whisper> whisper;
    std::vector<std::vector<std::size_t>> prompts;
    WhisperOptions options;
};

static void write_error_message(
    const char *source_message,
    Error *error_out
) noexcept;

// ─── Model Handle Operations ───────────────────────────────────────────────

extern "C" ModelHandle *model_create(
    const char *model_directory_path,
    const uint32_t inference_threads_count,
    Error *error_out
) {
    if (error_out == nullptr) {
        return nullptr;
    }
    error_out->message[0] = '\0';

    if (model_directory_path == nullptr) {
        write_error_message("model directory path is required", error_out);

        return nullptr;
    }
    if (model_directory_path[0] == '\0') {
        write_error_message("model directory path is required", error_out);

        return nullptr;
    }
    if (inference_threads_count == 0) {
        write_error_message("inference thread count must be nonzero", error_out);

        return nullptr;
    }

    try {
        // ── Configure One Int8 CPU Replica ──

        ModelLoader model_loader(model_directory_path);
        model_loader.device = Device::CPU;
        model_loader.compute_type = ComputeType::INT8;

        // Device index zero creates one CPU replica. This worker serves one
        // synchronous request at a time, so it neither partitions tensors nor
        // creates concurrent model copies.
        model_loader.device_indices = {0};
        model_loader.tensor_parallel = false;

        // One request may wait behind active inference. A negative core offset
        // leaves thread placement to Linux instead of coupling the worker to
        // one machine's CPU topology.
        ReplicaPoolConfig replica_pool_config;
        replica_pool_config.num_threads_per_replica = inference_threads_count;
        replica_pool_config.max_queued_batches = 1;
        replica_pool_config.cpu_core_offset = -1;

        auto model = std::make_unique<ModelHandle>();
        model->whisper = std::make_unique<Whisper>(
            model_loader,
            replica_pool_config
        );

        if (model->whisper->is_multilingual()) {
            write_error_message("model must be an English-only Whisper model", error_out);

            return nullptr;
        }
        if (model->whisper->n_mels() != log_mel_bins_count) {
            write_error_message("model Mel band count does not match the bridge", error_out);

            return nullptr;
        }

        // ── Configure English Greedy Decoding ──

        // The English-only model needs no language or task token. The second
        // token requests plain text and prevents timestamp generation.
        model->prompts = {{
            whisper_start_of_transcript_token_id,
            whisper_no_timestamps_token_id,
        }};

        // A beam size and top-K of one select the highest-probability token.
        // One hypothesis avoids scoring or retaining unused alternatives.
        model->options.beam_size = 1;
        model->options.sampling_topk = 1;
        model->options.sampling_temperature = 1.0f;
        model->options.num_hypotheses = 1;

        // Neutral penalties preserve the model distribution. The 448-token
        // bound is Whisper's decoder context. Disabling the n-gram rule permits
        // ordinary repeated words instead of imposing another language rule.
        model->options.patience = 1.0f;
        model->options.length_penalty = 1.0f;
        model->options.repetition_penalty = 1.0f;
        model->options.no_repeat_ngram_size = 0;
        model->options.max_length = whisper_decoder_context_tokens_count;

        // The bridge needs token strings and IDs only. Scores, vocabulary
        // logits, and silence probability would return unused result fields.
        model->options.return_scores = false;
        model->options.return_logits_vocab = false;
        model->options.return_no_speech_prob = false;

        // The prompt makes the timestamp bound inactive. Blank suppression and
        // the model's default suppression list reject non-speech symbols.
        model->options.max_initial_timestamp_index = 50;
        model->options.suppress_blank = true;
        model->options.suppress_tokens = {-1};

        ModelHandle *model_handle = model.release();
        assert(model_handle != nullptr);

        return model_handle;
    } catch (const std::exception &exception) {
        write_error_message(exception.what(), error_out);

        return nullptr;
    } catch (...) {
        write_error_message("unknown exception while loading the model", error_out);

        return nullptr;
    }
}

extern "C" void model_destroy(ModelHandle *model) {
    if (model == nullptr) {
        return;
    }

    delete model;
}

extern "C" bool model_transcribe(
    ModelHandle *model,
    const float *log_mel_values,
    Gpt2EncodedText *gpt2_encoded_text_out,
    Error *error_out
) {
    if (error_out == nullptr) {
        return false;
    }
    error_out->message[0] = '\0';

    if (gpt2_encoded_text_out == nullptr) {
        write_error_message("GPT-2 encoded text output is required", error_out);

        return false;
    }
    gpt2_encoded_text_out->size = 0;

    if (model == nullptr) {
        write_error_message("model handle is required", error_out);

        return false;
    }
    if (log_mel_values == nullptr) {
        write_error_message("log-Mel values are required", error_out);

        return false;
    }
    if (gpt2_encoded_text_out->bytes == nullptr) {
        write_error_message("GPT-2 encoded text bytes are required", error_out);

        return false;
    }
    if (gpt2_encoded_text_out->capacity == 0) {
        write_error_message("GPT-2 encoded text capacity must be nonzero", error_out);

        return false;
    }

    assert(model->whisper != nullptr);

    try {
        // ── Generate One Transcript ──
        //
        // `StorageView` requires a mutable pointer, but Whisper reads this view.
        // Waiting on the returned future also ensures that CTranslate2 releases
        // the caller's buffer before this function returns.

        StorageView log_mel_view(
            {
                1,
                static_cast<dim_t>(log_mel_bins_count),
                static_cast<dim_t>(log_mel_frames_count),
            },
            const_cast<float *>(log_mel_values)
        );

        auto generation_futures = model->whisper->generate(
            log_mel_view,
            model->prompts,
            model->options
        );

        if (generation_futures.size() != 1) {
            write_error_message("CTranslate2 returned an unexpected future count", error_out);

            return false;
        }

        const WhisperGenerationResult generation_result = generation_futures[0].get();
        if (generation_result.sequences.size() != 1) {
            write_error_message("CTranslate2 returned an unexpected text result count", error_out);

            return false;
        }
        if (generation_result.sequences_ids.size() != 1) {
            write_error_message(
                "CTranslate2 returned an unexpected token ID result count",
                error_out
            );

            return false;
        }

        const std::vector<std::string> &vocabulary_tokens = generation_result.sequences[0];
        const std::vector<std::size_t> &vocabulary_token_ids = generation_result.sequences_ids[0];

        if (vocabulary_tokens.size() != vocabulary_token_ids.size()) {
            write_error_message("CTranslate2 returned inconsistent token arrays", error_out);

            return false;
        }
        if (vocabulary_tokens.size() > whisper_decoder_context_tokens_count) {
            write_error_message("CTranslate2 returned too many generated tokens", error_out);

            return false;
        }

        // ── Write Encoded Text ──

        std::size_t write_offset = 0;

        // The text-token threshold must also exclude both control tokens used
        // in the decoder prompt if CTranslate2 returns either one.
        static_assert(whisper_end_of_text_token_id < whisper_start_of_transcript_token_id);
        static_assert(whisper_end_of_text_token_id < whisper_no_timestamps_token_id);

        for (
            std::size_t token_index = 0;
            token_index < vocabulary_tokens.size();
            token_index += 1
        ) {
            // Original Whisper vocabulary:
            //   [0, end-of-text)  ordinary GPT-2 text
            //   end-of-text       generation terminator
            //   above end-of-text control and timestamp tokens
            const std::size_t vocabulary_token_id = vocabulary_token_ids[token_index];

            if (vocabulary_token_id == whisper_end_of_text_token_id) {
                break;
            }
            if (vocabulary_token_id > whisper_end_of_text_token_id) {
                // Control and timestamp tokens are not terminators and may be
                // followed by ordinary text, so skip only the current token.
                continue;
            }

            const std::string &token = vocabulary_tokens[token_index];
            assert(write_offset <= gpt2_encoded_text_out->capacity);

            const size_t token_size = token.size();

            if (token_size > gpt2_encoded_text_out->capacity - write_offset) {
                write_error_message(
                    "GPT-2 encoded text exceeds its output capacity",
                    error_out
                );

                return false;
            }

            // The C ABI carries the allocation as a pointer and capacity, so
            // Clang cannot infer its extent. The subtraction check above proves
            // that `[write_offset, write_offset + token_size)` is in bounds.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
            std::memcpy(
                gpt2_encoded_text_out->bytes + write_offset,
                token.data(),
                token_size
            );
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
            write_offset += token_size;
        }

        assert(write_offset <= gpt2_encoded_text_out->capacity);
        assert(write_offset <= UINT32_MAX);

        gpt2_encoded_text_out->size = static_cast<uint32_t>(write_offset);

        return true;
    } catch (const std::exception &exception) {
        write_error_message(exception.what(), error_out);

        return false;
    } catch (...) {
        write_error_message("unknown exception while transcribing features", error_out);

        return false;
    }
}

// ─── Bounded Error Message ─────────────────────────────────────────────────

static void write_error_message(
    const char *message,
    Error *error_out
) noexcept {
    assert(message != nullptr);
    assert(error_out != nullptr);

    constexpr std::size_t target_capacity = sizeof(error_out->message);
    constexpr std::size_t copy_size_max = target_capacity - 1;
    static_assert(target_capacity == error_message_capacity);
    static_assert(copy_size_max > 0);

    std::size_t write_offset = 0;


    // Every caller supplies a string literal or `std::exception::what()`, both
    // of which are null-terminated. This loop scans at most the output capacity
    // and truncates without a preliminary unbounded `strlen` pass. Clang cannot
    // associate those contracts with raw C-array subscripts.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
    while (write_offset < copy_size_max) {
        const char message_byte = message[write_offset];
        if (message_byte == '\0') {
            break;
        }

        error_out->message[write_offset] = message_byte;
        write_offset += 1;
    }
    error_out->message[write_offset] = '\0';
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}
