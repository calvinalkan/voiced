/*
 * This file is Voiced's narrow semantic boundary around PipeWire's C API.
 * Zig owns the main loop, stream lifetime, callbacks, file descriptors, and
 * sample storage. C performs only the operations that need immediate `errno`
 * capture, SPA's macro-based pod builders, or conversion of a callback-owned
 * diagnostic into caller-owned bytes.
 *
 * The public functions describe Voiced operations rather than mirroring every
 * PipeWire call:
 *
 *     environment    report local capabilities and remote server version
 *     main loop      create, run
 *     callback wake  register, unregister
 *     source choice  resolve a stable Device serial before capture
 *     capture stream create, connect, verify its source, read timeline,
 *                    disconnect
 *     negotiation    parse the selected raw-audio format, configure buffers
 *
 * Every fallible operation requires an error output and clears it on entry. A
 * successful return leaves stage, domain, code, and message size at zero. An
 * error records the exact internal stage, native error domain and code, and a
 * bounded owned message before control returns to Zig. The asynchronous stream
 * state adapter emits the same structure, so borrowed PipeWire strings never
 * cross the boundary.
 *
 * SPA pods are compact binary descriptions, not sample buffers. Voiced offers
 * native float32/16 kHz/mono audio and asks for bounded buffers plus optional
 * `spa_meta_header` records. PipeWire may omit Header metadata even after
 * accepting that request; Zig validates and applies it only when supplied.
 * PipeWire 1.0.5 also exposes each capture buffer's graph-cycle timestamp. The
 * guarded timeline operation exists only when the build headers provide that
 * field; older Ubuntu builds expose Header-only validation without importing a
 * function or reading a struct member absent from their PipeWire version.
 */
#include "audio_pipewire.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <spa/param/param.h>
#include <spa/utils/string.h>

/** Clear a required error output before one semantic operation begins. */
static void clear_error(struct voiced_audio_pipewire_error *error_out) {
    if (error_out == NULL) {
        return;
    }

    memset(error_out, 0, sizeof(*error_out));
}

/** Copy a native or callback-owned message into the fixed error record. */
static void copy_error_message(
    struct voiced_audio_pipewire_error *error_out,
    const char *message
) {
    size_t message_size;

    if (error_out == NULL || message == NULL) {
        return;
    }

    message_size = strnlen(
        message,
        VOICED_AUDIO_PIPEWIRE_ERROR_MESSAGE_CAPACITY
    );
    memcpy(error_out->message, message, message_size);
    error_out->message_size = (uint32_t)message_size;
}

/** Capture thread-local libc errno at the stage that observed a native error. */
static void capture_errno(
    const enum voiced_audio_pipewire_error_stage stage,
    struct voiced_audio_pipewire_error *error_out
) {
    const int saved_errno = errno;

    clear_error(error_out);
    if (error_out == NULL) {
        return;
    }

    error_out->stage = (uint32_t)stage;
    error_out->domain = VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO;
    error_out->code = saved_errno;
    copy_error_message(
        error_out,
        saved_errno == 0 ? "No native errno was provided" : strerror(saved_errno)
    );
}

/** Capture one stable negative SPA/PipeWire result and its platform message. */
static void capture_result(
    const enum voiced_audio_pipewire_error_stage stage,
    const int result,
    struct voiced_audio_pipewire_error *error_out
) {
    clear_error(error_out);
    if (error_out == NULL) {
        return;
    }

    error_out->stage = (uint32_t)stage;
    error_out->domain = VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT;
    error_out->code = result;
    copy_error_message(error_out, strerror(-result));
}

/** Record one stable validation code for an error detected at this boundary. */
static void capture_validation(
    const enum voiced_audio_pipewire_error_stage stage,
    const enum voiced_audio_pipewire_boundary_error_code code,
    const char *message,
    struct voiced_audio_pipewire_error *error_out
) {
    clear_error(error_out);
    if (error_out == NULL) {
        return;
    }

    error_out->stage = (uint32_t)stage;
    error_out->domain = VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION;
    error_out->code = (int32_t)code;
    copy_error_message(error_out, message);
}

/** Copy one version string into a fixed caller-owned field. */
static uint32_t copy_version(
    char destination[VOICED_AUDIO_PIPEWIRE_VERSION_CAPACITY],
    const char *version
) {
    size_t version_size;

    if (version == NULL) {
        return 0;
    }

    version_size = strnlen(version, VOICED_AUDIO_PIPEWIRE_VERSION_CAPACITY);
    memcpy(destination, version, version_size);
    return (uint32_t)version_size;
}

#if PW_CHECK_VERSION(1, 0, 5)
/** Compare a dotted PipeWire version while permitting a package suffix. */
static bool version_is_at_least(
    const char *version,
    const uint32_t required_major,
    const uint32_t required_minor,
    const uint32_t required_micro
) {
    unsigned int major;
    unsigned int minor;
    unsigned int micro;

    if (version == NULL ||
        sscanf(version, "%u.%u.%u", &major, &minor, &micro) != 3) {
        return false;
    }

    if (major != required_major) {
        return major > required_major;
    }
    if (minor != required_minor) {
        return minor > required_minor;
    }
    return micro >= required_micro;
}
#endif

/**
 * Report capabilities shared by every stream created in this process.
 *
 * The header version determines which struct fields and inline SPA helpers this
 * host build could compile. The loaded library version independently guards
 * fields owned by that runtime. Only the intersection can authorize access to
 * `pw_buffer.time`; a host-matched older build remains usable in Header-only
 * mode. This is not a promise that a binary built on a newer Ubuntu release can
 * load against every older client library.
 */
void voiced_audio_pipewire_environment_read(
    struct voiced_audio_pipewire_environment *environment_out
) {
    const char *headers_version;
    const char *library_version;

    if (environment_out == NULL) {
        return;
    }

    memset(environment_out, 0, sizeof(*environment_out));
    headers_version = pw_get_headers_version();
    library_version = pw_get_library_version();
    environment_out->headers_version_size = copy_version(
        environment_out->headers_version,
        headers_version
    );
    environment_out->library_version_size = copy_version(
        environment_out->library_version,
        library_version
    );
    environment_out->timeline_validation =
        VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_HEADER_ONLY;

#if PW_CHECK_VERSION(1, 0, 5)
    if (version_is_at_least(library_version, 1, 0, 5)) {
        environment_out->timeline_validation =
            VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_FULL;
    }
#endif
}

/**
 * Create one PipeWire main loop.
 *
 * Zig owns a successful loop and must destroy it after all registered sources
 * and streams are gone. Null reports the constructor's immediate libc errno.
 */
struct pw_main_loop *voiced_audio_pipewire_main_loop_create(
    struct voiced_audio_pipewire_error *error_out
) {
    struct pw_main_loop *main_loop;

    clear_error(error_out);
    if (error_out == NULL) {
        return NULL;
    }

    main_loop = pw_main_loop_new(NULL);
    if (main_loop == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_MAIN_LOOP_CREATE,
            error_out
        );
    }
    return main_loop;
}

/**
 * Run a main loop until a callback quits it or PipeWire returns an error.
 *
 * This operation has no internal deadline. The supervising process must contain
 * a connected stream that stops producing progress without quitting the loop.
 */
int voiced_audio_pipewire_main_loop_run(
    struct pw_main_loop *main_loop,
    struct voiced_audio_pipewire_error *error_out
) {
    int result;

    clear_error(error_out);
    if (error_out == NULL) {
        return -EINVAL;
    }
    if (main_loop == NULL) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_MAIN_LOOP_RUN,
            -EINVAL,
            error_out
        );
        return -EINVAL;
    }

    result = pw_main_loop_run(main_loop);
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_MAIN_LOOP_RUN,
            result,
            error_out
        );
    }
    return result;
}

/**
 * Register one caller-owned descriptor with the PipeWire loop.
 *
 * PipeWire retains `context` and invokes `callback` on the loop thread whenever
 * `mask` is active. Passing `false` for close-on-destroy keeps descriptor
 * ownership with Zig. The public operations below supply semantic error stages
 * for the realtime callback eventfd and supervisor control socket.
 */
static struct spa_source *register_loop_io(
    struct pw_loop *loop,
    const int file_descriptor,
    const uint32_t mask,
    spa_source_io_func_t callback,
    void *context,
    const enum voiced_audio_pipewire_error_stage error_stage,
    struct voiced_audio_pipewire_error *error_out
) {
    struct spa_source *source;

    clear_error(error_out);
    if (error_out == NULL) {
        return NULL;
    }
    if (loop == NULL || file_descriptor < 0 || callback == NULL) {
        errno = EINVAL;
        capture_errno(error_stage, error_out);
        return NULL;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    source = pw_loop_add_io(
        loop,
        file_descriptor,
        mask,
        false,
        callback,
        context
    );
#pragma clang diagnostic pop

    if (source == NULL) {
        capture_errno(error_stage, error_out);
    }
    return source;
}

/** Register the realtime callback's terminal-outcome eventfd. */
struct spa_source *voiced_audio_pipewire_callback_event_register(
    struct pw_loop *loop,
    const int file_descriptor,
    const uint32_t mask,
    spa_source_io_func_t callback,
    void *context,
    struct voiced_audio_pipewire_error *error_out
) {
    return register_loop_io(
        loop,
        file_descriptor,
        mask,
        callback,
        context,
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CALLBACK_EVENT_REGISTER,
        error_out
    );
}

/** Register the supervisor socket that carries normal stop or cancel. */
struct spa_source *voiced_audio_pipewire_control_socket_register(
    struct pw_loop *loop,
    const int file_descriptor,
    const uint32_t mask,
    spa_source_io_func_t callback,
    void *context,
    struct voiced_audio_pipewire_error *error_out
) {
    return register_loop_io(
        loop,
        file_descriptor,
        mask,
        callback,
        context,
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CONTROL_SOCKET_REGISTER,
        error_out
    );
}

/** Remove a registered loop source without closing its caller-owned descriptor. */
void voiced_audio_pipewire_loop_io_unregister(
    struct pw_loop *loop,
    struct spa_source *source
) {
    if (loop == NULL || source == NULL) {
        return;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    pw_loop_destroy_source(loop, source);
#pragma clang diagnostic pop
}

/**
 * Set one property while retaining the semantic creation stage on error.
 *
 * PipeWire copies both strings during the call. The containing capture-stream
 * operation frees the property dictionary if this helper reports an error.
 */
static bool set_capture_property(
    struct pw_properties *properties,
    const char *key,
    const char *value,
    const enum voiced_audio_pipewire_error_stage error_stage,
    struct voiced_audio_pipewire_error *error_out
) {
    const int result = pw_properties_set(properties, key, value);

    if (result < 0) {
        capture_result(error_stage, result, error_out);
        return false;
    }
    return true;
}

/** Convert PipeWire's borrowed state message before dispatching to Zig. */
static void stream_state_changed(
    void *data,
    const enum pw_stream_state old_state,
    const enum pw_stream_state new_state,
    const char *message
) {
    const struct voiced_audio_pipewire_stream_callbacks *callbacks =
        (const struct voiced_audio_pipewire_stream_callbacks *)data;
    struct voiced_audio_pipewire_error error;

    clear_error(&error);
    if (message != NULL || new_state == PW_STREAM_STATE_ERROR) {
        error.stage = VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_STATE;
        error.domain = VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_CALLBACK;
        copy_error_message(&error, message);
    }
    callbacks->state_changed(
        callbacks->context,
        old_state,
        new_state,
        &error
    );
}

/** Forward a borrowed format event through the stable callback table. */
static void stream_parameter_changed(
    void *data,
    const uint32_t parameter_id,
    const struct spa_pod *parameter
) {
    const struct voiced_audio_pipewire_stream_callbacks *callbacks =
        (const struct voiced_audio_pipewire_stream_callbacks *)data;

    callbacks->format_changed(callbacks->context, parameter_id, parameter);
}

/** Forward one process notification through the stable callback table. */
static void stream_process(void *data) {
    const struct voiced_audio_pipewire_stream_callbacks *callbacks =
        (const struct voiced_audio_pipewire_stream_callbacks *)data;

    callbacks->process(callbacks->context);
}

/* PipeWire retains this immutable dispatch table for every Voiced stream. */
static const struct pw_stream_events stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .destroy = NULL,
    .state_changed = stream_state_changed,
    .control_info = NULL,
    .io_changed = NULL,
    .param_changed = stream_parameter_changed,
    .add_buffer = NULL,
    .remove_buffer = NULL,
    .process = stream_process,
    .drained = NULL,
    .command = NULL,
    .trigger_done = NULL,
};

/**
 * Create one unconnected Voiced capture stream.
 *
 * This single semantic operation creates and configures the private PipeWire
 * property dictionary, applies the optional target, then transfers dictionary
 * ownership to `pw_stream_new_simple`. Realtime processing selects PipeWire's
 * standard `client-rt.conf`; that configuration loads the thread utility which
 * asks RTKit for scheduler priority when direct Linux limits do not permit it.
 * The later `PW_STREAM_FLAG_RT_PROCESS` flag chooses the data thread but cannot
 * promote that thread by itself. `error_out.stage` identifies the exact failed
 * sub-operation without exposing those implementation calls as public APIs.
 * `callbacks` and its context must outlive the returned stream.
 */
struct pw_stream *voiced_audio_pipewire_capture_stream_create(
    struct pw_loop *loop,
    const char *target,
    struct voiced_audio_pipewire_stream_callbacks *callbacks,
    struct voiced_audio_pipewire_error *error_out
) {
    struct pw_properties *properties;
    struct pw_stream *stream;

    clear_error(error_out);
    if (error_out == NULL) {
        return NULL;
    }
    if (loop == NULL || callbacks == NULL ||
        callbacks->state_changed == NULL || callbacks->format_changed == NULL ||
        callbacks->process == NULL) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CREATE,
            -EINVAL,
            error_out
        );
        return NULL;
    }

    properties = pw_properties_new(NULL, NULL);
    if (properties == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTIES_CREATE,
            error_out
        );
        return NULL;
    }

    if (!set_capture_property(
        properties,
        PW_KEY_CONFIG_NAME,
        "client-rt.conf",
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_CONFIG,
        error_out
    )) {
        pw_properties_free(properties);
        return NULL;
    }

    if (!set_capture_property(
        properties,
        PW_KEY_MEDIA_TYPE,
        "Audio",
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_TYPE,
        error_out
    ) || !set_capture_property(
        properties,
        PW_KEY_MEDIA_CATEGORY,
        "Capture",
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_CATEGORY,
        error_out
    ) || !set_capture_property(
        properties,
        PW_KEY_MEDIA_ROLE,
        "Communication",
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_ROLE,
        error_out
    )) {
        pw_properties_free(properties);
        return NULL;
    }

    if (target != NULL) {
        if (!set_capture_property(
            properties,
            PW_KEY_TARGET_OBJECT,
            target,
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_TARGET,
            error_out
        )) {
            pw_properties_free(properties);
            return NULL;
        }

#if !PW_CHECK_VERSION(0, 3, 64)
        /*
         * Ubuntu 22.04's media-session reads `node.target`, not the newer
         * `target.object`. Supplying only the newer key lets an unknown name
         * silently fall back to the default source—the worst possible result
         * for explicit microphone selection. Old builds therefore publish both
         * spellings. PipeWire deprecated the legacy key in 0.3.64, so modern
         * builds omit it and retain only `target.object`.
         */
        if (!set_capture_property(
            properties,
            PW_KEY_NODE_TARGET,
            target,
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_TARGET,
            error_out
        )) {
            pw_properties_free(properties);
            return NULL;
        }
#endif
    }

    /*
     * `pw_stream_new_simple` takes ownership as soon as it is called, including
     * every error path. Do not free `properties` after this point.
     */
    stream = pw_stream_new_simple(
        loop,
        "voiced-audio",
        properties,
        &stream_events,
        callbacks
    );
    if (stream == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CREATE,
            error_out
        );
    }
    return stream;
}

/** Build Voiced's fixed raw-audio offer in caller-owned SPA pod storage. */
static const struct spa_pod *build_format_offer(
    struct spa_pod_builder *builder,
    const uint32_t sample_rate_hz
) {
    struct spa_audio_info_raw format;

    if (builder == NULL || sample_rate_hz == 0) {
        return NULL;
    }

    memset(&format, 0, sizeof(format));
    format.format = SPA_AUDIO_FORMAT_F32;
    format.rate = sample_rate_hz;
    format.channels = 1;
    format.position[0] = SPA_AUDIO_CHANNEL_MONO;

    return spa_format_audio_raw_build(
        builder,
        SPA_PARAM_EnumFormat,
        &format
    );
}

/** Build bounded buffer and optional Header requests in one SPA pod builder. */
static bool build_audio_buffer_parameters(
    struct spa_pod_builder *builder,
    const uint32_t samples_per_buffer_max,
    const struct spa_pod **buffer_parameter_out,
    const struct spa_pod **metadata_parameter_out
) {
    int32_t buffer_bytes_max;

    if (builder == NULL || samples_per_buffer_max == 0 ||
        samples_per_buffer_max > INT32_MAX / (int32_t)sizeof(float) ||
        buffer_parameter_out == NULL || metadata_parameter_out == NULL) {
        return false;
    }

    buffer_bytes_max = (int32_t)(samples_per_buffer_max * sizeof(float));

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    *buffer_parameter_out =
        (const struct spa_pod *)spa_pod_builder_add_object(
            builder,
            SPA_TYPE_OBJECT_ParamBuffers,
            SPA_PARAM_Buffers,
            SPA_PARAM_BUFFERS_buffers,
            SPA_POD_CHOICE_RANGE_Int(8, 2, 32),
            SPA_PARAM_BUFFERS_blocks,
            SPA_POD_Int(1),
            SPA_PARAM_BUFFERS_size,
            SPA_POD_CHOICE_RANGE_Int(
                buffer_bytes_max,
                (int32_t)sizeof(float),
                buffer_bytes_max
            ),
            SPA_PARAM_BUFFERS_stride,
            SPA_POD_Int((int32_t)sizeof(float))
        );
    *metadata_parameter_out =
        (const struct spa_pod *)spa_pod_builder_add_object(
            builder,
            SPA_TYPE_OBJECT_ParamMeta,
            SPA_PARAM_Meta,
            SPA_PARAM_META_type,
            SPA_POD_Id(SPA_META_Header),
            SPA_PARAM_META_size,
            SPA_POD_Int((int32_t)sizeof(struct spa_meta_header))
        );
#pragma clang diagnostic pop

    return *buffer_parameter_out != NULL && *metadata_parameter_out != NULL;
}

/**
 * Connect one capture stream with Voiced's complete initial negotiation policy.
 *
 * The operation offers native float32/mono at `sample_rate_hz`, bounds each
 * callback buffer, requests optional Header metadata, enables automatic target
 * linking and mapped buffers, disables implicit reconnect, and asks
 * PipeWire to dispatch processing on its realtime data thread. A zero result
 * accepts only this asynchronous request; callbacks and supervisor deadlines
 * establish actual readiness.
 */
int voiced_audio_pipewire_capture_stream_connect(
    struct pw_stream *stream,
    const uint32_t sample_rate_hz,
    const uint32_t samples_per_buffer_max,
    struct voiced_audio_pipewire_error *error_out
) {
    uint64_t storage[96];
    struct spa_pod_builder builder;
    const struct spa_pod *format_offer;
    const struct spa_pod *buffer_parameter;
    const struct spa_pod *metadata_parameter;
    const struct spa_pod *parameters[3];
    enum pw_stream_flags stream_flags;
    int result;

    clear_error(error_out);
    if (error_out == NULL) {
        return -EINVAL;
    }
    if (stream == NULL || sample_rate_hz == 0 ||
        samples_per_buffer_max == 0 ||
        samples_per_buffer_max > INT32_MAX / (int32_t)sizeof(float)) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CONNECT,
            -EINVAL,
            error_out
        );
        return -EINVAL;
    }

    spa_pod_builder_init(&builder, storage, sizeof(storage));
    format_offer = build_format_offer(&builder, sample_rate_hz);
    if (format_offer == NULL || !build_audio_buffer_parameters(
        &builder,
        samples_per_buffer_max,
        &buffer_parameter,
        &metadata_parameter
    )) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CONNECT,
            -ENOSPC,
            error_out
        );
        return -ENOSPC;
    }

    stream_flags = (enum pw_stream_flags)(
        PW_STREAM_FLAG_AUTOCONNECT |
        PW_STREAM_FLAG_MAP_BUFFERS |
        PW_STREAM_FLAG_DONT_RECONNECT |
        PW_STREAM_FLAG_RT_PROCESS
    );

    /* PipeWire's connect API requires a raw C array of borrowed pod pointers. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
    parameters[0] = format_offer;
    parameters[1] = buffer_parameter;
    parameters[2] = metadata_parameter;
    result = pw_stream_connect(
        stream,
        PW_DIRECTION_INPUT,
        PW_ID_ANY,
        stream_flags,
        parameters,
        3
    );
#pragma clang diagnostic pop

    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CONNECT,
            result,
            error_out
        );
    }
    return result;
}

/** Copy the server version delivered by the core's asynchronous info event. */
static void server_info_received(
    void *data,
    const struct pw_core_info *info
) {
    struct voiced_audio_pipewire_server_observer *observer =
        (struct voiced_audio_pipewire_server_observer *)data;

    if (observer == NULL || info == NULL) {
        return;
    }

    memset(observer->version, 0, sizeof(observer->version));
    observer->version_size = copy_version(observer->version, info->version);
}

/* Only the core info event is needed to retain the remote server version. */
static const struct pw_core_events server_events = {
    .version = PW_VERSION_CORE_EVENTS,
    .info = server_info_received,
};

/**
 * Observe the PipeWire server used by one capture stream.
 *
 * `pw_get_library_version` describes the local client library, not the remote
 * daemon. The stream's core emits its server version asynchronously after the
 * main loop starts. The observer owns its hook and fixed version bytes until
 * the caller unregisters it before destroying the stream.
 */
int voiced_audio_pipewire_capture_server_observer_register(
    struct pw_stream *stream,
    struct voiced_audio_pipewire_server_observer *observer,
    struct voiced_audio_pipewire_error *error_out
) {
    struct pw_core *core;
    int result;

    clear_error(error_out);
    if (error_out == NULL) {
        return -EINVAL;
    }
    if (stream == NULL || observer == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SERVER_OBSERVER_REGISTER,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SERVER_OBSERVER_ARGUMENTS,
            "Server observation requires a stream and observer storage",
            error_out
        );
        return -EINVAL;
    }

    memset(observer, 0, sizeof(*observer));
    core = pw_stream_get_core(stream);
    if (core == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SERVER_OBSERVER_REGISTER,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SERVER_CORE,
            "The capture stream did not expose its PipeWire core",
            error_out
        );
        return -EINVAL;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_core_add_listener(
        core,
        &observer->listener,
        &server_events,
        observer
    );
#pragma clang diagnostic pop
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SERVER_OBSERVER_REGISTER,
            result,
            error_out
        );
        return result;
    }

    observer->is_registered = 1;
    return 0;
}

/** Stop server-version callbacks before the observed stream is destroyed. */
void voiced_audio_pipewire_capture_server_observer_unregister(
    struct voiced_audio_pipewire_server_observer *observer
) {
    if (observer == NULL || observer->is_registered == 0) {
        return;
    }

    spa_hook_remove(&observer->listener);
    observer->is_registered = 0;
}

/*
 * A capture stream is itself one PipeWire node. Its incoming Link identifies
 * the node currently supplying samples:
 *
 *     microphone/source node -- Link --> Voiced capture-stream node
 *
 * `target.object` expresses a routing request, not an enduring identity. On a
 * real WirePlumber 0.5 graph, unplugging an explicitly targeted USB microphone
 * removed its Link and silently created another from the new default source,
 * despite `node.dont-reconnect=true`. Audio buffers do not carry their source
 * identity, so Voiced must observe the graph beside the stream and close the
 * realtime copy gate as soon as that relationship changes.
 *
 * Registry properties are callback-owned. This observer retains only bounded
 * source, device, and Link records needed to join the actual source node to its
 * stable `device.serial`. PipeWire global IDs and `object.serial` values remain
 * useful diagnostics, but both are expected to change after unplug/replug.
 */
#define SOURCE_NODES_CAPACITY 64
#define SOURCE_DEVICES_CAPACITY 64
#define SOURCE_LINKS_CAPACITY 128
#define ACTIVE_SOURCE_LINKS_CAPACITY 8

/* All indexing below is bounded by these fixed capacities and checked first. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"

struct source_node_record {
    uint64_t object_serial;
    uint32_t is_present;
    uint32_t has_invalid_identity;
    uint32_t id;
    uint32_t version;
    uint32_t device_id;
    uint32_t name_size;
    uint32_t description_size;
    uint32_t reserved;
    char name[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
    char description[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
};

struct source_device_record {
    uint64_t object_serial;
    uint32_t is_present;
    uint32_t has_invalid_identity;
    uint32_t id;
    uint32_t version;
    uint32_t serial_size;
    uint32_t description_size;
    char serial[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
    char description[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
};

struct source_link_record {
    uint32_t is_present;
    uint32_t id;
    uint32_t input_node_id;
    uint32_t output_node_id;
};

struct voiced_audio_pipewire_source_observer {
    struct pw_stream *stream;
    struct pw_registry *registry;
    struct pw_node *source_node_proxy;
    struct pw_device *source_device_proxy;
    struct spa_hook registry_listener;
    struct spa_hook source_node_listener;
    struct spa_hook source_device_listener;
    struct voiced_audio_pipewire_source_callbacks callbacks;
    struct voiced_audio_pipewire_source_identity identity;
    struct source_node_record source_nodes[SOURCE_NODES_CAPACITY];
    struct source_device_record source_devices[SOURCE_DEVICES_CAPACITY];
    struct source_link_record links[SOURCE_LINKS_CAPACITY];
    char expected_device_serial[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
    uint32_t active_link_ids[ACTIVE_SOURCE_LINKS_CAPACITY];
    uint32_t listener_is_registered;
    uint32_t source_node_listener_is_registered;
    uint32_t source_device_listener_is_registered;
    uint32_t terminal_event_was_sent;
    uint32_t active_source_node_id;
    uint32_t active_source_device_id;
    uint32_t active_links_count;
    uint32_t expected_device_serial_size;
    uint32_t linked_event_was_sent;
    uint32_t reserved;
    uint32_t reserved_2;
    uint32_t reserved_3;
};

/** Copy one complete property value into observer-owned identity storage. */
static bool copy_identity_text(
    char destination[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY],
    uint32_t *destination_size,
    const char *source
) {
    size_t source_size;

    memset(destination, 0, VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY);
    *destination_size = 0;
    if (source == NULL) {
        return true;
    }

    source_size = strnlen(
        source,
        VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY
    );
    if (source_size == VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY) {
        return false;
    }

    memcpy(destination, source, source_size);
    *destination_size = (uint32_t)source_size;
    return true;
}

/** Find a retained source node by its current PipeWire global ID. */
static struct source_node_record *find_source_node(
    struct voiced_audio_pipewire_source_observer *observer,
    const uint32_t id
) {
    uint32_t index;

    for (index = 0; index < SOURCE_NODES_CAPACITY; index += 1) {
        if (observer->source_nodes[index].is_present &&
            observer->source_nodes[index].id == id) {
            return &observer->source_nodes[index];
        }
    }
    return NULL;
}

/** Find a retained physical or virtual audio device by global ID. */
static struct source_device_record *find_source_device(
    struct voiced_audio_pipewire_source_observer *observer,
    const uint32_t id
) {
    uint32_t index;

    for (index = 0; index < SOURCE_DEVICES_CAPACITY; index += 1) {
        if (observer->source_devices[index].is_present &&
            observer->source_devices[index].id == id) {
            return &observer->source_devices[index];
        }
    }
    return NULL;
}

/** Find one observed Link by its global ID. */
static struct source_link_record *find_source_link(
    struct voiced_audio_pipewire_source_observer *observer,
    const uint32_t id
) {
    uint32_t index;

    for (index = 0; index < SOURCE_LINKS_CAPACITY; index += 1) {
        if (observer->links[index].is_present &&
            observer->links[index].id == id) {
            return &observer->links[index];
        }
    }
    return NULL;
}

/** Send one graph event through the observer's stable semantic callback. */
static void send_source_event(
    struct voiced_audio_pipewire_source_observer *observer,
    const enum voiced_audio_pipewire_source_event event,
    const uint32_t previous_source_node_id,
    const uint32_t current_source_node_id,
    const struct voiced_audio_pipewire_error *error
) {
    struct voiced_audio_pipewire_error empty_error;

    if (observer->terminal_event_was_sent) {
        return;
    }
    if (event != VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINKED) {
        observer->terminal_event_was_sent = true;
    }

    clear_error(&empty_error);
    observer->callbacks.event(
        observer->callbacks.context,
        (uint32_t)event,
        previous_source_node_id,
        current_source_node_id,
        error == NULL ? &empty_error : error
    );
}

/** Stop capture when bounded registry state can no longer prove its source. */
static void fail_source_observation(
    struct voiced_audio_pipewire_source_observer *observer,
    const enum voiced_audio_pipewire_boundary_error_code code,
    const char *message
) {
    struct voiced_audio_pipewire_error error;

    capture_validation(
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVATION,
        code,
        message,
        &error
    );
    send_source_event(
        observer,
        VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_OBSERVATION_ERROR,
        observer->active_source_node_id,
        PW_ID_ANY,
        &error
    );
}

/** Refresh the immutable first-linked identity as registry joins arrive. */
static bool refresh_source_identity(
    struct voiced_audio_pipewire_source_observer *observer
) {
    struct voiced_audio_pipewire_source_identity *identity =
        &observer->identity;
    struct source_node_record *source_node;
    struct source_device_record *source_device;

    if (identity->is_resolved == 0) {
        return true;
    }

    source_node = find_source_node(observer, identity->node_id);
    if (source_node == NULL) {
        return true;
    }
    if (source_node->has_invalid_identity) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_IDENTITY_TEXT,
            "The linked source published malformed or oversized identity properties"
        );
        return false;
    }

    identity->node_object_serial = source_node->object_serial;
    identity->device_id = source_node->device_id;
    identity->node_name_size = source_node->name_size;
    identity->node_description_size = source_node->description_size;
    memcpy(identity->node_name, source_node->name, sizeof(identity->node_name));
    memcpy(
        identity->node_description,
        source_node->description,
        sizeof(identity->node_description)
    );

    if (source_node->device_id == PW_ID_ANY) {
        return true;
    }
    source_device = find_source_device(observer, source_node->device_id);
    if (source_device == NULL) {
        return true;
    }
    if (source_device->has_invalid_identity) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_IDENTITY_TEXT,
            "The linked source device published malformed or oversized identity properties"
        );
        return false;
    }

    identity->device_object_serial = source_device->object_serial;
    identity->device_serial_size = source_device->serial_size;
    identity->device_description_size = source_device->description_size;
    memcpy(
        identity->device_serial,
        source_device->serial,
        sizeof(identity->device_serial)
    );
    memcpy(
        identity->device_description,
        source_device->description,
        sizeof(identity->device_description)
    );
    return true;
}

/** Open the sample gate only after the configured Device identity matches. */
static void authorize_observed_source(
    struct voiced_audio_pipewire_source_observer *observer
) {
    if (observer->terminal_event_was_sent || observer->linked_event_was_sent ||
        observer->active_source_node_id == PW_ID_ANY) {
        return;
    }

    if (observer->expected_device_serial_size > 0) {
        if (observer->identity.device_serial_size == 0) {
            return;
        }
        if (observer->identity.device_serial_size !=
                observer->expected_device_serial_size ||
            memcmp(
                observer->identity.device_serial,
                observer->expected_device_serial,
                observer->expected_device_serial_size
            ) != 0) {
            fail_source_observation(
                observer,
                VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_CONFIGURED_DEVICE_MISMATCH,
                "PipeWire linked a source from a different Device than configured"
            );
            return;
        }
    }

    observer->linked_event_was_sent = true;
    send_source_event(
        observer,
        VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINKED,
        PW_ID_ANY,
        observer->active_source_node_id,
        NULL
    );
}

/** Retain the full properties emitted by the bound active Device proxy. */
static void source_device_info_received(
    void *data,
    const struct pw_device_info *info
) {
    struct voiced_audio_pipewire_source_observer *observer =
        (struct voiced_audio_pipewire_source_observer *)data;
    struct source_device_record *record;
    const char *object_serial;
    const char *device_serial;
    const char *device_description;

    if (observer == NULL || info == NULL || info->props == NULL ||
        observer->terminal_event_was_sent) {
        return;
    }

    record = find_source_device(observer, observer->active_source_device_id);
    if (record == NULL || info->id != observer->active_source_device_id) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_PROPERTY_VALUE,
            "The bound source Device reported an unexpected global ID"
        );
        return;
    }

    record->has_invalid_identity = false;
    object_serial = spa_dict_lookup(info->props, PW_KEY_OBJECT_SERIAL);
    device_serial = spa_dict_lookup(info->props, PW_KEY_DEVICE_SERIAL);
    device_description = spa_dict_lookup(
        info->props,
        PW_KEY_DEVICE_DESCRIPTION
    );
    if ((object_serial != NULL &&
         !spa_atou64(object_serial, &record->object_serial, 10)) ||
        (device_serial != NULL &&
         !copy_identity_text(
             record->serial,
             &record->serial_size,
             device_serial
         )) ||
        (device_description != NULL &&
         !copy_identity_text(
             record->description,
             &record->description_size,
             device_description
         ))) {
        record->has_invalid_identity = true;
    }
    if (observer->expected_device_serial_size > 0 && record->serial_size == 0) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_CONFIGURED_DEVICE_MISMATCH,
            "PipeWire linked a source Device without the configured stable serial"
        );
        return;
    }
    if (refresh_source_identity(observer)) {
        authorize_observed_source(observer);
    }
}

/* The bound Device's info event supplies properties omitted by registry globals. */
static const struct pw_device_events source_device_events = {
    .version = PW_VERSION_DEVICE_EVENTS,
    .info = source_device_info_received,
    .param = NULL,
};

/** Bind only the active source's Device to obtain its stable serial property. */
static bool bind_active_source_device(
    struct voiced_audio_pipewire_source_observer *observer
) {
    struct source_device_record *record;
    uint32_t bound_version;
    int result;

    if (observer->identity.is_resolved == 0 ||
        observer->identity.device_id == PW_ID_ANY) {
        return true;
    }
    if (observer->source_device_proxy != NULL) {
        if (observer->active_source_device_id != observer->identity.device_id) {
            fail_source_observation(
                observer,
                VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_DEVICE_BIND,
                "The active source changed its Device global ID"
            );
            return false;
        }
        return true;
    }

    record = find_source_device(observer, observer->identity.device_id);
    if (record == NULL) {
        return true;
    }
    bound_version = record->version < PW_VERSION_DEVICE
        ? record->version
        : PW_VERSION_DEVICE;
    observer->source_device_proxy = (struct pw_device *)pw_registry_bind(
        observer->registry,
        record->id,
        PW_TYPE_INTERFACE_Device,
        bound_version,
        0
    );
    if (observer->source_device_proxy == NULL) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_DEVICE_BIND,
            "PipeWire could not bind the active source Device"
        );
        return false;
    }
    observer->active_source_device_id = record->id;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_device_add_listener(
        observer->source_device_proxy,
        &observer->source_device_listener,
        &source_device_events,
        observer
    );
#pragma clang diagnostic pop
    if (result < 0) {
        struct voiced_audio_pipewire_error error;

        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVATION,
            result,
            &error
        );
        pw_proxy_destroy((struct pw_proxy *)observer->source_device_proxy);
        observer->source_device_proxy = NULL;
        observer->active_source_device_id = PW_ID_ANY;
        send_source_event(
            observer,
            VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_OBSERVATION_ERROR,
            observer->active_source_node_id,
            PW_ID_ANY,
            &error
        );
        return false;
    }

    observer->source_device_listener_is_registered = true;
    return true;
}

/** Retain full source properties omitted by older registry global events. */
static void source_node_info_received(
    void *data,
    const struct pw_node_info *info
) {
    struct voiced_audio_pipewire_source_observer *observer =
        (struct voiced_audio_pipewire_source_observer *)data;
    struct source_node_record *record;
    const char *object_serial;
    const char *device_id;
    const char *node_name;
    const char *node_description;

    if (observer == NULL || info == NULL || info->props == NULL ||
        observer->terminal_event_was_sent) {
        return;
    }

    record = find_source_node(observer, observer->active_source_node_id);
    if (record == NULL || info->id != observer->active_source_node_id) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_PROPERTY_VALUE,
            "The bound source Node reported an unexpected global ID"
        );
        return;
    }

    record->has_invalid_identity = false;
    object_serial = spa_dict_lookup(info->props, PW_KEY_OBJECT_SERIAL);
    device_id = spa_dict_lookup(info->props, PW_KEY_DEVICE_ID);
    node_name = spa_dict_lookup(info->props, PW_KEY_NODE_NAME);
    node_description = spa_dict_lookup(info->props, PW_KEY_NODE_DESCRIPTION);
    if ((object_serial != NULL &&
         !spa_atou64(object_serial, &record->object_serial, 10)) ||
        (device_id != NULL && !spa_atou32(device_id, &record->device_id, 10)) ||
        (node_name != NULL &&
         !copy_identity_text(record->name, &record->name_size, node_name)) ||
        (node_description != NULL &&
         !copy_identity_text(
             record->description,
             &record->description_size,
             node_description
         ))) {
        record->has_invalid_identity = true;
    }
    if (observer->expected_device_serial_size > 0 &&
        record->device_id == PW_ID_ANY) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_CONFIGURED_DEVICE_MISMATCH,
            "PipeWire linked a source Node without the configured Device"
        );
        return;
    }
    if (refresh_source_identity(observer) &&
        bind_active_source_device(observer)) {
        authorize_observed_source(observer);
    }
}

/* The bound Node's info event supplies its complete identity and Device join. */
static const struct pw_node_events source_node_events = {
    .version = PW_VERSION_NODE_EVENTS,
    .info = source_node_info_received,
    .param = NULL,
};

/** Bind the active source Node without subscribing to media parameters. */
static bool bind_active_source_node(
    struct voiced_audio_pipewire_source_observer *observer
) {
    struct source_node_record *record;
    uint32_t bound_version;
    int result;

    if (observer->active_source_node_id == PW_ID_ANY) {
        return true;
    }
    if (observer->source_node_proxy != NULL) {
        return true;
    }

    record = find_source_node(observer, observer->active_source_node_id);
    if (record == NULL) {
        return true;
    }
    bound_version = record->version < PW_VERSION_NODE
        ? record->version
        : PW_VERSION_NODE;
    observer->source_node_proxy = (struct pw_node *)pw_registry_bind(
        observer->registry,
        record->id,
        PW_TYPE_INTERFACE_Node,
        bound_version,
        0
    );
    if (observer->source_node_proxy == NULL) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_NODE_BIND,
            "PipeWire could not bind the active source Node"
        );
        return false;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_node_add_listener(
        observer->source_node_proxy,
        &observer->source_node_listener,
        &source_node_events,
        observer
    );
#pragma clang diagnostic pop
    if (result < 0) {
        struct voiced_audio_pipewire_error error;

        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVATION,
            result,
            &error
        );
        pw_proxy_destroy((struct pw_proxy *)observer->source_node_proxy);
        observer->source_node_proxy = NULL;
        send_source_event(
            observer,
            VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_OBSERVATION_ERROR,
            observer->active_source_node_id,
            PW_ID_ANY,
            &error
        );
        return false;
    }

    observer->source_node_listener_is_registered = true;
    return true;
}

/** Add one Link from the already selected source without changing identity. */
static bool retain_active_source_link(
    struct voiced_audio_pipewire_source_observer *observer,
    const uint32_t link_id
) {
    uint32_t index;

    for (index = 0; index < observer->active_links_count; index += 1) {
        if (observer->active_link_ids[index] == link_id) {
            return true;
        }
    }
    if (observer->active_links_count == ACTIVE_SOURCE_LINKS_CAPACITY) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_REGISTRY_CAPACITY,
            "The capture stream exceeded its bounded active-Link capacity"
        );
        return false;
    }

    observer->active_link_ids[observer->active_links_count] = link_id;
    observer->active_links_count += 1;
    return true;
}

/** Lock capture to the source named by one incoming stream Link. */
static void consider_source_link(
    struct voiced_audio_pipewire_source_observer *observer,
    const struct source_link_record *link
) {
    const uint32_t stream_node_id = pw_stream_get_node_id(observer->stream);

    if (observer->terminal_event_was_sent || stream_node_id == PW_ID_ANY ||
        link->input_node_id != stream_node_id) {
        return;
    }

    if (observer->active_source_node_id == PW_ID_ANY) {
        observer->active_source_node_id = link->output_node_id;
        observer->identity.is_resolved = 1;
        observer->identity.node_id = link->output_node_id;
        if (!retain_active_source_link(observer, link->id) ||
            !refresh_source_identity(observer) ||
            !bind_active_source_node(observer) ||
            !bind_active_source_device(observer)) {
            return;
        }
        authorize_observed_source(observer);
        return;
    }

    if (link->output_node_id != observer->active_source_node_id) {
        send_source_event(
            observer,
            VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_CHANGED,
            observer->active_source_node_id,
            link->output_node_id,
            NULL
        );
        return;
    }

    (void)retain_active_source_link(observer, link->id);
}

/** Reconsider retained Links after the stream receives its own global ID. */
static void discover_source_links(
    struct voiced_audio_pipewire_source_observer *observer
) {
    uint32_t index;

    for (index = 0; index < SOURCE_LINKS_CAPACITY; index += 1) {
        if (observer->links[index].is_present) {
            consider_source_link(observer, &observer->links[index]);
        }
    }
}

/** Retain one Audio/Source global's diagnostic and device join properties. */
static void retain_source_node(
    struct voiced_audio_pipewire_source_observer *observer,
    const uint32_t id,
    const uint32_t version,
    const struct spa_dict *properties
) {
    struct source_node_record *record;
    const char *object_serial;
    const char *device_id;
    uint32_t index;

    record = find_source_node(observer, id);
    if (record == NULL) {
        for (index = 0; index < SOURCE_NODES_CAPACITY; index += 1) {
            if (!observer->source_nodes[index].is_present) {
                record = &observer->source_nodes[index];
                break;
            }
        }
    }
    if (record == NULL) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_REGISTRY_CAPACITY,
            "The PipeWire graph exceeded Voiced's bounded Audio/Source capacity"
        );
        return;
    }

    memset(record, 0, sizeof(*record));
    record->is_present = true;
    record->id = id;
    record->version = version;
    record->device_id = PW_ID_ANY;
    object_serial = spa_dict_lookup(properties, PW_KEY_OBJECT_SERIAL);
    device_id = spa_dict_lookup(properties, PW_KEY_DEVICE_ID);
    if ((object_serial != NULL &&
         !spa_atou64(object_serial, &record->object_serial, 10)) ||
        (device_id != NULL && !spa_atou32(device_id, &record->device_id, 10)) ||
        !copy_identity_text(
            record->name,
            &record->name_size,
            spa_dict_lookup(properties, PW_KEY_NODE_NAME)
        ) ||
        !copy_identity_text(
            record->description,
            &record->description_size,
            spa_dict_lookup(properties, PW_KEY_NODE_DESCRIPTION)
        )) {
        record->has_invalid_identity = true;
    }

    if (observer->active_source_node_id == id &&
        refresh_source_identity(observer) &&
        bind_active_source_node(observer) &&
        bind_active_source_device(observer)) {
        authorize_observed_source(observer);
    }
}

/** Retain one Audio/Device global for the stable device-serial join. */
static void retain_source_device(
    struct voiced_audio_pipewire_source_observer *observer,
    const uint32_t id,
    const uint32_t version,
    const struct spa_dict *properties
) {
    struct source_device_record *record;
    const char *object_serial;
    uint32_t index;

    record = find_source_device(observer, id);
    if (record == NULL) {
        for (index = 0; index < SOURCE_DEVICES_CAPACITY; index += 1) {
            if (!observer->source_devices[index].is_present) {
                record = &observer->source_devices[index];
                break;
            }
        }
    }
    if (record == NULL) {
        fail_source_observation(
            observer,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_REGISTRY_CAPACITY,
            "The PipeWire graph exceeded Voiced's bounded Audio/Device capacity"
        );
        return;
    }

    memset(record, 0, sizeof(*record));
    record->is_present = true;
    record->id = id;
    record->version = version;
    object_serial = spa_dict_lookup(properties, PW_KEY_OBJECT_SERIAL);
    if ((object_serial != NULL &&
         !spa_atou64(object_serial, &record->object_serial, 10)) ||
        !copy_identity_text(
            record->serial,
            &record->serial_size,
            spa_dict_lookup(properties, PW_KEY_DEVICE_SERIAL)
        ) ||
        !copy_identity_text(
            record->description,
            &record->description_size,
            spa_dict_lookup(properties, PW_KEY_DEVICE_DESCRIPTION)
        )) {
        record->has_invalid_identity = true;
    }

    if (observer->identity.is_resolved != 0 &&
        observer->identity.device_id == id &&
        refresh_source_identity(observer) &&
        bind_active_source_device(observer)) {
        authorize_observed_source(observer);
    }
}

/** Retain one Link until the stream's own node ID makes it relevant or not. */
static void retain_source_link(
    struct voiced_audio_pipewire_source_observer *observer,
    const uint32_t id,
    const struct spa_dict *properties
) {
    struct source_link_record *record;
    const char *input_node;
    const char *output_node;
    const uint32_t stream_node_id = pw_stream_get_node_id(observer->stream);
    uint32_t parsed_input_node;
    uint32_t parsed_output_node;
    uint32_t index;

    input_node = spa_dict_lookup(properties, PW_KEY_LINK_INPUT_NODE);
    if (input_node == NULL || !spa_atou32(input_node, &parsed_input_node, 10)) {
        return;
    }
    if (stream_node_id != PW_ID_ANY && parsed_input_node != stream_node_id) {
        return;
    }

    output_node = spa_dict_lookup(properties, PW_KEY_LINK_OUTPUT_NODE);
    if (output_node == NULL ||
        !spa_atou32(output_node, &parsed_output_node, 10)) {
        if (stream_node_id != PW_ID_ANY && parsed_input_node == stream_node_id) {
            fail_source_observation(
                observer,
                VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_PROPERTY_VALUE,
                "The capture stream's Link omitted a valid output-node ID"
            );
        }
        return;
    }

    record = find_source_link(observer, id);
    if (record == NULL) {
        for (index = 0; index < SOURCE_LINKS_CAPACITY; index += 1) {
            if (!observer->links[index].is_present) {
                record = &observer->links[index];
                break;
            }
        }
    }
    if (record == NULL) {
        if (stream_node_id != PW_ID_ANY && parsed_input_node == stream_node_id) {
            fail_source_observation(
                observer,
                VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_REGISTRY_CAPACITY,
                "The capture stream exceeded Voiced's bounded Link capacity"
            );
        }
        return;
    }

    record->is_present = true;
    record->id = id;
    record->input_node_id = parsed_input_node;
    record->output_node_id = parsed_output_node;
    consider_source_link(observer, record);
}

/** Classify one registry global and retain only source-identity coordinates. */
static void source_registry_global(
    void *data,
    const uint32_t id,
    const uint32_t permissions,
    const char *type,
    const uint32_t version,
    const struct spa_dict *properties
) {
    struct voiced_audio_pipewire_source_observer *observer =
        (struct voiced_audio_pipewire_source_observer *)data;
    const char *media_class;

    (void)permissions;
    (void)version;
    if (observer == NULL || type == NULL || properties == NULL ||
        observer->terminal_event_was_sent) {
        return;
    }

    media_class = spa_dict_lookup(properties, PW_KEY_MEDIA_CLASS);
    if (strcmp(type, PW_TYPE_INTERFACE_Node) == 0 && media_class != NULL &&
        strncmp(media_class, "Audio/Source", 12) == 0 &&
        (media_class[12] == '\0' || media_class[12] == '/')) {
        retain_source_node(observer, id, version, properties);
    } else if (strcmp(type, PW_TYPE_INTERFACE_Device) == 0 &&
               media_class != NULL &&
               strcmp(media_class, "Audio/Device") == 0) {
        retain_source_device(observer, id, version, properties);
    } else if (strcmp(type, PW_TYPE_INTERFACE_Link) == 0) {
        retain_source_link(observer, id, properties);
    }

    discover_source_links(observer);
}

/** Remove cached globals and terminate if the active source relationship left. */
static void source_registry_global_removed(void *data, const uint32_t id) {
    struct voiced_audio_pipewire_source_observer *observer =
        (struct voiced_audio_pipewire_source_observer *)data;
    struct source_link_record *link;
    struct source_node_record *source_node;
    struct source_device_record *source_device;
    uint32_t active_index;

    if (observer == NULL) {
        return;
    }

    link = find_source_link(observer, id);
    if (link != NULL) {
        link->is_present = false;
    }
    for (active_index = 0;
         active_index < observer->active_links_count;
         active_index += 1) {
        if (observer->active_link_ids[active_index] != id) {
            continue;
        }

        observer->active_links_count -= 1;
        observer->active_link_ids[active_index] =
            observer->active_link_ids[observer->active_links_count];
        if (observer->active_links_count == 0) {
            send_source_event(
                observer,
                VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINK_REMOVED,
                observer->active_source_node_id,
                PW_ID_ANY,
                NULL
            );
        }
        break;
    }

    source_node = find_source_node(observer, id);
    if (source_node != NULL) {
        source_node->is_present = false;
    }
    if (id == observer->active_source_node_id) {
        send_source_event(
            observer,
            VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_NODE_REMOVED,
            observer->active_source_node_id,
            PW_ID_ANY,
            NULL
        );
    }

    source_device = find_source_device(observer, id);
    if (source_device != NULL) {
        source_device->is_present = false;
    }
    if (id == observer->active_source_device_id) {
        send_source_event(
            observer,
            VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_DEVICE_REMOVED,
            observer->active_source_node_id,
            PW_ID_ANY,
            NULL
        );
    }
}

/* Registry globals and removals are the complete source-observation surface. */
static const struct pw_registry_events source_registry_events = {
    .version = PW_VERSION_REGISTRY_EVENTS,
    .global = source_registry_global,
    .global_remove = source_registry_global_removed,
};

/**
 * Begin observing the concrete source linked to one connecting capture stream.
 *
 * `pw_stream_connect` creates the stream's private core, so this operation must
 * follow a successful connect request. It must still precede the first main-loop
 * run: no graph event or process callback can dispatch in that window, and the
 * registry then enumerates existing sources, devices, and the new stream Link
 * before Zig is allowed to copy samples. When `expected_device_serial` is
 * supplied, the observer keeps the sample gate closed until the linked Node's
 * Device reports that exact stable serial. The returned observer owns its
 * registry proxy and bounded caches.
 */
struct voiced_audio_pipewire_source_observer *
voiced_audio_pipewire_capture_source_observer_create(
    struct pw_stream *stream,
    const char *expected_device_serial,
    const struct voiced_audio_pipewire_source_callbacks *callbacks,
    struct voiced_audio_pipewire_error *error_out
) {
    struct voiced_audio_pipewire_source_observer *observer;
    struct pw_core *core;
    size_t expected_device_serial_size = 0;
    int result;

    clear_error(error_out);
    if (error_out == NULL) {
        return NULL;
    }
    if (stream == NULL || callbacks == NULL || callbacks->event == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_CREATE,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_OBSERVER_ARGUMENTS,
            "Source observation requires a stream and event callback",
            error_out
        );
        return NULL;
    }
    if (expected_device_serial != NULL) {
        expected_device_serial_size = strnlen(
            expected_device_serial,
            VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY
        );
        if (expected_device_serial_size == 0 ||
            expected_device_serial_size ==
                VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY) {
            capture_validation(
                VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_CREATE,
                VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_OBSERVER_ARGUMENTS,
                "The expected Device serial is empty or exceeds its fixed bound",
                error_out
            );
            return NULL;
        }
    }

    core = pw_stream_get_core(stream);
    if (core == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_CREATE,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_OBSERVER_CORE,
            "The capture stream did not expose its PipeWire core",
            error_out
        );
        return NULL;
    }

    observer = (struct voiced_audio_pipewire_source_observer *)calloc(
        1,
        sizeof(*observer)
    );
    if (observer == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_CREATE,
            error_out
        );
        return NULL;
    }
    observer->stream = stream;
    observer->callbacks = *callbacks;
    observer->active_source_node_id = PW_ID_ANY;
    observer->active_source_device_id = PW_ID_ANY;
    observer->identity.node_id = PW_ID_ANY;
    observer->identity.device_id = PW_ID_ANY;
    if (expected_device_serial != NULL) {
        memcpy(
            observer->expected_device_serial,
            expected_device_serial,
            expected_device_serial_size
        );
        observer->expected_device_serial_size =
            (uint32_t)expected_device_serial_size;
    }

    observer->registry = pw_core_get_registry(core, PW_VERSION_REGISTRY, 0);
    if (observer->registry == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_REGISTRY_CREATE,
            error_out
        );
        free(observer);
        return NULL;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_registry_add_listener(
        observer->registry,
        &observer->registry_listener,
        &source_registry_events,
        observer
    );
#pragma clang diagnostic pop
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_REGISTER,
            result,
            error_out
        );
        pw_proxy_destroy((struct pw_proxy *)observer->registry);
        free(observer);
        return NULL;
    }

    observer->listener_is_registered = true;
    return observer;
}

/** Stop graph callbacks, destroy the registry proxy, and release its caches. */
void voiced_audio_pipewire_capture_source_observer_destroy(
    struct voiced_audio_pipewire_source_observer *observer
) {
    if (observer == NULL) {
        return;
    }

    if (observer->source_node_listener_is_registered) {
        spa_hook_remove(&observer->source_node_listener);
        observer->source_node_listener_is_registered = false;
    }
    if (observer->source_node_proxy != NULL) {
        pw_proxy_destroy((struct pw_proxy *)observer->source_node_proxy);
        observer->source_node_proxy = NULL;
    }
    if (observer->source_device_listener_is_registered) {
        spa_hook_remove(&observer->source_device_listener);
        observer->source_device_listener_is_registered = false;
    }
    if (observer->source_device_proxy != NULL) {
        pw_proxy_destroy((struct pw_proxy *)observer->source_device_proxy);
        observer->source_device_proxy = NULL;
    }
    if (observer->listener_is_registered) {
        spa_hook_remove(&observer->registry_listener);
        observer->listener_is_registered = false;
    }
    if (observer->registry != NULL) {
        pw_proxy_destroy((struct pw_proxy *)observer->registry);
        observer->registry = NULL;
    }
    free(observer);
}

/**
 * Copy the first concrete source linked during this capture.
 *
 * The snapshot deliberately survives a later Link removal or source change. It
 * therefore describes where retained samples originated instead of whatever
 * source WirePlumber may have selected while capture was stopping. False with
 * an empty error means no source Link was ever observed.
 */
bool voiced_audio_pipewire_capture_source_snapshot(
    const struct voiced_audio_pipewire_source_observer *observer,
    struct voiced_audio_pipewire_source_identity *identity_out,
    struct voiced_audio_pipewire_error *error_out
) {
    clear_error(error_out);
    if (error_out == NULL) {
        return false;
    }
    if (observer == NULL || identity_out == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_SNAPSHOT,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_SNAPSHOT_ARGUMENTS,
            "Source snapshot requires an observer and output structure",
            error_out
        );
        return false;
    }

    *identity_out = observer->identity;
    return identity_out->is_resolved != 0;
}

/*
 * Configured-device selection happens before the capture stream exists. A
 * separate, short-lived PipeWire connection inventories source Nodes and their
 * Devices, then resolves the configured stable `device.serial` to exactly one
 * current `node.name`. The capture stream can pass that concrete node name to
 * `target.object` without ever opening the current default microphone.
 *
 * Registry globals omit some identity properties on supported PipeWire
 * versions. Discovery therefore binds each Audio/Source Node and Audio/Device,
 * waits through a second core synchronization barrier for their info events,
 * and only then joins Node `device.id` to Device `device.serial`.
 */
struct source_discovery;

struct source_discovery_node {
    struct source_discovery *discovery;
    struct pw_node *proxy;
    struct spa_hook listener;
    struct source_node_record identity;
    uint32_t listener_is_registered;
    uint32_t reserved;
};

struct source_discovery_device {
    struct source_discovery *discovery;
    struct pw_device *proxy;
    struct spa_hook listener;
    struct source_device_record identity;
    uint32_t listener_is_registered;
    uint32_t reserved;
};

struct source_discovery {
    struct pw_main_loop *main_loop;
    struct pw_context *context;
    struct pw_core *core;
    struct pw_registry *registry;
    struct spa_hook core_listener;
    struct spa_hook registry_listener;
    struct voiced_audio_pipewire_error error;
    struct source_discovery_node nodes[
        VOICED_AUDIO_PIPEWIRE_SOURCE_CATALOG_CAPACITY
    ];
    struct source_discovery_device devices[
        VOICED_AUDIO_PIPEWIRE_SOURCE_CATALOG_CAPACITY
    ];
    int sync_sequence;
    uint32_t sync_round;
    uint32_t nodes_count;
    uint32_t devices_count;
    uint32_t core_listener_is_registered;
    uint32_t registry_listener_is_registered;
};

/** Retain the first discovery failure and stop its private main loop. */
static void fail_source_discovery(
    struct source_discovery *discovery,
    const struct voiced_audio_pipewire_error *error
) {
    if (discovery == NULL || error == NULL ||
        discovery->error.stage != VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE) {
        return;
    }

    discovery->error = *error;
    if (discovery->main_loop != NULL) {
        (void)pw_main_loop_quit(discovery->main_loop);
    }
}

/** Report one malformed discovery property with a stable boundary code. */
static void fail_source_discovery_validation(
    struct source_discovery *discovery,
    const enum voiced_audio_pipewire_boundary_error_code code,
    const char *message
) {
    struct voiced_audio_pipewire_error error;

    capture_validation(
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION,
        code,
        message,
        &error
    );
    fail_source_discovery(discovery, &error);
}

/** Merge callback-owned source Node properties into one bounded record. */
static bool update_discovered_source_node(
    struct source_discovery_node *node,
    const struct spa_dict *properties
) {
    const char *object_serial;
    const char *device_id;
    const char *node_name;
    const char *node_description;

    if (properties == NULL) {
        return true;
    }

    object_serial = spa_dict_lookup(properties, PW_KEY_OBJECT_SERIAL);
    device_id = spa_dict_lookup(properties, PW_KEY_DEVICE_ID);
    node_name = spa_dict_lookup(properties, PW_KEY_NODE_NAME);
    node_description = spa_dict_lookup(properties, PW_KEY_NODE_DESCRIPTION);
    if ((object_serial != NULL &&
         !spa_atou64(object_serial, &node->identity.object_serial, 10)) ||
        (device_id != NULL &&
         !spa_atou32(device_id, &node->identity.device_id, 10)) ||
        (node_name != NULL &&
         !copy_identity_text(
             node->identity.name,
             &node->identity.name_size,
             node_name
         )) ||
        (node_description != NULL &&
         !copy_identity_text(
             node->identity.description,
             &node->identity.description_size,
             node_description
         ))) {
        fail_source_discovery_validation(
            node->discovery,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_PROPERTY,
            "An Audio/Source Node published a malformed identity property"
        );
        return false;
    }
    return true;
}

/** Merge callback-owned Device properties needed for stable selection. */
static bool update_discovered_source_device(
    struct source_discovery_device *device,
    const struct spa_dict *properties
) {
    const char *object_serial;
    const char *device_serial;
    const char *device_description;

    if (properties == NULL) {
        return true;
    }

    object_serial = spa_dict_lookup(properties, PW_KEY_OBJECT_SERIAL);
    device_serial = spa_dict_lookup(properties, PW_KEY_DEVICE_SERIAL);
    device_description = spa_dict_lookup(
        properties,
        PW_KEY_DEVICE_DESCRIPTION
    );
    if ((object_serial != NULL &&
         !spa_atou64(object_serial, &device->identity.object_serial, 10)) ||
        (device_serial != NULL &&
         !copy_identity_text(
             device->identity.serial,
             &device->identity.serial_size,
             device_serial
         )) ||
        (device_description != NULL &&
         !copy_identity_text(
             device->identity.description,
             &device->identity.description_size,
             device_description
         ))) {
        fail_source_discovery_validation(
            device->discovery,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_PROPERTY,
            "An Audio/Device published a malformed identity property"
        );
        return false;
    }
    return true;
}

/** Receive complete properties from one bound Audio/Source Node. */
static void discovered_source_node_info(
    void *data,
    const struct pw_node_info *info
) {
    struct source_discovery_node *node =
        (struct source_discovery_node *)data;

    if (node == NULL || info == NULL || !node->identity.is_present ||
        node->discovery->error.stage !=
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE) {
        return;
    }
    if (info->id != node->identity.id) {
        fail_source_discovery_validation(
            node->discovery,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_PROPERTY,
            "A bound Audio/Source Node reported an unexpected global ID"
        );
        return;
    }
    (void)update_discovered_source_node(node, info->props);
}

/* Discovery needs only each source Node's info properties. */
static const struct pw_node_events discovered_source_node_events = {
    .version = PW_VERSION_NODE_EVENTS,
    .info = discovered_source_node_info,
    .param = NULL,
};

/** Receive complete properties from one bound Audio/Device. */
static void discovered_source_device_info(
    void *data,
    const struct pw_device_info *info
) {
    struct source_discovery_device *device =
        (struct source_discovery_device *)data;

    if (device == NULL || info == NULL || !device->identity.is_present ||
        device->discovery->error.stage !=
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE) {
        return;
    }
    if (info->id != device->identity.id) {
        fail_source_discovery_validation(
            device->discovery,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_PROPERTY,
            "A bound Audio/Device reported an unexpected global ID"
        );
        return;
    }
    (void)update_discovered_source_device(device, info->props);
}

/* Discovery needs only each Device's info properties. */
static const struct pw_device_events discovered_source_device_events = {
    .version = PW_VERSION_DEVICE_EVENTS,
    .info = discovered_source_device_info,
    .param = NULL,
};

/** Bind one discovered source Node so old runtimes reveal its full identity. */
static bool bind_discovered_source_node(
    struct source_discovery_node *node
) {
    const uint32_t bound_version = node->identity.version < PW_VERSION_NODE
        ? node->identity.version
        : PW_VERSION_NODE;
    int result;

    node->proxy = (struct pw_node *)pw_registry_bind(
        node->discovery->registry,
        node->identity.id,
        PW_TYPE_INTERFACE_Node,
        bound_version,
        0
    );
    if (node->proxy == NULL) {
        struct voiced_audio_pipewire_error error;

        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_NODE_BIND,
            &error
        );
        fail_source_discovery(node->discovery, &error);
        return false;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_node_add_listener(
        node->proxy,
        &node->listener,
        &discovered_source_node_events,
        node
    );
#pragma clang diagnostic pop
    if (result < 0) {
        struct voiced_audio_pipewire_error error;

        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_NODE_BIND,
            result,
            &error
        );
        fail_source_discovery(node->discovery, &error);
        return false;
    }

    node->listener_is_registered = true;
    return true;
}

/** Bind one Audio/Device so its stable serial becomes available. */
static bool bind_discovered_source_device(
    struct source_discovery_device *device
) {
    const uint32_t bound_version = device->identity.version < PW_VERSION_DEVICE
        ? device->identity.version
        : PW_VERSION_DEVICE;
    int result;

    device->proxy = (struct pw_device *)pw_registry_bind(
        device->discovery->registry,
        device->identity.id,
        PW_TYPE_INTERFACE_Device,
        bound_version,
        0
    );
    if (device->proxy == NULL) {
        struct voiced_audio_pipewire_error error;

        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_DEVICE_BIND,
            &error
        );
        fail_source_discovery(device->discovery, &error);
        return false;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_device_add_listener(
        device->proxy,
        &device->listener,
        &discovered_source_device_events,
        device
    );
#pragma clang diagnostic pop
    if (result < 0) {
        struct voiced_audio_pipewire_error error;

        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_DEVICE_BIND,
            result,
            &error
        );
        fail_source_discovery(device->discovery, &error);
        return false;
    }

    device->listener_is_registered = true;
    return true;
}

/** Inventory and bind one relevant Node or Device registry global. */
static void source_discovery_global(
    void *data,
    const uint32_t id,
    const uint32_t permissions,
    const char *type,
    const uint32_t version,
    const struct spa_dict *properties
) {
    struct source_discovery *discovery = (struct source_discovery *)data;
    const char *media_class;

    (void)permissions;
    if (discovery == NULL || type == NULL || properties == NULL ||
        discovery->error.stage != VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE) {
        return;
    }

    media_class = spa_dict_lookup(properties, PW_KEY_MEDIA_CLASS);
    if (strcmp(type, PW_TYPE_INTERFACE_Node) == 0 && media_class != NULL &&
        strncmp(media_class, "Audio/Source", 12) == 0 &&
        (media_class[12] == '\0' || media_class[12] == '/')) {
        struct source_discovery_node *node;

        if (discovery->nodes_count ==
            VOICED_AUDIO_PIPEWIRE_SOURCE_CATALOG_CAPACITY) {
            fail_source_discovery_validation(
                discovery,
                VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_CAPACITY,
                "The PipeWire graph exceeded the configured-source catalog capacity"
            );
            return;
        }

        node = &discovery->nodes[discovery->nodes_count];
        discovery->nodes_count += 1;
        node->discovery = discovery;
        node->identity.is_present = true;
        node->identity.id = id;
        node->identity.version = version;
        node->identity.device_id = PW_ID_ANY;
        if (update_discovered_source_node(node, properties)) {
            (void)bind_discovered_source_node(node);
        }
        return;
    }

    if (strcmp(type, PW_TYPE_INTERFACE_Device) == 0 && media_class != NULL &&
        strcmp(media_class, "Audio/Device") == 0) {
        struct source_discovery_device *device;

        if (discovery->devices_count ==
            VOICED_AUDIO_PIPEWIRE_SOURCE_CATALOG_CAPACITY) {
            fail_source_discovery_validation(
                discovery,
                VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_CAPACITY,
                "The PipeWire graph exceeded the configured-device catalog capacity"
            );
            return;
        }

        device = &discovery->devices[discovery->devices_count];
        discovery->devices_count += 1;
        device->discovery = discovery;
        device->identity.is_present = true;
        device->identity.id = id;
        device->identity.version = version;
        if (update_discovered_source_device(device, properties)) {
            (void)bind_discovered_source_device(device);
        }
    }
}

/** Mark a source or Device removed before the discovery barriers complete. */
static void source_discovery_global_removed(void *data, const uint32_t id) {
    struct source_discovery *discovery = (struct source_discovery *)data;
    uint32_t index;

    if (discovery == NULL) {
        return;
    }

    for (index = 0; index < discovery->nodes_count; index += 1) {
        if (discovery->nodes[index].identity.id == id) {
            discovery->nodes[index].identity.is_present = false;
        }
    }
    for (index = 0; index < discovery->devices_count; index += 1) {
        if (discovery->devices[index].identity.id == id) {
            discovery->devices[index].identity.is_present = false;
        }
    }
}

/* Registry discovery consumes only global creation and removal events. */
static const struct pw_registry_events source_discovery_registry_events = {
    .version = PW_VERSION_REGISTRY_EVENTS,
    .global = source_discovery_global,
    .global_remove = source_discovery_global_removed,
};

/** Finish two ordered round trips: globals first, bound info events second. */
static void source_discovery_core_done(
    void *data,
    const uint32_t id,
    const int sequence
) {
    struct source_discovery *discovery = (struct source_discovery *)data;
    int next_sequence;

    if (discovery == NULL || id != PW_ID_CORE ||
        sequence != discovery->sync_sequence ||
        discovery->error.stage != VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE) {
        return;
    }

    if (discovery->sync_round == 0) {
        discovery->sync_round = 1;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
        next_sequence = pw_core_sync(discovery->core, PW_ID_CORE, sequence);
#pragma clang diagnostic pop
        if (next_sequence < 0) {
            struct voiced_audio_pipewire_error error;

            capture_result(
                VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_SYNC,
                next_sequence,
                &error
            );
            fail_source_discovery(discovery, &error);
            return;
        }
        discovery->sync_sequence = next_sequence;
        return;
    }

    (void)pw_main_loop_quit(discovery->main_loop);
}

/** Retain an asynchronous core failure before stopping discovery. */
static void source_discovery_core_error(
    void *data,
    const uint32_t id,
    const int sequence,
    const int result,
    const char *message
) {
    struct source_discovery *discovery = (struct source_discovery *)data;
    struct voiced_audio_pipewire_error error;

    (void)id;
    (void)sequence;
    clear_error(&error);
    error.stage = VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION;
    error.domain = VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_CALLBACK;
    error.code = result;
    copy_error_message(&error, message);
    fail_source_discovery(discovery, &error);
}

/* Core synchronization and fatal errors delimit the discovery operation. */
static const struct pw_core_events source_discovery_core_events = {
    .version = PW_VERSION_CORE_EVENTS,
    .done = source_discovery_core_done,
    .error = source_discovery_core_error,
};

/** Find the retained Device joined to one source Node. */
static const struct source_device_record *source_discovery_find_device(
    const struct source_discovery *discovery,
    const uint32_t device_id
) {
    uint32_t index;

    for (index = 0; index < discovery->devices_count; index += 1) {
        if (discovery->devices[index].identity.is_present &&
            discovery->devices[index].identity.id == device_id) {
            return &discovery->devices[index].identity;
        }
    }
    return NULL;
}

/** Convert one joined discovery record into the public fixed identity. */
static void source_discovery_copy_identity(
    const struct source_discovery *discovery,
    const struct source_node_record *node,
    struct voiced_audio_pipewire_source_identity *identity_out
) {
    const struct source_device_record *device;

    memset(identity_out, 0, sizeof(*identity_out));
    identity_out->is_resolved = 1;
    identity_out->node_id = node->id;
    identity_out->node_object_serial = node->object_serial;
    identity_out->device_id = node->device_id;
    identity_out->node_name_size = node->name_size;
    identity_out->node_description_size = node->description_size;
    memcpy(identity_out->node_name, node->name, sizeof(identity_out->node_name));
    memcpy(
        identity_out->node_description,
        node->description,
        sizeof(identity_out->node_description)
    );

    if (node->device_id == PW_ID_ANY) {
        return;
    }
    device = source_discovery_find_device(discovery, node->device_id);
    if (device == NULL) {
        return;
    }

    identity_out->device_object_serial = device->object_serial;
    identity_out->device_serial_size = device->serial_size;
    identity_out->device_description_size = device->description_size;
    memcpy(identity_out->device_serial, device->serial, sizeof(identity_out->device_serial));
    memcpy(
        identity_out->device_description,
        device->description,
        sizeof(identity_out->device_description)
    );
}

/** Destroy every discovery proxy before its owning core and context. */
static void source_discovery_destroy(struct source_discovery *discovery) {
    uint32_t index;

    if (discovery == NULL) {
        return;
    }

    for (index = 0; index < discovery->nodes_count; index += 1) {
        if (discovery->nodes[index].listener_is_registered) {
            spa_hook_remove(&discovery->nodes[index].listener);
        }
        if (discovery->nodes[index].proxy != NULL) {
            pw_proxy_destroy(
                (struct pw_proxy *)discovery->nodes[index].proxy
            );
        }
    }
    for (index = 0; index < discovery->devices_count; index += 1) {
        if (discovery->devices[index].listener_is_registered) {
            spa_hook_remove(&discovery->devices[index].listener);
        }
        if (discovery->devices[index].proxy != NULL) {
            pw_proxy_destroy(
                (struct pw_proxy *)discovery->devices[index].proxy
            );
        }
    }
    if (discovery->registry_listener_is_registered) {
        spa_hook_remove(&discovery->registry_listener);
    }
    if (discovery->registry != NULL) {
        pw_proxy_destroy((struct pw_proxy *)discovery->registry);
    }
    if (discovery->core_listener_is_registered) {
        spa_hook_remove(&discovery->core_listener);
    }
    if (discovery->core != NULL) {
        (void)pw_core_disconnect(discovery->core);
    }
    if (discovery->context != NULL) {
        pw_context_destroy(discovery->context);
    }
    if (discovery->main_loop != NULL) {
        pw_main_loop_destroy(discovery->main_loop);
    }
    free(discovery);
}

/**
 * Resolve one stable Device serial to exactly one current source Node.
 *
 * A successful native discovery returns true for all three semantic results:
 * resolved, absent, or ambiguous. `resolution_out.sources` always lists the
 * observed Audio/Source Nodes so the caller can explain either failure without
 * reconnecting or parsing PipeWire text. Native setup, protocol, capacity, and
 * malformed-property failures return false with a structured error.
 */
bool voiced_audio_pipewire_source_resolve_device_serial(
    const char *device_serial,
    struct voiced_audio_pipewire_source_resolution *resolution_out,
    struct voiced_audio_pipewire_error *error_out
) {
    struct source_discovery *discovery;
    size_t device_serial_size;
    uint32_t node_index;
    int result;

    clear_error(error_out);
    if (error_out == NULL) {
        return false;
    }
    if (device_serial == NULL || resolution_out == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_ARGUMENTS,
            "Configured-source resolution requires a Device serial and output",
            error_out
        );
        return false;
    }
    device_serial_size = strnlen(
        device_serial,
        VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY
    );
    if (device_serial_size == 0 ||
        device_serial_size == VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_ARGUMENTS,
            "The configured Device serial is empty or exceeds its fixed bound",
            error_out
        );
        return false;
    }

    memset(resolution_out, 0, sizeof(*resolution_out));
    discovery = (struct source_discovery *)calloc(1, sizeof(*discovery));
    if (discovery == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CONTEXT_CREATE,
            error_out
        );
        return false;
    }

    discovery->main_loop = pw_main_loop_new(NULL);
    if (discovery->main_loop == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_MAIN_LOOP_CREATE,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }
    discovery->context = pw_context_new(
        pw_main_loop_get_loop(discovery->main_loop),
        NULL,
        0
    );
    if (discovery->context == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CONTEXT_CREATE,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }
    discovery->core = pw_context_connect(discovery->context, NULL, 0);
    if (discovery->core == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CORE_CONNECT,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_core_add_listener(
        discovery->core,
        &discovery->core_listener,
        &source_discovery_core_events,
        discovery
    );
#pragma clang diagnostic pop
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CORE_OBSERVER_REGISTER,
            result,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }
    discovery->core_listener_is_registered = true;

    discovery->registry = pw_core_get_registry(
        discovery->core,
        PW_VERSION_REGISTRY,
        0
    );
    if (discovery->registry == NULL) {
        capture_errno(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_REGISTRY_CREATE,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_registry_add_listener(
        discovery->registry,
        &discovery->registry_listener,
        &source_discovery_registry_events,
        discovery
    );
#pragma clang diagnostic pop
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_REGISTRY_OBSERVER_REGISTER,
            result,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }
    discovery->registry_listener_is_registered = true;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
    result = pw_core_sync(discovery->core, PW_ID_CORE, 0);
#pragma clang diagnostic pop
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_SYNC,
            result,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }
    discovery->sync_sequence = result;

    result = pw_main_loop_run(discovery->main_loop);
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_MAIN_LOOP_RUN,
            result,
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    }
    if (discovery->error.stage !=
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE) {
        *error_out = discovery->error;
        source_discovery_destroy(discovery);
        return false;
    }

    for (node_index = 0;
         node_index < discovery->nodes_count;
         node_index += 1) {
        struct source_discovery_node *node = &discovery->nodes[node_index];
        struct voiced_audio_pipewire_source_identity *source;

        if (!node->identity.is_present) {
            continue;
        }
        source = &resolution_out->sources[resolution_out->sources_count];
        source_discovery_copy_identity(discovery, &node->identity, source);
        resolution_out->sources_count += 1;

        if (source->device_serial_size == device_serial_size &&
            memcmp(
                source->device_serial,
                device_serial,
                device_serial_size
            ) == 0) {
            resolution_out->matches_count += 1;
            if (resolution_out->matches_count == 1) {
                resolution_out->resolved_source = *source;
            }
        }
    }

    if (resolution_out->matches_count == 0) {
        resolution_out->status =
            VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_NOT_FOUND;
    } else if (resolution_out->matches_count > 1) {
        resolution_out->status =
            VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_AMBIGUOUS;
    } else if (resolution_out->resolved_source.node_name_size == 0) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_NODE_NAME,
            "The matching source Node did not publish a targetable node.name",
            error_out
        );
        source_discovery_destroy(discovery);
        return false;
    } else {
        resolution_out->status =
            VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_RESOLVED;
    }

    source_discovery_destroy(discovery);
    return true;
}

#pragma clang diagnostic pop

/**
 * Read the graph clock and the exact cycle that produced one capture buffer.
 *
 * PipeWire added `pw_buffer.time` in 1.0.5. Building against older headers
 * removes every reference to both that field and `pw_stream_get_time_n`, so an
 * Ubuntu 22.04 build has no unavailable symbol or unsafe struct access. Zig
 * calls this operation only when the environment selected full validation.
 */
bool voiced_audio_pipewire_capture_buffer_timeline(
    struct pw_stream *stream,
    const struct pw_buffer *buffer,
    const uint32_t timeline_validation,
    struct voiced_audio_pipewire_timeline *timeline_out,
    struct voiced_audio_pipewire_error *error_out
) {
    clear_error(error_out);
    if (error_out == NULL) {
        return false;
    }
    if (stream == NULL || buffer == NULL || timeline_out == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_TIMELINE_QUERY,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_TIMELINE_ARGUMENTS,
            "Timeline reading requires a stream, buffer, and output structure",
            error_out
        );
        return false;
    }

    memset(timeline_out, 0, sizeof(*timeline_out));
    if (timeline_validation !=
        VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_FULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_TIMELINE_QUERY,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_TIMELINE_UNAVAILABLE,
            "Per-buffer timeline reading is unavailable in Header-only mode",
            error_out
        );
        return false;
    }

#if PW_CHECK_VERSION(1, 0, 5)
    {
        struct pw_time stream_time;
        const int result = pw_stream_get_time_n(
            stream,
            &stream_time,
            sizeof(stream_time)
        );

        if (result < 0) {
            capture_result(
                VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_TIMELINE_QUERY,
                result,
                error_out
            );
            return false;
        }

        timeline_out->graph_now_ns = stream_time.now;
        timeline_out->graph_rate_num = stream_time.rate.num;
        timeline_out->graph_rate_denom = stream_time.rate.denom;
        timeline_out->graph_ticks = stream_time.ticks;
        timeline_out->buffer_cycle_ns = buffer->time;
        return true;
    }
#else
    capture_validation(
        VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_TIMELINE_QUERY,
        VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_TIMELINE_UNAVAILABLE,
        "The compiled PipeWire headers do not provide per-buffer timing",
        error_out
    );
    return false;
#endif
}

/**
 * Decode one fixed raw-audio Format pod for Zig to validate.
 *
 * The callback's parameter ID identifies the event, while a fixed pod may retain
 * its EnumFormat object ID. The stable content identity is therefore the Format
 * object type plus fixation. Missing properties remain zero for Zig to reject.
 */
bool voiced_audio_pipewire_parse_negotiated_format(
    const struct spa_pod *parameter,
    struct spa_audio_info_raw *format_out,
    struct voiced_audio_pipewire_error *error_out
) {
    struct spa_audio_info_raw parsed_format;
    int parse_result;

    clear_error(error_out);
    if (error_out == NULL) {
        return false;
    }
    if (parameter == NULL || format_out == NULL) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_FORMAT_PARSE,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_FORMAT_ARGUMENTS,
            "Format parsing requires a pod and an output structure",
            error_out
        );
        return false;
    }
    if (!spa_pod_is_object_type(parameter, SPA_TYPE_OBJECT_Format) ||
        spa_pod_is_fixated(parameter) != 1) {
        capture_validation(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_FORMAT_PARSE,
            VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_FORMAT_OBJECT,
            "PipeWire returned a pod that was not a fixed SPA Format object",
            error_out
        );
        return false;
    }

    memset(&parsed_format, 0, sizeof(parsed_format));
    parse_result = spa_format_audio_raw_parse(parameter, &parsed_format);
    if (parse_result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_FORMAT_PARSE,
            parse_result,
            error_out
        );
        return false;
    }

    *format_out = parsed_format;
    return true;
}

/**
 * Reaffirm buffer bounds and optional Header metadata after Format negotiation.
 *
 * PipeWire asks clients to update allocation parameters from the Format
 * callback. Repeating the initial requirements covers renegotiation and graphs
 * that wait for a selected format before reading ParamBuffers.
 */
int voiced_audio_pipewire_capture_stream_configure_buffers(
    struct pw_stream *stream,
    const uint32_t samples_per_buffer_max,
    struct voiced_audio_pipewire_error *error_out
) {
    uint64_t storage[64];
    struct spa_pod_builder builder;
    const struct spa_pod *buffer_parameter;
    const struct spa_pod *metadata_parameter;
    const struct spa_pod *parameters[2];
    int result;

    clear_error(error_out);
    if (error_out == NULL) {
        return -EINVAL;
    }
    if (stream == NULL || samples_per_buffer_max == 0 ||
        samples_per_buffer_max > INT32_MAX / (int32_t)sizeof(float)) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_BUFFERS_CONFIGURE,
            -EINVAL,
            error_out
        );
        return -EINVAL;
    }

    spa_pod_builder_init(&builder, storage, sizeof(storage));
    if (!build_audio_buffer_parameters(
        &builder,
        samples_per_buffer_max,
        &buffer_parameter,
        &metadata_parameter
    )) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_BUFFERS_CONFIGURE,
            -ENOSPC,
            error_out
        );
        return -ENOSPC;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
    parameters[0] = buffer_parameter;
    parameters[1] = metadata_parameter;
    result = pw_stream_update_params(stream, parameters, 2);
#pragma clang diagnostic pop

    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_BUFFERS_CONFIGURE,
            result,
            error_out
        );
    }
    return result;
}

/** Disconnect one capture stream while retaining any native error details. */
int voiced_audio_pipewire_capture_stream_disconnect(
    struct pw_stream *stream,
    struct voiced_audio_pipewire_error *error_out
) {
    int result;

    clear_error(error_out);
    if (error_out == NULL) {
        return -EINVAL;
    }
    if (stream == NULL) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_DISCONNECT,
            -EINVAL,
            error_out
        );
        return -EINVAL;
    }

    result = pw_stream_disconnect(stream);
    if (result < 0) {
        capture_result(
            VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_DISCONNECT,
            result,
            error_out
        );
    }
    return result;
}
