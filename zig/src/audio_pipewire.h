#ifndef VOICED_AUDIO_PIPEWIRE_H
#define VOICED_AUDIO_PIPEWIRE_H

/* See audio_pipewire.c for the boundary's documentation and implementation. */

#include <stdbool.h>
#include <stdint.h>

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>

#if !PW_CHECK_VERSION(0, 3, 48)
#error "Voiced requires PipeWire headers from Ubuntu 22.04 or newer"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VOICED_AUDIO_PIPEWIRE_ERROR_MESSAGE_CAPACITY 256
#define VOICED_AUDIO_PIPEWIRE_VERSION_CAPACITY 64
#define VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY 256
#define VOICED_AUDIO_PIPEWIRE_SOURCE_CATALOG_CAPACITY 32

#if PW_CHECK_VERSION(0, 3, 53)
#define VOICED_AUDIO_PIPEWIRE_CHUNK_FLAG_EMPTY SPA_CHUNK_FLAG_EMPTY
#else
#define VOICED_AUDIO_PIPEWIRE_CHUNK_FLAG_EMPTY (1u << 1)
#endif

enum voiced_audio_pipewire_error_stage {
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_NONE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_MAIN_LOOP_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_MAIN_LOOP_RUN,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CALLBACK_EVENT_REGISTER,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CONTROL_SOCKET_REGISTER,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTIES_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_CONFIG,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_TYPE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_CATEGORY,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_MEDIA_ROLE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_PROPERTY_TARGET,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_CONNECT,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_FORMAT_PARSE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_BUFFERS_CONFIGURE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_DISCONNECT,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_STREAM_STATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SERVER_OBSERVER_REGISTER,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_REGISTRY_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVER_REGISTER,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_OBSERVATION,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_SOURCE_SNAPSHOT,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_MAIN_LOOP_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CONTEXT_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CORE_CONNECT,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_CORE_OBSERVER_REGISTER,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_REGISTRY_CREATE,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_REGISTRY_OBSERVER_REGISTER,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_NODE_BIND,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_DEVICE_BIND,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_SYNC,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION_MAIN_LOOP_RUN,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_SOURCE_RESOLUTION,
    VOICED_AUDIO_PIPEWIRE_ERROR_STAGE_CAPTURE_TIMELINE_QUERY,
};

enum voiced_audio_pipewire_error_domain {
    VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_NONE,
    VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_LIBC_ERRNO,
    VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_RESULT,
    VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_PIPEWIRE_CALLBACK,
    VOICED_AUDIO_PIPEWIRE_ERROR_DOMAIN_BOUNDARY_VALIDATION,
};

enum voiced_audio_pipewire_boundary_error_code {
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_NONE,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_FORMAT_ARGUMENTS,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_FORMAT_OBJECT,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SERVER_OBSERVER_ARGUMENTS,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SERVER_CORE,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_OBSERVER_ARGUMENTS,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_OBSERVER_CORE,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_REGISTRY_CAPACITY,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_PROPERTY_VALUE,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_IDENTITY_TEXT,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_NODE_BIND,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_DEVICE_BIND,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_CONFIGURED_DEVICE_MISMATCH,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_SNAPSHOT_ARGUMENTS,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_ARGUMENTS,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_CAPACITY,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_PROPERTY,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_SOURCE_RESOLUTION_NODE_NAME,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_TIMELINE_ARGUMENTS,
    VOICED_AUDIO_PIPEWIRE_BOUNDARY_ERROR_TIMELINE_UNAVAILABLE,
};

struct voiced_audio_pipewire_error {
    uint32_t stage;
    uint32_t domain;
    int32_t code;
    uint32_t message_size;
    char message[VOICED_AUDIO_PIPEWIRE_ERROR_MESSAGE_CAPACITY];
};

enum voiced_audio_pipewire_timeline_validation {
    VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_HEADER_ONLY,
    VOICED_AUDIO_PIPEWIRE_TIMELINE_VALIDATION_FULL,
};

struct voiced_audio_pipewire_environment {
    uint32_t timeline_validation;
    uint32_t headers_version_size;
    uint32_t library_version_size;
    char headers_version[VOICED_AUDIO_PIPEWIRE_VERSION_CAPACITY];
    char library_version[VOICED_AUDIO_PIPEWIRE_VERSION_CAPACITY];
};

struct voiced_audio_pipewire_server_observer {
    struct spa_hook listener;
    uint32_t is_registered;
    uint32_t version_size;
    char version[VOICED_AUDIO_PIPEWIRE_VERSION_CAPACITY];
};

struct voiced_audio_pipewire_timeline {
    int64_t graph_now_ns;
    uint32_t graph_rate_num;
    uint32_t graph_rate_denom;
    uint64_t graph_ticks;
    uint64_t buffer_cycle_ns;
};

struct voiced_audio_pipewire_source_identity {
    uint64_t node_object_serial;
    uint64_t device_object_serial;
    uint32_t is_resolved;
    uint32_t node_id;
    uint32_t device_id;
    uint32_t node_name_size;
    uint32_t node_description_size;
    uint32_t device_serial_size;
    uint32_t device_description_size;
    uint32_t reserved;
    char node_name[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
    char node_description[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
    char device_serial[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
    char device_description[VOICED_AUDIO_PIPEWIRE_IDENTITY_TEXT_CAPACITY];
};

enum voiced_audio_pipewire_source_resolution_status {
    VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_RESOLVED,
    VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_NOT_FOUND,
    VOICED_AUDIO_PIPEWIRE_SOURCE_RESOLUTION_AMBIGUOUS,
};

struct voiced_audio_pipewire_source_resolution {
    uint32_t status;
    uint32_t sources_count;
    uint32_t matches_count;
    uint32_t reserved;
    struct voiced_audio_pipewire_source_identity resolved_source;
    struct voiced_audio_pipewire_source_identity
        sources[VOICED_AUDIO_PIPEWIRE_SOURCE_CATALOG_CAPACITY];
};

enum voiced_audio_pipewire_source_event {
    VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINKED,
    VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_LINK_REMOVED,
    VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_NODE_REMOVED,
    VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_DEVICE_REMOVED,
    VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_CHANGED,
    VOICED_AUDIO_PIPEWIRE_SOURCE_EVENT_OBSERVATION_ERROR,
};

struct voiced_audio_pipewire_source_observer;

typedef void (*voiced_audio_pipewire_source_event_func_t)(
    void *context,
    uint32_t event,
    uint32_t previous_source_node_id,
    uint32_t current_source_node_id,
    const struct voiced_audio_pipewire_error *error
);

struct voiced_audio_pipewire_source_callbacks {
    void *context;
    voiced_audio_pipewire_source_event_func_t event;
};

typedef void (*voiced_audio_pipewire_state_changed_func_t)(
    void *context,
    enum pw_stream_state old_state,
    enum pw_stream_state new_state,
    const struct voiced_audio_pipewire_error *error
);

typedef void (*voiced_audio_pipewire_format_changed_func_t)(
    void *context,
    uint32_t parameter_id,
    const struct spa_pod *parameter
);

typedef void (*voiced_audio_pipewire_process_func_t)(void *context);

struct voiced_audio_pipewire_stream_callbacks {
    void *context;
    voiced_audio_pipewire_state_changed_func_t state_changed;
    voiced_audio_pipewire_format_changed_func_t format_changed;
    voiced_audio_pipewire_process_func_t process;
};

void voiced_audio_pipewire_environment_read(
    struct voiced_audio_pipewire_environment *environment_out
);

struct pw_main_loop *voiced_audio_pipewire_main_loop_create(
    struct voiced_audio_pipewire_error *error_out
);

int voiced_audio_pipewire_main_loop_run(
    struct pw_main_loop *main_loop,
    struct voiced_audio_pipewire_error *error_out
);

struct spa_source *voiced_audio_pipewire_callback_event_register(
    struct pw_loop *loop,
    int file_descriptor,
    uint32_t mask,
    spa_source_io_func_t callback,
    void *context,
    struct voiced_audio_pipewire_error *error_out
);

struct spa_source *voiced_audio_pipewire_control_socket_register(
    struct pw_loop *loop,
    int file_descriptor,
    uint32_t mask,
    spa_source_io_func_t callback,
    void *context,
    struct voiced_audio_pipewire_error *error_out
);

void voiced_audio_pipewire_loop_io_unregister(
    struct pw_loop *loop,
    struct spa_source *source
);

struct pw_stream *voiced_audio_pipewire_capture_stream_create(
    struct pw_loop *loop,
    const char *target,
    struct voiced_audio_pipewire_stream_callbacks *callbacks,
    struct voiced_audio_pipewire_error *error_out
);

int voiced_audio_pipewire_capture_stream_connect(
    struct pw_stream *stream,
    uint32_t sample_rate_hz,
    uint32_t samples_per_buffer_max,
    struct voiced_audio_pipewire_error *error_out
);

int voiced_audio_pipewire_capture_server_observer_register(
    struct pw_stream *stream,
    struct voiced_audio_pipewire_server_observer *observer,
    struct voiced_audio_pipewire_error *error_out
);

void voiced_audio_pipewire_capture_server_observer_unregister(
    struct voiced_audio_pipewire_server_observer *observer
);

bool voiced_audio_pipewire_source_resolve_device_serial(
    const char *device_serial,
    struct voiced_audio_pipewire_source_resolution *resolution_out,
    struct voiced_audio_pipewire_error *error_out
);

struct voiced_audio_pipewire_source_observer *
voiced_audio_pipewire_capture_source_observer_create(
    struct pw_stream *stream,
    const char *expected_device_serial,
    const struct voiced_audio_pipewire_source_callbacks *callbacks,
    struct voiced_audio_pipewire_error *error_out
);

void voiced_audio_pipewire_capture_source_observer_destroy(
    struct voiced_audio_pipewire_source_observer *observer
);

bool voiced_audio_pipewire_capture_source_snapshot(
    const struct voiced_audio_pipewire_source_observer *observer,
    struct voiced_audio_pipewire_source_identity *identity_out,
    struct voiced_audio_pipewire_error *error_out
);

bool voiced_audio_pipewire_capture_buffer_timeline(
    struct pw_stream *stream,
    const struct pw_buffer *buffer,
    uint32_t timeline_validation,
    struct voiced_audio_pipewire_timeline *timeline_out,
    struct voiced_audio_pipewire_error *error_out
);

bool voiced_audio_pipewire_parse_negotiated_format(
    const struct spa_pod *parameter,
    struct spa_audio_info_raw *format_out,
    struct voiced_audio_pipewire_error *error_out
);

int voiced_audio_pipewire_capture_stream_configure_buffers(
    struct pw_stream *stream,
    uint32_t samples_per_buffer_max,
    struct voiced_audio_pipewire_error *error_out
);

int voiced_audio_pipewire_capture_stream_disconnect(
    struct pw_stream *stream,
    struct voiced_audio_pipewire_error *error_out
);

#ifdef __cplusplus
}
#endif

#endif
