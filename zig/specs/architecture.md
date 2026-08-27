# Architecture

## Direction

Voiced ships as one executable and runs as one systemd user service. The
executable selects one of three internal runtime roles:

```text
systemd --user
└── voiced serve
    ├── supervisor
    ├── voiced --role=audio
    └── voiced --role=whisper
```

The role arguments are implementation details. Users interact with one service
and one small command-line interface.

## Runtime ownership

```text
                         control socket
hotkey or CLI ─────────────────────────────────┐
                                               ▼
                                    ┌─────────────────────┐
                                    │ supervisor          │
                                    │                     │
                                    │ lifecycle           │
                                    │ deadlines           │
                                    │ final commit/output │
                                    └──────┬───────┬──────┘
                                           │       │
                              audio link   │       │ Whisper link
                                           │       │
                        ┌──────────────────▼─┐   ┌─▼──────────────────┐
MV7 → PipeWire ────────▶│ audio process      │   │ Whisper process    │
                        │ writes PCM          │   │ reads sealed PCM   │
                        └──────────┬──────────┘   └─────────┬─────────┘
                                   │                        │
                        ┌──────────▼──────────┐   ┌─────────▼─────────┐
                        │ fixed audio memfd   │   │ fixed text memfd  │
                        └─────────────────────┘   └───────────────────┘
```

The supervisor is the only authority for session phase, generation, slot
ownership, deadlines, transcript acceptance, persistence, and user-visible
output. It must not call PipeWire or Whisper.

The audio process owns PipeWire and the active microphone connection. The
supervisor may kill and replace it without reloading the model.

The Whisper process owns the resident model and reusable inference state. The
supervisor normally cancels it cooperatively and may kill and replace it when
native inference stops honoring progress or cancellation deadlines.

Processes are used because a thread blocked inside native code cannot be
safely destroyed. Killing a worker process guarantees that its threads, file
descriptors, private memory, and ability to touch later sessions disappear.

## Communication

The supervisor creates all shared mappings before it starts either worker.
PCM and transcript bytes move through fixed `memfd` regions rather than control
sockets.

Each worker has one private Unix `SOCK_SEQPACKET` link to the supervisor. The
audio and Whisper protocols remain separate and contain only messages meaningful
to that role. Protocol messages carry a version, role, payload size, and
session generation. Receivers reject stale or malformed messages.

The supervisor uses Linux primitives such as `epoll`, `pidfd`, `timerfd`, and
`signalfd` to keep one visible event loop. Systemd remains the final recovery
boundary for a supervisor invariant failure.

## Principal lifecycle

```text
idle
  │ start
  ▼
starting
  │ audio ready
  ▼
recording
  │ stop                    │ cancel
  ▼                         ▼
stopping                 canceling
  │ audio sealed             │ audio stopped + Whisper returned
  ▼                          ▼
draining transcription     idle
  │ all chunks produced
  ▼
persisting
  │ latest transcript durable
  ▼
outputting
  │ clipboard/paste attempted
  ▼
idle
```

A capture or inference fault enters a bounded recovery path within the current
phase. It does not invent a second lifecycle owner.

## Shared representation

The initial eager design uses a small fixed number of audio slots. A provisional
starting point is three slots with at most thirty seconds of float32 mono PCM
per slot. Each slot carries a generation, chunk ordinal, and atomically
published sample count.

The transcript exchange has a fixed UTF-8 capacity. Chunk text is produced into
the active session but is not user-visible until normal stop completes. Cancel
discards all produced text for that generation.

Shared representations contain fixed-width values, arrays, counts, and offsets.
They contain no pointers, slices, allocators, or process-local handles.

## Simplicity constraints

The architecture does not require a generic worker framework, message broker,
plugin API, storage database, or hierarchy of service abstractions. The three
roles exist for fault containment; they should otherwise remain direct and
specific to voiced.
