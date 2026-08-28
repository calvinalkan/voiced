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

The supervisor is the only authority for session phase, session identity, slot
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
audio and transcription protocols remain separate and contain only fixed records
meaningful to that role. The first versioned record transfers exactly two
role-specific descriptors with `SCM_RIGHTS`; the receiver obtains close-on-exec
descriptors for the same kernel objects. Later transcription records carry only
the command kind and irreducible physical slot selection. Ordinals, sample
counts, and result text remain authoritative in shared memory rather than being
copied into notification packets.

A private socket and pidfd already identify one worker. Signaling, readiness,
and reaping all use that pidfd; the supervisor stores no second PID identity. It
consumes a complete epoll batch before starting a replacement, so old events
never need a worker-incarnation field. One session ID belongs to the active session and
exchange headers rather than every slot. A worker must reach its ordered
end-session barrier, or be reaped, before another session can use reset storage.

The supervisor uses `epoll`, `pidfd`, one `timerfd`, and `signalfd` in one visible
event loop. Timer readiness causes it to evaluate current absolute deadlines;
timer events carry no deadline incarnation. Systemd remains the final recovery
boundary for a supervisor invariant failure.

## Principal lifecycle

```text
idle
  │ start
  ▼
starting
  │ first audio progress
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

The initial eager design uses three audio slots with at most thirty seconds of
float32 mono PCM per slot. Each slot carries a chunk ordinal and one atomic
published sample count. Zero exposes no complete payload; a positive count
release-publishes the immutable prefix. Private audio and supervisor state name
the filling and in-flight slots, so those states are not duplicated in shared
memory.

The transcript exchange has a fixed UTF-8 capacity and one atomic mailbox state:
zero is empty, one is cancelled, and values from two encode a committed byte
count plus two. Publication uses an empty-to-published CAS; cancellation uses an
atomic exchange, so both terminal states cannot coexist. Chunk text remains
private to the active session until normal stop completes.

Shared representations contain fixed-width values, arrays, counts, and offsets.
They contain no pointers, slices, allocators, or process-local handles.

## Simplicity constraints

The architecture does not require a generic worker framework, message broker,
plugin API, storage database, or hierarchy of service abstractions. The three
roles exist for fault containment; they should otherwise remain direct and
specific to voiced.
