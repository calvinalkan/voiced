# Voiced in Zig

These documents describe the intended shape of a Zig implementation of voiced.
They are a north star, not a frozen implementation contract or a checklist that
must be followed literally. Prototypes, measurements, library behavior, and
failure testing may change the details as implementation proceeds.

The specifications should preserve decisions that materially affect reliability,
ownership, latency, or user experience. They should leave ordinary code
structure and locally reversible choices to the implementation.

## Specifications

- [Architecture](architecture.md) — runtime roles, process boundaries, shared
  memory, lifecycle ownership, and the principal state machine.
- [Audio and transcription](audio-and-transcription.md) — direct PipeWire
  capture, eager Whisper transcription, bounded audio slots, cancellation, and
  recovery.
- [Control, output, and persistence](control-output-persistence.md) — user
  commands, the local control socket, clipboard delivery, optional paste, and
  the latest-transcript fallback.
- [Engineering principles](engineering.md) — TigerStyle-inspired limits,
  allocation policy, assertions, naming, dependencies, observability, and
  testing.
- [Milestones and open questions](milestones.md) — a provisional implementation
  sequence and the decisions that should be resolved through prototypes or
  measurement.

## Product boundary

The Zig version is a local Linux and PipeWire dictation tool. It should feel
immediate after the user stops speaking, survive microphone disappearance, and
avoid losing a transcript because clipboard or paste delivery failed.

The initial scope does not include meetings, live captions, engine plugins, a
configuration UI, a transcript database, or general desktop portability. It
may retain useful behavior from the Python implementation, but compatibility
must not force a brittle design.

## Working rules

- Current implementation evidence wins over speculative detail in these files.
- A changed architectural decision should update the owning specification in
  the same change.
- Provisional constants should be labeled as such and measured before they
  become compatibility promises.
- Open questions should remain explicit rather than being hidden behind a
  premature abstraction.
- The implementation should prefer one understandable path over configurable
  alternatives that have no demonstrated use.
