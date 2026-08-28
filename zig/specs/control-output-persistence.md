# Control, Output, and Persistence

## Control surface

Voiced exposes a small local command-line interface intended for compositor
hotkeys and direct shell use. The useful operations include starting or
toggling dictation, stopping normally, canceling, and reporting status. Exact
command spelling may be refined while preserving those semantics.

The client sends one bounded request and receives one bounded response over a
permission-restricted Unix `SOCK_SEQPACKET` socket under
`$XDG_RUNTIME_DIR/voiced/`. The server verifies `SO_PEERCRED` and accepts only
the owning user.

Internal `audio` and `whisper` roles are launched by the supervisor and are not
part of the public command surface.

## Stop, cancel, and busy behavior

Normal stop means “finish this dictation.” It seals audio, drains eager
transcription, persists the final text, and performs output.

Cancel means “discard this dictation.” It stores no transcript and performs no
output.

Commands received during stopping, canceling, persistence, or output follow one
explicit policy. The implementation should prefer a clear busy or accepted
response over silently queuing another recording behind an incomplete one.

Status reports the authoritative supervisor phase and enough failure context
to explain why a new recording cannot start. It does not expose worker-private
state as a second source of truth.

## Output

Clipboard delivery is the primary output. Sending a paste key is an optional
convenience performed only after clipboard ownership has been established.

```text
final transcript
  │
  ▼
atomically persist latest transcript
  │
  ▼
offer clipboard contents
  │
  ├── paste disabled → done
  │
  └── paste enabled → send configured paste key
```

Persistence, clipboard, and paste are separate outcomes. A paste failure does
not invalidate successful transcription or clipboard delivery. A clipboard
failure does not remove the persisted fallback.

The initial implementation delegates Wayland clipboard ownership to a bounded
`wl-copy --foreground` process. `libwayland-client` is a transport API rather
than a portable `setClipboard` API: a background client must select among
compositor-specific data-control protocols or obtain an input serial through a
surface. `wl-copy` already handles those protocol differences and remains alive
to serve selection requests. A direct implementation is justified only if this
process boundary causes a measured reliability or latency problem.

External output processes receive fixed arguments, closed unrelated file
descriptors, and bounded completion deadlines. Voiced must not shell-expand
transcript text.

## Latest-transcript fallback

The initial version stores only the most recent successful, nonempty transcript:

```text
$XDG_STATE_HOME/voiced/transcript.txt
```

When `XDG_STATE_HOME` is unset, the default is
`~/.local/state/voiced/transcript.txt`.

The file contains exactly the final UTF-8 transcript. It contains no JSON,
clipboard status, paste status, model metadata, or history index.

Persistence uses a temporary file in the same directory, bounded writes,
`fdatasync`, atomic rename, and directory `fsync`. The directory is mode `0700`
and the transcript is mode `0600`.

Canceled and empty sessions do not replace the previous transcript. If
persistence fails, voiced still attempts clipboard delivery and reports that
the durable fallback is unavailable.

Timestamped history, search, history retention, replay commands, and failed
audio storage are outside the initial scope. They can be added without changing
the capture or inference pipeline.

## Notifications and logs

Notifications should identify the current operation and actionable failures
without accumulating stale “transcribing” notifications. A failed paste should
say that clipboard and persisted text remain available when that is true.

Operational logs use real journal priorities and include session ID, phase,
worker role, deadline, and concise failure reason. They must not include PCM or
transcript text.

## Shutdown

Shutdown first prevents new control operations, then cancels or drains the
active session according to the explicit shutdown policy, stops audio, stops
Whisper, reaps both workers, removes the control socket, and exits. Systemd may
kill the complete cgroup after its outer stop deadline.
