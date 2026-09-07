#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest="$root/manifest.tsv"
expected_count=$(($(wc -l < "$manifest") - 1))
wav_count=$(find "$root/audio" -maxdepth 1 -type f -name '*.wav' | wc -l)
transcript_count=$(find "$root/audio" -maxdepth 1 -type f -name '*.transcript' | wc -l)

if (( wav_count != expected_count || transcript_count != expected_count )); then
  printf 'expected %d pairs, found %d WAV and %d transcript files\n' \
    "$expected_count" "$wav_count" "$transcript_count" >&2
  exit 1
fi

while IFS=$'\t' read -r \
  id split duration_bin duration speaker gender source_id source_hash wav_hash transcript_hash; do
  wav="$root/audio/$id.wav"
  transcript="$root/audio/$id.transcript"

  [[ -s "$wav" && -s "$transcript" ]] || {
    printf 'missing pair for %s\n' "$id" >&2
    exit 1
  }
  "${PYTHON:-python3}" - "$wav_hash" "$wav" "$transcript_hash" "$transcript" <<'PY'
import hashlib
import sys
from blake3 import blake3

for expected, path in zip(sys.argv[1::2], sys.argv[2::2]):
    with open(path, "rb") as source:
        actual = hashlib.file_digest(source, blake3).hexdigest()
    if actual != expected:
        sys.exit(f"BLAKE3 mismatch: {path}: expected {expected}, got {actual}")
PY

  format=$(ffprobe -v error \
    -show_entries stream=codec_name,sample_rate,channels \
    -of csv=p=0 "$wav")
  [[ "$format" == "pcm_s16le,16000,1" ]] || {
    printf '%s has unexpected format: %s\n' "$id" "$format" >&2
    exit 1
  }
done < <(tail -n +2 "$manifest")

printf 'verified %d audio/transcript pairs\n' "$expected_count"
