#!/bin/sh

if [ "$#" -ne 0 ]; then
  printf '%s\n' \
    'build-release.sh does not accept arguments.' \
    'Run: ./build-release.sh' >&2
  exit 2
fi

exec zig build \
  -Ddeveloper=false \
  -Doptimize=ReleaseSmall \
  -Doptimize-inference-runtime=ReleaseFast \
  --summary all
