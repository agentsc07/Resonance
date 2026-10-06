#!/usr/bin/env bash
# Fetch the Rice University copy of JFK's 12 Sep 1962 address (Internet Archive, Public Domain Mark 1.0), extract 16 kHz mono audio and cut
# the two baseline excerpts B09/B10. The full speech is cached in .cache/ and is not part of the repository or the dataset release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/flawline-dataset/generator/.cache/jfk"
mkdir -p "$CACHE"
if [ ! -f "$CACHE/rice16k.wav" ]; then
  curl -L -A "Mozilla/5.0" -o "$CACHE/rice720.mp4" \
    "https://archive.org/download/president-john-f.-kennedy-09-12-1962/President%20John%20F.%20Kennedy%20%28720p%29.mp4"
  ffmpeg -y -loglevel error -i "$CACHE/rice720.mp4" -vn -ac 1 -ar 16000 "$CACHE/rice16k.wav"
fi
cd "$ROOT/flawline-dataset/generator" && python ingest_jfk.py "$CACHE/rice16k.wav"
