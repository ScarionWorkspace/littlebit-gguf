#!/usr/bin/env bash
# Write the LittleBit llama.cpp changes (fork branch `littlebit` vs. the tag Ollama pins) as an
# Ollama compat patch, so Ollama's normal build applies it over upstream llama.cpp.
# usage: scripts/make_ollama_patch.sh [llama.cpp dir] [ollama dir]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
LLAMA="${1:-$here/../../llama.cpp}"
OLLAMA="${2:-$here/../../ollama}"
TAG="$(cat "$OLLAMA/LLAMA_CPP_VERSION")"
OUT="$OLLAMA/llama/compat/002-littlebit.patch"
# runtime code only: src/ and ggml/src/; tests stay in the fork
git -C "$LLAMA" diff --no-color "$TAG"..littlebit -- src ggml/src > "$OUT"
echo "wrote $OUT ($(grep -c '^diff --git' "$OUT") files, base $TAG, littlebit = $(git -C "$LLAMA" rev-parse --short littlebit))"
