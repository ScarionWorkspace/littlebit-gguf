#!/usr/bin/env bash
# Build the helper tools against a llama.cpp build directory.
# usage: tools/build_tools.sh [llama.cpp dir] [build dir name]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
LLAMA="${1:-$here/../../llama.cpp}"
BUILD="${2:-build-cpu}"
mkdir -p "$here/bin"
g++ -O2 -std=c++17 "$here/lb_dump_logits.cpp" -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
    -L"$LLAMA/$BUILD/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA/$BUILD/bin" \
    -o "$here/bin/lb_dump_logits-$BUILD"
echo "built $here/bin/lb_dump_logits-$BUILD"
