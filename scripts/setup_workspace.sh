#!/usr/bin/env bash
# Clone the sibling repositories next to this one (../LittleBit, ../llama.cpp, ../ollama)
# and create the Python venv in ../.venv. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/../.."
GH_USER="${GH_USER:-ScarionWorkspace}"
[ -d LittleBit ] || git clone https://github.com/SamsungLabs/LittleBit.git
if [ ! -d llama.cpp ]; then
  git clone -b littlebit "https://github.com/${GH_USER}/llama.cpp.git"
  git -C llama.cpp remote add upstream https://github.com/ggml-org/llama.cpp.git
fi
if [ ! -d ollama ]; then
  git clone -b littlebit "https://github.com/${GH_USER}/ollama.git"
  git -C ollama remote add upstream https://github.com/ollama/ollama.git
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -r littlebit-gguf/requirements.txt
fi
