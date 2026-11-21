#!/usr/bin/env bash
# Dense comparison models of similar file size as the LittleBit 8B (e.g. Qwen3-1.7B, Qwen3-4B),
# same pipeline and same PPL method as run_baselines.sh. All Qwen3 sizes share one tokenizer
# (tokenizer.json byte-identical), so the PPL values are directly comparable.
#
# usage: scripts/run_dense_baselines.sh <hf model dir name under ../models> <gguf prefix> <quant>...
#   e.g. scripts/run_dense_baselines.sh Qwen3-1.7B qwen3-1.7b Q8_0 Q4_K_M
set -euo pipefail
cd "$(dirname "$0")/.."
W=..
L=$W/llama.cpp/build-hip/bin
export HSA_ENABLE_DXG_DETECTION=1
HF=$W/models/$1; P=$2; shift 2
BF16=$W/gguf/$P-bf16.gguf
IMAT=$W/gguf/$P.imatrix.gguf

[ -f $BF16 ] || $W/.venv/bin/python $W/llama.cpp/convert_hf_to_gguf.py $HF --outtype bf16 --outfile $BF16
# imatrix on the BF16 model (small models fit into VRAM), same data as for the 8B
[ -f $IMAT ] || $L/llama-imatrix -m $BF16 -f $W/datasets/wikitext2-train.txt -o $IMAT -ngl 99 -c 512 --chunks 200
mkdir -p results/baselines
args=()
for t in "$@"; do
  out=$W/gguf/$P-$t.gguf
  if [ ! -f $out ]; then
    if [ "$t" = Q8_0 ]; then $L/llama-quantize $BF16 $out $t 16
    else $L/llama-quantize --imatrix $IMAT $BF16 $out $t 16; fi
  fi
  $W/.venv/bin/python scripts/ppl_littlebit_method.py --gguf $out --tokenizer $HF --ngl 99 \
      --tool tools/bin/lb_dump_logits-build-hip --out results/baselines/ppl_wikitext2_s2048_$P-$t.json --resume
  args+=(-m $out)
done
$L/llama-bench "${args[@]}" -ngl 99 -p 512 -n 128 -r 3 -o json > results/baselines/llama-bench_hip_$P.json
