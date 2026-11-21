#!/usr/bin/env bash
# Baseline comparison: dense Qwen3-8B llama.cpp quants vs. LittleBit GGUFs, same GPU, same PPL method.
# Expects the workspace layout from README (../llama.cpp with build-hip, ../models, ../gguf, ../datasets).
#
# Steps (the first ones only if the output does not exist yet):
#   1. convert_hf_to_gguf.py Qwen3-8B -> BF16
#   2. Q8_0 (fits into 16 GB VRAM) -> importance matrix on WikiText-2 *train* (200 x 512 tokens)
#   3. quants Q4_K_M Q2_K IQ2_XXS IQ1_S TQ1_0, all with the same imatrix
#   4. WikiText-2 test PPL with scripts/ppl_littlebit_method.py (seqlen 2048, all positions)
#   5. llama-bench pp512 / tg128, ngl 99
set -euo pipefail
cd "$(dirname "$0")/.."
W=..
L=$W/llama.cpp/build-hip/bin
export HSA_ENABLE_DXG_DETECTION=1
TOK=$W/models/Qwen3-8B

[ -f $W/gguf/qwen3-8b-bf16.gguf ] || $W/.venv/bin/python $W/llama.cpp/convert_hf_to_gguf.py $TOK --outtype bf16 --outfile $W/gguf/qwen3-8b-bf16.gguf
[ -f $W/gguf/qwen3-8b-Q8_0.gguf ] || $L/llama-quantize $W/gguf/qwen3-8b-bf16.gguf $W/gguf/qwen3-8b-Q8_0.gguf Q8_0 16
[ -f $W/gguf/qwen3-8b.imatrix.gguf ] || $L/llama-imatrix -m $W/gguf/qwen3-8b-Q8_0.gguf -f $W/datasets/wikitext2-train.txt \
    -o $W/gguf/qwen3-8b.imatrix.gguf -ngl 99 -c 512 --chunks 200
for t in Q4_K_M Q2_K IQ2_XXS IQ1_S TQ1_0; do
  [ -f $W/gguf/qwen3-8b-$t.gguf ] || $L/llama-quantize --imatrix $W/gguf/qwen3-8b.imatrix.gguf \
      $W/gguf/qwen3-8b-bf16.gguf $W/gguf/qwen3-8b-$t.gguf $t 16
done

MODELS="qwen3-8b-Q8_0 qwen3-8b-Q4_K_M qwen3-8b-Q2_K qwen3-8b-IQ2_XXS qwen3-8b-IQ1_S qwen3-8b-TQ1_0
        littlebit-qwen3-8b-0.55bpw-lnone-Q4_K_M littlebit-qwen3-8b-0.55bpw-lnone-Q2_K"
mkdir -p results/baselines
for m in $MODELS; do
  $W/.venv/bin/python scripts/ppl_littlebit_method.py --gguf $W/gguf/$m.gguf --tokenizer $TOK --ngl 99 \
      --tool tools/bin/lb_dump_logits-build-hip --out results/baselines/ppl_wikitext2_s2048_$m.json --resume
done
args=(); for m in $MODELS; do args+=(-m $W/gguf/$m.gguf); done
$L/llama-bench "${args[@]}" -ngl 99 -p 512 -n 128 -r 3 -o json > results/baselines/llama-bench_hip.json
