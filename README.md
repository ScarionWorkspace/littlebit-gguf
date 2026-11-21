# littlebit-gguf

Native inference for [LittleBit](https://github.com/SamsungLabs/LittleBit)-compressed LLMs
(< 1 bit per weight) in **llama.cpp** and **Ollama**: the binary factors stay packed in GGUF and
run on llama.cpp's existing `Q1_0` kernels. Nothing is expanded back to dense weights.

> **Status: experimental research project.** LittleBit models run natively in llama.cpp and Ollama on CPU
> and on AMD GPUs (HIP/ROCm), verified against a reference implementation. Formats and interfaces may still
> change. Not suitable for production use.

> **License notice.** Our own code in this repository is MIT-licensed (see `LICENSE`). LittleBit itself
> (paper, reference code, and all checkpoints trained with it) is licensed under
> **CC BY-NC 4.0**, so **non-commercial use only**. This repository contains no LittleBit code.
> Scripts that need it (`make_littlebit_init.py`, the comparison scripts) import it from a
> separate checkout at run time. No model weights, checkpoints or GGUF files are distributed here.

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Analysis (`docs/ANALYSIS.md`) | done |
| 1 | Python reference + golden files | done |
| 2 | GGUF format + converter (`docs/FORMAT.md`) | done |
| 3 | llama.cpp: CPU + HIP (RX 9070 XT), correctness, PPL, baselines, profiling, fused GPU kernels (`docs/PROFILING.md`) | done; further kernel tuning possible |
| 4 | Ollama: `ollama create` / `ollama run` via a compat patch, no Go changes | done |
| 5 | Larger models, more architectures | open |

Details and the decisions behind them: `docs/PROGRESS.md`. Related projects: `docs/RELATED.md`.

## Results (Qwen3-8B, LittleBit-2 at 0.55 bpw)

Checkpoint: [`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw)
(community release, QAT for 0.5 epochs). All numbers on one AMD Radeon RX 9070 XT (ROCm 7.2.4, WSL2).
PPL is WikiText-2 test, measured exactly like LittleBit's `eval.py` (non-overlapping 2048-token windows,
all positions, 299,078 tokens). See `scripts/ppl_littlebit_method.py`.

**Correctness.** llama.cpp output matches our Python reference within tolerances derived from measured
numerics (`littlebit_ref/metrics.py`). On the GPU, a factorized layer equals the simulation of llama.cpp's
arithmetic to 6e-8. The 8B reaches **PPL 24.493**, or 24.490 with LittleBit's own BF16 rounding of the
latent scale, compared to 24.475 on the model card (+0.06 %).

**Comparison at similar file size** (full table: `results/baselines/TABLE.md`):

| Model | File | bpw | PPL ↓ | Generation t/s |
|---|---:|---:|---:|---:|
| **Qwen3-1.7B Q4_K_M** (dense) | 1223 MiB | 5.05 | **16.90** | **241** |
| Qwen3-4B IQ2_XXS (dense) | 1189 MiB | 2.48 | 27.00 | 169 |
| **LittleBit Qwen3-8B**, embeddings Q2_K | 1225 MiB | 1.25 | 24.70 | 88 (unfused) |
| **LittleBit Qwen3-8B**, embeddings Q4_K / lm_head Q6_K | 1364 MiB | 1.40 | 24.58 | 69 (unfused), **140 (fused)** |
| Qwen3-8B IQ1_S | 2018 MiB | 2.07 | 32.63 | 155 |
| Qwen3-8B Q2_K | 3130 MiB | 3.21 | 10.84 | 130 |
| Qwen3-8B Q4_K_M | 4795 MiB | 4.91 | 9.82 | 102 |

bpw counts the whole file against the dense model's parameters (embeddings and lm_head included).

**Honest conclusion so far.** At this size, a smaller dense model (Qwen3-1.7B Q4_K_M) is both clearly
better and faster than LittleBit Qwen3-8B. LittleBit only beats the 8B quants below 2 bpw (IQ1_S, TQ1_0).
Even fully trained (the paper's Llama-3-8B ratio of 2.3× FP16 PPL), an 8B would likely stay behind. The
more promising case is **larger models** (27B and up), which tolerate extreme compression better. That
needs a trained checkpoint (none published yet).

**Speed.** Unfused, generation is limited by the number of GPU kernels (about 2300 small kernels per
token). The llama.cpp fork now fuses each LittleBit projection into 2 GPU kernels for decoding
(`ggml/src/ggml-cuda/littlebit.cu`): **82 -> 140 t/s** on the RX 9070 XT, and since activations stay F32
it is also more exact than the unfused chain (KL vs. FP64 reference 3e-7 instead of 9e-4). Still slower than
Qwen3-1.7B Q4_K_M (241 t/s); the large projections reach only ~100 GB/s so far (`docs/PROFILING.md`).

## How it works

A LittleBit linear layer is two low-rank chains of binary matrices with three scale vectors each:

```
y = h ⊙ (U · (l ⊙ (V · (g ⊙ x))))  +  (same with U_R, V_R, h_R, l_R, g_R)
    U ∈ {±1}^(out×r),  V ∈ {±1}^(r×in)
```

`U` and `V` are stored as llama.cpp's existing `Q1_0` type with `h` and `l` folded into its block scales,
`g` as a small F32 tensor (`docs/FORMAT.md`). In llama.cpp each projection becomes
`mul(g) → mul_mat(V) → pad → mul_mat(U)` plus the residual path. The patch adds no new ggml type or op.

## Repositories

- llama.cpp fork, branch `littlebit`: https://github.com/ScarionWorkspace/llama.cpp/tree/littlebit
  (based on tag `b12257`, the version Ollama pins)
- Ollama fork, branch `littlebit`: https://github.com/ScarionWorkspace/ollama/tree/littlebit
  (adds `llama/compat/002-littlebit.patch`, generated from the llama.cpp fork by `scripts/make_ollama_patch.sh`)

They are linked rather than included as submodules. Both are large and track upstream. Ollama
consumes llama.cpp via a local source path (`OLLAMA_LLAMA_CPP_SOURCE`), so a submodule pin
would add a third place to keep in sync. `scripts/setup_workspace.sh` clones everything
side by side:

```
littlebit-work/
  littlebit-gguf/   (this repo)
  LittleBit/        (upstream, read-only)
  llama.cpp/        (fork, branch littlebit)
  ollama/           (fork, branch littlebit)
  models/ gguf/ golden/ datasets/   (local data, never committed)
```

## Quickstart

### 1. Build llama.cpp (fork, branch `littlebit`)

```bash
scripts/setup_workspace.sh            # clones the forks and creates ../.venv
cd ../llama.cpp
# CPU
HF_UI_VERSION=b12257 cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
HF_UI_VERSION=b12257 cmake --build build-cpu -j
# AMD GPU (ROCm; RX 9070 XT = gfx1201). Under WSL2 see docs/SETUP.md (librocdxg, HSA_ENABLE_DXG_DETECTION=1)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" HF_UI_VERSION=b12257 \
  cmake -B build-hip -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
HF_UI_VERSION=b12257 cmake --build build-hip -j
cd ../littlebit-gguf
```

`HF_UI_VERSION=b12257` works around a failing web-UI asset download for the patched build number.

### 2. Convert a checkpoint and run it

```bash
../.venv/bin/hf download LittleBitLLM/littlebit-qwen3-8b-0.55bpw --local-dir ../models/littlebit-qwen3-8b-0.55bpw
../.venv/bin/python convert_littlebit_to_gguf.py ../models/littlebit-qwen3-8b-0.55bpw \
    --outfile ../gguf/lb-qwen3-8b-f16.gguf          # --l-round bf16 = LittleBit's own rounding of l
# shrink embeddings/lm_head (LittleBit tensors are left untouched):
../llama.cpp/build-hip/bin/llama-quantize ../gguf/lb-qwen3-8b-f16.gguf ../gguf/lb-qwen3-8b-Q4_K_M.gguf Q4_K_M
../.venv/bin/python scripts/gguf_size_report.py ../gguf/lb-qwen3-8b-Q4_K_M.gguf   # effective bpw
../llama.cpp/build-hip/bin/llama-completion -m ../gguf/lb-qwen3-8b-Q4_K_M.gguf -ngl 99 -no-cnv \
    -p "The capital of France is" -n 32 --temp 0
```

### 3. Ollama

```bash
cd ../ollama   # fork, branch littlebit (contains llama/compat/002-littlebit.patch)
cmake -B build . -G Ninja -DOLLAMA_LLAMA_BACKENDS=rocm_v7_2 -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --parallel 16
# under WSL2 + librocdxg the bundled HSA runtime must be replaced by the system one (docs/SETUP.md):
LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 ./ollama serve &
./ollama create littlebit-qwen3-8b -f ../littlebit-gguf/modelfiles/littlebit-qwen3-8b.Modelfile
./ollama run littlebit-qwen3-8b "The capital of France is"
```

Ollama reaches ~117 t/s generation with this model. The checkpoint is trained on plain text only:
completion works, chat does not (see the Modelfile).

### 4. Verify against the reference

```bash
tools/build_tools.sh ../llama.cpp build-hip          # builds tools/bin/lb_dump_logits-build-hip
# whole model: tolerance levels 1 (vs. simulation of llama.cpp's numerics) and 2 (vs. exact reference)
../.venv/bin/python scripts/check_llamacpp.py --model ../models/littlebit-qwen3-8b-0.55bpw \
    --gguf ../gguf/lb-qwen3-8b-f16.gguf --ngl 99 --tool tools/bin/lb_dump_logits-build-hip --out check.json
# block 0 step by step (isolates the LittleBit path)
../.venv/bin/python scripts/check_llamacpp_steps.py ../models/littlebit-qwen3-8b-0.55bpw \
    ../gguf/lb-qwen3-8b-f16.gguf off tools/bin/lb_dump_logits-build-hip 99
# perplexity, LittleBit method
../.venv/bin/python scripts/ppl_littlebit_method.py --gguf ../gguf/lb-qwen3-8b-f16.gguf \
    --tokenizer ../models/littlebit-qwen3-8b-0.55bpw --ngl 99 --tool tools/bin/lb_dump_logits-build-hip --out ppl.json
# baseline comparison (dense quants and smaller dense models): scripts/run_baselines.sh, run_dense_baselines.sh
```

### 5. Python reference and golden files

```bash
cd .. && .venv/bin/hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B && cd littlebit-gguf
# untrained LittleBit-2 init (SVD + Joint-ITQ, 0 QAT steps): correctness testing only
../.venv/bin/python scripts/make_littlebit_init.py --model ../models/Qwen3-0.6B \
    --littlebit-dir ../LittleBit --out ../models/qwen3-0.6b-lb-init-0.55 --eff-bit 0.55
# reference vs. LittleBit's own inference
../.venv/bin/python scripts/compare_original.py --mode littlebit --emulate-bf16-l \
    --model ../models/qwen3-0.6b-lb-init-0.55 --littlebit-dir ../LittleBit
# golden files (45-85 MB per model, never committed; the init is deterministic, meta.json records the model's SHA256)
../.venv/bin/python scripts/make_golden.py --model ../models/qwen3-0.6b-lb-init-0.55 \
    --out ../golden/qwen3-0.6b-lb-init-0.55
# tests (the LittleBit/golden/GGUF groups only run when the env vars are set)
LITTLEBIT_DIR=../LittleBit LB_MODEL=../models/qwen3-0.6b-lb-init-0.55 \
LB_GOLDEN=../golden/qwen3-0.6b-lb-init-0.55 ../.venv/bin/python -m pytest -q tests
```

## Documentation

| File | Content |
|---|---|
| `docs/ANALYSIS.md` | Phase 0: LittleBit parametrization, llama.cpp and Ollama internals, design options |
| `docs/FORMAT.md` | GGUF tensor layout and metadata |
| `docs/PROFILING.md` | Where the time goes on the GPU, optimization options |
| `docs/RELATED.md` | Parallel and neighbouring projects |
| `docs/SETUP.md` | Environment, ROCm under WSL2 |
| `docs/PROGRESS.md` | Status, decisions, open points (German) |

## Citation

From the LittleBit README:

```bibtex
@inproceedings{lee2025littlebit,
  title={LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author={Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
@inproceedings{lee2026littlebit2,
  title={LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},
  author={Lee, Banseok and Kim, Youngmin},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

Reference project for evaluation methodology:
[littlebit-bench](https://github.com/christopherrobertbrooks-tech/littlebit-bench) (CC BY-NC 4.0).
