# littlebit-gguf

Native inference for [LittleBit](https://github.com/SamsungLabs/LittleBit)-compressed LLMs
(< 1 bit per weight) in **llama.cpp** and **Ollama**: packed binary factors in GGUF and
existing ggml kernels, without expanding them back to dense weights.

> **Status: experimental.** Work in progress, formats and interfaces may still change.
> Nothing here runs in llama.cpp or Ollama yet. Not suitable for production use.

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
| 3 | llama.cpp (CPU, then HIP/ROCm on RX 9070 XT) | next |
| 4 | Ollama | open |
| 5 | More architectures, benchmarks | open |

Details: `docs/PROGRESS.md`.

## How it works

A LittleBit linear layer is two low-rank chains of binary matrices with three scale vectors each:

```
y = h ⊙ (U · (l ⊙ (Vᵀ · (g ⊙ x))))  +  (same with U_R, V_R, h_R, l_R, g_R)
    U ∈ {±1}^(out×r),  V ∈ {±1}^(r×in)
```

The plan (see `docs/ANALYSIS.md` §5) stores `U` and `V` as llama.cpp's existing `Q1_0` type,
folds `h` and `l` into its block scales, and keeps `g` as a small separate tensor.

## Related repositories

- llama.cpp fork, branch `littlebit`: https://github.com/ScarionWorkspace/llama.cpp/tree/littlebit
- Ollama fork, branch `littlebit`: https://github.com/ScarionWorkspace/ollama/tree/littlebit

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
  models/ golden/   (local data, never committed)
```

## Converting a checkpoint

```bash
../.venv/bin/python convert_littlebit_to_gguf.py ../models/littlebit-qwen3-8b-0.55bpw \
    --outfile ../gguf/lb-qwen3-8b-f16.gguf          # --l-round bf16 = LittleBit's own rounding of l
# shrink embeddings/lm_head (LittleBit tensors are left untouched):
../llama.cpp/build-cpu/bin/llama-quantize ../gguf/lb-qwen3-8b-f16.gguf ../gguf/lb-qwen3-8b-Q4_K_M.gguf Q4_K_M
../.venv/bin/python scripts/gguf_size_report.py ../gguf/lb-qwen3-8b-Q4_K_M.gguf   # effective bpw
```

Qwen3-8B (LittleBit 0.55 bpw): 1364 MiB with Q4_K_M embeddings, i.e. 1.40 bpw for the whole file.
Running it in llama.cpp is Phase 3 (not done yet).

## Quickstart (Phase 1: reference)

```bash
scripts/setup_workspace.sh
cd .. && .venv/bin/hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B && cd littlebit-gguf
# untrained LittleBit-2 init (SVD + Joint-ITQ, 0 QAT steps): correctness testing only
../.venv/bin/python scripts/make_littlebit_init.py --model ../models/Qwen3-0.6B \
    --littlebit-dir ../LittleBit --out ../models/qwen3-0.6b-lb-init-0.55 --eff-bit 0.55
# reference vs. LittleBit's own inference
../.venv/bin/python scripts/compare_original.py --mode littlebit --emulate-bf16-l \
    --model ../models/qwen3-0.6b-lb-init-0.55 --littlebit-dir ../LittleBit
# golden files for llama.cpp
../.venv/bin/python scripts/make_golden.py --model ../models/qwen3-0.6b-lb-init-0.55 \
    --out ../golden/qwen3-0.6b-lb-init-0.55
# golden files are 45-85 MB per model and are never committed: regenerate them with
# make_golden.py. The init is deterministic (byte-identical on re-run);
# meta.json records the SHA256 of the model it was made from.
# tests (the LittleBit/golden groups only run when the env vars are set)
LITTLEBIT_DIR=../LittleBit LB_MODEL=../models/qwen3-0.6b-lb-init-0.55 \
LB_GOLDEN=../golden/qwen3-0.6b-lb-init-0.55 ../.venv/bin/python -m pytest -q tests
```

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
