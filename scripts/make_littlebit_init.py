#!/usr/bin/env python3
"""Create an *untrained* LittleBit checkpoint (SVD + optional Joint-ITQ init, 0 QAT steps).

The result has exactly the packed on-disk format of a trained LittleBit checkpoint
(`*_packed`, `*_shape`, `u1/u2/v1/v2[_R]`, `littlebit_config.json`). It is only useful
for *correctness* testing of inference code, not for quality measurements.

This script does not copy LittleBit code. It imports the LittleBit repository
(CC BY-NC 4.0) from `--littlebit-dir` and calls `apply_littlebit_patch` the same way
`LittleBit/main.py` does, then saves like `main.py:save_artifacts` (non-DeepSpeed path).

Example:
    python scripts/make_littlebit_init.py --model ../models/Qwen3-0.6B \
        --littlebit-dir ../LittleBit --out ../models/qwen3-0.6b-lb-init-0.55 \
        --eff-bit 0.55 --residual 1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (dense)")
    ap.add_argument("--littlebit-dir", required=True, help="checkout of SamsungLabs/LittleBit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eff-bit", type=float, default=0.55)
    ap.add_argument("--residual", type=int, default=1)
    ap.add_argument("--use-itq", type=int, default=1)
    ap.add_argument("--itq-n-iter", type=int, default=50)
    ap.add_argument("--kv-factor", type=float, default=1.0)
    ap.add_argument("--split-dim", type=int, default=1024)
    ap.add_argument("--min-split-dim", type=int, default=8)
    ap.add_argument("--quant-func", default="SmoothSign")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.littlebit_dir).resolve()))
    from quantization.utils import apply_littlebit_patch  # noqa: E402  (LittleBit, CC BY-NC 4.0)
    from transformers import AutoModelForCausalLM, set_seed

    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    qargs = argparse.Namespace(
        quant_func=args.quant_func, quant_mod="LittleBitLinear", residual=bool(args.residual),
        split_dim=args.split_dim, eff_bit=args.eff_bit, kv_factor=args.kv_factor,
        min_split_dim=args.min_split_dim, use_itq=bool(args.use_itq), itq_n_iter=args.itq_n_iter,
        model_id=args.model,
    )
    with torch.no_grad():
        model = apply_littlebit_patch(model, qargs, do_train=True)
    model.eval()

    # Trained checkpoints store every float tensor as BF16 (integer sign words and shapes stay
    # as they are). Do the same so the init has exactly the on-disk format of a trained model.
    def as_stored(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.bfloat16) if t.is_floating_point() and t.dtype == torch.float32 else t

    sd = {name: as_stored(t.detach().cpu()) for name, t in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)  # tied to embed_tokens, safetensors refuses shared storage
    sd = {k: v.contiguous() for k, v in sd.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})

    quant_params = {
        "quant_func": args.quant_func, "eff_bit": args.eff_bit, "split_dim": args.split_dim,
        "residual": bool(args.residual), "kv_factor": args.kv_factor,
        "min_split_dim": args.min_split_dim, "quant_mod": "LittleBitLinear",
        "use_itq": bool(args.use_itq), "itq_n_iter": args.itq_n_iter,
    }
    (out / "littlebit_config.json").write_text(json.dumps(quant_params, indent=2))
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    cfg.update(quant_params)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json", "special_tokens_map.json"]:
        if (Path(args.model) / f).exists():
            shutil.copy(Path(args.model) / f, out / f)
    ranks = sorted({int(v) for k, v in sd.items() if k.endswith("_split_dim_final")})
    print(f"saved {len(sd)} tensors to {out}; ranks used: {ranks}")


if __name__ == "__main__":
    main()
