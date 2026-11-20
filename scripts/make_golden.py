#!/usr/bin/env python3
"""Write golden files from the reference implementation (computed in float64, stored as float32).

Output (not committed, regenerate with this script):
  <out>/meta.json           model hash, prompts, token ids, settings
  <out>/layers.npz          for selected blocks and all 7 projections:
                              <prefix>.x, <prefix>.y, <prefix>.y_main  (real activations of prompt 0)
                            plus random inputs <prefix>.x_rand / .y_rand (N(0,1), seed 0)
  <out>/prompt_<i>.npz      ids [T], logits [T, vocab], hidden [n_layer+1, T, d] (embd + every block),
                            norm [T, d]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from littlebit_ref import Qwen3Ref, lb_linear, lb_path_forward, load_checkpoint  # noqa: E402
from littlebit_ref.prompts import PROMPTS  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="packed LittleBit checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="0,mid,last")
    args = ap.parse_args()

    model_dir = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)

    ck = load_checkpoint(model_dir, dtype=torch.float64, keep_dense_dtype=True)
    ref = Qwen3Ref.from_checkpoint(ck)
    n = ref.n_layer
    blocks = sorted({{"mid": n // 2, "last": n - 1}.get(b, None) if b in ("mid", "last") else int(b)
                     for b in args.blocks.split(",")})

    f32 = lambda t: t.detach().to(torch.float32).numpy()  # noqa: E731
    prompt_ids = []
    layer_data = {}
    for pi, text in enumerate(PROMPTS):
        ids = tok(text, return_tensors="pt").input_ids[0]
        prompt_ids.append(ids.tolist())
        cap = {}
        logits = ref.forward(ids, hook=lambda k, t: cap.__setitem__(k, t))
        hidden = np.stack([f32(cap["embd"])] + [f32(cap[f"model.layers.{i}.out"]) for i in range(n)])
        np.savez(out / f"prompt_{pi}.npz", ids=ids.numpy(), logits=f32(logits), hidden=hidden,
                 norm=f32(cap["norm"]))
        if pi == 0:
            g = torch.Generator().manual_seed(0)
            for b in blocks:
                for p in Qwen3Ref.PROJ:
                    pre = f"model.layers.{b}.{p}"
                    lin = ck.linears[pre]
                    x = cap[f"{pre}.in"]
                    xr = torch.randn(8, lin.in_features, generator=g, dtype=torch.float64)
                    layer_data[f"{pre}.x"] = f32(x)
                    layer_data[f"{pre}.y"] = f32(cap[f"{pre}.out"])
                    layer_data[f"{pre}.y_main"] = f32(lb_path_forward(x, lin.main))
                    layer_data[f"{pre}.x_rand"] = f32(xr)
                    layer_data[f"{pre}.y_rand"] = f32(lb_linear(xr, lin))
        print(f"prompt {pi}: {len(ids)} tokens, top-1 next = {tok.decode(int(logits[-1].argmax()))!r}")
    np.savez(out / "layers.npz", **layer_data)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    weights = sorted(model_dir.glob("*.safetensors"))
    meta = {
        "model_dir_name": model_dir.name,
        "weights_sha256": {w.name: sha256(w) for w in weights},
        "config": ck.config, "littlebit_config": ck.lb_config,
        "compute_dtype": "float64", "stored_dtype": "float32",
        "l": "exact product v1*u2 of the stored BF16 values (no re-rounding)",
        "prompts": PROMPTS, "token_ids": prompt_ids, "blocks": blocks,
        "ranks": {k: v.main.rank for k, v in ck.linears.items() if ".layers.0." in k},
        "generator_commit": commit,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"golden files written to {out}")


if __name__ == "__main__":
    main()
