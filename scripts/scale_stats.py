#!/usr/bin/env python3
"""How well do the LittleBit scales fit into FP16 (Q1_0 block scale d)?

h = u1 and g = v2 are BF16 values: FP16 represents them exactly unless they are outside
the FP16 range (|x| < 6.1e-5 becomes subnormal and loses bits, |x| > 65504 overflows).
l = v1*u2 is a product of two BF16 values (16 significant bits) and gets rounded to FP16's 11.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from littlebit_ref import load_checkpoint  # noqa: E402

ck = load_checkpoint(sys.argv[1], dtype=torch.float64, keep_dense_dtype=True)
stats = {}
for name, lin in ck.linears.items():
    for pth in [lin.main] + ([lin.residual] if lin.residual is not None else []):
        for key, v in (("h", pth.h), ("l", pth.l), ("g", pth.g)):
            a = v.abs()
            rt = v.to(torch.float16).double()
            rel = ((rt - v).abs() / a.clamp_min(1e-300)).max().item()
            s = stats.setdefault(key, {"min": 1e300, "max": 0, "rel_err_fp16": 0, "n_subnormal": 0, "n": 0})
            s["min"] = min(s["min"], a.min().item()); s["max"] = max(s["max"], a.max().item())
            s["rel_err_fp16"] = max(s["rel_err_fp16"], rel)
            s["n_subnormal"] += int((a < 6.103515625e-05).sum()); s["n"] += a.numel()
for k, s in stats.items():
    print(f"{k}: |x| in [{s['min']:.3e}, {s['max']:.3e}]  max rel. FP16 rounding error {s['rel_err_fp16']:.2e}"
          f"  FP16-subnormal {s['n_subnormal']}/{s['n']}")
