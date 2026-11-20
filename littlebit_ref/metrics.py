"""Comparison metrics between a reference and a test implementation.

Pass/fail is decided on **logits** (max abs/rel deviation, KL divergence), never on generated
text or argmax: untrained LittleBit inits have nearly tied top tokens, so rounding noise alone
flips the argmax. `argmax_agree` and `top12_gap_min` are reported for information only.
"""
from __future__ import annotations

import torch


def tensor_stats(ref: torch.Tensor, test: torch.Tensor) -> dict:
    d = test.double() - ref.double()
    r = ref.double()
    return {
        "max_abs": d.abs().max().item(),
        "rel_max": (d.abs().max() / r.abs().max()).item(),
        "rel_rms": (d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item(),
    }


def logits_stats(ref: torch.Tensor, test: torch.Tensor) -> dict:
    """ref, test: [T, vocab]. KL(ref || test) per position, natural log."""
    s = tensor_stats(ref, test)
    lp_r = torch.log_softmax(ref.double(), -1)
    lp_t = torch.log_softmax(test.double(), -1)
    kl = (lp_r.exp() * (lp_r - lp_t)).sum(-1)
    s["kl_max"] = kl.max().item()
    s["kl_mean"] = kl.mean().item()
    # informational only
    top2 = ref.double().topk(2, dim=-1).values
    s["top12_gap_min"] = (top2[:, 0] - top2[:, 1]).min().item()
    s["argmax_agree"] = (ref.argmax(-1) == test.argmax(-1)).double().mean().item()
    return s


# Tolerances (see docs/PROGRESS.md, "Toleranzen"). Keys checked by `within`.
TOL_FP_REFERENCE = {"rel_max": 1e-4, "kl_max": 1e-6}      # same math, FP32 vs FP64/other FP32 impl.


def within(stats: dict, tol: dict) -> bool:
    return all(stats[k] <= v for k, v in tol.items())
