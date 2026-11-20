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
# Level 0: Python vs. Python, same math (FP32/FP64, different implementations).
TOL_FP_REFERENCE = {"rel_max": 1e-4, "kl_max": 1e-6}

# Levels for llama.cpp with Q1_0, derived from results/numerics/littlebit-qwen3-8b-0.55bpw.json
# (trained 8B, 4 prompts + 4x512 WikiText-2 tokens). The simulation of llama.cpp's numerics
# (Q8_0 activations, FP16 scales, F16 KV cache) vs. the exact FP64 reference measured:
#   KL mean 8.9e-4, KL max 5.8e-2, PPL -0.10 %. Q8_0 activations are ~100 % of that.
# Level 1: llama.cpp vs. the *simulation* (Numerics.llamacpp_q1_0). First set to 10 % of the Q8
#   effect on the assumption that only summation order differs; that was wrong: Q8 rounding is
#   discontinuous, so any FP32 difference flips single quantization steps, which then propagate.
#   Measured noise floor (same simulation in FP32 vs FP64, 8B, 2108 positions): KL mean 2.38e-4,
#   KL max 5.6e-3. llama.cpp vs simulation sits exactly on it (2.38e-4). Limit = 2x floor.
#   Bugs are caught more sharply by the block-wise check (llama.cpp == simulation to ~1e-7).
TOL_LLAMACPP_VS_SIM = {"kl_mean": 5e-4, "kl_max": 2e-2, "ppl_delta_abs": 2.5e-3}
# Level 2: llama.cpp vs. the *exact* reference: about 2x the measured Q8 effect.
TOL_LLAMACPP_VS_EXACT = {"kl_mean": 2e-3, "kl_max": 1e-1, "ppl_delta_abs": 5e-3}
# Per single layer (teacher forced, golden layers.npz inputs), relative L2 error vs. exact.
# Measured Q8 effect: 0.41-1.0 % (0.6B init and 8B alike) -> 2x.
TOL_LAYER_LLAMACPP_VS_EXACT_REL_L2 = 2e-2


def within(stats: dict, tol: dict) -> bool:
    """`ppl_delta_abs` is compared as |ppl_test/ppl_ref - 1|; all other keys directly."""
    def val(k):
        return abs(stats["ppl_delta_rel"]) if k == "ppl_delta_abs" else stats[k]
    return all(val(k) <= v for k, v in tol.items())
