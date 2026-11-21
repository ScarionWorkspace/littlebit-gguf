"""Simulation of llama.cpp's numerics for LittleBit layers stored as Q1_0.

Mirrors ggml (tag b12257, CPU backend) as read in the source:
  * Q1_0 weight block: 128 signs + one FP16 scale d. We put h (rows of U) and l (rows of V)
    into d, so they get rounded to FP16 (`scales_fp16`).
  * Q1_0 x activation dot product: the activation row is first quantized to Q8_0
    (`quantize_row_q8_0_ref`: blocks of 32, d = amax/127 in FP32, q = roundf(x * (1/d)),
    d stored as FP16). The integer dot product is exact, so the result equals
    dequant(Q8_0(x)) @ W up to FP32 summation order (`act_q8_0`).
  * KV cache defaults to F16 (`common.h: cache_type_k/v = F16`) (`kv_f16`).
  * F16 weights keep F32 activations (`vec_dot_type = F32`), nothing to simulate.
CUDA/HIP quantize to Q8_1 instead: same q values and d, so the same simulation applies.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

QK8_0 = 32
QK1_0 = 128


@dataclass(frozen=True)
class Numerics:
    act_q8_0: bool = False
    scales_fp16: bool = False
    kv_f16: bool = False

    @classmethod
    def exact(cls) -> "Numerics":
        return cls()

    @classmethod
    def llamacpp_q1_0(cls) -> "Numerics":
        return cls(act_q8_0=True, scales_fp16=True, kv_f16=True)

    @classmethod
    def llamacpp_lb_fused(cls) -> "Numerics":
        """Fused LittleBit kernels (ggml-cuda/littlebit.cu): activations stay F32, no Q8."""
        return cls(act_q8_0=False, scales_fp16=True, kv_f16=True)

    @property
    def is_exact(self) -> bool:
        return not (self.act_q8_0 or self.scales_fp16 or self.kv_f16)


def _roundf(x: torch.Tensor) -> torch.Tensor:
    """C roundf: half away from zero (torch.round is half to even).

    floor(a + 0.5) would be wrong in FP32 for a = 0.49999997 (the sum rounds up to 1.0);
    floor(a) + floor(2*(a - floor(a))) is exact, same formulation as gguf-py's np_roundf.
    """
    a = x.abs()
    fl = torch.floor(a)
    return torch.sign(x) * (fl + torch.floor(2 * (a - fl)))


def q8_0_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Quantize the last dim to Q8_0 and back, computed in FP32 like ggml. Pads with zeros to 32."""
    dtype = x.dtype
    n = x.shape[-1]
    pad = (-n) % QK8_0
    xf = x.to(torch.float32)
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    xb = xf.reshape(*xf.shape[:-1], -1, QK8_0)
    d = xb.abs().amax(-1, keepdim=True) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = _roundf(xb * inv).clamp(-128, 127)
    d16 = d.to(torch.float16).to(torch.float32)
    out = (q * d16).reshape(*xf.shape[:-1], -1)[..., :n]
    return out.to(dtype)


def fp16_roundtrip(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)
