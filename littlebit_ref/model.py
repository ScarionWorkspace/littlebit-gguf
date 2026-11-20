"""Plain-PyTorch reference forward pass for Qwen3 with LittleBit linear layers.

Deliberately simple and independent of `transformers` and of LittleBit's code:
every op is written out so it can be mirrored 1:1 in llama.cpp.
Linear layers can be LittleBit factors (`LBLinear`) or dense weight matrices.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from .checkpoint import Checkpoint, LBLinear, LBPath
from .numerics import QK1_0, Numerics, fp16_roundtrip, q8_0_roundtrip

EXACT = Numerics.exact()


def lb_path_forward(x: torch.Tensor, p: LBPath, num: Numerics = EXACT) -> torch.Tensor:
    """((((x * g) @ V^T) * l) @ U^T) * h   for x [..., in]

    With `num` set, mirrors the llama.cpp graph  mul(x, g) -> mul_mat(V_q1_0, .) -> mul_mat(U_q1_0, .)
    where l and h are the Q1_0 block scales of V and U.
    """
    U, V = p.U.to(x.dtype), p.V.to(x.dtype)  # signs are stored as int8 to save memory
    l, h = p.l, p.h
    if num.scales_fp16:
        l, h = fp16_roundtrip(l), fp16_roundtrip(h)
    t = x * p.g
    if num.act_q8_0:
        t = q8_0_roundtrip(t)
    t = (t @ V.T) * l
    if num.act_q8_0:
        # U's rank dim is padded to a multiple of 128 in GGUF; the padding entries of t are 0
        r = t.shape[-1]
        t = q8_0_roundtrip(torch.nn.functional.pad(t, (0, (-r) % QK1_0)))[..., :r]
    return (t @ U.T) * h


def lb_linear(x: torch.Tensor, lin: LBLinear, num: Numerics = EXACT) -> torch.Tensor:
    y = lb_path_forward(x, lin.main, num)
    if lin.residual is not None:
        y = y + lb_path_forward(x, lin.residual, num)
    if lin.bias is not None:
        y = y + lin.bias
    return y


def linear(x: torch.Tensor, w, num: Numerics = EXACT) -> torch.Tensor:
    if isinstance(w, LBLinear):
        return lb_linear(x, w, num)
    return x @ w.T.to(x.dtype)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.to(x.dtype)


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """NeoX-style rotary embedding (rotate halves). x: [T, H, D]"""
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = pos.to(torch.float64)[:, None] * inv_freq[None, :]          # [T, D/2]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Qwen3Ref:
    PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    def __init__(self, config: dict, weights: dict, linears: dict, dtype=None, numerics: Numerics = EXACT):
        """weights: dense tensors by HF name; linears: HF prefix -> LBLinear or dense W [out, in]."""
        self.c = config
        self.w = weights
        self.lin = linears
        self.num = numerics
        self.n_layer = config["num_hidden_layers"]
        self.n_head = config["num_attention_heads"]
        self.n_kv = config["num_key_value_heads"]
        self.hd = config.get("head_dim") or config["hidden_size"] // self.n_head
        self.eps = config["rms_norm_eps"]
        self.theta = config.get("rope_theta", 10000.0)
        emb = weights["model.embed_tokens.weight"]
        self.dtype = dtype or emb.dtype
        self.lm_head = weights.get("lm_head.weight", emb if config.get("tie_word_embeddings") else None)
        if self.lm_head is None:
            raise ValueError("no lm_head.weight and embeddings are not tied")

    @classmethod
    def from_checkpoint(cls, ck: Checkpoint, dtype=None, numerics: Numerics = EXACT) -> "Qwen3Ref":
        """dtype: compute dtype (default: dtype of the scales). numerics: llama.cpp simulation."""
        any_lin = next(iter(ck.linears.values()))
        return cls(ck.config, ck.dense, ck.linears, dtype=dtype or any_lin.main.h.dtype, numerics=numerics)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, hook: Callable[[str, torch.Tensor], None] | None = None) -> torch.Tensor:
        """ids: [T] int64 -> logits [T, vocab]. hook(name, tensor) sees intermediate values."""
        hook = hook or (lambda n, t: None)
        T = ids.shape[0]
        pos = torch.arange(T)
        h = self.w["model.embed_tokens.weight"][ids].to(self.dtype)
        hook("embd", h)
        mask = torch.full((T, T), float("-inf"), dtype=h.dtype).triu(1)
        rep = self.n_head // self.n_kv
        for i in range(self.n_layer):
            P = f"model.layers.{i}."

            def proj(name, x):
                hook(f"{P}{name}.in", x)
                y = linear(x, self.lin[P + name], self.num)
                hook(f"{P}{name}.out", y)
                return y

            x = rms_norm(h, self.w[P + "input_layernorm.weight"], self.eps)
            q = proj("self_attn.q_proj", x).view(T, self.n_head, self.hd)
            k = proj("self_attn.k_proj", x).view(T, self.n_kv, self.hd)
            v = proj("self_attn.v_proj", x).view(T, self.n_kv, self.hd)
            q = rms_norm(q, self.w[P + "self_attn.q_norm.weight"], self.eps)
            k = rms_norm(k, self.w[P + "self_attn.k_norm.weight"], self.eps)
            q, k = rope(q, pos, self.theta), rope(k, pos, self.theta)
            if self.num.kv_f16:  # K is cached after RoPE
                k, v = fp16_roundtrip(k), fp16_roundtrip(v)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            att = torch.einsum("thd,shd->hts", q, k) / math.sqrt(self.hd) + mask
            att = torch.softmax(att, dim=-1)
            o = torch.einsum("hts,shd->thd", att, v).reshape(T, self.n_head * self.hd)
            h = h + proj("self_attn.o_proj", o)
            x = rms_norm(h, self.w[P + "post_attention_layernorm.weight"], self.eps)
            gate = proj("mlp.gate_proj", x)
            up = proj("mlp.up_proj", x)
            h = h + proj("mlp.down_proj", torch.nn.functional.silu(gate) * up)
            hook(f"{P}out", h)
        h = rms_norm(h, self.w["model.norm.weight"], self.eps)
        hook("norm", h)
        out = torch.empty(T, self.lm_head.shape[0], dtype=h.dtype)
        for s in range(0, self.lm_head.shape[0], 16384):  # chunked: no full FP32 copy of lm_head
            out[:, s:s + 16384] = h @ self.lm_head[s:s + 16384].to(h.dtype).T
        return out
