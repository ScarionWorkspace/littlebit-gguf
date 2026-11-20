"""Load a packed LittleBit checkpoint into plain float32 tensors (no LittleBit code involved)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .packing import unpack_signs


@dataclass
class LBPath:
    """One factor chain:  y = ((((x*g) @ V^T) * l) @ U^T) * h"""
    U: torch.Tensor  # [out, r]  ±1 (int8 when loaded from a checkpoint, cast on use)
    V: torch.Tensor  # [r, in]   ±1
    h: torch.Tensor  # [out]     = u1
    l: torch.Tensor  # [r]       = v1 * u2
    g: torch.Tensor  # [in]      = v2
    # raw factors kept for exactness checks (l is a product of two stored vectors)
    v1: torch.Tensor | None = None
    u2: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        return self.V.shape[0]


@dataclass
class LBLinear:
    main: LBPath
    residual: LBPath | None = None
    bias: torch.Tensor | None = None

    @property
    def in_features(self) -> int:
        return self.main.V.shape[1]

    @property
    def out_features(self) -> int:
        return self.main.U.shape[0]

    def dense(self) -> torch.Tensor:
        """Reconstruct W [out, in] (only for tests / debugging)."""
        def one(p: LBPath):
            U, V = p.U.to(p.h.dtype), p.V.to(p.h.dtype)
            return (p.h[:, None] * U) @ (p.l[:, None] * V * p.g[None, :])
        W = one(self.main)
        if self.residual is not None:
            W = W + one(self.residual)
        return W


@dataclass
class Checkpoint:
    config: dict
    lb_config: dict
    dense: dict[str, torch.Tensor] = field(default_factory=dict)   # embeddings, norms, lm_head
    linears: dict[str, LBLinear] = field(default_factory=dict)     # prefix -> factors


_PACKED = re.compile(r"^(.*)\.(U|V|U_R|V_R)_packed$")


def _open_all(path: Path):
    index = path / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    for f in files:
        with safe_open(str(path / f), framework="pt", device="cpu") as fh:
            for k in fh.keys():
                yield k, fh.get_tensor(k)


def load_checkpoint(path: str | Path, dtype=torch.float32, l_round: torch.dtype | None = None,
                    keep_dense_dtype: bool = False) -> Checkpoint:
    """Load scales as `dtype`, signs as int8.

    keep_dense_dtype: keep embeddings/norms/lm_head in their stored dtype (e.g. BF16) to save
    memory on big models; the model casts them on use (exact, BF16 -> FP32 is lossless).

    l_round: if set (e.g. torch.bfloat16), round l = v1*u2 to that dtype. This emulates
    LittleBit's own loader, which keeps the scales in BF16 so the product is rounded again.
    Default None = exact product of the stored values (what we treat as the reference).
    """
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    lbc_path = path / "littlebit_config.json"
    lb_config = json.loads(lbc_path.read_text()) if lbc_path.exists() else {}

    raw: dict[str, torch.Tensor] = dict(_open_all(path))
    prefixes = sorted({m.group(1) for k in raw if (m := _PACKED.match(k))})

    ck = Checkpoint(config=config, lb_config=lb_config)

    def take(name):
        return raw.pop(name)

    def vec(name):
        return take(name).reshape(-1).to(dtype)

    def path_for(prefix, sfx):
        out, r = take(f"{prefix}.U{sfx}_shape").tolist()
        r2, inp = take(f"{prefix}.V{sfx}_shape").tolist()
        assert r == r2, (prefix, r, r2)
        U = torch.from_numpy(unpack_signs(take(f"{prefix}.U{sfx}_packed").numpy(), r)).to(torch.int8)
        V = torch.from_numpy(unpack_signs(take(f"{prefix}.V{sfx}_packed").numpy(), inp)).to(torch.int8)
        v1_raw, u2_raw = take(f"{prefix}.v1{sfx}"), take(f"{prefix}.u2{sfx}")
        v1, u2 = v1_raw.reshape(-1).to(dtype), u2_raw.reshape(-1).to(dtype)
        l = v1 * u2
        if l_round is not None:
            l = (v1_raw.float() * u2_raw.float()).to(l_round).reshape(-1).to(dtype)
        return LBPath(U=U, V=V, h=vec(f"{prefix}.u1{sfx}"), l=l, g=vec(f"{prefix}.v2{sfx}"),
                      v1=v1, u2=u2)

    for p in prefixes:
        main = path_for(p, "")
        res = path_for(p, "_R") if f"{p}.U_R_packed" in raw else None
        bias = vec(f"{p}.bias") if f"{p}.bias" in raw else None
        for extra in ("_split_dim_final", "_eff_bit_actual", "_eff_bit_target"):
            raw.pop(f"{p}.{extra}", None)
        ck.linears[p] = LBLinear(main=main, residual=res, bias=bias)

    for k, v in raw.items():
        ck.dense[k] = v if keep_dense_dtype else v.to(dtype)
    return ck
