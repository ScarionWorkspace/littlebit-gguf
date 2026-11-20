"""Load a LittleBit GGUF (written by convert_littlebit_to_gguf.py) back into reference structures.

Used for the roundtrip test GGUF -> Python reference. Needs gguf-py (from llama.cpp) on sys.path.
"""
from __future__ import annotations

import numpy as np
import torch

from .checkpoint import Checkpoint, LBLinear, LBPath

# GGUF name -> HF name (Qwen3)
PROJ = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
NORMS = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
}


def _signs_and_scale(deq: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Q1_0 rows dequantized to ±d -> (signs int8, d >= 0). Rows must have one d for all blocks."""
    d = np.abs(deq[:, :1])
    if not np.array_equal(np.abs(deq), np.broadcast_to(d, deq.shape)):
        raise ValueError("Q1_0 row with differing block scales: not a LittleBit factor")
    signs = np.where(deq < 0, -1, 1).astype(np.int8)   # d == 0 rows: sign is irrelevant
    return torch.from_numpy(signs), torch.from_numpy(d[:, 0].astype(np.float64))


def load_gguf(path, dtype=torch.float64) -> Checkpoint:
    import gguf
    r = gguf.GGUFReader(str(path))
    arch = r.fields["general.architecture"].contents()
    kv = lambda k, default=None: r.fields[k].contents() if k in r.fields else default  # noqa: E731
    n_head = kv(f"{arch}.attention.head_count")
    config = {
        "num_hidden_layers": kv(f"{arch}.block_count"),
        "num_attention_heads": n_head,
        "num_key_value_heads": kv(f"{arch}.attention.head_count_kv"),
        "head_dim": kv(f"{arch}.attention.key_length"),
        "hidden_size": kv(f"{arch}.embedding_length"),
        "rms_norm_eps": kv(f"{arch}.attention.layer_norm_rms_epsilon"),
        "rope_theta": kv(f"{arch}.rope.freq_base"),
    }
    lb_config = {k: f.contents() for k, f in r.fields.items() if k.startswith("littlebit.")}

    tensors = {}
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape)]  # numpy order: [rows, cols]
        arr = gguf.quants.dequantize(t.data, t.tensor_type).reshape(shape)
        tensors[t.name] = (t.tensor_type, arr)

    dense = {}
    lin_parts: dict[str, dict] = {}
    for name, (_, arr) in tensors.items():
        if name == "token_embd.weight":
            dense["model.embed_tokens.weight"] = torch.from_numpy(arr.astype(np.float32))
        elif name == "output.weight":
            dense["lm_head.weight"] = torch.from_numpy(arr.astype(np.float32))
        elif name == "output_norm.weight":
            dense["model.norm.weight"] = torch.from_numpy(arr.astype(np.float32))
        elif name.startswith("blk."):
            _, bid, base, suffix = name.split(".", 3)
            if base in NORMS and suffix == "weight":
                dense[f"model.layers.{bid}.{NORMS[base]}.weight"] = torch.from_numpy(arr.astype(np.float32))
            elif suffix.startswith("lb_"):
                lin_parts.setdefault(f"model.layers.{bid}.{PROJ[base]}", {})[suffix] = arr
            else:
                raise ValueError(f"unexpected tensor {name}")
    if "lm_head.weight" not in dense:
        config["tie_word_embeddings"] = True

    linears = {}
    for prefix, parts in lin_parts.items():
        def path(sfx):
            if f"lb_u{sfx}" not in parts:
                return None
            U, h = _signs_and_scale(parts[f"lb_u{sfx}"])
            V, l = _signs_and_scale(parts[f"lb_v{sfx}"])
            # V holds the true rank r, U is padded to r_pad: mirror ggml_pad of the latent by
            # zero rows in V (l = 0), so the reference can multiply directly
            r, r_pad = V.shape[0], U.shape[1]
            if r_pad > r:
                V = torch.cat([V, torch.ones(r_pad - r, V.shape[1], dtype=V.dtype)])
                l = torch.cat([l, torch.zeros(r_pad - r, dtype=l.dtype)])
            g = torch.from_numpy(parts[f"lb_g{sfx}"].reshape(-1).astype(np.float64))
            return LBPath(U=U, V=V, h=h.to(dtype), l=l.to(dtype), g=g.to(dtype))
        linears[prefix] = LBLinear(main=path(""), residual=path("_r"))
    return Checkpoint(config=config, lb_config=lb_config, dense=dense, linears=linears)
