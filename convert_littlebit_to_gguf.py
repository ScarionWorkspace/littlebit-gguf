#!/usr/bin/env python3
"""Convert a packed LittleBit checkpoint (Qwen3) to GGUF without expanding it to dense weights.

Layout (docs/ANALYSIS.md 5.1, decisions in docs/PROGRESS.md), per projection P and path S ("" or "_r"):
  blk.N.P.lb_v{S}  Q1_0  ne = [in, r]       row i = sign(V[i,:]) with block scale |l[i]|
  blk.N.P.lb_u{S}  Q1_0  ne = [r_pad, out]  row o = sign(U[o,:]) with block scale |h[o]|,
                                           padding columns are +|h[o]|
  blk.N.P.lb_g{S}  F32   ne = [in]          input scale g (exact, BF16 -> F32)
Signs of h and l are folded into the sign bits. r_pad = r rounded up to a multiple of 128.
Everything else (embeddings, norms, lm_head, tokenizer, hparams) comes from llama.cpp's own
Qwen3 converter, which this script imports from a llama.cpp checkout (MIT).

Forward per path:  y = mul_mat(lb_u, pad(mul_mat(lb_v, x * lb_g), r_pad - r))
Q1_0 needs the input dim (ne0) to be a multiple of 128. V's input dim is the hidden size, so V
keeps its true rank r (no padding bits in the file). U's input dim is the rank, so U is padded;
the latent vector is zero-padded at run time (ggml_pad), so U's padding columns contribute 0.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from littlebit_ref.checkpoint import LBPath, load_checkpoint  # noqa: E402

# Shared format (GeniusDevam/littlebit-gguf#1): version 1 = lb_v padded to r_pad,
# version 2 = lb_v at the true rank r. This converter never pads lb_v, so a file is version 1
# only when every rank is already a multiple of 128 (then both layouts are the same bytes).
FORMAT_VERSION_PADDED_V = 1
FORMAT_VERSION_UNPADDED_V = 2
QK1_0 = 128
LB_RAW = re.compile(r"\.(U|V|U_R|V_R)_(packed|shape)$|\.(u1|u2|v1|v2)(_R)?$|\._(split_dim_final|eff_bit_actual|eff_bit_target)$")
logger = logging.getLogger("littlebit")


def pad_to(n: int, k: int) -> int:
    return (n + k - 1) // k * k


def lb_rows(p: LBPath) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense float32 rows whose Q1_0 quantization is exactly the LittleBit factors.

    Q1_0 quantizes a block of 128 values to d = mean|x| and bit = (x >= 0), so a row of
    ±s quantizes to d = |s| (exact: s has <= 16 significant bits, 128*s fits FP32) and the
    sign of each value. Only the FP16 rounding of d remains.
    """
    r = p.rank
    r_pad = pad_to(r, QK1_0)
    U = p.U.to(torch.float32)
    V = p.V.to(torch.float32)
    h = p.h.to(torch.float32)
    l = p.l.to(torch.float32)
    v = V * l[:, None]
    u = h[:, None].abs().expand(-1, r_pad).clone()       # padding columns: +|h|
    u[:, :r] = U * h[:, None]
    return u.numpy(), v.numpy(), p.g.to(torch.float32).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="packed LittleBit checkpoint directory")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=["f32", "f16", "bf16"], default="f16",
                    help="type for embeddings/lm_head (norms and g are always F32)")
    ap.add_argument("--l-round", choices=["none", "bf16"], default="none",
                    help="round l = v1*u2 to BF16 before storing (reproduces LittleBit's own inference)")
    ap.add_argument("--eff-bits", type=float, default=None,
                    help="target bpw for littlebit.eff_bits (default: eff_bit from littlebit_config.json)")
    ap.add_argument("--llama-cpp-dir", type=Path,
                    default=Path(os.environ.get("LLAMA_CPP_DIR", ROOT.parent / "llama.cpp")))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    sys.path.insert(0, str(args.llama_cpp_dir / "gguf-py"))
    sys.path.insert(0, str(args.llama_cpp_dir))
    import gguf
    from conversion import get_model_class  # llama.cpp (MIT)

    ck = load_checkpoint(args.model, dtype=torch.float64, keep_dense_dtype=True,
                         l_round=torch.bfloat16 if args.l_round == "bf16" else None)
    if ck.config.get("architectures", [None])[0] != "Qwen3ForCausalLM":
        raise SystemExit(f"only Qwen3ForCausalLM is supported so far, got {ck.config.get('architectures')}")
    if any(lin.bias is not None for lin in ck.linears.values()):
        raise SystemExit("LittleBit layers with bias are not supported yet")
    residual = all(lin.residual is not None for lin in ck.linears.values())
    if not residual and any(lin.residual is not None for lin in ck.linears.values()):
        raise SystemExit("mixed residual / non-residual layers are not supported")
    eff_bits = args.eff_bits if args.eff_bits is not None else ck.lb_config.get("eff_bit")
    if not isinstance(eff_bits, (int, float)) or isinstance(eff_bits, bool) \
            or not np.isfinite(eff_bits) or eff_bits <= 0:
        raise SystemExit(f"littlebit_config.json has no usable eff_bit ({eff_bits!r}); pass --eff-bits")
    use_itq = ck.lb_config.get("use_itq", False)
    if not isinstance(use_itq, bool):
        raise SystemExit(f"littlebit_config.json: use_itq must be true or false, got {use_itq!r}")
    ranks = [p.rank for lin in ck.linears.values() for p in (lin.main, lin.residual) if p is not None]
    format_version = (FORMAT_VERSION_PADDED_V if all(r % QK1_0 == 0 for r in ranks)
                      else FORMAT_VERSION_UNPADDED_V)

    base_cls = get_model_class("Qwen3ForCausalLM")
    stats = {"h_fp16_zero": 0, "h_fp16_subnormal": 0, "n_h": 0}

    class LittleBitQwen3(base_cls):
        model_arch = base_cls.model_arch
        @classmethod
        def filter_tensors(cls, item):
            if LB_RAW.search(item[0]):
                return None  # handled in generate_extra_tensors, must not be cast to float32
            return super().filter_tensors(item)

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            w = self.gguf_writer
            # the four shared keys, then two of our own (informational, other loaders ignore them)
            w.add_uint32("littlebit.format_version", format_version)
            w.add_bool("littlebit.residual", residual)
            w.add_float32("littlebit.eff_bits", float(eff_bits))
            w.add_bool("littlebit.use_itq", use_itq)
            w.add_string("littlebit.l_round", args.l_round)
            w.add_string("littlebit.quant_func", str(ck.lb_config.get("quant_func", "")))

        def generate_extra_tensors(self):
            yield from super().generate_extra_tensors()
            for prefix, lin in ck.linears.items():
                base = self.map_tensor_name(prefix + ".weight").removesuffix(".weight")
                for sfx, p in (("", lin.main), ("_r", lin.residual)):
                    if p is None:
                        continue
                    u, v, g = lb_rows(p)
                    h16 = np.abs(p.h.numpy()).astype(np.float16)
                    stats["n_h"] += h16.size
                    stats["h_fp16_zero"] += int(((h16 == 0) & (p.h.numpy() != 0)).sum())
                    stats["h_fp16_subnormal"] += int(((h16 != 0) & (h16 < np.float16(6.103515625e-05))).sum())
                    yield f"{base}.lb_v{sfx}", torch.from_numpy(v)
                    yield f"{base}.lb_u{sfx}", torch.from_numpy(u)
                    yield f"{base}.lb_g{sfx}", torch.from_numpy(g)

        def modify_tensors(self, data_torch, name, bid):
            if ".lb_" in name:
                yield name, data_torch
                return
            yield from super().modify_tensors(data_torch, name, bid)

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            if re.search(r"\.lb_[uv](_r)?$", new_name):
                return gguf.GGMLQuantizationType.Q1_0
            if re.search(r"\.lb_g(_r)?$", new_name):
                return gguf.GGMLQuantizationType.F32
            return super().tensor_force_quant(name, new_name, bid, n_dims)

    ftype = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
             "bf16": gguf.LlamaFileType.MOSTLY_BF16}[args.outtype]
    model = LittleBitQwen3(args.model, ftype, args.outfile, use_temp_file=False, eager=True)
    model.write()
    logger.info("h -> FP16: %d of %d values flush to 0, %d become subnormal",
                stats["h_fp16_zero"], stats["n_h"], stats["h_fp16_subnormal"])
    logger.info("littlebit.format_version = %d (%s lb_v)", format_version,
                "padded" if format_version == FORMAT_VERSION_PADDED_V else "unpadded")
    logger.info("wrote %s (%.1f MiB)", args.outfile, args.outfile.stat().st_size / 2**20)


if __name__ == "__main__":
    main()
