#!/usr/bin/env python3
"""Compare llama.cpp logits (via tools/lb_dump_logits) with the Python reference.

Level 1: llama.cpp vs. simulation of its numerics (Numerics.llamacpp_q1_0, FP32)
Level 2: llama.cpp vs. exact reference (FP64)
Tolerances: littlebit_ref.metrics (derived in docs/PROGRESS.md). Pass/fail on logits/KL/PPL only.
The l variant (exact / BF16-rounded) is taken from the GGUF metadata `littlebit.l_round`.
"""
import argparse
import glob
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from littlebit_ref import Qwen3Ref, load_checkpoint  # noqa: E402
from littlebit_ref.metrics import (TOL_LLAMACPP_VS_EXACT, TOL_LLAMACPP_VS_SIM,  # noqa: E402
                                   logits_stats, within)
from littlebit_ref.numerics import Numerics  # noqa: E402
from littlebit_ref.prompts import PROMPTS  # noqa: E402


def nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits[:-1].double(), -1)
    return -lp.gather(-1, ids[1:, None])[:, 0]


class Agg:
    def __init__(self):
        self.kl_max = self.rel_max = 0.0
        self.kl_sum = self.nll_sum = 0.0
        self.n_pos = self.n_tok = 0
        self.argmax_agree = 0.0

    def add(self, ref, test, ids):
        s = logits_stats(ref, test)
        self.kl_max = max(self.kl_max, s["kl_max"])
        self.rel_max = max(self.rel_max, s["rel_max"])
        self.kl_sum += s["kl_mean"] * len(ids)
        self.argmax_agree += s["argmax_agree"] * len(ids)
        self.n_pos += len(ids)
        self.nll_sum += nll(test, ids).sum().item()
        self.n_tok += len(ids) - 1

    def result(self, ref_ppl):
        ppl = float(np.exp(self.nll_sum / self.n_tok))
        return {"kl_max": self.kl_max, "kl_mean": self.kl_sum / self.n_pos, "rel_max": self.rel_max,
                "ppl": ppl, "ppl_delta_rel": ppl / ref_ppl - 1,
                "argmax_agree_info_only": self.argmax_agree / self.n_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="packed LittleBit checkpoint dir (reference)")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tool", default=str(ROOT / "tools/bin/lb_dump_logits-build-cpu"))
    ap.add_argument("--wikitext", default=str(ROOT.parent / "datasets/wikitext"))
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--flash-attn", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--ngl", type=int, default=0, help="GPU layers for llama.cpp")
    ap.add_argument("--llama-cpp-dir", default=str(ROOT.parent / "llama.cpp"))
    ap.add_argument("--noise-floor", action="store_true",
                    help="also run the simulation in FP64 and report sim FP32 vs FP64 (Q8 noise floor)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.llama_cpp_dir) / "gguf-py"))
    import gguf
    r = gguf.GGUFReader(args.gguf)
    l_round = r.fields["littlebit.l_round"].contents() if "littlebit.l_round" in r.fields else "none"
    del r

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    inputs = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    if args.chunks:
        text = "".join(pq.read_table(glob.glob(f"{args.wikitext}/wikitext-2-raw-v1/test-*.parquet")[0])
                       .column("text").to_pylist())
        wt = tok(text, return_tensors="pt").input_ids[0]
        inputs += [wt[i * args.ctx:(i + 1) * args.ctx] for i in range(args.chunks)]

    lr = torch.bfloat16 if l_round == "bf16" else None
    exact = Qwen3Ref.from_checkpoint(load_checkpoint(args.model, dtype=torch.float64, keep_dense_dtype=True,
                                                     l_round=lr))
    sim = Qwen3Ref.from_checkpoint(load_checkpoint(args.model, dtype=torch.float32, keep_dense_dtype=True,
                                                   l_round=lr), numerics=Numerics.llamacpp_q1_0())
    # noise floor of the Q8 quantization: same simulation in FP64. FP rounding differences alone flip
    # single Q8 rounding decisions, so two correct implementations differ by about this much.
    sim64 = Qwen3Ref.from_checkpoint(load_checkpoint(args.model, dtype=torch.float64, keep_dense_dtype=True,
                                                     l_round=lr), numerics=Numerics.llamacpp_q1_0()) \
        if args.noise_floor else None

    a_exact, a_sim, a_simref, a_floor = Agg(), Agg(), Agg(), Agg()
    ref_nll, n_tok = 0.0, 0
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        for i, ids in enumerate(inputs):
            idf, outf = Path(td) / "ids.i32", Path(td) / "logits.f32"
            ids.to(torch.int32).numpy().astype("<i4").tofile(idf)
            subprocess.run([args.tool, args.gguf, str(idf), str(outf), str(args.threads), args.flash_attn,
                            str(args.ngl)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            lc = torch.from_numpy(np.fromfile(outf, dtype="<f4").reshape(len(ids), -1).copy())
            ref = exact.forward(ids)
            sref = sim.forward(ids)
            lc = lc[:, :ref.shape[1]]  # GGUF vocab may be padded? (same for Qwen3, kept for safety)
            a_exact.add(ref, lc, ids)
            a_sim.add(sref, lc, ids)
            a_simref.add(ref, sref, ids)
            if sim64 is not None:
                a_floor.add(sim64.forward(ids), sref, ids)
            ref_nll += nll(ref, ids).sum().item()
            n_tok += len(ids) - 1
            print(f"input {i}: {len(ids)} tokens ({time.time() - t0:.0f}s)", flush=True)

    ref_ppl = float(np.exp(ref_nll / n_tok))
    vs_exact, vs_sim, sim_vs_exact = a_exact.result(ref_ppl), a_sim.result(a_simref.result(ref_ppl)["ppl"]), \
        a_simref.result(ref_ppl)
    res = {
        "gguf": Path(args.gguf).name, "l_round": l_round, "flash_attn": args.flash_attn, "ngl": args.ngl,
        "n_inputs": len(inputs), "n_positions": a_exact.n_pos, "ref_ppl": ref_ppl,
        "llamacpp_vs_sim": vs_sim, "llamacpp_vs_exact": vs_exact, "sim_vs_exact": sim_vs_exact,
        "noise_floor_sim32_vs_sim64": a_floor.result(ref_ppl) if sim64 is not None else None,
        "level1_pass": within(vs_sim, TOL_LLAMACPP_VS_SIM),
        "level2_pass": within(vs_exact, TOL_LLAMACPP_VS_EXACT),
        "tolerances": {"level1": TOL_LLAMACPP_VS_SIM, "level2": TOL_LLAMACPP_VS_EXACT},
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
