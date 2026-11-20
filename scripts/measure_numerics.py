#!/usr/bin/env python3
"""Measure how far llama.cpp's Q1_0 numerics move the logits away from the exact reference.

Reference: exact LittleBit math in float64. Variants run in float32 (like llama.cpp) with the
switches from littlebit_ref.numerics turned on one at a time and all together.
Data: the 4 fixed prompts plus N chunks of WikiText-2 test (raw, rows concatenated).
Reports per variant: rel_max / rel_rms of logits, KL max / mean, and NLL-based PPL of the
evaluated tokens (reference vs. variant) so the tolerance can be expressed in PPL too.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from littlebit_ref import Qwen3Ref, load_checkpoint  # noqa: E402
from littlebit_ref.metrics import logits_stats  # noqa: E402
from littlebit_ref.numerics import Numerics  # noqa: E402
from littlebit_ref.prompts import PROMPTS  # noqa: E402

VARIANTS = {
    "fp32_exact": Numerics(),
    "scales_fp16": Numerics(scales_fp16=True),
    "kv_f16": Numerics(kv_f16=True),
    "act_q8_0": Numerics(act_q8_0=True),
    "llamacpp_q1_0": Numerics.llamacpp_q1_0(),
}


def nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Per-token NLL of ids[1:] given logits[:-1]."""
    lp = torch.log_softmax(logits[:-1].double(), -1)
    return -lp.gather(-1, ids[1:, None])[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wikitext", required=True, help="dir containing wikitext-2-raw-v1/test-*.parquet")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    text = "".join(pq.read_table(glob.glob(f"{args.wikitext}/wikitext-2-raw-v1/test-*.parquet")[0])
                   .column("text").to_pylist())
    wt = tok(text, return_tensors="pt").input_ids[0]
    inputs = [("prompt", tok(p, return_tensors="pt").input_ids[0]) for p in PROMPTS]
    inputs += [("wikitext", wt[i * args.ctx:(i + 1) * args.ctx]) for i in range(args.chunks)]

    ck = load_checkpoint(args.model, dtype=torch.float64, keep_dense_dtype=True)
    ref64 = Qwen3Ref.from_checkpoint(ck)
    # float32 copy of the scales for the variants (signs stay int8, dense stays BF16)
    ck32 = load_checkpoint(args.model, dtype=torch.float32, keep_dense_dtype=True)
    models = {name: Qwen3Ref.from_checkpoint(ck32, numerics=num) for name, num in VARIANTS.items()}

    agg = {name: {"rel_max": 0.0, "rel_rms_max": 0.0, "kl_max": 0.0, "kl_sum": 0.0, "nll_sum": 0.0}
           for name in VARIANTS}
    ref_nll_sum, n_pos, n_tok = 0.0, 0, 0
    t0 = time.time()
    for kind, ids in inputs:
        ref = ref64.forward(ids)
        ref_nll = nll(ref, ids)
        ref_nll_sum += ref_nll.sum().item()
        n_tok += len(ids) - 1
        n_pos += len(ids)
        for name, m in models.items():
            out = m.forward(ids)
            s = logits_stats(ref, out)
            a = agg[name]
            a["rel_max"] = max(a["rel_max"], s["rel_max"])
            a["rel_rms_max"] = max(a["rel_rms_max"], s["rel_rms"])
            a["kl_max"] = max(a["kl_max"], s["kl_max"])
            a["kl_sum"] += s["kl_mean"] * len(ids)
            a["nll_sum"] += nll(out, ids).sum().item()
        print(f"{kind} {len(ids)} tok done ({time.time() - t0:.0f}s)", flush=True)

    ref_ppl = float(torch.tensor(ref_nll_sum / n_tok).exp())
    res = {"model": Path(args.model).name, "ctx": args.ctx, "chunks": args.chunks, "n_positions": n_pos,
           "ref_ppl_over_all_inputs": ref_ppl, "variants": {}}
    for name, a in agg.items():
        ppl = float(torch.tensor(a["nll_sum"] / n_tok).exp())
        res["variants"][name] = {"rel_max": a["rel_max"], "rel_rms_max": a["rel_rms_max"],
                                 "kl_max": a["kl_max"], "kl_mean": a["kl_sum"] / n_pos,
                                 "ppl": ppl, "ppl_delta_rel": ppl / ref_ppl - 1}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
