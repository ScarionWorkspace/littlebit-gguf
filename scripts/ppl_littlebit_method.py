#!/usr/bin/env python3
"""WikiText-2 perplexity of a GGUF in llama.cpp, measured exactly like LittleBit's eval.py.

LittleBit (utils/datautils.py + eval.py): text = "\\n\\n".join(test rows), tokenized once,
non-overlapping windows of `seqlen` tokens (default 2048, as used for the published 8B numbers),
every position of a window except the first is scored, PPL = exp(mean NLL).
This differs from llama-perplexity (which scores only the second half of each window), so the
numbers are directly comparable with LittleBit's / the model card's values.

Writes per-window NLL sums so an interrupted run can be resumed (--resume).
"""
import argparse
import glob
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True, help="HF dir with the tokenizer (the checkpoint dir)")
    ap.add_argument("--wikitext", default=str(ROOT.parent / "datasets/wikitext"))
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--flash-attn", default="auto")
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--tool", default=str(ROOT / "tools/bin/lb_dump_logits-build-cpu"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    rows = pq.read_table(glob.glob(f"{args.wikitext}/wikitext-2-raw-v1/test-*.parquet")[0]).column("text").to_pylist()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    enc = np.array(tok("\n\n".join(rows)).input_ids, dtype=np.int32)
    n_chunks = len(enc) // args.seqlen
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)

    out = Path(args.out)
    state = json.loads(out.read_text()) if args.resume and out.exists() else {}
    done = state.get("chunk_nll_sums", [])
    start = len(done)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        for i in range(len(done), n_chunks):
            ids = enc[i * args.seqlen:(i + 1) * args.seqlen]
            idf, nllf = Path(td) / "ids.i32", Path(td) / "out.nll"
            ids.astype("<i4").tofile(idf)
            subprocess.run([args.tool, args.gguf, str(idf), str(nllf), str(args.threads), args.flash_attn,
                            str(args.ngl)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            nll = np.fromfile(nllf, dtype="<f8")
            assert len(nll) == args.seqlen - 1
            done.append(float(nll.sum()))
            ppl = math.exp(sum(done) / (len(done) * (args.seqlen - 1)))
            state = {"gguf": Path(args.gguf).name, "seqlen": args.seqlen, "n_tokens_total": int(len(enc)),
                     "n_chunks_total": len(enc) // args.seqlen, "n_chunks_done": len(done),
                     "flash_attn": args.flash_attn, "ngl": args.ngl, "ppl": ppl, "chunk_nll_sums": done}
            out.write_text(json.dumps(state, indent=1))
            n_run = i + 1 - start
            print(f"chunk {len(done)}/{n_chunks}  ppl {ppl:.4f}  ({(time.time() - t0) / n_run:.0f} s/chunk)", flush=True)
    print(json.dumps({k: v for k, v in state.items() if k != "chunk_nll_sums"}, indent=1))


if __name__ == "__main__":
    main()
