#!/usr/bin/env python3
"""Markdown comparison table from results/baselines (PPL json files + llama-bench json).

bpw = file size in bits / parameter count of the dense model (8,190,735,360 for Qwen3-8B,
incl. embeddings and lm_head), the same definition as scripts/gguf_size_report.py.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN3_8B_PARAMS = 8_190_735_360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/baselines"))
    ap.add_argument("--gguf-dir", default=str(ROOT.parent / "gguf"))
    ap.add_argument("--params", type=int, default=QWEN3_8B_PARAMS)
    args = ap.parse_args()
    res = Path(args.results)

    bench = {}
    bf = res / "llama-bench_hip.json"
    if bf.exists():
        for row in json.loads(bf.read_text()):
            name = Path(row["model_filename"]).stem
            key = "pp" if row["n_prompt"] > 0 else "tg"
            bench.setdefault(name, {})[key] = (row["avg_ts"], row["stddev_ts"])

    rows = []
    for f in sorted(res.glob("ppl_wikitext2_s2048_*.json")):
        r = json.loads(f.read_text())
        name = f.stem.removeprefix("ppl_wikitext2_s2048_")
        size = (Path(args.gguf_dir) / f"{name}.gguf").stat().st_size
        done = r["n_chunks_done"] == r["n_chunks_total"]
        b = bench.get(name, {})
        rows.append((size, name, r["ppl"], done, b.get("pp"), b.get("tg")))

    print("| Modell | Größe (MiB) | bpw (ganze Datei) | PPL WikiText-2 | pp512 t/s | tg128 t/s |")
    print("|---|---:|---:|---:|---:|---:|")
    fmt = lambda v: f"{v[0]:.0f} ± {v[1]:.0f}" if v else "–"  # noqa: E731
    for size, name, ppl, done, pp, tg in sorted(rows):
        star = "" if done else " (unvollständig)"
        print(f"| {name} | {size / 2**20:.0f} | {size * 8 / args.params:.2f} | {ppl:.2f}{star} | {fmt(pp)} | "
              f"{(f'{tg[0]:.1f}' if tg else '–')} |")


if __name__ == "__main__":
    main()
