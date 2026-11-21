#!/usr/bin/env python3
"""Markdown comparison table from results/baselines (PPL json files + llama-bench json).

bpw = file size in bits / parameter count of the *dense* model the file was made from
(incl. embeddings and lm_head, tied embeddings counted once), the same definition as
scripts/gguf_size_report.py. The count is read from <family>-bf16.gguf (sum of tensor elements);
LittleBit files use the dense Qwen3-8B count.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def family(name: str) -> str:
    if name.startswith("littlebit-qwen3-8b"):
        return "qwen3-8b"
    return name.rsplit("-", 1)[0]


def dense_params(gguf_dir: Path, fam: str, cache: dict) -> int:
    if fam not in cache:
        import sys
        sys.path.insert(0, str(ROOT.parent / "llama.cpp" / "gguf-py"))
        import gguf
        r = gguf.GGUFReader(str(gguf_dir / f"{fam}-bf16.gguf"))
        cache[fam] = sum(int(t.n_elements) for t in r.tensors)
    return cache[fam]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/baselines"))
    ap.add_argument("--gguf-dir", default=str(ROOT.parent / "gguf"))
    args = ap.parse_args()
    res = Path(args.results)

    bench = {}
    for bf in sorted(res.glob("llama-bench_hip*.json")):
        for row in json.loads(bf.read_text()):
            name = Path(row["model_filename"]).stem
            key = "pp" if row["n_prompt"] > 0 else "tg"
            bench.setdefault(name, {})[key] = (row["avg_ts"], row["stddev_ts"])

    pcache: dict = {}
    rows = []
    for f in sorted(res.glob("ppl_wikitext2_s2048_*.json")):
        r = json.loads(f.read_text())
        name = f.stem.removeprefix("ppl_wikitext2_s2048_")
        size = (Path(args.gguf_dir) / f"{name}.gguf").stat().st_size
        done = r["n_chunks_done"] == r["n_chunks_total"]
        b = bench.get(name, {})
        n_par = dense_params(Path(args.gguf_dir), family(name), pcache)
        part = "" if done else f" ({r['n_chunks_done']}/{r['n_chunks_total']} Fenster)"
        rows.append((size, name, n_par, r["ppl"], part, b.get("pp"), b.get("tg")))

    print("| Modell | Parameter | Größe (MiB) | bpw (ganze Datei) | PPL WikiText-2 | pp512 t/s | tg128 t/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    fmt = lambda v: f"{v[0]:.0f} ± {v[1]:.0f}" if v else "–"  # noqa: E731
    for size, name, n_par, ppl, part, pp, tg in sorted(rows):
        ppl_s = f"{ppl:.4g}" if ppl > 1e4 else f"{ppl:.2f}"
        print(f"| {name} | {n_par / 1e9:.2f} B | {size / 2**20:.0f} | {size * 8 / n_par:.2f} | {ppl_s}{part} | "
              f"{fmt(pp)} | {(f'{tg[0]:.1f}' if tg else '–')} |")


if __name__ == "__main__":
    main()
