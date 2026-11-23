#!/usr/bin/env python3
"""Size breakdown of a LittleBit GGUF and its effective bits per weight.

bpw is always relative to the parameter count of the *original dense* model
(embeddings + linear layers + norms + lm_head if untied), computed from the GGUF hparams.
Reported: whole-file bpw (incl. GGUF header and metadata), bpw of the LittleBit linear layers
(Q1_0 factors + g), and LittleBit's own bit budget for comparison (eff = 2r(in+out+16)+32(in+out)).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--llama-cpp-dir", type=Path, default=Path(__file__).resolve().parents[2] / "llama.cpp")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.llama_cpp_dir / "gguf-py"))
    import gguf

    r = gguf.GGUFReader(str(args.gguf))
    arch = r.fields["general.architecture"].contents()
    kv = lambda k: r.fields[k].contents()  # noqa: E731
    d, n_layer = kv(f"{arch}.embedding_length"), kv(f"{arch}.block_count")
    n_ff, vocab = kv(f"{arch}.feed_forward_length"), None
    n_head, n_kv, hd = kv(f"{arch}.attention.head_count"), kv(f"{arch}.attention.head_count_kv"), \
        kv(f"{arch}.attention.key_length")

    groups = defaultdict(lambda: [0, 0])  # name -> [bytes, n_tensors]
    lb = defaultdict(dict)
    has_output = False
    for t in r.tensors:
        n = t.name
        if n == "token_embd.weight":
            vocab = int(t.shape[1]); key = "embeddings"
        elif n == "output.weight":
            key, has_output = "lm_head", True
        elif ".lb_" in n:
            key = "littlebit_factors" if n.split(".lb_")[1][0] in "uv" else "littlebit_g"
            base = n.rsplit(".lb_", 1)[0]; part = n.rsplit(".lb_", 1)[1]
            lb[base][part] = [int(x) for x in t.shape]
        elif "norm" in n:
            key = "norms"
        else:
            key = "other"
        groups[key][0] += int(t.n_bytes); groups[key][1] += 1

    proj_shapes = {"attn_q": (d, n_head * hd), "attn_k": (d, n_kv * hd), "attn_v": (d, n_kv * hd),
                   "attn_output": (n_head * hd, d), "ffn_gate": (d, n_ff), "ffn_up": (d, n_ff),
                   "ffn_down": (n_ff, d)}
    linear_params = n_layer * sum(i * o for i, o in proj_shapes.values())
    norm_params = n_layer * (2 * d + 2 * hd) + d
    emb_params = vocab * d
    dense_params = linear_params + norm_params + emb_params * (2 if has_output else 1)

    # LittleBit's own budget, from the true ranks: true rank is not stored, use r_pad as upper bound
    file_bytes = args.gguf.stat().st_size
    lin_bytes = groups["littlebit_factors"][0] + groups["littlebit_g"][0]
    rep = {
        "file": args.gguf.name, "file_MiB": file_bytes / 2**20,
        "dense_params": dense_params, "linear_params": linear_params,
        "bpw_whole_file": file_bytes * 8 / dense_params,
        "bpw_linear_layers": lin_bytes * 8 / linear_params,
        # littlebit.eff_bits since the shared format; littlebit.eff_bit in older files of ours
        "littlebit_eff_bit_target": next((float(kv(k)) for k in ("littlebit.eff_bits", "littlebit.eff_bit")
                                          if k in r.fields), None),
        "parts_MiB": {k: v[0] / 2**20 for k, v in sorted(groups.items())},
        "parts_share": {k: v[0] / file_bytes for k, v in sorted(groups.items())},
        "metadata_and_header_MiB": (file_bytes - sum(v[0] for v in groups.values())) / 2**20,
    }
    print(json.dumps(rep, indent=1))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
