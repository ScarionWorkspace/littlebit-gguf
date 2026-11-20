#!/usr/bin/env python3
"""Diagnostic: compare intermediate tensors of block 0 in llama.cpp with the reference, step by step.

usage: check_llamacpp_steps.py <checkpoint dir> <gguf> <flash_attn auto|on|off> [tool]
With flash attention off, llama.cpp should equal the simulation (Numerics.llamacpp_q1_0) to
~1e-7 up to the first place where a single Q8 rounding flips; a real bug (wrong scale, padding,
factor order) shows up as a large deviation at the first affected step.
"""
import sys, subprocess, tempfile, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from littlebit_ref import Qwen3Ref, load_checkpoint
from littlebit_ref.numerics import Numerics
from transformers import AutoTokenizer
m, g, fa = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(m)
sm = Qwen3Ref.from_checkpoint(load_checkpoint(m, dtype=torch.float32, keep_dense_dtype=True), numerics=Numerics.llamacpp_q1_0())
ex = Qwen3Ref.from_checkpoint(load_checkpoint(m, dtype=torch.float64, keep_dense_dtype=True))
ids = tok("The capital of France is", return_tensors='pt').input_ids[0]
cs, ce = {}, {}
sm.forward(ids, hook=lambda k, t: cs.__setitem__(k, t)); ex.forward(ids, hook=lambda k, t: ce.__setitem__(k, t))
P = 'model.layers.0.'
pairs = {'attn_norm-0': P+'self_attn.q_proj.in', 'kqv_out-0': P+'self_attn.o_proj.in',
         'ffn_norm-0': P+'mlp.gate_proj.in', 'ffn_out-0': P+'mlp.down_proj.out', 'l_out-0': P+'out'}
rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
with tempfile.TemporaryDirectory() as td:
    idf, outf = Path(td) / 'ids', Path(td) / 'out'
    ids.to(torch.int32).numpy().astype('<i4').tofile(idf)
    tool = sys.argv[4] if len(sys.argv) > 4 else str(ROOT / 'tools/bin/lb_dump_logits-build-cpu')
    r = subprocess.run([tool, g, str(idf), str(outf), '8', fa, '0', ','.join(pairs)], capture_output=True, text=True)
    for name, key in pairs.items():
        f = Path(f'{outf}.{name}')
        if not f.exists(): print(name, 'not dumped'); continue
        lc = torch.from_numpy(np.fromfile(f, dtype='<f4').reshape(len(ids), -1).copy())
        print(f'{name:12s} llama.cpp vs sim {rel(lc, cs[key]):.2e}   vs exact {rel(lc, ce[key]):.2e}   (sim vs exact {rel(cs[key], ce[key]):.2e})')
