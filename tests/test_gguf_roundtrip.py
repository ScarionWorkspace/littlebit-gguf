"""GGUF written by convert_littlebit_to_gguf.py -> read back -> must equal the checkpoint
up to the FP16 rounding of the block scales h and l (and F16 embeddings).
Skipped unless LB_MODEL and LB_GGUF are set (and gguf-py is available)."""
import os

import numpy as np
import pytest
import torch

from littlebit_ref import Qwen3Ref, load_checkpoint
from littlebit_ref.metrics import logits_stats
from littlebit_ref.numerics import Numerics
from littlebit_ref.prompts import PROMPTS


@pytest.fixture(scope="module")
def pair(request):
    m, g = os.environ.get("LB_MODEL"), os.environ.get("LB_GGUF")
    if not m or not g:
        pytest.skip("LB_MODEL / LB_GGUF not set")
    request.getfixturevalue("gguf_py")
    from littlebit_ref.gguf_loader import load_gguf
    l_round = torch.bfloat16 if os.environ.get("LB_L_ROUND") == "bf16" else None
    return m, load_checkpoint(m, dtype=torch.float64, keep_dense_dtype=True, l_round=l_round), load_gguf(g)


def test_factors_bit_exact(pair):
    _, ck, gk = pair
    assert ck.linears.keys() == gk.linears.keys()
    f16 = lambda t: t.to(torch.float16).double()  # noqa: E731
    for name, lin in ck.linears.items():
        glin = gk.linears[name]
        for p, q in ((lin.main, glin.main), (lin.residual, glin.residual)):
            if p is None:
                assert q is None
                continue
            r = p.rank
            # shared spec: the bits are the checkpoint's signs, the scales keep their sign
            assert torch.equal(q.U[:, :r], p.U.to(q.U.dtype)), name
            assert torch.all(q.U[:, r:] == -1), name            # padding bits cleared
            assert torch.equal(q.V[:r], p.V.to(q.V.dtype)), name
            assert torch.equal(q.h, f16(p.h)), name
            assert torch.equal(q.l[:r], f16(p.l)), name
            assert torch.all(q.l[r:] == 0), name               # padding rows contribute nothing
            assert torch.equal(q.g, p.g), name                  # g is F32: exact


def test_logits_match_reference_with_fp16_scales(pair):
    m, ck, gk = pair
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m)
    ref = Qwen3Ref.from_checkpoint(ck, numerics=Numerics(scales_fp16=True))
    test = Qwen3Ref.from_checkpoint(gk)
    for p in PROMPTS[:2]:
        ids = tok(p, return_tensors="pt").input_ids[0]
        s = logits_stats(ref.forward(ids), test.forward(ids))
        # remaining difference: BF16 -> F16 embeddings
        assert s["rel_max"] < 1e-5 and s["kl_max"] < 1e-9, s
