"""Re-run the reference against golden files (skipped unless LB_MODEL and LB_GOLDEN are set)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from littlebit_ref import Qwen3Ref, lb_linear, load_checkpoint
from littlebit_ref.metrics import TOL_FP_REFERENCE, logits_stats, within


@pytest.fixture(scope="module")
def setup():
    m, gdir = os.environ.get("LB_MODEL"), os.environ.get("LB_GOLDEN")
    if not m or not gdir:
        pytest.skip("LB_MODEL / LB_GOLDEN not set")
    ck = load_checkpoint(m, dtype=torch.float64, keep_dense_dtype=True)
    return ck, Path(gdir), json.loads((Path(gdir) / "meta.json").read_text())


def test_layers(setup):
    ck, gdir, _ = setup
    data = np.load(gdir / "layers.npz")
    for k in [k for k in data.files if k.endswith(".x_rand")]:
        pre = k[: -len(".x_rand")]
        y = lb_linear(torch.from_numpy(data[k]).double(), ck.linears[pre]).numpy()
        np.testing.assert_allclose(y, data[pre + ".y_rand"], rtol=1e-5, atol=1e-5 * np.abs(y).max())


def test_logits(setup):
    ck, gdir, meta = setup
    ref = Qwen3Ref.from_checkpoint(ck)
    for i, ids in enumerate(meta["token_ids"]):
        g = np.load(gdir / f"prompt_{i}.npz")
        logits = ref.forward(torch.tensor(ids))
        s = logits_stats(torch.from_numpy(g["logits"]), logits)
        assert within(s, TOL_FP_REFERENCE), s
