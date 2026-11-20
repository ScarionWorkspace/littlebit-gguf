import numpy as np
import pytest
import torch

from littlebit_ref.checkpoint import LBLinear, LBPath
from littlebit_ref.model import lb_linear
from littlebit_ref.numerics import Numerics, _roundf, q8_0_roundtrip


def test_roundf_half_away_from_zero_and_fp32_edge():
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, -0.49999997, 0.0], dtype=torch.float32)
    assert _roundf(x).tolist() == [1, 2, 3, -1, -3, 0, 0, 0]


def test_q8_0_scalar_loop():
    # direct transcription of quantize_row_q8_0_ref + dequant, element by element
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64).astype(np.float32)
    exp = np.empty_like(x)
    for b in range(2):
        blk = x[b * 32:(b + 1) * 32]
        d = np.float32(np.abs(blk).max() / np.float32(127))
        inv = np.float32(1) / d
        for j, v in enumerate(blk):
            a = abs(np.float32(v * inv))
            q = np.sign(v) * (np.floor(a) + np.floor(2 * (a - np.floor(a))))
            exp[b * 32 + j] = np.float32(q) * np.float32(np.float16(d))
    assert np.array_equal(q8_0_roundtrip(torch.from_numpy(x)).numpy(), exp)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 300.0])
def test_q8_0_matches_gguf_py(gguf_py, scale):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((7, 256)) * scale).astype(np.float32)
    x[0, :32] = 0  # all-zero block
    q = gguf_py.quants.quantize(x, gguf_py.GGMLQuantizationType.Q8_0)
    ref = gguf_py.quants.dequantize(q, gguf_py.GGMLQuantizationType.Q8_0)
    assert np.array_equal(q8_0_roundtrip(torch.from_numpy(x)).numpy(), ref)


def test_q8_0_padding_does_not_change_values():
    x = torch.randn(3, 40)
    y = q8_0_roundtrip(x)
    assert y.shape == x.shape
    assert torch.equal(y[:, :32], q8_0_roundtrip(x[:, :32]))


def test_exact_numerics_is_default_and_sim_is_close():
    g = torch.Generator().manual_seed(0)
    sgn = lambda *s: torch.randint(0, 2, s, generator=g).float() * 2 - 1  # noqa: E731
    p = LBPath(U=sgn(64, 40), V=sgn(40, 128), h=torch.rand(64, generator=g), l=torch.rand(40, generator=g),
               g=torch.rand(128, generator=g))
    lin = LBLinear(main=p)
    x = torch.randn(4, 128, generator=g)
    y0 = lb_linear(x, lin)
    assert torch.equal(y0, lb_linear(x, lin, Numerics.exact()))
    y1 = lb_linear(x, lin, Numerics.llamacpp_q1_0())
    rel = ((y1 - y0).norm() / y0.norm()).item()
    assert 0 < rel < 0.02
