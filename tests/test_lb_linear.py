import torch

from littlebit_ref.checkpoint import LBLinear, LBPath
from littlebit_ref.model import lb_linear, rope


def rand_path(out, inp, r, g):
    sgn = lambda *s: torch.randint(0, 2, s, generator=g).double() * 2 - 1  # noqa: E731
    v1, u2 = torch.rand(r, generator=g, dtype=torch.float64), torch.rand(r, generator=g, dtype=torch.float64)
    return LBPath(U=sgn(out, r), V=sgn(r, inp), h=torch.rand(out, generator=g, dtype=torch.float64),
                  l=v1 * u2, g=torch.rand(inp, generator=g, dtype=torch.float64), v1=v1, u2=u2)


def test_factorized_equals_dense():
    g = torch.Generator().manual_seed(0)
    lin = LBLinear(main=rand_path(48, 64, 24, g), residual=rand_path(48, 64, 24, g),
                   bias=torch.rand(48, generator=g, dtype=torch.float64))
    x = torch.randn(5, 64, generator=g, dtype=torch.float64)
    y = lb_linear(x, lin)
    y_dense = x @ lin.dense().T + lin.bias
    assert torch.allclose(y, y_dense, rtol=1e-12, atol=1e-12)


def test_formula_matches_paper_notation():
    # W = diag(h) U diag(l) V diag(g)   (LittleBit-2, Eq. 1)
    g = torch.Generator().manual_seed(1)
    p = rand_path(16, 32, 8, g)
    W = torch.diag(p.h) @ p.U @ torch.diag(p.l) @ p.V @ torch.diag(p.g)
    assert torch.allclose(LBLinear(main=p).dense(), W, rtol=1e-12, atol=1e-12)


def test_rope_position_zero_is_identity_and_norm_preserving():
    x = torch.randn(3, 2, 8, dtype=torch.float64)
    y = rope(x, torch.tensor([0, 1, 7]), 1e6)
    assert torch.allclose(y[0], x[0])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1))


def test_metrics_ignore_argmax_flip():
    # two nearly tied tokens: a tiny perturbation flips argmax but must still pass
    from littlebit_ref.metrics import TOL_FP_REFERENCE, logits_stats, within
    ref = torch.tensor([[5.0, 5.0 + 1e-7, -3.0, 0.0]], dtype=torch.float64)
    test = ref.clone(); test[0, 0] += 2e-7
    s = logits_stats(ref, test)
    assert s["argmax_agree"] == 0.0 and within(s, TOL_FP_REFERENCE)
