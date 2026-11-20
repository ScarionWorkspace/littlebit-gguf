import sys

import numpy as np
import pytest
import torch

from littlebit_ref.packing import pack_signs, unpack_signs


def test_known_pattern():
    # word 0: bits 0 and 3 set -> columns 0 and 3 are -1; LSB first
    packed = np.array([[0b1001]], dtype=np.int32)
    s = unpack_signs(packed, 5)
    assert s.tolist() == [[-1, 1, 1, -1, 1]]


def test_sign_bit_31_and_second_word():
    packed = np.array([[np.int32(-2**31), 1]], dtype=np.int32)  # bit 31 of word 0, bit 0 of word 1
    s = unpack_signs(packed, 40)
    assert s[0, 31] == -1 and s[0, 32] == -1
    assert (s[0, :31] == 1).all() and (s[0, 33:] == 1).all()


@pytest.mark.parametrize("shape", [(1, 1), (3, 31), (5, 32), (7, 33), (64, 1000), (192, 3072)])
def test_roundtrip(shape):
    rng = np.random.default_rng(0)
    s = rng.choice([-1.0, 1.0], size=shape).astype(np.float32)
    p = pack_signs(s)
    assert p.dtype == np.int32 and p.shape == (shape[0], (shape[1] + 31) // 32)
    assert np.array_equal(unpack_signs(p, shape[1]), s)


def test_padding_is_plus_one():
    s = -np.ones((2, 33), dtype=np.float32)
    p = pack_signs(s)
    assert p[0, 1] == 1  # only bit 0 (column 32) set in the second word, padding bits 0


def test_rejects_wrong_width():
    with pytest.raises(ValueError):
        unpack_signs(np.zeros((2, 2), dtype=np.int32), 100)


@pytest.mark.parametrize("shape", [(3, 31), (17, 100), (8, 64)])
def test_matches_littlebit_packer(littlebit_dir, shape):
    sys.path.insert(0, str(littlebit_dir))
    from quantization.utils.binary_packer import binary_packer  # LittleBit (CC BY-NC 4.0), test only
    g = torch.Generator().manual_seed(1)
    s = torch.randint(0, 2, shape, generator=g, dtype=torch.int8) * 2 - 1
    theirs = binary_packer(s).numpy()
    assert np.array_equal(pack_signs(s.numpy()), theirs)
    assert np.array_equal(unpack_signs(theirs, shape[1]), s.numpy().astype(np.float32))
