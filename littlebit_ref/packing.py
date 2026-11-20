"""Bit layout of LittleBit's packed sign matrices.

Derived from the on-disk format (see docs/ANALYSIS.md 1.3), independently implemented:
  * int32 words, row-major, ``words_per_row = ceil(ncols / 32)``
  * bit j of word w (LSB first) holds column ``32*w + j``
  * bit 1 means -1, bit 0 means +1; padding columns are +1 (bit 0)
"""
import numpy as np


def unpack_signs(packed: np.ndarray, ncols: int) -> np.ndarray:
    """int32 [rows, words] -> float32 ±1 [rows, ncols]."""
    packed = np.ascontiguousarray(packed)
    if packed.ndim != 2:
        raise ValueError(f"expected 2D packed tensor, got shape {packed.shape}")
    rows, words = packed.shape
    if words != (ncols + 31) // 32:
        raise ValueError(f"{words} words per row do not match ncols={ncols}")
    as_bytes = packed.astype("<i4", copy=False).view(np.uint8).reshape(rows, words * 4)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")[:, :ncols]
    return (1.0 - 2.0 * bits).astype(np.float32)


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """float/int ±1 [rows, ncols] -> int32 [rows, ceil(ncols/32)] (inverse of unpack_signs).

    Zero is treated as +1, matching ``sign(0) := +1`` in LittleBit.
    """
    signs = np.asarray(signs)
    rows, ncols = signs.shape
    words = (ncols + 31) // 32
    bits = np.zeros((rows, words * 32), dtype=np.uint8)
    bits[:, :ncols] = signs < 0
    as_bytes = np.packbits(bits, axis=1, bitorder="little")
    return as_bytes.view("<i4").reshape(rows, words).astype(np.int32)
