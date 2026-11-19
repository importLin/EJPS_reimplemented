"""Tests for ejps/metrics.py."""

import dataclasses
import hashlib

import make_fixtures as MF
import numpy as np
import pytest

from ejps import metrics as M
from ejps.geometry import (
    KeyGroups,
    dihedral,
    dihedral_inverse,
    image_to_blocks,
)


def _plain_blocks(seed=0, size=32):
    return image_to_blocks(M.synthetic_natural_image(size=size, seed=seed), 16)


def _scramble(plain, al):
    """A reconstruction that ``al`` maps back onto ``plain``."""
    q = np.empty_like(plain)
    for c in range(3):
        ch = plain[..., c]
        q[..., al.perm[c]] = 255 - ch if al.invert[c] else ch
    return np.ascontiguousarray(dihedral(q, dihedral_inverse(al.dihedral), axes=(1, 2)))


def test_best_alignment_recovers_the_residual_element():
    plain = _plain_blocks()
    want = M.Alignment(3, (2, 0, 1), (True, False, True), 0.0)
    recon = _scramble(plain, want)
    assert not np.array_equal(recon, plain)
    got = M.best_alignment(recon, plain)
    assert got == want
    np.testing.assert_array_equal(M.apply_alignment(recon, got), plain)
    assert M.is_equivalent(recon, plain)
    recon[0, 0, 0, 0] ^= 1
    assert not M.is_equivalent(recon, plain)


def test_best_alignment_mse_and_tie_order():
    plain = _plain_blocks()
    shifted = np.clip(plain.astype(int) + 2, 0, 255).astype(np.uint8)
    al = M.best_alignment(shifted, plain)
    assert (al.dihedral, al.perm, al.invert) == (0, (0, 1, 2), (False, False, False))
    assert al.mse == pytest.approx(np.mean((shifted.astype(float) - plain) ** 2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        al.mse = 0.0
    # a tie between "inverted" and "not inverted" keeps the channel as is
    half = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    half[0, 0] = 255
    tie = M.best_alignment(np.zeros_like(half), half)
    assert tie.invert == (False, False, False) and tie.mse == 255.0**2 / 2
    flat = np.full((2, 16, 16, 3), 7, dtype=np.uint8)
    assert M.best_alignment(flat, flat) == M.Alignment(0, (0, 1, 2), (False, False, False), 0.0)


def test_align_groups_per_group():
    plain = _plain_blocks(seed=2)
    a1 = M.Alignment(5, (1, 0, 2), (False, True, False), 0.0)
    a2 = M.Alignment(2, (0, 2, 1), (True, True, True), 0.0)
    recon = np.concatenate([_scramble(plain[:2], a1), _scramble(plain[2:], a2)])
    groups = KeyGroups(4, (np.array([0, 1]), np.array([2, 3])))
    aligned, als = M.align_groups(recon, plain, groups)
    np.testing.assert_array_equal(aligned, plain)
    assert als == [a1, a2]
    single, single_al = M.align_groups(recon, plain)
    assert M.exact_block_rate(single, plain) == 0.0  # no common element fits both halves
    assert single_al[0].mse == 662.509765625
    with pytest.raises(ValueError):
        M.align_groups(recon[:3], plain)
    ev = M.evaluate_blocks(recon, plain, 2, 2, groups=groups)
    assert ev["psnr"] == float("inf") and ev["ssim"] == 1.0
    assert ev["block_rate"] == 1.0 and ev["sub_block_rate"] == 1.0
    ev1 = M.evaluate_blocks(recon, plain, 2, 2)
    assert ev1["block_rate"] == 0.0 and ev1["sub_block_rate"] == 0.0
    assert ev1["psnr"] == pytest.approx(19.918880765597567)
    assert ev1["ssim"] == pytest.approx(0.6009291410777611)


def test_exact_rates():
    plain = _plain_blocks()
    aligned = plain.copy()
    aligned[2, 12, 3, 1] ^= 4  # block 2, sub-block dl
    assert M.exact_block_rate(aligned, plain) == 0.75
    assert M.exact_sub_block_rate(aligned, plain, 8) == 15 / 16
    assert M.exact_block_rate(plain, plain) == 1.0


def test_psnr_ssim_match_golden(golden):
    # golden: the reference PSNR / SSIM implementation on the same inputs
    a, b = MF.metrics_input()
    assert M.psnr(a, b) == golden["metrics"]["psnr"] == 33.35667663682227
    assert M.ssim(a, b) == golden["metrics"]["ssim"]
    assert M.ssim(a[..., 0], b[..., 0]) == golden["metrics"]["ssim_gray"]
    assert golden["metrics"]["ssim"] == pytest.approx(0.99709994166)
    assert M.psnr(a, a) == float("inf")
    assert M.psnr(np.zeros(4), np.ones(4)) == pytest.approx(48.130803608679)
    assert M.ssim(a, a) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.ssim(a, b[:-1])


def test_direct_and_neighbor_comparison():
    true = np.arange(4)
    assert M.direct_comparison(true, true) == 1.0
    assert M.neighbor_comparison(true, true, 2, 2) == 1.0
    swapped = np.array([1, 0, 3, 2])  # columns swapped: only vertical pairs survive
    assert M.direct_comparison(swapped, true) == 0.0
    assert M.neighbor_comparison(swapped, true, 2, 2) == 0.5
    wrapped = np.array([2, 0, 1, 3])  # piece 1 right after piece 0 but on the next row
    assert M.neighbor_comparison(wrapped, true, 2, 2) == 0.0
    assert M.direct_comparison(wrapped, true) == 0.25
    assert M.neighbor_comparison(np.array([2, 0, 1]), np.arange(3), 1, 3) == 0.5
    shuffled_truth = np.array([3, 1, 0, 2])
    assert M.neighbor_comparison(shuffled_truth, shuffled_truth, 2, 2) == 1.0
    assert M.neighbor_comparison(np.zeros(1, int), np.zeros(1, int), 1, 1) == 1.0
    with pytest.raises(ValueError):
        M.direct_comparison(true, true[:3])
    with pytest.raises(ValueError, match="orders must have 6 entries"):
        M.neighbor_comparison(true, true, 3, 2)
    with pytest.raises(ValueError, match="must have 6"):
        M.neighbor_comparison(true, np.arange(6), 3, 2)
    with pytest.raises(ValueError, match="must have 6"):
        M.neighbor_comparison(np.arange(6), true, 3, 2)


def test_synthetic_natural_image():
    img = M.synthetic_natural_image()
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8
    digest = hashlib.sha256(img.tobytes()).hexdigest()
    assert digest == "eb2c62967346a54fd610b4f6aa6d80e434fb9746bd5c24d9df28577f93ef1b0d"
    np.testing.assert_array_equal(img, M.synthetic_natural_image(seed=0))
    assert not np.array_equal(img, M.synthetic_natural_image(seed=1))
    small = M.synthetic_natural_image(size=48, seed=4, texture=0.0)
    assert small.shape == (48, 48, 3)
    assert int(img.min()) >= 20 and int(img.max()) <= 235
    x = img.reshape(-1, 3).astype(float).T
    corr = np.corrcoef(x)
    assert corr[0, 1] > 0.6 and corr[0, 2] > 0.6 and corr[1, 2] > 0.9
    assert (img[..., 0] != img[..., 1]).mean() > 0.5
