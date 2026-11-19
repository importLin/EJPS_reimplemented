"""Evaluation of ciphertext-only reconstruction attacks against the plaintext.

Residual ambiguity (README): from edge continuity alone an attack recovers each
key group only up to

* one dihedral move of every block of the group (8),
* the absolute order of the three channels (6),
* the inversion ``255 - x`` of each channel (2^3),

so ``best_alignment`` searches this group of 384 elements for the element that
brings the reconstruction closest (MSE) to the plaintext, and every quality
number below is computed *after* that alignment, per key group.

Metrics
-------
``evaluate_blocks``   PSNR / SSIM of the aligned image, exact block rate (blocks
                      equal to the plaintext block after alignment) and exact
                      sub-block rate.
``direct_comparison`` / ``neighbor_comparison``  jigsaw accuracy of a
                      ``JigsawResult.order``-style permutation.
``synthetic_natural_image``  deterministic smooth test image (no downloads).

SSIM is the Gaussian-window SSIM (sigma 1.5, truncate 3.5, the usual constants
``K1 = 0.01``, ``K2 = 0.03``), averaged over channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import (
    CHANNEL_PERMS,
    N_DIHEDRAL,
    KeyGroups,
    blocks_to_image,
    dihedral,
    split_sub_blocks,
)

# ---------------------------------------------------------------------------
# Residual-ambiguity alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """``aligned[..., c] = inv_c(dihedral(recon, k)[..., perm[c]])``."""

    dihedral: int
    perm: tuple[int, int, int]
    invert: tuple[bool, bool, bool]
    mse: float


def apply_alignment(blocks: np.ndarray, al: Alignment) -> np.ndarray:
    """Apply ``al`` to a block stack ``(n, B, B, 3)`` uint8."""
    out = dihedral(blocks, al.dihedral, axes=(1, 2))[..., list(al.perm)]
    out = np.where(np.asarray(al.invert), 255 - out, out)
    return np.ascontiguousarray(out.astype(blocks.dtype))


def best_alignment(recon: np.ndarray, plain: np.ndarray) -> Alignment:
    """The residual-group element minimising the MSE to ``plain`` (one key group).

    Ties go to the first element in the order dihedral -> permutation ->
    no inversion before inversion.
    """
    p = plain.astype(np.float64)
    count = p[..., 0].size  # pragma: no mutate (any channel)
    best = None
    for k in range(N_DIHEDRAL):
        r = dihedral(recon, k, axes=(1, 2)).astype(np.float64)
        # err[c_out, c_src, inverted]
        err = np.empty((3, 3, 2))  # pragma: no mutate (fully written)
        for co in range(3):
            for cs in range(3):
                err[co, cs, 0] = np.sum((p[..., co] - r[..., cs]) ** 2)
                err[co, cs, 1] = np.sum((p[..., co] - (255 - r[..., cs])) ** 2)
        for perm in CHANNEL_PERMS:
            inv = tuple(bool(err[c, perm[c], 1] < err[c, perm[c], 0]) for c in range(3))
            total = sum(err[c, perm[c], int(inv[c])] for c in range(3)) / (3 * count)
            if best is None or total < best.mse:
                best = Alignment(k, tuple(perm), inv, float(total))
    return best


def align_groups(
    recon: np.ndarray, plain: np.ndarray, groups: KeyGroups | None = None
) -> tuple[np.ndarray, list[Alignment]]:
    """Align every key group separately; ``None`` = a single group."""
    if recon.shape != plain.shape:
        raise ValueError(f"shape mismatch {recon.shape} vs {plain.shape}")
    groups = groups or KeyGroups.common(len(recon))
    out = np.empty_like(recon)
    als = []
    for g in groups:
        al = best_alignment(recon[g], plain[g])
        out[g] = apply_alignment(recon[g], al)
        als.append(al)
    return out, als


def is_equivalent(recon: np.ndarray, plain: np.ndarray) -> bool:
    """True iff ``recon`` equals ``plain`` exactly under some residual-group element."""
    return best_alignment(recon, plain).mse == 0.0


# ---------------------------------------------------------------------------
# Quality numbers
# ---------------------------------------------------------------------------


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def _ssim_plane(a: np.ndarray, b: np.ndarray, data_range: float, sigma: float = 1.5) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def blur(x):
        return gaussian_filter(x, sigma=sigma, truncate=3.5)

    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a * mu_a
    sbb = blur(b * b) - mu_b * mu_b
    sab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (saa + sbb + c2)
    return float(np.mean(num / den))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Gaussian SSIM averaged over channels of ``(H, W)`` or ``(H, W, C)`` images."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.ndim == 2:
        return _ssim_plane(a, b, data_range)
    return float(
        np.mean([_ssim_plane(a[..., c], b[..., c], data_range) for c in range(a.shape[-1])])
    )


def exact_block_rate(aligned: np.ndarray, plain: np.ndarray) -> float:
    """Fraction of blocks identical to the plaintext block."""
    return float(np.mean(np.all(aligned == plain, axis=(1, 2, 3))))


def exact_sub_block_rate(aligned: np.ndarray, plain: np.ndarray, sub_block_size: int) -> float:
    """Fraction of sub-blocks identical to the plaintext sub-block."""
    a = split_sub_blocks(aligned, sub_block_size)
    p = split_sub_blocks(plain, sub_block_size)
    return float(np.mean(np.all(a == p, axis=(2, 3, 4))))


def evaluate_blocks(
    recon: np.ndarray,
    plain: np.ndarray,
    rows: int,
    cols: int,
    *,
    groups: KeyGroups | None = None,
    sub_block_size: int = 8,
) -> dict:
    """All block-level numbers after per-group residual alignment.

    ``recon`` / ``plain`` are ``(rows * cols, B, B, 3)`` stacks of one image.
    """
    aligned, als = align_groups(recon, plain, groups)
    img_a = blocks_to_image(aligned, rows, cols)
    img_p = blocks_to_image(plain, rows, cols)
    return {
        "psnr": psnr(img_a, img_p),
        "ssim": ssim(img_a, img_p),
        "block_rate": exact_block_rate(aligned, plain),
        "sub_block_rate": exact_sub_block_rate(aligned, plain, sub_block_size),
        "alignments": als,
    }


# ---------------------------------------------------------------------------
# Jigsaw accuracy
# ---------------------------------------------------------------------------


def direct_comparison(order: np.ndarray, true_order: np.ndarray) -> float:
    """Fraction of output positions holding the correct piece.

    Both arrays are ``JigsawResult.order``-style: ``order[pos]`` = index of the
    input piece placed at output position ``pos`` (row-major).
    """
    order, true_order = np.asarray(order), np.asarray(true_order)
    if order.shape != true_order.shape:
        raise ValueError(f"shape mismatch {order.shape} vs {true_order.shape}")
    return float(np.mean(order == true_order))


def neighbor_comparison(order: np.ndarray, true_order: np.ndarray, rows: int, cols: int) -> float:
    """Fraction of true right / below neighbour pairs that are neighbours alike in ``order``."""
    order, true_order = np.asarray(order), np.asarray(true_order)
    if order.size != rows * cols or true_order.size != rows * cols:
        raise ValueError(f"orders must have {rows * cols} entries")
    pos = np.empty(rows * cols, dtype=np.int64)
    pos[order] = np.arange(rows * cols)  # piece -> output position
    grid = true_order.reshape(rows, cols)
    pairs = [(grid[:, :-1], grid[:, 1:], 1), (grid[:-1, :], grid[1:, :], cols)]
    good = total = 0
    for a, b, step in pairs:
        pa, pb = pos[a.ravel()], pos[b.ravel()]
        ok = pb == pa + step
        if step == 1:
            ok &= pa % cols != cols - 1  # no wrap-around to the next row
        good += int(ok.sum())
        total += a.size
    return good / total if total else 1.0


# ---------------------------------------------------------------------------
# Synthetic plaintext
# ---------------------------------------------------------------------------


def synthetic_natural_image(size: int = 224, seed: int = 0, texture: float = 6.0) -> np.ndarray:
    """A smooth natural-looking RGB image: colour ramp + Gaussian blobs + faint texture.

    Channels are strongly but not perfectly correlated (each blob has its own
    tint), which is what the attacks and the candidate selector rely on.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size
    ang = rng.uniform(0, 2 * np.pi)
    img = (0.35 * (np.cos(ang) * xx + np.sin(ang) * yy))[..., None] * rng.uniform(0.6, 1.4, 3)
    for _ in range(14):
        cy, cx = rng.uniform(0, 1, 2)
        s = rng.uniform(0.04, 0.2)
        amp = rng.uniform(-0.6, 0.6)
        tint = rng.uniform(0.5, 1.5, 3)
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))[..., None] * tint
    tex = np.zeros_like(xx)
    for _ in range(3):
        th = rng.uniform(0, np.pi)
        period = rng.uniform(5, 14) / size
        ph = rng.uniform(0, 2 * np.pi)
        tex += np.sin(2 * np.pi * (np.cos(th) * xx + np.sin(th) * yy) / period + ph)
    img = (img - img.min()) / (img.max() - img.min())
    img = 30 + 190 * img + texture * tex[..., None]
    return np.clip(np.round(img), 0, 255).astype(np.uint8)  # pragma: no mutate (range 20..235)
