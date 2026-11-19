"""Automatic ranking of EJPS candidates (replaces picking 1 of 64 images by eye).

Criterion
---------
In natural images the three colour channels share their structure: inside a
small block, R, G and B rise and fall together, so their Pearson correlation is
strongly positive. For a candidate we compute, for each channel pair
``(R,G), (R,B), (G,B)`` and each block with non-constant channels, the
Pearson correlation of the two ``B x B`` planes (``feature="pixel"``) or of
their discrete Laplacians (``feature="laplacian"``: insensitive to the
linear-ramp part of a block, see below), and average over the blocks of the
key group: ``rho = (rho_RG, rho_RB, rho_GB)``.

* A layer in the **wrong dihedral orientation** relative to the others moves
  its structure to other pixels: its correlations with the other layers drop
  towards 0.
* A layer that is **inverted** relative to the others has negative
  correlations. Because inversion only flips signs, the solver's candidates
  need not enumerate it: for each candidate we choose the signs
  ``s = (1, s_G, s_B)`` maximising ``s_R s_G rho_RG + s_R s_B rho_RB + s_G s_B rho_GB``,
  that maximum is the candidate's score, and the returned blocks have G / B
  inverted where ``s = -1``.

Candidates are ranked by that score, highest first (stable for ties).

What the criterion cannot tell apart (residual ambiguity, README)
-----------------------------------------------------------------
* the absolute channel order: the score is symmetric in the three channels;
* inverting *all* channels: pairwise products are unchanged (so R is never
  inverted here, G / B are inverted *relative to R*);
* the same dihedral move applied to all three layers: within-block
  correlations do not see it;
* per block, a pure linear ramp is invariant under the flip along the ramp
  and maps to its own inversion under a 180 degree turn, so ramp-like blocks
  give the same evidence to several candidates; the decision comes from the
  blocks with curvature or texture, aggregated over the whole group. With
  ``feature="laplacian"`` ramps contribute nothing at all.

Usage (stages 0-2 of the README pipeline)::

    from ejps import EJPS_SOLVERS, KeyGroups, image_to_blocks, map_groups, select_best

    blocks = image_to_blocks(cipher_img, 16)
    solver = EJPS_SOLVERS.build("layer_greedy")
    rec = map_groups(blocks, KeyGroups.common(len(blocks)),
                     lambda g: select_best(solver.solve(g, sub_block_size=8)))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry import SubBlockCandidate

CHANNEL_PAIRS = ((0, 1), (0, 2), (1, 2))
# Sign patterns (s_R, s_G, s_B) in preference order: ties keep "no inversion".
SIGN_PATTERNS = ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1))
FEATURES = ("pixel", "laplacian")


@dataclass(frozen=True)
class RankedCandidate:
    """A candidate with its structure score; ``blocks`` has the inversion applied."""

    candidate: SubBlockCandidate
    blocks: np.ndarray
    score: float
    rho: tuple[float, float, float]
    invert: tuple[bool, bool, bool]
    index: int  # position in the solver's list


def block_features(blocks: np.ndarray, feature: str = "pixel") -> np.ndarray:
    """``(n, B, B, 3)`` -> ``(n, P, 3)`` float64, the per-block samples that get correlated."""
    x = np.asarray(blocks, dtype=np.float64)
    if feature == "pixel":
        f = x
    elif feature == "laplacian":
        f = (
            4 * x[:, 1:-1, 1:-1]
            - x[:, :-2, 1:-1]
            - x[:, 2:, 1:-1]
            - x[:, 1:-1, :-2]
            - x[:, 1:-1, 2:]
        )
    else:
        raise ValueError(f"unknown feature {feature!r}; choose from {FEATURES}")
    n, h, w, c = f.shape
    return f.reshape(n, h * w, c)


def channel_correlations(blocks: np.ndarray, feature: str = "pixel") -> tuple[float, float, float]:
    """Mean per-block Pearson correlation of ``(R,G), (R,B), (G,B)``.

    Blocks where either channel of a pair is constant are skipped for that
    pair; a pair with no usable block gets 0.
    """
    f = block_features(blocks, feature)
    f = f - f.mean(axis=1, keepdims=True)
    out = []
    for i, j in CHANNEL_PAIRS:
        num = np.sum(f[..., i] * f[..., j], axis=1)
        den = np.sqrt(np.sum(f[..., i] ** 2, axis=1) * np.sum(f[..., j] ** 2, axis=1))
        ok = den > 0
        out.append(float(np.mean(num[ok] / den[ok])) if ok.any() else 0.0)
    return tuple(out)


def best_signs(rho: Sequence[float]) -> tuple[float, tuple[int, int, int]]:
    """Maximise ``s_R s_G rho_RG + s_R s_B rho_RB + s_G s_B rho_GB`` over ``SIGN_PATTERNS``."""
    best_score, best_s = -np.inf, SIGN_PATTERNS[0]  # pragma: no mutate (overwritten)
    for s in SIGN_PATTERNS:
        signs = [s[i] * s[j] for i, j in CHANNEL_PAIRS]  # pragma: no mutate (+-1 products)
        val = float(np.dot(signs, rho))
        if val > best_score:
            best_score, best_s = val, s
    return float(best_score), best_s


def rank_candidates(
    candidates: Sequence[SubBlockCandidate],
    *,
    feature: str = "pixel",
    resolve_inversion: bool = True,
) -> list[RankedCandidate]:
    """Score every candidate and return them best first.

    With ``resolve_inversion=False`` the score is the plain sum of the three
    correlations and no channel is inverted.
    """
    ranked = []
    for idx, cand in enumerate(candidates):
        rho = channel_correlations(cand.blocks, feature)
        if resolve_inversion:
            score, signs = best_signs(rho)
        else:
            score, signs = float(sum(rho)), SIGN_PATTERNS[0]
        invert = tuple(s < 0 for s in signs)  # pragma: no mutate (s is +-1)
        blocks = cand.blocks
        if any(invert):
            blocks = np.where(np.asarray(invert), 255 - blocks, blocks).astype(blocks.dtype)
        ranked.append(RankedCandidate(cand, blocks, score, rho, invert, idx))
    return sorted(ranked, key=lambda r: -r.score)


def select_best(candidates: Sequence[SubBlockCandidate], **kwargs) -> np.ndarray:
    """Blocks of the top-ranked candidate (inversion resolved)."""
    return rank_candidates(candidates, **kwargs)[0].blocks
