"""EJPS solver "edge_ref": edge matching against a reference sub-block.

A vectorised port of the author's original (loop-based) notebook
implementation, which is not part of this repository. It runs the same seven
stages on an arbitrary block stack ``(n, 2m, 2m, 3)`` (one key group) with the
per-block Python loops replaced by array operations; the output was verified
bit-identical to the original (``tests/test_restore_edge.py`` locks it with
golden fixtures).

Stages
------
1. **Sub-block scramble.** For each of ``ur, dl, dr`` take the best edge match
   to ``ul`` (any edge pair, edge of ``ul`` in any of the 4 variants
   plain / reversed / nega / reversed+nega, per ``ul`` channel against every
   channel), summed over blocks. The *worst* of the three is ``ul``'s diagonal
   partner, which is swapped into ``dr``.
2.-7. **Channels R, G, B in turn** (target channel ``t`` of the current image):

   * *main*: choose transforms ``(ul_t, ur_t, dl_t)`` of the 16-element plane
     group and source channels ``(c2, c3)`` minimising
     ``MSE(right(ul), left(ur)) + MSE(down(ul), top(dl))`` summed over blocks;
     write the transformed planes into channel ``t`` (``ur`` / ``dl`` swap
     channels ``t`` and ``c2`` / ``c3``). Source channels are restricted to
     not-yet-solved ones: ``{0,1,2}`` for R, ``{1,2}`` for G, ``{2}`` for B.
   * *corner*: choose the transform and source channel of ``dr`` matching the
     solved ``down(ur)`` and ``right(dl)``.

Ties are broken exactly as in the original (first strict minimum in its
enumeration order). The result is the plaintext up to the residual ambiguity
of the reference sub-block ``ul`` (a dihedral move shared by all channels, the
absolute channel order, per-channel inversion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import (
    EJPS_SOLVERS,
    SubBlockCandidate,
    all_plane_transforms,
    merge_sub_blocks,
    plane_edges,
    plane_transform,
    split_sub_blocks,
    sub_block_edges,
)

UL, UR, DL, DR = 0, 1, 2, 3
TOP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
# (target channel, admissible source channels) of the three channel rounds.
CHANNEL_ROUNDS = ((0, (0, 1, 2)), (1, (1, 2)), (2, (2,)))
# Sub-block order after stage 1, by the index (into ur, dl, dr) of the diagonal.
_DIAGONAL_ORDER = {0: (UL, DR, DL, UR), 1: (UL, UR, DR, DL), 2: (UL, UR, DL, DR)}


def _mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean squared difference over the last axis (same rounding as ``np.mean``)."""
    return np.mean((a - b) ** 2, axis=-1)


def scramble_compat(subs: np.ndarray) -> np.ndarray:
    """Stage-1 compatibility of ``ur, dl, dr`` with ``ul`` -> ``(3,)``; max = diagonal.

    subs ``(n, 4, m, m, 3)``.
    """
    e = np.moveaxis(sub_block_edges(subs), -1, -2)  # n, sub, edge, ch, m
    ul = e[:, UL]
    rev = ul[..., ::-1]
    var = np.stack([ul, rev, 255 - ul, 255 - rev])
    var = np.moveaxis(var, 0, 1)  # pragma: no mutate (n, var, pa, ca, m; min is over var & pa)
    a = var[:, :, :, :, None, None, :]  # n, var, pa, ca, 1, 1, m
    comp = []
    for x in (UR, DL, DR):
        b = e[:, x][:, None, None, None]  # n, 1, 1, 1, pb, cb, m
        acc = _mse(a, b).sum(axis=0)  # var, pa, ca, pb, cb
        comp.append(acc.min(axis=(0, 1, 3, 4)).sum())
    return np.array(comp)


def _transformed_edges(planes: np.ndarray) -> np.ndarray:
    """``(n, m, m)`` -> ``(n, 16, 4, m)``: edges of every plane transform."""
    return plane_edges(all_plane_transforms(planes))


def main_scores(subs: np.ndarray, target: int, cands: tuple[int, ...]) -> np.ndarray:
    """``(k*k, 16, 16, 16)`` over ``((c2, c3), ul_t, ur_t, dl_t)``, pairs in reference order."""
    ul = _transformed_edges(subs[:, UL, :, :, target])
    ur = {c: _transformed_edges(subs[:, UR, :, :, c])[:, :, LEFT] for c in cands}
    dl = {c: _transformed_edges(subs[:, DL, :, :, c])[:, :, TOP] for c in cands}
    right = {c: _mse(ul[:, :, None, RIGHT], ur[c][:, None]) for c in cands}  # n, i, j
    down = {c: _mse(ul[:, :, None, DOWN], dl[c][:, None]) for c in cands}  # n, i, k
    return np.stack(
        [
            (right[c2][:, :, :, None] + down[c3][:, :, None, :]).sum(axis=0)
            for c2 in cands
            for c3 in cands
        ]
    )


def corner_scores(subs: np.ndarray, target: int, cands: tuple[int, ...]) -> np.ndarray:
    """``(k, 16)`` over ``(c, dr_t)``: dr against the solved ``down(ur)`` / ``right(dl)``."""
    ur_down = plane_edges(subs[:, UR, :, :, target])[:, DOWN][:, None]
    dl_right = plane_edges(subs[:, DL, :, :, target])[:, RIGHT][:, None]
    out = []
    for c in cands:
        dr = _transformed_edges(subs[:, DR, :, :, c])
        out.append((_mse(ur_down, dr[:, :, TOP]) + _mse(dl_right, dr[:, :, LEFT])).sum(axis=0))
    return np.stack(out)


def _put(subs: np.ndarray, pos: int, target: int, src: int, idx: int) -> None:
    """Transform channel ``src`` of sub-block ``pos`` by ``idx`` and swap it into ``target``."""
    plane = plane_transform(subs[:, pos, :, :, src], idx).copy()  # rot90 / flip are views
    if src != target:
        subs[:, pos, :, :, src] = subs[:, pos, :, :, target]
    subs[:, pos, :, :, target] = plane


@EJPS_SOLVERS.register("edge_ref")
@dataclass(frozen=True)
class EdgeRefSolver:
    """The edge-matching EJPS as an ``EjpsSolver``.

    stat_blocks: use only the first N blocks of the group for the statistics;
    ``None`` = all blocks.
    """

    stat_blocks: int | None = None

    def solve(self, blocks: np.ndarray, *, sub_block_size: int = 8) -> list[SubBlockCandidate]:
        subs = split_sub_blocks(np.asarray(blocks, dtype=np.uint8), sub_block_size)
        n_stat = len(subs) if self.stat_blocks is None else min(self.stat_blocks, len(subs))
        if n_stat <= 0:
            raise ValueError("edge_ref: no blocks available for the statistics")

        diagonal = int(np.argmax(scramble_compat(subs[:n_stat])))
        subs = np.ascontiguousarray(subs[:, list(_DIAGONAL_ORDER[diagonal])])
        meta: dict = {"diagonal": ("ur", "dl", "dr")[diagonal], "channels": []}
        cost = 0.0
        for target, cands in CHANNEL_ROUNDS:
            sc = main_scores(subs[:n_stat], target, cands)
            flat = int(np.argmin(sc))
            pair, rest = divmod(flat, 16**3)
            ul_t, ur_t, dl_t = np.unravel_index(rest, (16, 16, 16))  # pragma: no mutate
            c2, c3 = cands[pair // len(cands)], cands[pair % len(cands)]
            cost += float(sc.flat[flat])
            _put(subs, UL, target, target, int(ul_t))
            _put(subs, UR, target, c2, int(ur_t))
            _put(subs, DL, target, c3, int(dl_t))

            cs = corner_scores(subs[:n_stat], target, cands)
            flat = int(np.argmin(cs))
            c_idx, dr_t = divmod(flat, 16)
            c4 = cands[c_idx]
            cost += float(cs.flat[flat])
            _put(subs, DR, target, c4, dr_t)
            meta["channels"].append(
                {
                    "target": target,
                    "transforms": (int(ul_t), int(ur_t), int(dl_t), dr_t),
                    "sources": (target, c2, c3, c4),
                }
            )
        return [SubBlockCandidate(merge_sub_blocks(subs), cost, meta)]
