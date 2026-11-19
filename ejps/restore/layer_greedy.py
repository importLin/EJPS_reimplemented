"""EJPS solver "layer_greedy": greedy layer-by-layer assembly of the 12 planes.

A refactor of the author's original notebook implementation (not part of this
repository); "the notebook" below refers to that original.

Idea
----
A cipher block holds 4 sub-blocks x 3 channels = 12 ``m x m`` *planes*. Each
plane is an arbitrary dihedral transform (and maybe a negative) of one colour
channel of one plaintext sub-block, sitting in an arbitrary sub-block slot.
Treat the 12 planes as jigsaw pieces and assemble three 2x2 *layers*, one at a
time, greedily:

1. **Neighbour search.** Take the layer's anchor plane (kept untransformed)
   and score every other candidate plane on every (anchor edge, candidate edge,
   variant) triple, variant in ``{plain, reversed, nega, reversed+nega}``; the
   cost is the edge MSE summed over all blocks of the key group. The global
   minimum is the first neighbour ("best"); after forbidding the anchor's edges
   on that axis, the minimum over the other planes is the second ("sbest"),
   on the orthogonal axis. The two edges fix where the anchor sits in the 2x2.
2. **Placement.** Each neighbour gets its variant (flip *along* the matched
   edge, nega) and then the unique dihedral move that turns the matched edge
   to face the anchor, preserving the pixel order along it.
3. **Fourth plane.** The empty diagonal slot is filled by the unused plane,
   dihedral transform (8) and variant (4) whose two edges best match the two
   placed neighbours.

Pass R uses the raw 12 planes, anchor = ``ul`` channel 0. The 8 leftovers are
then dealt, in index order, into the G / B slots (first four -> G slots, next
four -> B slots) and pass G runs over those 8 slots (anchor = G slot of ``ul``);
its 4 leftovers form pass B. The G and B layers are each internally consistent
but their dihedral orientation relative to R is unknown, so the solver emits
**64 candidates** ``(g, b)``: G layer under ``dihedral(g)``, B under
``dihedral(b)``, candidate ``8 * g + b``. Picking one is the job of
:mod:`ejps.restore.select`. Inversion of a whole layer and the absolute channel
order stay ambiguous (see README, residual ambiguity).

Fidelity
--------
With the default flags the output is bit-identical to the notebook (checked
against the notebook's 64 candidate images on synthetic ciphertexts; locked by
golden hashes in ``tests/test_restore_layer_greedy.py``), including its
tie-breaking: candidate lists are scanned in the notebook's order and the first
strict minimum wins.

Notebook issues kept by default, fixable by flag
------------------------------------------------
``fix_fourth_flip``
    In the fourth-plane search the notebook's "flip" variant reverses *both*
    matched edges in place, which is not the edge set of any transform of the
    plane, yet a winning flip variant is applied as ``fliplr``. With the flag
    the flip variants are excluded there (the 8 dihedral moves already cover
    every geometric case), so the scored and the applied plane agree.
``one_plane_per_sub_block``
    The notebook lets a layer take two planes of the same cipher sub-block
    (e.g. ``ul.G`` as the anchor's neighbour in pass R), although a channel
    shuffle never splits a sub-block, so such pairs are impossible and
    co-located channels are a strong false match. With the flag each layer
    draws its planes from four distinct cipher sub-blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import (
    EJPS_SOLVERS,
    N_DIHEDRAL,
    SubBlockCandidate,
    all_dihedral,
    dihedral,
    merge_sub_blocks,
    plane_edges,
    split_sub_blocks,
)

TOP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
# Neighbour offset (drow, dcol) of the plane adjacent to each anchor edge.
_EDGE_OFFSET = {TOP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
# Side of the neighbour that faces the anchor across each anchor edge.
_FACING = {TOP: DOWN, DOWN: TOP, LEFT: RIGHT, RIGHT: LEFT}
# Fourth plane at slot p: ((neighbour slot, its edge), candidate edge) x 2, in
# the notebook's summation order. Slots: 0 ul, 1 ur, 2 dl, 3 dr.
_FOURTH_PAIRS = {
    0: (((1, LEFT), RIGHT), ((2, TOP), DOWN)),
    1: (((0, RIGHT), LEFT), ((3, TOP), DOWN)),
    2: (((0, DOWN), TOP), ((3, LEFT), RIGHT)),
    3: (((2, RIGHT), LEFT), ((1, DOWN), TOP)),
}


def _edge_move_table() -> np.ndarray:
    """``table[src, dst]`` = dihedral k whose edge ``dst`` equals edge ``src`` of the input."""
    probe = np.arange(9).reshape(3, 3)
    src = plane_edges(probe)
    table = np.full((4, 4), -1)  # pragma: no mutate (every entry is overwritten)
    for k in range(N_DIHEDRAL):
        dst = plane_edges(dihedral(probe, k))
        for s in range(4):
            for d in range(4):
                if np.array_equal(src[s], dst[d]):
                    table[s, d] = k
    return table


EDGE_MOVE = _edge_move_table()


def edge_variants(edges: np.ndarray) -> np.ndarray:
    """``(..., m)`` -> ``(..., 4, m)``: plain, reversed, nega, reversed + nega."""
    rev = edges[..., ::-1]
    return np.stack([edges, rev, 255 - edges, 255 - rev], axis=-2)


def _masked(scores: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Copy of ``scores`` (candidates on axis 0) with inadmissible candidates set to inf."""
    keep = np.asarray(ok, dtype=bool).reshape(len(scores), *([1] * (scores.ndim - 1)))
    return np.where(keep, scores, np.inf)


def _first_min(scores) -> tuple[int, int, float]:
    """Notebook scan: first array whose minimum is strictly smallest -> (array, flat argmin, min)."""
    best_k, best_v = -1, np.inf  # pragma: no mutate (any negative start index)
    for k, s in enumerate(scores):
        v = float(np.min(s))
        if v < best_v:
            best_k, best_v = k, v
    if best_k < 0:
        raise RuntimeError("layer_greedy: no admissible candidate plane left")
    return best_k, int(np.argmin(scores[best_k])), best_v


def neighbour_scores(anchor: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Edge cost of every candidate next to the anchor.

    anchor ``(n, m, m)``, cands ``(n, K, m, m)`` -> ``(K, 4 variants, 4 anchor
    edges, 4 candidate edges)``, summed over the n blocks.
    """
    a = plane_edges(anchor)[:, None, None, :, None, :]  # n,1,1,pa,1,m
    v = edge_variants(plane_edges(cands)).swapaxes(-3, -2)[:, :, :, None]  # n,K,var,1,pb,m
    return np.mean((a - v) ** 2, axis=-1).sum(axis=0)


def fourth_scores(layer: np.ndarray, slot: int, cands: np.ndarray) -> np.ndarray:
    """Cost of each candidate in the empty ``slot`` of ``layer`` ``(n, 4, m, m)``.

    Returns ``(K, 4 variants, 8 dihedral)``: edges of ``dihedral(cand, r)``, in
    variant ``v``, against the facing edges of the two placed neighbours.
    """
    ce = edge_variants(plane_edges(all_dihedral(cands)))  # n,K,8,4edge,4var,m
    total = 0.0
    for (nb_slot, nb_edge), c_edge in _FOURTH_PAIRS[slot]:
        target = plane_edges(layer[:, nb_slot])[:, nb_edge][:, None, None, None, :]
        total = total + np.mean((target - ce[:, :, :, c_edge]) ** 2, axis=-1)
    return np.asarray(total).sum(axis=0).swapaxes(-1, -2)  # K, var, rot


def apply_variant(planes: np.ndarray, variant: int, flip_axis: int) -> np.ndarray:
    """Variant 0..3 = plain / flip / nega / flip+nega (flip along ``flip_axis``)."""
    out = np.flip(planes, axis=flip_axis) if variant % 2 else planes
    return 255 - out if variant >= 2 else out


@dataclass
class LayerResult:
    layer: np.ndarray  # (n, 4, m, m) uint8, slots ul, ur, dl, dr
    slots: tuple[int, int, int, int]  # pool index of the plane at ul, ur, dl, dr
    cost: float  # best + sbest + fourth matching cost
    meta: dict


def assemble_layer(
    pool: np.ndarray,
    groups: np.ndarray,
    *,
    fix_fourth_flip: bool = False,
    one_plane_per_sub_block: bool = False,
) -> LayerResult:
    """One greedy pass over ``pool`` ``(n, P, m, m)``; ``pool[:, 0]`` is the anchor.

    ``groups[p]`` is the cipher sub-block that pool plane ``p`` came from (only
    used with ``one_plane_per_sub_block``).
    """
    n, n_pool, m, _ = pool.shape
    anchor = pool[:, 0]

    def admissible(chosen: list[int]) -> np.ndarray:
        ok = np.ones(n_pool, dtype=bool)
        ok[chosen] = False
        if one_plane_per_sub_block:
            ok &= ~np.isin(groups, groups[chosen])
        return ok

    # -- 1. neighbour search: best, then sbest on the orthogonal axis ----------
    cand_idx = np.arange(1, n_pool)
    nb = neighbour_scores(anchor, pool[:, 1:])  # K, var, pa, pb
    nb_scan = _masked(nb, admissible([0])[1:])
    k1, f1, c1 = _first_min(nb_scan)
    v1, pa1, pb1 = np.unravel_index(f1, nb.shape[1:])
    # forbid the anchor's other edge on the same axis (the layer must be 2x2)
    nb_scan[:, :, slice(0, 2) if pa1 < 2 else slice(2, None)] = np.inf
    k2, f2, c2 = _first_min(_masked(nb_scan, admissible([0, int(cand_idx[k1])])[1:]))
    v2, pa2, pb2 = np.unravel_index(f2, nb.shape[1:])

    # -- 2. placement ------------------------------------------------------------
    row = 1 if TOP in (pa1, pa2) else 0
    col = 1 if LEFT in (pa1, pa2) else 0
    anchor_slot = 2 * row + col
    layer = np.zeros((n, 4, m, m), dtype=pool.dtype)
    layer[:, anchor_slot] = anchor
    slots = [-1, -1, -1, -1]  # pragma: no mutate (every entry is overwritten)
    slots[anchor_slot] = 0
    placed = []
    for k, v, pa, pb in ((k1, v1, pa1, pb1), (k2, v2, pa2, pb2)):
        dr, dc = _EDGE_OFFSET[int(pa)]
        slot = 2 * (row + dr) + (col + dc)
        plane = apply_variant(pool[:, 1 + k], int(v), flip_axis=2 if pb < 2 else 1)
        layer[:, slot] = dihedral(plane, int(EDGE_MOVE[pb, _FACING[int(pa)]]), axes=(1, 2))
        slots[slot] = int(cand_idx[k])
        placed.append(
            {
                "plane": int(cand_idx[k]),
                "slot": slot,
                "variant": int(v),
                "anchor_edge": int(pa),
                "edge": int(pb),
            }
        )

    # -- 3. fourth plane -------------------------------------------------------
    hole = 3 - anchor_slot
    ok4 = admissible([0, int(cand_idx[k1]), int(cand_idx[k2])])
    fs = fourth_scores(layer, hole, pool)  # P, var, rot
    if fix_fourth_flip:
        fs[:, 1::2, :] = np.inf
    k4, f4, c4 = _first_min(_masked(fs, ok4))
    v4, r4 = divmod(f4, N_DIHEDRAL)
    layer[:, hole] = apply_variant(dihedral(pool[:, k4], r4, axes=(1, 2)), v4, flip_axis=2)
    slots[hole] = k4

    cost = c1 + c2 + c4
    meta = {
        "anchor_slot": anchor_slot,
        "neighbours": placed,
        "fourth": {"plane": k4, "slot": hole, "variant": int(v4), "dihedral": int(r4)},
    }
    return LayerResult(layer, tuple(slots), cost, meta)


def _interleave(a: list[int]) -> list[int]:
    """Deal the 8 leftovers of pass R into G/B slots and list them in slot order."""
    return [q for pair in zip(a[:4], a[4:], strict=True) for q in pair]  # pragma: no mutate


@EJPS_SOLVERS.register("layer_greedy")
@dataclass(frozen=True)
class LayerGreedySolver:
    """The layer-greedy EJPS as an ``EjpsSolver``; see the module docstring for the flags."""

    fix_fourth_flip: bool = False
    one_plane_per_sub_block: bool = False

    def solve(self, blocks: np.ndarray, *, sub_block_size: int = 8) -> list[SubBlockCandidate]:
        subs = split_sub_blocks(np.asarray(blocks, dtype=np.uint8), sub_block_size)
        n, _, m, _, _ = subs.shape
        planes = subs.transpose(0, 1, 4, 2, 3).reshape(n, 12, m, m)  # index 3 * slot + ch
        sub_of = np.arange(12) // 3  # pragma: no mutate (only indices < 12 are read)

        order = list(range(12))
        layers, metas, cost = [], [], 0.0
        for name in ("R", "G", "B"):
            res = assemble_layer(
                planes[:, order],
                sub_of[order],
                fix_fourth_flip=self.fix_fourth_flip,
                one_plane_per_sub_block=self.one_plane_per_sub_block,
            )
            layers.append(merge_sub_blocks(res.layer[..., None])[..., 0])
            used = [order[p] for p in res.slots]
            meta = {"layer": name, "planes": used, **res.meta}
            for step in (*meta["neighbours"], meta["fourth"]):  # pool index -> raw plane
                step["plane"] = order[step["plane"]]
            metas.append(meta)
            cost += res.cost
            left = [q for q in order if q not in used]
            order = _interleave(left) if name == "R" else left

        r, g, b = layers
        out = []
        for gi in range(N_DIHEDRAL):
            g_rot = dihedral(g, gi, axes=(1, 2))
            for bi in range(N_DIHEDRAL):
                stacked = np.stack([r, g_rot, dihedral(b, bi, axes=(1, 2))], axis=-1)
                meta = {"g_dihedral": gi, "b_dihedral": bi, "layers": metas}
                out.append(SubBlockCandidate(np.ascontiguousarray(stacked), cost, meta))
        return out
