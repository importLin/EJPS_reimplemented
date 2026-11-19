"""Geometry of EtC-style block ciphers, plus the types shared by the EJPS solvers.

Everything an EJPS (extended jigsaw puzzle solver) attack needs to move pixels
around: image <-> block stacks, the 2x2 sub-block split, the plane transform
groups, channel permutations, edge extraction, and *key groups* -- which
blocks share one sub-block key and therefore pool their statistics.

Layout conventions (used by every module in this package)
---------------------------------------------------------
* A **block stack** is ``(n, B, B, C)`` uint8, blocks in raster order of the
  image (row-major), channel last.
* A block of size ``B = 2m`` splits into four ``m x m`` **sub-blocks** in the
  order ``ul, ur, dl, dr`` (index 0..3 = row-major in the 2x2 grid).
* A **plane** is one channel of one sub-block, ``(m, m)``.
* **Edges** of a plane, index 0..3 = ``top, down, left, right``; the top / down
  rows read left -> right, the left / right columns read top -> bottom.

The two transform groups
------------------------
``plane_transform(p, idx)`` -- the 16-element group used by the ``edge_ref``
solver (the numbering of the original notebook implementation)::

    idx = nega * 8 + flip * 4 + rot      (rot in 0..3, flip / nega in {0, 1})
    p -> rotate clockwise by rot * 90 deg -> flipud if flip -> 255 - p if nega

``dihedral(x, k)`` -- the 8-element group used by the ``layer_greedy`` solver::

    k in 0..3 : np.rot90(x, k)            (counter-clockwise)
    k in 4..7 : np.flipud(np.rot90(x, k - 4))

``plane_transform_to_dihedral`` converts between the two numberings.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .registry import Registry

SUB_POSITIONS = ("ul", "ur", "dl", "dr")
EDGE_NAMES = ("top", "down", "left", "right")
N_PLANE_TRANSFORMS = 16
N_DIHEDRAL = 8
CHANNEL_PERMS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(3)))

# ---------------------------------------------------------------------------
# Shared solver types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubBlockCandidate:
    """One candidate reconstruction of a group of blocks that share a key.

    blocks: ``(n, B, B, 3)`` uint8, the reconstructed blocks.
    score:  the solver's own fitting cost (edge mismatch), lower is better.
    meta:   the solved parameters (sub-block positions, transforms, mapping).
    """

    blocks: np.ndarray
    score: float
    meta: dict = field(default_factory=dict)


class EjpsSolver(Protocol):
    def solve(self, blocks: np.ndarray, *, sub_block_size: int) -> list[SubBlockCandidate]: ...


EJPS_SOLVERS: Registry = Registry("ejps_solver")

# ---------------------------------------------------------------------------
# Images, videos and block stacks
# ---------------------------------------------------------------------------


def image_to_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    """``(H, W, C)`` -> ``(rows * cols, B, B, C)`` in raster order.

    H and W must be multiples of ``block_size`` (crop first, see ``crop_to_blocks``).
    """
    h, w, c = img.shape
    if h % block_size or w % block_size:
        raise ValueError(f"image {h}x{w} is not a multiple of block size {block_size}")
    rows, cols = h // block_size, w // block_size
    v = img.reshape(rows, block_size, cols, block_size, c).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(v.reshape(rows * cols, block_size, block_size, c))


def blocks_to_image(blocks: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Inverse of ``image_to_blocks``."""
    n, b, b2, c = blocks.shape
    if n != rows * cols or b != b2:
        raise ValueError(f"{n} blocks of {b}x{b2} cannot fill a {rows}x{cols} grid")
    v = blocks.reshape(rows, cols, b, b, c).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(v.reshape(rows * b, cols * b, c))


def crop_to_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    """Crop the bottom / right margin so that both sides are multiples of B."""
    h = img.shape[0] // block_size * block_size
    w = img.shape[1] // block_size * block_size
    return img[:h, :w]


def video_to_blocks(frames: np.ndarray, block_size: int) -> np.ndarray:
    """``(T, H, W, C)`` -> ``(T * rows * cols, B, B, C)``, frame-major."""
    return np.concatenate([image_to_blocks(f, block_size) for f in frames], axis=0)


def blocks_to_video(blocks: np.ndarray, n_frames: int, rows: int, cols: int) -> np.ndarray:
    """Inverse of ``video_to_blocks``."""
    per = rows * cols
    if blocks.shape[0] != n_frames * per:
        raise ValueError(f"{blocks.shape[0]} blocks != {n_frames} frames x {per}")
    return np.stack(
        [blocks_to_image(blocks[t * per : (t + 1) * per], rows, cols) for t in range(n_frames)]
    )


# ---------------------------------------------------------------------------
# 2x2 sub-blocks
# ---------------------------------------------------------------------------


def check_sub_block_size(block_size: int, sub_block_size: int) -> None:
    """The EJPS solvers assume exactly a 2x2 sub-block grid."""
    if sub_block_size <= 0 or 2 * sub_block_size != block_size:
        raise ValueError(
            f"EJPS needs a 2x2 sub-block grid: sub_block_size ({sub_block_size}) must be "
            f"block_size // 2 (block_size = {block_size})"
        )


def split_sub_blocks(blocks: np.ndarray, sub_block_size: int) -> np.ndarray:
    """``(n, 2m, 2m, C)`` -> ``(n, 4, m, m, C)``, sub-blocks in ``ul, ur, dl, dr`` order."""
    n, b, _, c = blocks.shape
    m = sub_block_size
    check_sub_block_size(b, m)
    v = blocks.reshape(n, 2, m, 2, m, c).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(v.reshape(n, 4, m, m, c))


def merge_sub_blocks(subs: np.ndarray) -> np.ndarray:
    """Inverse of ``split_sub_blocks``: ``(n, 4, m, m, C)`` -> ``(n, 2m, 2m, C)``."""
    n, four, m, _, c = subs.shape
    if four != 4:
        raise ValueError(f"expected 4 sub-blocks, got {four}")
    v = subs.reshape(n, 2, 2, m, m, c).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(v.reshape(n, 2 * m, 2 * m, c))


# ---------------------------------------------------------------------------
# Transform groups
# ---------------------------------------------------------------------------


def plane_transform(planes: np.ndarray, idx: int, axes: tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Apply transform ``idx`` (0..15, ``edge_ref`` numbering) to the square ``axes``.

    ``nega`` uses ``255 - x``, so feed uint8 (or integer-valued) arrays.
    """
    if not 0 <= idx < N_PLANE_TRANSFORMS:
        raise ValueError(f"plane transform index {idx} not in 0..15")
    out = np.rot90(planes, -(idx % 4), axes=axes)
    if (idx // 4) % 2:
        out = np.flip(out, axis=axes[0])
    if idx // 8:
        out = 255 - out
    return out


def dihedral(x: np.ndarray, k: int, axes: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Apply dihedral element ``k`` (0..7, ``layer_greedy`` numbering) to the square ``axes``."""
    if not 0 <= k < N_DIHEDRAL:
        raise ValueError(f"dihedral index {k} not in 0..7")
    out = np.rot90(x, k % 4, axes=axes)
    if k >= 4:
        out = np.flip(out, axis=axes[0])
    return out


def dihedral_inverse(k: int) -> int:
    """The index ``j`` with ``dihedral(dihedral(x, k), j) == x``."""
    # rotations invert to the opposite rotation; flips are involutions.
    return (4 - k) % 4 if k < 4 else k


def plane_transform_to_dihedral(idx: int) -> tuple[int, bool]:
    """``plane_transform`` index -> ``(dihedral k, nega)`` describing the same map."""
    rot, flip, nega = idx % 4, (idx // 4) % 2, idx // 8
    return (4 - rot) % 4 + 4 * flip, bool(nega)


def all_plane_transforms(planes: np.ndarray) -> np.ndarray:
    """``(..., m, m)`` uint8 -> ``(..., 16, m, m)``: every transform of every plane."""
    return np.stack([plane_transform(planes, i) for i in range(N_PLANE_TRANSFORMS)], axis=-3)


def all_dihedral(x: np.ndarray, axes: tuple[int, int] = (-2, -1)) -> np.ndarray:
    """``(..., m, m)`` -> ``(..., 8, m, m)`` (new axis right before the square axes)."""
    return np.stack([dihedral(x, k, axes=axes) for k in range(N_DIHEDRAL)], axis=-3)


def permute_channels(x: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Output channel ``c`` = input channel ``perm[c]``."""
    return x[..., list(perm)]


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def plane_edges(planes: np.ndarray) -> np.ndarray:
    """``(..., m, m)`` -> ``(..., 4, m)`` float64: top, down, left, right."""
    p = np.asarray(planes, dtype=np.float64)
    return np.stack([p[..., 0, :], p[..., -1, :], p[..., :, 0], p[..., :, -1]], axis=-2)


def sub_block_edges(subs: np.ndarray) -> np.ndarray:
    """``(..., m, m, C)`` -> ``(..., 4, m, C)`` float64: the edges of every channel."""
    return np.moveaxis(plane_edges(np.moveaxis(subs, -1, -3)), -3, -1)


# ---------------------------------------------------------------------------
# Key groups
# ---------------------------------------------------------------------------

KEY_GROUP_KINDS = ("common", "independent", "per_position")


@dataclass(frozen=True)
class KeyGroups:
    """A partition of a block stack into groups that share one sub-block key.

    ``common``       -- one key for all blocks (Com-B/S): one group.
    ``independent``  -- one key per block (Ind-B/S): ``n`` singleton groups.
    ``per_position`` -- frame stack, one key per block *position* reused by every
                        frame: group ``p`` = position ``p`` of every frame, for a
                        frame-major stack (``video_to_blocks``).
    """

    n_blocks: int
    groups: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        seen = np.concatenate([np.asarray(g, dtype=np.int64) for g in self.groups])
        if not np.array_equal(np.sort(seen), np.arange(self.n_blocks)):
            raise ValueError("key groups must partition range(n_blocks) exactly")

    @classmethod
    def common(cls, n_blocks: int) -> KeyGroups:
        return cls(n_blocks, (np.arange(n_blocks),))

    @classmethod
    def independent(cls, n_blocks: int) -> KeyGroups:
        return cls(n_blocks, tuple(np.array([i]) for i in range(n_blocks)))

    @classmethod
    def per_position(cls, n_frames: int, blocks_per_frame: int) -> KeyGroups:
        idx = np.arange(n_frames * blocks_per_frame).reshape(n_frames, blocks_per_frame)
        return cls(n_frames * blocks_per_frame, tuple(idx[:, p] for p in range(blocks_per_frame)))

    @classmethod
    def build(cls, kind: str, n_blocks: int, n_frames: int = 1) -> KeyGroups:
        """Factory by name (``KEY_GROUP_KINDS``); ``n_blocks`` counts all frames."""
        if kind == "common":
            return cls.common(n_blocks)
        if kind == "independent":
            return cls.independent(n_blocks)
        if kind == "per_position":
            if n_frames <= 0 or n_blocks % n_frames:
                raise ValueError(f"{n_blocks} blocks do not split into {n_frames} frames")
            return cls.per_position(n_frames, n_blocks // n_frames)
        raise ValueError(f"unknown key-group kind {kind!r}; choose from {KEY_GROUP_KINDS}")

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.groups)


def map_groups(
    blocks: np.ndarray, groups: KeyGroups, fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Run ``fn`` on the blocks of each key group and write the results back in place order.

    ``fn`` receives ``(k, B, B, C)`` and must return the same shape, e.g.
    ``lambda g: select(solver.solve(g, sub_block_size=8))[0].blocks``.
    """
    if blocks.shape[0] != groups.n_blocks:
        raise ValueError(f"{blocks.shape[0]} blocks but key groups cover {groups.n_blocks}")
    out = np.empty_like(blocks)
    for g in groups:
        res = np.asarray(fn(blocks[g]))
        if res.shape != blocks[g].shape:
            raise ValueError(f"group fn returned {res.shape}, expected {blocks[g].shape}")
        out[g] = res
    return out
