"""The ciphertext-only attack pipeline: EJPS sub-block restoration -> candidate
selection -> block jigsaw -> evaluation.  See README.md.

Pure array logic, no file IO: ``run_attack.py`` / :mod:`ejps.cli` is the thin launcher.

Conventions
-----------
* Images are ``(H, W, 3)`` uint8 with sides that are multiples of
  ``block_size``; videos are ``(T, H, W, 3)``. Blocks are row-major
  (``geometry.image_to_blocks``), frame-major for videos.
* Block scrambling is described by ``sigma``: ``cipher_block[i] =
  plain_block[sigma[i]]``. The jigsaw answer in ``JigsawResult.order`` form
  (``order[pos]`` = input piece placed at output position ``pos``) is then
  ``argsort(sigma)`` -- see :func:`true_order`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    EJPS_SOLVERS,
    N_DIHEDRAL,
    KeyGroups,
    blocks_to_image,
    blocks_to_video,
    dihedral,
    image_to_blocks,
    video_to_blocks,
)
from .jigsaw import solve_jigsaw
from .metrics import (
    apply_alignment,
    best_alignment,
    direct_comparison,
    evaluate_blocks,
    neighbor_comparison,
    psnr,
    ssim,
)
from .restore import rank_candidates  # importing restore registers the EJPS solvers


@dataclass(frozen=True)
class AttackConfig:
    """Every knob of one attack run.

    solver        : EJPS solver name in ``EJPS_SOLVERS`` (``edge_ref`` | ``layer_greedy``).
    solver_options: keyword arguments for the solver, e.g. ``layer_greedy``'s
                    ``one_plane_per_sub_block`` / ``fix_fourth_flip`` (default {} =
                    the original notebook behaviour).
    key_groups    : which blocks share a key (``common`` | ``independent`` | ``per_position``).
    jigsaw        : run the block-level jigsaw (only meaningful if blocks were scrambled).
    keep_top      : how many ranked candidates of the FIRST key group to keep for inspection.
    """

    solver: str = "edge_ref"
    solver_options: dict = field(default_factory=dict)
    key_groups: str = "common"
    block_size: int = 16
    sub_block_size: int = 8
    select_feature: str = "pixel"
    resolve_inversion: bool = True
    jigsaw: bool = False
    jigsaw_population: int = 30
    jigsaw_generations: int = 20
    seed: int = 0
    keep_top: int = 1

    @classmethod
    def from_dict(cls, options: dict) -> AttackConfig:
        """Build from a dict of field names (case-insensitive); unknown keys are an error."""
        kw = {k.lower(): v for k, v in options.items()}
        unknown = set(kw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown AttackConfig keys: {sorted(unknown)}")
        return cls(**kw)


@dataclass
class AttackResult:
    restored: np.ndarray  # after EJPS + selection, blocks still in cipher positions
    reconstruction: np.ndarray  # after the jigsaw (== restored when jigsaw is off)
    orders: list[np.ndarray] | None  # one JigsawResult.order per frame, or None
    group_scores: np.ndarray  # selection score of the chosen candidate, per key group
    top_candidates: list[np.ndarray]  # first group's top-k restorations (blocks)
    timings: dict = field(default_factory=dict)


def true_order(sigma: np.ndarray) -> np.ndarray:
    """Jigsaw ground truth (``JigsawResult.order`` form) for ``cipher[i] = plain[sigma[i]]``."""
    return np.argsort(np.asarray(sigma))


def restore_blocks(
    blocks: np.ndarray, groups: KeyGroups, cfg: AttackConfig
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """EJPS + automatic selection on every key group.

    Returns (restored blocks in input order, chosen score per group, top-k
    blocks of the first group).
    """
    solver = EJPS_SOLVERS.build(cfg.solver, **cfg.solver_options)
    scores, top = [], []

    def one_group(g: np.ndarray) -> np.ndarray:
        cands = solver.solve(g, sub_block_size=cfg.sub_block_size)
        ranked = rank_candidates(
            cands, feature=cfg.select_feature, resolve_inversion=cfg.resolve_inversion
        )
        if not top:
            top.extend(r.blocks for r in ranked[: cfg.keep_top])
        scores.append(ranked[0].score)
        return ranked[0].blocks

    out = np.empty_like(blocks)
    if blocks.shape[0] != groups.n_blocks:
        raise ValueError(f"{blocks.shape[0]} blocks but key groups cover {groups.n_blocks}")
    for g in groups:
        out[g] = one_group(blocks[g])
    return out, np.asarray(scores, dtype=np.float64), top


def _jigsaw_frame(img: np.ndarray, cfg: AttackConfig) -> tuple[np.ndarray, np.ndarray]:
    res = solve_jigsaw(
        img,
        cfg.block_size,
        population=cfg.jigsaw_population,
        generations=cfg.jigsaw_generations,
        seed=cfg.seed,
    )
    return res.image, res.order


def attack_frames(frames: np.ndarray, cfg: AttackConfig) -> AttackResult:
    """Attack a ``(T, H, W, 3)`` stack (``T = 1`` for a single image)."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.dtype != np.uint8:
        raise ValueError("frames must be a (T, H, W, 3) uint8 array")
    t, h, w, _ = frames.shape
    bs = cfg.block_size
    if h % bs or w % bs:
        raise ValueError(f"frame {h}x{w} is not a multiple of block_size={bs}")
    rows, cols = h // bs, w // bs
    blocks = video_to_blocks(frames, bs)
    groups = KeyGroups.build(cfg.key_groups, blocks.shape[0], n_frames=t)

    t0 = time.perf_counter()
    restored_blocks, scores, top = restore_blocks(blocks, groups, cfg)
    t1 = time.perf_counter()
    restored = blocks_to_video(restored_blocks, t, rows, cols)

    orders = None
    recon = restored
    if cfg.jigsaw:
        solved = [_jigsaw_frame(f, cfg) for f in restored]
        recon = np.stack([s[0] for s in solved])
        orders = [s[1] for s in solved]
    t2 = time.perf_counter()
    return AttackResult(
        restored=restored,
        reconstruction=recon,
        orders=orders,
        group_scores=scores,
        top_candidates=top,
        timings={"ejps_s": t1 - t0, "jigsaw_s": t2 - t1},
    )


def attack_image(img: np.ndarray, cfg: AttackConfig) -> AttackResult:
    """Attack one ``(H, W, 3)`` cipher image; result arrays keep the leading frame axis."""
    return attack_frames(np.asarray(img)[None], cfg)


def evaluate(
    result: AttackResult,
    plain: np.ndarray,
    cfg: AttackConfig,
    sigma: np.ndarray | None = None,
) -> dict:
    """Score an attack against the plaintext ``(T, H, W, 3)`` (or ``(H, W, 3)``).

    ``sigma`` (per-frame block scrambling, ``cipher[i] = plain[sigma[i]]``; one
    array for all frames or ``(T, n)``) is needed whenever blocks were scrambled.

    EJPS numbers compare the restored blocks with the plain blocks that sit at
    the same cipher positions, so they isolate stage 1 from the jigsaw:

    * ``ejps_global``  -- ONE residual alignment for the whole stack: what an
      attacker actually sees (independent keys leave every block with its own
      residual rotation/inversion, and that damage counts here).
    * ``ejps_oracle``  -- the best alignment per key group: an upper bound that
      forgives every residual ambiguity.

    Jigsaw numbers: mean direct / neighbour comparison over frames, each the
    best over the whole-image symmetries (:func:`symmetric_truths`).
    ``final``: the reconstruction vs the plaintext under one alignment per frame
    (:func:`final_image_metrics`; after a jigsaw only for square frames).
    """
    plain = np.asarray(plain)
    if plain.ndim == 3:
        plain = plain[None]
    t, h, w, _ = plain.shape
    bs = cfg.block_size
    rows, cols = h // bs, w // bs
    n = rows * cols
    if sigma is None:
        sig = np.tile(np.arange(n), (t, 1))
    else:
        sig = np.broadcast_to(np.asarray(sigma), (t, n))

    restored = video_to_blocks(result.restored, bs)
    plain_at_cipher = np.concatenate(
        [image_to_blocks(plain[f], bs)[sig[f]] for f in range(t)], axis=0
    )
    groups = KeyGroups.build(cfg.key_groups, restored.shape[0], n_frames=t)

    def summary(per_group: bool) -> dict:
        r = evaluate_blocks(
            restored,
            plain_at_cipher,
            t * rows,
            cols,
            groups=groups if per_group else None,
            sub_block_size=cfg.sub_block_size,
        )
        r.pop("alignments")
        return r

    out = {"ejps_global": summary(False), "ejps_oracle": summary(True)}
    if result.orders is not None:
        direct, neighbor = [], []
        for f in range(t):
            truths = symmetric_truths(true_order(sig[f]), rows, cols)
            direct.append(max(direct_comparison(result.orders[f], tr) for tr in truths))
            neighbor.append(
                max(neighbor_comparison(result.orders[f], tr, rows, cols) for tr in truths)
            )
        out["jigsaw_direct"] = float(np.mean(direct))
        out["jigsaw_neighbor"] = float(np.mean(neighbor))
    if result.orders is None:
        out["final"] = final_image_metrics(result.reconstruction, plain, bs)
    elif h == w:
        out["final"] = final_image_metrics(result.reconstruction, plain)
    out["timings"] = dict(result.timings)
    return out


def symmetric_truths(truth: np.ndarray, rows: int, cols: int) -> list[np.ndarray]:
    """The jigsaw answer under every whole-image dihedral symmetry of the grid.

    A common key leaves every restored block with the SAME residual dihedral
    transform; the jigsaw then (correctly) assembles the equally transformed
    WHOLE image, whose pieces sit at transformed positions. Output position
    ``q`` of ``dihedral(image, k)`` holds the plain block at
    ``dihedral(grid, k).flat[q]``, so the expected order is ``truth[that]``.
    Only symmetries that keep the ``rows x cols`` shape are listed.
    """
    grid = np.arange(rows * cols).reshape(rows, cols)
    out = []
    for k in range(N_DIHEDRAL):
        g = dihedral(grid, k, axes=(0, 1))
        if g.shape == grid.shape:
            out.append(np.asarray(truth)[g.ravel()])
    return out


def final_image_metrics(
    recon: np.ndarray, plain: np.ndarray, block_size: int | None = None
) -> dict:
    """PSNR / SSIM of the final reconstruction under ONE residual alignment per frame.

    ``block_size=None`` (after a jigsaw): the whole frame is aligned as one piece
    -- the jigsaw assembles the equally transformed WHOLE image. Square frames only
    (a 90-degree turn changes a non-square shape).
    ``block_size=B`` (no jigsaw): the residual transform sits on every block IN
    PLACE, so one alignment is applied to every ``B x B`` block of the frame.
    Either way it is a single alignment per frame: the honest "how close is what
    the attacker got" number.
    """
    recon, plain = np.asarray(recon), np.asarray(plain)
    ps, ss = [], []
    for r, p in zip(recon, plain, strict=True):
        if block_size is None:
            rb, pb = r[None], p[None]
        else:
            rb, pb = image_to_blocks(r, block_size), image_to_blocks(p, block_size)
        a = apply_alignment(rb, best_alignment(rb, pb))
        if block_size is not None:
            a = blocks_to_image(a, r.shape[0] // block_size, r.shape[1] // block_size)
        else:
            a = a[0]
        ps.append(psnr(a, p))
        ss.append(ssim(a, p))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}


def display_upright(
    img: np.ndarray,
    plain: np.ndarray,
    block_size: int | None = None,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Turn a reconstruction upright FOR DISPLAY; colours are left as attacked.

    The residual ambiguity includes a dihedral transform (4 rotations x flip),
    which makes figures awkward to read. This picks, from the 8, the one of the
    best residual alignment against the plaintext (an oracle: only for showing
    results, never part of the attack) and applies ONLY that geometric part.

    ``block_size=None``: the whole ``(H, W, 3)`` image is turned (after a jigsaw
    the reconstruction is the equally transformed whole image; square only).
    ``block_size=B``: every ``B x B`` block is turned in place by the same
    transform (EJPS output before the jigsaw); ``sigma`` gives the plain block
    at each cipher position (``cipher[i] = plain[sigma[i]]``). Independent keys
    leave every block its own residual, which one global turn cannot (and
    should not) undo.
    """
    img, plain = np.asarray(img), np.asarray(plain)
    if block_size is None:
        k = best_alignment(img[None], plain[None]).dihedral
        return np.ascontiguousarray(dihedral(img, k, axes=(0, 1)))
    rows, cols = img.shape[0] // block_size, img.shape[1] // block_size
    rb = image_to_blocks(img, block_size)
    pb = image_to_blocks(plain, block_size)
    if sigma is not None:
        pb = pb[np.asarray(sigma)]
    k = best_alignment(rb, pb).dihedral
    return blocks_to_image(np.ascontiguousarray(dihedral(rb, k, axes=(1, 2))), rows, cols)


def candidates_to_images(top: list[np.ndarray], rows: int, cols: int) -> list[np.ndarray]:
    """Top-k restorations of the first key group as images (single-image, common-key case)."""
    return [blocks_to_image(b, rows, cols) for b in top if b.shape[0] == rows * cols]
