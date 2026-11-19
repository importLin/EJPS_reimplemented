"""A standalone block-wise EtC (encryption-then-compression) encryptor with ground truth.

Written from the published description of block scrambling-based EtC images
(block scrambling, block rotation / inversion, negative-positive transformation,
colour component shuffling), extended with the 2x2 sub-block layer that the
EJPS attack targets. It exists to produce ciphertexts with known ground truth
for ``run_attack.py --simulate`` and the tests; it is not meant as a secure
cipher.

Per-block transform (block ``B x B``, sub-blocks ``S x S`` with ``B = 2S``)
--------------------------------------------------------------------------
The block is split into four sub-blocks in the order ``ul, ur, dl, dr``. With
the block key ``k`` (all parameters are drawn from the key, see
:class:`KeyTable`), cipher slot ``j`` (0..3) is computed as::

    x = plain_sub_block[k.sub_perm[j]]          # 1. sub-block shuffle
    x = np.rot90(x, k.rot[j])                   # 2. rotation by 0/90/180/270 deg (CCW)
    x = x[:, ::-1] if k.flip[j] else x          # 3. horizontal flip
    x = 255 - x if k.nega[j] else x             # 4. negative-positive transform
    x = x[..., k.chan[j]]                       # 5. colour channel shuffle
    cipher_sub_block[j] = x

Every op acts on a whole sub-block (all three channels together), so the three
channels of a cipher sub-block always come from one plaintext sub-block and
share one geometric transform; this is the structure both EJPS solvers rely on.
``ops`` selects which of the five ops are applied (the key parameters are drawn
regardless, so disabling one op does not change the others).

Block scrambling
----------------
Optionally the encrypted blocks are permuted: ``cipher_block[i] =
enc(plain_block[sigma[i]])``. Block keys follow the *plaintext* position, so an
independent key stays attached to its block wherever the block moves. One
``sigma`` is used for every frame of a stack.

Key groups (which blocks share one block key)
---------------------------------------------
``common``       one key for every block of every frame (Com-B/S).
``independent``  one key per block of the stack (Ind-B/S).
``per_position`` frame stacks: one key per block position, reused by every frame.

These match :class:`ejps.geometry.KeyGroups` of the same name, which tells the
attack which blocks pool their statistics.

Determinism: everything is drawn from ``np.random.SeedSequence(seed)``: its first
spawned child drives the block keys, the second one the block scrambling.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from .geometry import (
    KEY_GROUP_KINDS,
    blocks_to_image,
    check_sub_block_size,
    image_to_blocks,
    merge_sub_blocks,
    split_sub_blocks,
)

OPS = ("shuffle", "rotate", "flip", "negate", "channel")


@dataclass(frozen=True)
class KeyTable:
    """The parameters of ``k`` block keys (row ``r`` = key ``r``).

    sub_perm: ``(k, 4)``    cipher slot ``j`` holds plaintext sub-block ``sub_perm[r, j]``.
    rot:      ``(k, 4)``    quarter turns (``np.rot90``, counter-clockwise) per cipher slot.
    flip:     ``(k, 4)``    bool, horizontal flip per cipher slot.
    nega:     ``(k, 4)``    bool, ``255 - x`` per cipher slot.
    chan:     ``(k, 4, 3)`` cipher channel ``c`` of slot ``j`` = channel ``chan[r, j, c]``.
    """

    sub_perm: np.ndarray
    rot: np.ndarray
    flip: np.ndarray
    nega: np.ndarray
    chan: np.ndarray

    def __len__(self) -> int:
        return len(self.sub_perm)

    @classmethod
    def draw(cls, n_keys: int, rng: np.random.Generator) -> KeyTable:
        """Draw ``n_keys`` independent block keys, one key after another."""
        sub_perm = np.empty((n_keys, 4), dtype=np.int64)
        rot = np.empty((n_keys, 4), dtype=np.int64)
        flip = np.empty((n_keys, 4), dtype=bool)
        nega = np.empty((n_keys, 4), dtype=bool)
        chan = np.empty((n_keys, 4, 3), dtype=np.int64)
        for r in range(n_keys):
            sub_perm[r] = rng.permutation(4)
            rot[r] = rng.integers(0, 4, 4)
            flip[r] = rng.integers(0, 2, 4).astype(bool)
            nega[r] = rng.integers(0, 2, 4).astype(bool)
            chan[r] = [rng.permutation(3) for _ in range(4)]
        return cls(sub_perm, rot, flip, nega, chan)


@dataclass(frozen=True)
class EtcKey:
    """A complete key for a stack of ``n_frames`` frames of ``blocks_per_frame`` blocks.

    table:     the distinct block keys.
    key_index: ``(n_frames, blocks_per_frame)``, the row of ``table`` used by the
               block at each *plaintext* position.
    sigma:     ``(blocks_per_frame,)`` block scrambling, ``cipher[i] = plain[sigma[i]]``.
    """

    table: KeyTable
    key_index: np.ndarray
    sigma: np.ndarray


def check_ops(ops: Collection[str]) -> frozenset[str]:
    unknown = set(ops) - set(OPS)
    if unknown:
        raise ValueError(f"unknown ops {sorted(unknown)}; choose from {OPS}")
    return frozenset(ops)


def key_index(kind: str, n_frames: int, blocks_per_frame: int) -> np.ndarray:
    """``(n_frames, blocks_per_frame)`` key rows for the key-group ``kind``."""
    if kind == "common":
        return np.zeros((n_frames, blocks_per_frame), dtype=np.int64)
    if kind == "independent":
        return np.arange(n_frames * blocks_per_frame, dtype=np.int64).reshape(
            n_frames, blocks_per_frame
        )
    if kind == "per_position":
        return np.tile(np.arange(blocks_per_frame, dtype=np.int64), (n_frames, 1))
    raise ValueError(f"unknown key-group kind {kind!r}; choose from {KEY_GROUP_KINDS}")


def generate_key(
    kind: str,
    n_frames: int,
    blocks_per_frame: int,
    seed: int = 0,
    *,
    scramble: bool = False,
) -> EtcKey:
    """Draw the key of the key-group ``kind`` (deterministic in ``seed``)."""
    index = key_index(kind, n_frames, blocks_per_frame)
    key_seq, perm_seq = np.random.SeedSequence(seed).spawn(2)
    table = KeyTable.draw(int(index.max()) + 1, np.random.default_rng(key_seq))
    if scramble:
        sigma = np.random.default_rng(perm_seq).permutation(blocks_per_frame)
    else:
        sigma = np.arange(blocks_per_frame)
    return EtcKey(table, index, sigma)


def encrypt_sub_blocks(
    subs: np.ndarray, table: KeyTable, row: int, ops: frozenset[str]
) -> np.ndarray:
    """Encrypt the four sub-blocks ``(4, S, S, C)`` of one block with key ``row``."""
    out = np.empty_like(subs)
    for j in range(4):
        x = subs[table.sub_perm[row, j] if "shuffle" in ops else j]
        if "rotate" in ops:
            x = np.rot90(x, int(table.rot[row, j]))
        if "flip" in ops and table.flip[row, j]:
            x = x[:, ::-1]
        if "negate" in ops and table.nega[row, j]:
            x = 255 - x
        if "channel" in ops:
            x = x[..., table.chan[row, j]]
        out[j] = x
    return out


def decrypt_sub_blocks(
    subs: np.ndarray, table: KeyTable, row: int, ops: frozenset[str]
) -> np.ndarray:
    """Inverse of :func:`encrypt_sub_blocks`."""
    out = np.empty_like(subs)
    for j in range(4):
        x = subs[j]
        if "channel" in ops:
            x = x[..., np.argsort(table.chan[row, j])]
        if "negate" in ops and table.nega[row, j]:
            x = 255 - x
        if "flip" in ops and table.flip[row, j]:
            x = x[:, ::-1]
        if "rotate" in ops:
            x = np.rot90(x, -int(table.rot[row, j]))
        out[table.sub_perm[row, j] if "shuffle" in ops else j] = x
    return out


def _check_frames(frames: np.ndarray, block_size: int, sub_block_size: int) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
        raise ValueError("frames must be a (T, H, W, 3) uint8 array")
    check_sub_block_size(block_size, sub_block_size)
    h, w = frames.shape[1:3]
    if h % block_size or w % block_size:
        raise ValueError(f"frame {h}x{w} is not a multiple of block_size={block_size}")
    return frames


def _check_key(key: EtcKey, t: int, n: int) -> None:
    if key.key_index.shape != (t, n) or key.sigma.shape != (n,):
        raise ValueError(f"key is for {key.key_index.shape} blocks, frames have {(t, n)}")


def encrypt_with_key(
    frames: np.ndarray,
    key: EtcKey,
    *,
    block_size: int = 16,
    sub_block_size: int = 8,
    ops: Collection[str] = OPS,
) -> np.ndarray:
    """Encrypt ``(T, H, W, 3)`` uint8 frames with an explicit :class:`EtcKey`."""
    frames = _check_frames(frames, block_size, sub_block_size)
    ops = check_ops(ops)
    t, h, w, _ = frames.shape
    rows, cols = h // block_size, w // block_size
    _check_key(key, t, rows * cols)
    out = np.empty_like(frames)
    for f in range(t):
        subs = split_sub_blocks(image_to_blocks(frames[f], block_size), sub_block_size)
        enc = np.stack(
            [
                encrypt_sub_blocks(s, key.table, int(key.key_index[f, i]), ops)
                for i, s in enumerate(subs)
            ]
        )
        out[f] = blocks_to_image(merge_sub_blocks(enc)[key.sigma], rows, cols)
    return out


def decrypt_with_key(
    frames: np.ndarray,
    key: EtcKey,
    *,
    block_size: int = 16,
    sub_block_size: int = 8,
    ops: Collection[str] = OPS,
) -> np.ndarray:
    """Inverse of :func:`encrypt_with_key`."""
    frames = _check_frames(frames, block_size, sub_block_size)
    ops = check_ops(ops)
    t, h, w, _ = frames.shape
    rows, cols = h // block_size, w // block_size
    _check_key(key, t, rows * cols)
    out = np.empty_like(frames)
    for f in range(t):
        blocks = np.empty((rows * cols, block_size, block_size, 3), dtype=frames.dtype)
        blocks[key.sigma] = image_to_blocks(frames[f], block_size)  # undo the scrambling
        subs = split_sub_blocks(blocks, sub_block_size)
        dec = np.stack(
            [
                decrypt_sub_blocks(s, key.table, int(key.key_index[f, i]), ops)
                for i, s in enumerate(subs)
            ]
        )
        out[f] = blocks_to_image(merge_sub_blocks(dec), rows, cols)
    return out


def encrypt_frames(
    frames: np.ndarray,
    kind: str = "common",
    seed: int = 0,
    *,
    block_size: int = 16,
    sub_block_size: int = 8,
    scramble: bool = False,
    ops: Collection[str] = OPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Encrypt ``(T, H, W, 3)`` uint8 frames; returns ``(cipher_frames, sigma)``.

    The key is :func:`generate_key` ``(kind, T, blocks_per_frame, seed, scramble=...)``.
    """
    frames = _check_frames(frames, block_size, sub_block_size)
    t, h, w, _ = frames.shape
    key = generate_key(kind, t, (h // block_size) * (w // block_size), seed, scramble=scramble)
    enc = encrypt_with_key(
        frames, key, block_size=block_size, sub_block_size=sub_block_size, ops=ops
    )
    return enc, key.sigma


def encrypt_image(
    img: np.ndarray, kind: str = "common", seed: int = 0, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`encrypt_frames` for one ``(H, W, 3)`` image; returns ``(cipher, sigma)``."""
    enc, sigma = encrypt_frames(np.asarray(img)[None], kind, seed, **kwargs)
    return enc[0], sigma
