"""Tests for ejps/encrypt.py (the standalone EtC encryptor)."""

import hashlib

import numpy as np
import pytest

from ejps.encrypt import (
    OPS,
    KeyTable,
    check_ops,
    decrypt_sub_blocks,
    decrypt_with_key,
    encrypt_frames,
    encrypt_image,
    encrypt_sub_blocks,
    encrypt_with_key,
    generate_key,
    key_index,
)
from ejps.geometry import image_to_blocks, merge_sub_blocks, split_sub_blocks
from ejps.metrics import synthetic_natural_image

ALL = frozenset(OPS)


def _toy_subs():
    """Four 2x2x3 sub-blocks, value = 40 * sub + 10 * channel + (2 * row + col)."""
    k, c, i, j = np.meshgrid(np.arange(4), np.arange(3), np.arange(2), np.arange(2), indexing="ij")
    return (40 * k + 10 * c + 2 * i + j).transpose(0, 2, 3, 1).astype(np.uint8)


def _toy_table():
    return KeyTable(
        sub_perm=np.array([[1, 0, 3, 2]]),
        rot=np.array([[1, 0, 0, 0]]),
        flip=np.array([[False, True, False, False]]),
        nega=np.array([[False, False, True, False]]),
        chan=np.array([[[0, 1, 2], [0, 1, 2], [0, 1, 2], [2, 0, 1]]]),
    )


def test_sub_block_ops_hand_computed():
    subs = _toy_subs()
    enc = encrypt_sub_blocks(subs, _toy_table(), 0, ALL)
    # slot 0: sub-block 1 turned 90 degrees counter-clockwise
    np.testing.assert_array_equal(enc[0, ..., 0], [[41, 43], [40, 42]])
    # slot 1: sub-block 0 flipped left-right
    np.testing.assert_array_equal(enc[1, ..., 0], [[1, 0], [3, 2]])
    np.testing.assert_array_equal(enc[1, ..., 2], [[21, 20], [23, 22]])
    # slot 2: sub-block 3 negated
    np.testing.assert_array_equal(enc[2, ..., 0], [[135, 134], [133, 132]])
    # slot 3: sub-block 2 with channels (2, 0, 1)
    np.testing.assert_array_equal(enc[3, ..., 0], [[100, 101], [102, 103]])
    np.testing.assert_array_equal(enc[3, ..., 1], [[80, 81], [82, 83]])
    np.testing.assert_array_equal(enc[3, ..., 2], [[90, 91], [92, 93]])
    np.testing.assert_array_equal(decrypt_sub_blocks(enc, _toy_table(), 0, ALL), subs)


def test_ops_selection():
    subs = _toy_subs()
    table = _toy_table()
    np.testing.assert_array_equal(encrypt_sub_blocks(subs, table, 0, frozenset()), subs)
    only_shuffle = encrypt_sub_blocks(subs, table, 0, frozenset({"shuffle"}))
    np.testing.assert_array_equal(only_shuffle, subs[[1, 0, 3, 2]])
    only_nega = encrypt_sub_blocks(subs, table, 0, frozenset({"negate"}))
    np.testing.assert_array_equal(only_nega[2], 255 - subs[2])
    np.testing.assert_array_equal(only_nega[[0, 1, 3]], subs[[0, 1, 3]])
    no_channel = encrypt_sub_blocks(subs, table, 0, ALL - {"channel"})
    np.testing.assert_array_equal(no_channel[3], subs[2])
    for ops in (frozenset({"rotate"}), frozenset({"flip", "channel"}), ALL - {"shuffle"}):
        enc = encrypt_sub_blocks(subs, table, 0, ops)
        np.testing.assert_array_equal(decrypt_sub_blocks(enc, table, 0, ops), subs)
    assert check_ops(["flip"]) == frozenset({"flip"})
    with pytest.raises(ValueError, match="unknown ops"):
        check_ops(["rotate", "xor"])


def test_key_draw_is_deterministic():
    key = generate_key("common", 1, 4, 0, scramble=True)
    t = key.table
    assert len(t) == 1
    assert t.sub_perm.tolist() == [[1, 0, 3, 2]]
    assert t.rot.tolist() == [[3, 2, 0, 0]]
    assert t.flip.tolist() == [[True, False, True, True]]
    assert t.nega.tolist() == [[True, False, False, True]]
    assert t.chan.tolist() == [[[2, 1, 0], [2, 1, 0], [1, 0, 2], [1, 2, 0]]]
    assert key.sigma.tolist() == [2, 0, 3, 1]
    assert key.key_index.tolist() == [[0, 0, 0, 0]]
    # the scrambling does not change the block keys
    plain_key = generate_key("common", 1, 4, 0)
    assert plain_key.sigma.tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(plain_key.table.chan, t.chan)
    assert generate_key("common", 1, 4, 1).table.sub_perm.tolist() != [[1, 0, 3, 2]]


def test_key_groups():
    assert key_index("common", 2, 3).tolist() == [[0, 0, 0], [0, 0, 0]]
    assert key_index("independent", 2, 3).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert key_index("per_position", 2, 3).tolist() == [[0, 1, 2], [0, 1, 2]]
    with pytest.raises(ValueError, match="unknown key-group kind"):
        key_index("per_pixel", 1, 1)
    ind = generate_key("independent", 2, 3, 5)
    assert len(ind.table) == 6
    assert ind.table.sub_perm[:3].tolist() == [[2, 0, 3, 1], [2, 3, 0, 1], [3, 0, 2, 1]]
    pos = generate_key("per_position", 2, 3, 5)
    assert len(pos.table) == 3
    # independent keys are drawn one after another from the same stream
    np.testing.assert_array_equal(pos.table.sub_perm, ind.table.sub_perm[:3])


def test_key_group_semantics_on_identical_blocks():
    block = synthetic_natural_image(size=16, seed=3)
    img = np.tile(block, (2, 3, 1))  # six identical plaintext blocks
    enc, _ = encrypt_image(img, "common", 4)
    cb = image_to_blocks(enc, 16)
    assert all(np.array_equal(cb[0], b) for b in cb)
    enc, _ = encrypt_image(img, "independent", 4)
    cb = image_to_blocks(enc, 16)
    assert len({b.tobytes() for b in cb}) == 6
    frames = np.stack([img, img])
    enc, _ = encrypt_frames(frames, "per_position", 4)
    np.testing.assert_array_equal(enc[0], enc[1])
    assert len({b.tobytes() for b in image_to_blocks(enc[0], 16)}) == 6
    enc, _ = encrypt_frames(frames, "independent", 4)
    assert not np.array_equal(enc[0], enc[1])


def test_sigma_semantics_and_keys_follow_the_plaintext():
    plain = synthetic_natural_image(size=64, seed=1)
    pb = image_to_blocks(plain, 16)
    enc, sigma = encrypt_image(plain, "common", 2, scramble=True, ops=())
    np.testing.assert_array_equal(image_to_blocks(enc, 16), pb[sigma])
    key = generate_key("independent", 1, 16, 2, scramble=True)
    enc = encrypt_with_key(plain[None], key)[0]
    cb = image_to_blocks(enc, 16)
    subs = split_sub_blocks(pb, 8)
    for i in (0, 5, 15):
        src = int(key.sigma[i])
        want = merge_sub_blocks(encrypt_sub_blocks(subs[src], key.table, src, ALL)[None])[0]
        np.testing.assert_array_equal(cb[i], want)


@pytest.mark.parametrize("kind", ["common", "independent", "per_position"])
@pytest.mark.parametrize("scramble", [False, True])
def test_roundtrip(kind, scramble):
    frames = np.stack([synthetic_natural_image(size=48, seed=s) for s in range(3)])[:, :32]
    key = generate_key(kind, 3, 6, 9, scramble=scramble)
    enc = encrypt_with_key(frames, key)
    assert enc.shape == frames.shape and enc.dtype == np.uint8
    assert not np.array_equal(enc, frames)
    np.testing.assert_array_equal(decrypt_with_key(enc, key), frames)
    key8 = generate_key(kind, 3, 24, 9, scramble=scramble)  # 8-pixel blocks: 4 x 6 per frame
    enc8 = encrypt_with_key(frames, key8, block_size=8, sub_block_size=4)
    np.testing.assert_array_equal(
        decrypt_with_key(enc8, key8, block_size=8, sub_block_size=4), frames
    )
    np.testing.assert_array_equal(enc, encrypt_frames(frames, kind, 9, scramble=scramble)[0])


def test_fixed_seed_ciphertext():
    plain = synthetic_natural_image(size=64, seed=0)
    enc, sigma = encrypt_image(plain, "common", 0, scramble=True)
    assert hashlib.sha256(enc.tobytes()).hexdigest() == (
        "564d6c1034d4cfc71632c633b36484298b21a27e0a2b3521e7cb7c58cfc08e09"
    )
    assert sigma.tolist() == [7, 11, 6, 4, 3, 9, 8, 15, 14, 2, 12, 13, 10, 0, 1, 5]
    assert enc[0, 0].tolist() == [128, 127, 172]
    assert enc[10, 20].tolist() == [90, 70, 85]


def test_input_validation():
    img = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        encrypt_frames(img, "common")
    with pytest.raises(ValueError, match="uint8"):
        encrypt_frames(img[None].astype(np.float32), "common")
    with pytest.raises(ValueError, match="uint8"):
        encrypt_frames(img[None, ..., :2], "common")
    with pytest.raises(ValueError, match="multiple of block_size"):
        encrypt_frames(np.zeros((1, 40, 32, 3), np.uint8), "common")
    with pytest.raises(ValueError, match="sub_block_size"):
        encrypt_frames(img[None], "common", sub_block_size=4)
    with pytest.raises(ValueError, match="unknown ops"):
        encrypt_frames(img[None], "common", ops=("rot",))
    key = generate_key("common", 1, 9, 0)
    with pytest.raises(ValueError, match="key is for"):
        encrypt_with_key(img[None], key)
    with pytest.raises(ValueError, match="key is for"):
        decrypt_with_key(img[None], key)
