"""Tests for ejps/geometry.py."""

import dataclasses

import numpy as np
import pytest

from ejps import geometry as G

P = np.array([[1, 2], [3, 4]], dtype=np.uint8)


def test_constants():
    assert G.SUB_POSITIONS == ("ul", "ur", "dl", "dr")
    assert G.EDGE_NAMES == ("top", "down", "left", "right")
    assert G.N_PLANE_TRANSFORMS == 16 and G.N_DIHEDRAL == 8
    assert G.CHANNEL_PERMS[0] == (0, 1, 2) and len(set(G.CHANNEL_PERMS)) == 6
    assert G.KEY_GROUP_KINDS == ("common", "independent", "per_position")


def test_image_blocks_roundtrip_and_order():
    img = np.arange(32 * 48 * 3, dtype=np.int64).reshape(32, 48, 3)
    blocks = G.image_to_blocks(img, 16)
    assert blocks.shape == (6, 16, 16, 3)
    np.testing.assert_array_equal(blocks[1], img[0:16, 16:32])
    np.testing.assert_array_equal(blocks[3], img[16:32, 0:16])
    np.testing.assert_array_equal(G.blocks_to_image(blocks, 2, 3), img)
    with pytest.raises(ValueError, match="image 30x48 is not a multiple"):
        G.image_to_blocks(img[:30], 16)
    with pytest.raises(ValueError, match="image 32x40 is not a multiple"):
        G.image_to_blocks(img[:, :40], 16)
    with pytest.raises(ValueError, match="6 blocks of 16x16 cannot fill a 3x3 grid"):
        G.blocks_to_image(blocks, 3, 3)
    with pytest.raises(ValueError, match="cannot fill"):
        G.blocks_to_image(blocks[:, :, :8], 2, 3)
    assert G.crop_to_blocks(np.zeros((40, 50, 3)), 16).shape == (32, 48, 3)


def test_video_blocks_are_frame_major():
    frames = np.arange(2 * 32 * 16 * 1).reshape(2, 32, 16, 1)
    blocks = G.video_to_blocks(frames, 16)
    assert blocks.shape == (4, 16, 16, 1)
    np.testing.assert_array_equal(blocks[2], frames[1, :16])
    np.testing.assert_array_equal(G.blocks_to_video(blocks, 2, 2, 1), frames)
    with pytest.raises(ValueError, match="4 blocks != 3 frames x 2"):
        G.blocks_to_video(blocks, 3, 2, 1)


def test_split_merge_sub_blocks():
    blocks = np.arange(2 * 16 * 16 * 3).reshape(2, 16, 16, 3)
    subs = G.split_sub_blocks(blocks, 8)
    assert subs.shape == (2, 4, 8, 8, 3)
    np.testing.assert_array_equal(subs[1, 0], blocks[1, :8, :8])
    np.testing.assert_array_equal(subs[1, 1], blocks[1, :8, 8:])
    np.testing.assert_array_equal(subs[1, 2], blocks[1, 8:, :8])
    np.testing.assert_array_equal(subs[0, 3], blocks[0, 8:, 8:])
    np.testing.assert_array_equal(G.merge_sub_blocks(subs), blocks)
    with pytest.raises(ValueError):
        G.split_sub_blocks(blocks, 4)
    with pytest.raises(ValueError):
        G.merge_sub_blocks(subs[:, :3])
    for b, m in ((16, 0), (0, 0), (16, -8)):
        with pytest.raises(ValueError):
            G.check_sub_block_size(b, m)
    G.check_sub_block_size(16, 8)
    G.check_sub_block_size(2, 1)


def test_plane_transform_known_values():
    np.testing.assert_array_equal(G.plane_transform(P, 0), P)
    np.testing.assert_array_equal(G.plane_transform(P, 1), [[3, 1], [4, 2]])  # clockwise
    np.testing.assert_array_equal(G.plane_transform(P, 2), [[4, 3], [2, 1]])
    np.testing.assert_array_equal(G.plane_transform(P, 4), [[3, 4], [1, 2]])  # flipud
    np.testing.assert_array_equal(G.plane_transform(P, 5), [[4, 2], [3, 1]])
    np.testing.assert_array_equal(G.plane_transform(P, 8), 255 - P)
    np.testing.assert_array_equal(G.plane_transform(P, 13), 255 - np.array([[4, 2], [3, 1]]))
    for bad in (-1, 16):
        with pytest.raises(ValueError):
            G.plane_transform(P, bad)


def test_plane_transform_matches_golden(golden_arrays):
    # golden: the original implementation's 16 transforms of this plane
    want = golden_arrays["plane_transforms"]
    p = np.random.default_rng(0).integers(0, 256, (8, 8)).astype(np.uint8)
    for i in range(16):
        np.testing.assert_array_equal(G.plane_transform(p, i), want[i])
    batch = np.stack([p.T, p])
    allt = G.all_plane_transforms(batch)
    assert allt.shape == (2, 16, 8, 8)
    np.testing.assert_array_equal(allt[1], want)
    np.testing.assert_array_equal(allt[0, 6], G.plane_transform(p.T, 6))
    # the transform acts on the requested axes only
    hwc = np.stack([p, 255 - p], axis=-1)
    np.testing.assert_array_equal(
        G.plane_transform(hwc, 7, axes=(0, 1))[..., 1], 255 - G.plane_transform(p, 7)
    )


def test_dihedral_and_conversions():
    np.testing.assert_array_equal(G.dihedral(P, 0), P)
    np.testing.assert_array_equal(G.dihedral(P, 1), np.rot90(P))
    np.testing.assert_array_equal(G.dihedral(P, 3), np.rot90(P, 3))
    np.testing.assert_array_equal(G.dihedral(P, 4), np.flipud(P))
    np.testing.assert_array_equal(G.dihedral(P, 5), np.flipud(np.rot90(P)))
    np.testing.assert_array_equal(G.dihedral(P, 7), np.flipud(np.rot90(P, 3)))
    for bad in (-1, 8):
        with pytest.raises(ValueError):
            G.dihedral(P, bad)
    q = np.arange(9).reshape(3, 3)
    assert [G.dihedral_inverse(k) for k in range(8)] == [0, 3, 2, 1, 4, 5, 6, 7]
    for k in range(8):
        np.testing.assert_array_equal(G.dihedral(G.dihedral(q, k), G.dihedral_inverse(k)), q)
    assert len({G.dihedral(q, k).tobytes() for k in range(8)}) == 8
    assert G.plane_transform_to_dihedral(1) == (3, False)
    assert G.plane_transform_to_dihedral(12) == (4, True)
    for i in range(16):
        k, nega = G.plane_transform_to_dihedral(i)
        want = G.dihedral(q, k)
        np.testing.assert_array_equal(G.plane_transform(q, i), 255 - want if nega else want)
    alld = G.all_dihedral(np.stack([q, 2 * q]))
    assert alld.shape == (2, 8, 3, 3)
    np.testing.assert_array_equal(alld[1, 6], G.dihedral(2 * q, 6))


def test_permute_channels_and_edges():
    x = np.arange(12).reshape(2, 2, 3)
    np.testing.assert_array_equal(G.permute_channels(x, (2, 0, 1))[0, 0], [2, 0, 1])
    q = np.arange(9).reshape(3, 3)
    e = G.plane_edges(q)
    assert e.dtype == np.float64
    np.testing.assert_array_equal(e, [[0, 1, 2], [6, 7, 8], [0, 3, 6], [2, 5, 8]])
    sub = np.stack([q, 10 * q], axis=-1)
    se = G.sub_block_edges(sub)
    assert se.shape == (4, 3, 2)
    np.testing.assert_array_equal(se[..., 1], 10 * e)
    np.testing.assert_array_equal(se[..., 0], e)


def test_key_groups():
    c = G.KeyGroups.common(5)
    assert len(c) == 1 and c.n_blocks == 5
    np.testing.assert_array_equal(c.groups[0], np.arange(5))
    ind = G.KeyGroups.independent(3)
    assert [g.tolist() for g in ind] == [[0], [1], [2]]
    fs = G.KeyGroups.per_position(3, 2)
    assert [g.tolist() for g in fs] == [[0, 2, 4], [1, 3, 5]]
    assert [g.tolist() for g in G.KeyGroups.build("per_position", 6, n_frames=2)] == [
        [0, 3],
        [1, 4],
        [2, 5],
    ]
    assert [g.tolist() for g in G.KeyGroups.build("per_position", 3)] == [[0], [1], [2]]
    assert len(G.KeyGroups.build("per_position", 3, n_frames=1)) == 3
    assert len(G.KeyGroups.build("independent", 4)) == 4
    assert len(G.KeyGroups.build("common", 4)) == 1
    for kind, n, t in (("bogus", 4, 1), ("per_position", 5, 2), ("per_position", 4, 0)):
        with pytest.raises(ValueError):
            G.KeyGroups.build(kind, n, n_frames=t)
    with pytest.raises(ValueError):
        G.KeyGroups(3, (np.array([0, 1]),))
    with pytest.raises(ValueError):
        G.KeyGroups(2, (np.array([0, 1]), np.array([1])))


def test_map_groups():
    blocks = np.zeros((6, 2, 2, 1), dtype=np.int64)
    groups = G.KeyGroups.per_position(3, 2)
    seen = []

    def fn(g):
        seen.append(len(g))
        return g + len(seen)

    out = G.map_groups(blocks, groups, fn)
    assert seen == [3, 3]
    assert out[:, 0, 0, 0].tolist() == [1, 2, 1, 2, 1, 2]
    with pytest.raises(ValueError, match="5 blocks but key groups cover 6"):
        G.map_groups(blocks[:5], groups, fn)
    with pytest.raises(ValueError):
        G.map_groups(blocks, groups, lambda g: g[:1])


def test_candidate_defaults():
    c = G.SubBlockCandidate(np.zeros(1), 1.5)
    assert c.meta == {} and c.score == 1.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.score = 2.0
    groups = G.KeyGroups.common(2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        groups.n_blocks = 3
    assert G.EJPS_SOLVERS.kind == "ejps_solver"
