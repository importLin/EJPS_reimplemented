"""Tests for ejps/restore/layer_greedy.py.

The golden hashes in ``tests/fixtures`` are sha256 digests of the 64 candidate
block stacks (candidate ``8 * g + b``) produced by the original implementation,
which was itself verified bit-identical to the author's original notebook.
"""

import dataclasses

import make_fixtures as MF
import numpy as np
import pytest

from ejps.geometry import EJPS_SOLVERS, dihedral
from ejps.metrics import is_equivalent
from ejps.restore import layer_greedy as L


def _plain_layers(layers):
    """Meta as stored in JSON (tuples -> lists)."""
    return MF.normalise(layers)


@pytest.mark.parametrize("name", MF.LAYER_CASES)
@pytest.mark.parametrize("flag", sorted(MF.LAYER_FLAGS))
def test_matches_golden(name, flag, golden):
    plain, blocks = MF.layer_input(name)
    want = golden["layer"][name]
    assert MF.sha(blocks) == want["input_sha"]
    cands = L.LayerGreedySolver(**MF.LAYER_FLAGS[flag]).solve(blocks, sub_block_size=8)
    assert len(cands) == 64
    assert MF.sha(np.stack([c.blocks for c in cands])) == want[flag]["sha"]
    assert cands[0].score == want[flag]["score"]
    assert _plain_layers(cands[0].meta["layers"]) == want[flag]["layers"]


def test_candidate_structure_and_meta():
    plain, blocks = MF.layer_input("synth0")
    cands = EJPS_SOLVERS.build("layer_greedy").solve(blocks, sub_block_size=8)
    assert [(c.meta["g_dihedral"], c.meta["b_dihedral"]) for c in cands[:3]] == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]
    assert cands[13].meta["g_dihedral"] == 1 and cands[13].meta["b_dihedral"] == 5
    base = cands[0].blocks
    for c in (cands[13], cands[63]):
        np.testing.assert_array_equal(c.blocks[..., 0], base[..., 0])
        g, b = c.meta["g_dihedral"], c.meta["b_dihedral"]
        np.testing.assert_array_equal(c.blocks[..., 1], dihedral(base[..., 1], g, axes=(1, 2)))
        np.testing.assert_array_equal(c.blocks[..., 2], dihedral(base[..., 2], b, axes=(1, 2)))
    assert {c.score for c in cands} == {22231.875}
    layers = cands[0].meta["layers"]
    assert [m["layer"] for m in layers] == ["R", "G", "B"]
    assert [m["planes"] for m in layers] == [[6, 0, 11, 5], [8, 1, 10, 4], [7, 9, 2, 3]]
    assert layers[0]["anchor_slot"] == 1
    assert layers[0]["fourth"] == {"plane": 11, "slot": 2, "variant": 2, "dihedral": 1}
    good = [k for k, c in enumerate(cands) if is_equivalent(c.blocks, plain)]
    assert good == [5]


def test_notebook_flaws_and_fix_flags(golden):
    _, blocks = MF.layer_input("noise")
    default = L.LayerGreedySolver().solve(blocks)[0].meta["layers"]
    # the notebook takes two planes of cipher sub-block 0 (planes 0 and 1) into layer R ...
    assert default[0]["planes"] == [6, 0, 9, 1]
    assert sorted(q // 3 for q in default[0]["planes"]) == [0, 0, 2, 3]
    # ... and applies a "flip" variant in the fourth-plane step
    assert default[0]["fourth"]["variant"] == 3

    fixed = L.LayerGreedySolver(fix_fourth_flip=True).solve(blocks)[0].meta["layers"]
    assert [m["planes"] for m in fixed] == [[6, 0, 8, 1], [9, 2, 3, 10], [5, 4, 7, 11]]
    assert all(m["fourth"]["variant"] in (0, 2) for m in fixed)

    distinct = L.LayerGreedySolver(one_plane_per_sub_block=True).solve(blocks)[0].meta["layers"]
    assert [m["planes"] for m in distinct] == [[6, 0, 9, 4], [1, 11, 8, 3], [5, 10, 7, 2]]
    for m in distinct:
        assert sorted(q // 3 for q in m["planes"]) == [0, 1, 2, 3]

    # assemble_layer's defaults are the notebook's behaviour
    planes = blocks.reshape(196, 2, 8, 2, 8, 3).transpose(0, 1, 3, 5, 2, 4).reshape(196, 12, 8, 8)
    res = L.assemble_layer(planes, np.arange(12) // 3)
    assert res.slots == (6, 0, 9, 1) and res.meta["fourth"]["variant"] == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        L.LayerGreedySolver().fix_fourth_flip = True


def test_edge_move_table_and_variants():
    assert L.EDGE_MOVE.tolist() == [[0, 4, 5, 3], [4, 0, 3, 5], [5, 1, 0, 6], [1, 5, 6, 0]]
    e = np.array([1.0, 2.0, 7.0])
    np.testing.assert_array_equal(
        L.edge_variants(e), [[1, 2, 7], [7, 2, 1], [254, 253, 248], [248, 253, 254]]
    )
    p = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)
    np.testing.assert_array_equal(L.apply_variant(p, 0, 2), p)
    np.testing.assert_array_equal(L.apply_variant(p, 1, 2), [[[2, 1], [4, 3]]])
    np.testing.assert_array_equal(L.apply_variant(p, 1, 1), [[[3, 4], [1, 2]]])
    np.testing.assert_array_equal(L.apply_variant(p, 2, 2), 255 - p)
    np.testing.assert_array_equal(L.apply_variant(p, 3, 2), [[[253, 254], [251, 252]]])
    assert L._interleave(list(range(10, 18))) == [10, 14, 11, 15, 12, 16, 13, 17]
    assert L._FACING == {0: 1, 1: 0, 2: 3, 3: 2}


def test_neighbour_and_fourth_scores_on_toy_planes():
    a = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    c = np.array([[[[2, 3], [9, 9]], [[0, 0], [0, 0]]]], dtype=np.uint8)
    s = L.neighbour_scores(a, c)
    assert s.shape == (2, 4, 4, 4)
    assert s[0, 0].tolist() == [
        [4.0, 72.5, 34.0, 36.5],
        [0.0, 42.5, 18.0, 18.5],
        [2.5, 65.0, 26.5, 29.0],
        [0.5, 50.0, 18.5, 20.0],
    ]
    assert s[0, 1, 1, 0] == 1.0 and s[0, 2, 1, 0] == (251**2 + 249**2) / 2

    # layer with ul empty: ur's left edge [5, 6], dl's top edge [7, 8]
    layer = np.zeros((1, 4, 2, 2), dtype=np.uint8)
    layer[0, 1] = [[5, 0], [6, 0]]
    layer[0, 2] = [[7, 8], [0, 0]]
    cand = np.array([[[[0, 5], [7, 6]]]], dtype=np.uint8)  # right [5, 6], down [7, 6]
    fs = L.fourth_scores(layer, 0, cand)
    assert fs.shape == (1, 4, 8)
    assert fs[0, 0, 0] == 2.0  # right matches exactly, down off by (0, 2)
    assert fs[0, 1, 0] == 1.0 + 1.0  # both edges reversed: [6,5] vs [5,6], [6,7] vs [7,8]
    assert fs[0, 0, 1] == 1.0 + 25.0  # rot90 -> right [6,7], down [0,7]
    assert fs[0, 2, 0] == 117618.0


def test_fourth_pairs_table_and_ur_hole():
    assert L._FOURTH_PAIRS == {
        0: (((1, 2), 3), ((2, 0), 1)),
        1: (((0, 3), 2), ((3, 0), 1)),
        2: (((0, 1), 0), ((3, 2), 3)),
        3: (((2, 3), 2), ((1, 1), 0)),
    }
    # hole at ur: ul's right edge [1, 2] and dr's top edge [3, 4]
    layer = np.zeros((1, 4, 2, 2), dtype=np.uint8)
    layer[0, 0] = [[0, 1], [0, 2]]
    layer[0, 3] = [[3, 4], [0, 0]]
    cand = np.array([[[[1, 9], [2, 4]], [[3, 4], [1, 2]]]], dtype=np.uint8)
    fs = L.fourth_scores(layer, 1, cand)
    assert fs[0, 0, 0] == 0.0 + 0.5  # left [1, 2] exact, down [2, 4] vs [3, 4]
    assert fs[1, 0, 0] == 2.5 + 4.0  # left [3, 1] vs [1, 2], down [1, 2] vs [3, 4]
    masked = L._masked(fs, np.array([False, True]))
    assert np.isinf(masked[0]).all() and np.array_equal(masked[1], fs[1])


def test_first_min_scan_order():
    a, b = np.array([3.0, 1.0]), np.array([1.0, 0.5])
    assert L._first_min([a, b]) == (1, 1, 0.5)
    assert L._first_min([np.array([2.0, 1.0]), np.array([1.0, 5.0])]) == (0, 1, 1.0)
    with pytest.raises(RuntimeError):
        L._first_min([np.full(2, np.inf)])
