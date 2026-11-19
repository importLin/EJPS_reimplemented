"""Tests for ejps/restore/edge.py.

The golden outputs in ``tests/fixtures`` were produced by the original
(loop-based) implementation, which the vectorised solver reproduces bit for bit.
"""

import dataclasses

import make_fixtures as MF
import numpy as np
import pytest

from ejps import encrypt_image
from ejps.geometry import EJPS_SOLVERS, image_to_blocks, split_sub_blocks
from ejps.metrics import evaluate_blocks, is_equivalent, synthetic_natural_image
from ejps.restore import edge as E


def _case(k):
    seed, key, inv, diag = MF.EDGE_CASES[k]
    plain, blocks = MF.edge_input(seed, key, inv)
    return plain, blocks, inv, diag


# the cases hit all three stage-1 branches and source-channel swaps in every round
@pytest.mark.parametrize("k", range(len(MF.EDGE_CASES)))
def test_bit_identical_to_golden(k, golden, golden_arrays):
    plain, blocks, _, diag = _case(k)
    want = golden["edge"][k]
    assert MF.sha(blocks) == want["input_sha"]  # the input recipe is unchanged
    cands = E.EdgeRefSolver().solve(blocks, sub_block_size=8)
    assert len(cands) == 1
    got = cands[0]
    assert got.blocks.dtype == np.uint8
    np.testing.assert_array_equal(got.blocks, golden_arrays[f"edge_case{k}"])
    assert got.meta["diagonal"] == diag
    assert got.score == want["score"]
    np.testing.assert_array_equal(
        E.scramble_compat(split_sub_blocks(blocks, 8)), want["scramble_compat"]
    )
    # (a global channel inversion of the cipher stays within the residual ambiguity)
    assert is_equivalent(got.blocks, plain)


def test_sources_cover_channel_swaps(golden):
    sources = [tuple(c["sources"]) for e in golden["edge"] for c in e["meta"]["channels"]]
    assert {s[0] for s in sources} == {0, 1, 2}
    assert any(s[1] != s[0] for s in sources) and any(s[3] != s[0] for s in sources)
    assert {e["meta"]["diagonal"] for e in golden["edge"]} == {"ur", "dl", "dr"}


def test_bit_identical_on_noise_and_partial_statistics(golden, golden_arrays):
    blocks = MF.noise_input()
    assert MF.sha(blocks) == golden["edge_noise"]["input_sha"]
    for stat in (None, 5, 2, 1):
        got = E.EdgeRefSolver(stat_blocks=stat).solve(blocks)[0]
        np.testing.assert_array_equal(got.blocks, golden_arrays[f"edge_noise_stat{stat}"])
        assert got.score == golden["edge_noise"]["scores"][str(stat)]
    # the stat_blocks=2 solution differs from the full one on this input
    full = E.EdgeRefSolver().solve(blocks)[0]
    part = E.EdgeRefSolver(stat_blocks=2).solve(blocks)[0]
    assert not np.array_equal(full.blocks, part.blocks)


def test_golden_meta_and_cost(golden):
    _, blocks, _, _ = _case(0)
    cand = E.EdgeRefSolver().solve(blocks)[0]
    assert cand.score == 3773.75
    np.testing.assert_array_equal(
        E.scramble_compat(split_sub_blocks(blocks, 8)), [19386.75, 885.875, 1394.25]
    )
    meta = golden["edge"][0]["meta"]
    assert cand.meta["diagonal"] == meta["diagonal"] == "ur"
    assert [
        {k: list(v) if isinstance(v, tuple) else v for k, v in c.items()}
        for c in cand.meta["channels"]
    ] == meta["channels"]


def test_recovers_full_image_under_residual_ambiguity():
    plain = synthetic_natural_image(size=224, seed=1)
    blocks = image_to_blocks(encrypt_image(plain, "common", 22)[0], 16)
    plain = image_to_blocks(plain, 16)
    cand = EJPS_SOLVERS.build("edge_ref").solve(blocks, sub_block_size=8)[0]
    ev = evaluate_blocks(cand.blocks, plain, 14, 14)
    assert ev["block_rate"] == 1.0 and ev["psnr"] == float("inf")


def test_scramble_compat_sees_reversed_negative_edges():
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 256, (1, 8, 8, 3)).astype(np.uint8)
    # ur's left column = 255 - reversed(ul's right column), channel by channel
    blocks[0, :4, 4] = 255 - blocks[0, 3::-1, 3]
    comp = E.scramble_compat(split_sub_blocks(blocks, 4))
    assert comp[0] == 0.0
    assert comp[1] > 100 and comp[2] > 100


def test_put_swaps_channels_without_aliasing():
    subs = np.zeros((1, 4, 2, 2, 3), dtype=np.uint8)
    subs[0, 1, ..., 0] = [[1, 2], [3, 4]]
    subs[0, 1, ..., 2] = [[5, 6], [7, 8]]
    E._put(subs, 1, 0, 2, 1)  # clockwise turn of channel 2 into channel 0
    np.testing.assert_array_equal(subs[0, 1, ..., 0], [[7, 5], [8, 6]])
    np.testing.assert_array_equal(subs[0, 1, ..., 2], [[1, 2], [3, 4]])
    E._put(subs, 1, 0, 0, 8)
    np.testing.assert_array_equal(subs[0, 1, ..., 0], [[248, 250], [247, 249]])
    assert subs[0, 0].sum() == 0


def test_scores_shapes_and_tables():
    _, blocks, _, _ = _case(1)
    subs = split_sub_blocks(blocks, 8)
    assert E.main_scores(subs, 1, (1, 2)).shape == (4, 16, 16, 16)
    assert E.corner_scores(subs, 2, (2,)).shape == (1, 16)
    assert E.CHANNEL_ROUNDS == ((0, (0, 1, 2)), (1, (1, 2)), (2, (2,)))
    assert E._DIAGONAL_ORDER[0] == (0, 3, 2, 1) and E._DIAGONAL_ORDER[1] == (0, 1, 3, 2)
    a = np.array([[1.0, 3.0], [0.0, 0.0]])
    np.testing.assert_array_equal(E._mse(a, np.zeros(2)), [5.0, 0.0])


def test_rejects_bad_inputs():
    _, blocks, _, _ = _case(0)
    with pytest.raises(ValueError):
        E.EdgeRefSolver().solve(blocks, sub_block_size=4)
    with pytest.raises(ValueError):
        E.EdgeRefSolver(stat_blocks=0).solve(blocks)
    assert "edge_ref" in EJPS_SOLVERS
    with pytest.raises(dataclasses.FrozenInstanceError):
        E.EdgeRefSolver().stat_blocks = 3
