"""Tests for ejps/restore/select.py."""

import dataclasses

import make_fixtures as MF
import numpy as np
import pytest

from ejps.geometry import SubBlockCandidate
from ejps.metrics import is_equivalent
from ejps.restore import select as S
from ejps.restore.edge import EdgeRefSolver
from ejps.restore.layer_greedy import LayerGreedySolver


@pytest.mark.parametrize("feature", S.FEATURES)
def test_top_candidate_is_plaintext_up_to_residual_ambiguity(feature, golden):
    hits, top_indices = 0, []
    for k, seed in enumerate(MF.SELECT_SEEDS):
        plain, blocks = MF.select_input(seed)
        want = golden["select"][feature][k]
        assert MF.sha(blocks) == want["input_sha"]
        ranked = S.rank_candidates(LayerGreedySolver().solve(blocks), feature=feature)
        assert len(ranked) == 64
        assert ranked[0].score > ranked[1].score + 0.5  # clear margin, not a tie
        assert [r.index for r in ranked[:3]] == want["top"]
        assert [r.score for r in ranked[:3]] == want["scores"]
        assert list(ranked[0].rho) == want["rho"]
        hits += is_equivalent(ranked[0].blocks, plain)
        top_indices.append(ranked[0].index)
    assert hits == len(MF.SELECT_SEEDS)
    assert top_indices == [0, 0, 3, 0, 6, 1]


def test_resolves_relative_inversion_of_edge_ref_output(golden):
    # a globally inverted cipher channel leaves edge_ref with one channel inverted
    plain, blocks = MF.edge_input(*MF.EDGE_CASES[3][:3])
    cand = EdgeRefSolver().solve(blocks)[0]
    raw = S.channel_correlations(cand.blocks)
    assert sum(r < 0 for r in raw) == 2  # one channel is inverted relative to the others
    ranked = S.rank_candidates([cand])
    assert list(ranked[0].rho) == golden["select_edge_inverted"]["rho"]
    assert (
        list(ranked[0].invert) == golden["select_edge_inverted"]["invert"] == [False, True, False]
    )
    assert is_equivalent(ranked[0].blocks, plain)
    # after resolution all three correlations are positive
    rho = S.channel_correlations(ranked[0].blocks)
    assert min(rho) > 0.5
    assert ranked[0].score == pytest.approx(sum(rho))


def _toy():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 200, (3, 4, 4)).astype(np.uint8)
    return np.stack([r, r + 10, 255 - r], axis=-1)


def test_correlations_signs_and_resolution():
    blocks = _toy()
    np.testing.assert_allclose(S.channel_correlations(blocks), (1.0, -1.0, -1.0))
    score, signs = S.best_signs((1.0, -1.0, -1.0))
    assert (score, signs) == (3.0, (1, 1, -1))
    ranked = S.rank_candidates([SubBlockCandidate(blocks, 0.0)])
    top = ranked[0]
    assert top.invert == (False, False, True) and top.index == 0
    np.testing.assert_array_equal(top.blocks[..., 2], blocks[..., 0])
    assert top.blocks.dtype == np.uint8
    np.testing.assert_allclose(top.rho, (1.0, -1.0, -1.0))
    plain = S.rank_candidates([SubBlockCandidate(blocks, 0.0)], resolve_inversion=False)[0]
    assert plain.invert == (False, False, False)
    assert plain.score == pytest.approx(-1.0)
    np.testing.assert_array_equal(plain.blocks, blocks)
    assert S.best_signs((0.0, 0.0, 0.0)) == (0.0, (1, 1, 1))
    assert S.best_signs((1.0, 1.0, 1.0)) == (3.0, (1, 1, 1))
    assert S.best_signs((-1.0, 1.0, -1.0)) == (3.0, (1, -1, 1))
    assert S.best_signs((-1.0, -1.0, 1.0)) == (3.0, (1, -1, -1))
    assert S.best_signs((-1.0, 0.5, 0.2)) == (pytest.approx(1.3), (1, -1, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        top.score = 0.0
    assert S.SIGN_PATTERNS[0] == (1, 1, 1) and len(S.SIGN_PATTERNS) == 4


def test_per_block_average_skips_constant_planes():
    r = np.arange(16, dtype=np.uint8).reshape(4, 4)
    b1 = np.stack([r, r, np.full_like(r, 9)], axis=-1)  # B constant
    b2 = np.stack([r, 15 - r, r], axis=-1)
    rho = S.channel_correlations(np.stack([b1, b2]))
    np.testing.assert_allclose(rho, (0.0, 1.0, -1.0))  # RG: (1 + -1) / 2; RB/GB from b2 only
    tiny = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    tiny[0, 0, 0, :2] = 1  # variance below 1 is still a valid block
    assert S.channel_correlations(tiny)[0] == pytest.approx(1.0)
    const = np.full((2, 4, 4, 3), 5, dtype=np.uint8)
    assert S.channel_correlations(const) == (0.0, 0.0, 0.0)


def test_laplacian_ignores_ramps():
    yy, xx = np.mgrid[0:6, 0:6]
    ramp = np.stack([10 * xx, 10 * xx + 5 * yy, 200 - 10 * xx], axis=-1).astype(np.uint8)[None]
    assert S.channel_correlations(ramp, "laplacian") == (0.0, 0.0, 0.0)
    rho = S.channel_correlations(ramp, "pixel")
    assert rho[1] == pytest.approx(-1.0) and rho[0] > 0.8
    assert S.block_features(ramp, "laplacian").shape == (1, 16, 3)
    assert S.block_features(ramp, "pixel").shape == (1, 36, 3)
    with pytest.raises(ValueError):
        S.block_features(ramp, "sobel")


def test_ranking_is_stable_and_select_best():
    good = _toy()
    good[..., 2] = good[..., 0]
    bad = good.copy()
    bad[..., 1] = good[..., 1][:, ::-1]
    cands = [
        SubBlockCandidate(bad, 0.0),
        SubBlockCandidate(good, 0.0),
        SubBlockCandidate(good, 1.0),
    ]
    ranked = S.rank_candidates(cands)
    assert [r.index for r in ranked][:2] == [1, 2]
    assert ranked[0].candidate is cands[1]
    np.testing.assert_array_equal(S.select_best(cands), good)
    np.testing.assert_array_equal(S.select_best(cands[:2][::-1]), good)
    assert S.rank_candidates(cands[:2])[1].index == 0
