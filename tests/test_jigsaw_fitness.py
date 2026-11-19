"""Tests for ejps/jigsaw/fitness.py (gaps edge dissimilarity, ported).

Golden numbers marked "upstream" were produced by running the original gaps
(commit 5aa0a0d, Python 3.12) on the same inputs.
"""

import numpy as np
import pytest

from ejps.jigsaw.domain import EdgeAxis, Piece, flatten_image
from ejps.jigsaw.fitness import (
    _coarse_line,
    _huber_mean,
    _huber_mean_batch,
    dissimilarity_measure,
    pairwise_dissimilarity,
    pairwise_dissimilarity_from_channels,
    prepare_piece_channels,
)

H, V = EdgeAxis.HORIZONTAL, EdgeAxis.VERTICAL


def huber(e):
    """Reference Huber term for a normalized absolute error (beta = 0.01)."""
    return 50.0 * e * e if e <= 0.01 else e - 0.005


def hand_pair():
    """a's right edge = 10, inner column 9; b's left edge = 12 (one pixel 1), inner 20."""
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.zeros((4, 4, 3), np.uint8)
    a[:, 3, :] = 10
    a[:, 2, :] = 9
    b[:, 0, :] = 12
    b[:, 1, :] = 20
    b[1, 0, 1] = 1
    return Piece(a, 0), Piece(b, 1)


def expected_hand_cost():
    # boundary: 11 x |10-12|=2 (quadratic side), 1 x |10-1|=9 (linear side); /255
    boundary = (11 * huber(2 / 255) + huber(9 / 255)) / 12
    # normal: (10-9) - (20-12) = -7 (linear side) x11, (10-9) - (20-1) = -18 x1; /510
    normal = (11 * huber(7 / 510) + huber(18 / 510)) / 12
    # tangent: diff(a)=0, diff(b ch1)=[-11, 11, 0]; 9 entries; /510
    tangent = 2 * huber(11 / 510) / 9
    # coarse (pool by 2): a=10; b ch1 = [6.5, 12], others [12, 12]; /255
    coarse = (5 * huber(2 / 255) + huber(3.5 / 255)) / 6
    base = 0.60 * boundary + 0.25 * normal + 0.15 * tangent
    return 0.85 * base + 0.15 * coarse


def test_huber_mean_covers_both_branches_exactly():
    # 2/255 < beta -> 0.5 e^2 / beta ; 9/255 > beta -> e - beta/2
    assert _huber_mean(np.array([2.0]), 255.0) == pytest.approx(50 * (2 / 255) ** 2, rel=1e-12)
    assert _huber_mean(np.array([-9.0]), 255.0) == pytest.approx(9 / 255 - 0.005, rel=1e-12)
    # exactly at beta: quadratic branch == 0.5 * beta
    assert _huber_mean(np.array([2.55]), 255.0) == pytest.approx(0.005, rel=1e-9)
    assert _huber_mean(np.array([2.0, -9.0]), 255.0) == pytest.approx(
        (huber(2 / 255) + huber(9 / 255)) / 2, rel=1e-12
    )
    batch = _huber_mean_batch(np.array([[[[2.0], [-9.0]]], [[[0.0], [5.1]]]]), 255.0)
    assert batch.shape == (2, 1)
    np.testing.assert_allclose(
        batch[:, 0], [(huber(2 / 255) + huber(9 / 255)) / 2, huber(5.1 / 255) / 2], rtol=1e-12
    )


def test_hand_example_exact_value_and_upstream_golden():
    pa, pb = hand_pair()
    value = dissimilarity_measure(pa, pb, H)
    assert value == pytest.approx(expected_hand_cost(), rel=1e-6)
    assert value == pytest.approx(0.006033569365972653, rel=1e-9)  # upstream
    assert dissimilarity_measure(pa, pb) == value  # default axis is horizontal


def test_direction_matters():
    pa, pb = hand_pair()
    assert dissimilarity_measure(pb, pa, H) == 0.0  # b's right edge and a's left edge are 0
    assert dissimilarity_measure(pa, pb, V) == pytest.approx(0.03265990785817848, rel=1e-9)
    assert dissimilarity_measure(pa, pb, V) != dissimilarity_measure(pa, pb, H)


def test_pairwise_matches_scalar_and_has_inf_diagonal():
    pa, pb = hand_pair()
    rng = np.random.default_rng(1)
    pc = Piece(rng.integers(0, 256, (4, 4, 3), dtype=np.uint8), 2)
    pieces = [pa, pb, pc]
    for axis in (H, V):
        m = pairwise_dissimilarity(pieces, axis)
        assert m.dtype == np.float32
        assert np.all(np.isinf(np.diag(m)))
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert m[i, j] == pytest.approx(
                        dissimilarity_measure(pieces[i], pieces[j], axis), rel=1e-5
                    )
    m = pairwise_dissimilarity(pieces, H)
    assert m[0, 1] == pytest.approx(0.006033569365972653, rel=1e-6)
    assert m[1, 0] == 0.0


def test_pairwise_block_size_does_not_change_result():
    rng = np.random.default_rng(2)
    pieces = [Piece(rng.integers(0, 256, (5, 5), dtype=np.uint8), i) for i in range(7)]
    channels = prepare_piece_channels(pieces)
    ref = pairwise_dissimilarity_from_channels(channels, V, block_size=7)
    for bs in (1, 2, 3, 32):
        np.testing.assert_array_equal(
            pairwise_dissimilarity_from_channels(channels, V, block_size=bs), ref
        )


def test_pairwise_gray_upstream_golden():
    gray = np.arange(36, dtype=np.uint8).reshape(6, 6) * 7
    pieces, _ = flatten_image(gray, 3, indexed=True)
    mh = pairwise_dissimilarity(pieces, H)
    mv = pairwise_dissimilarity(pieces, V)
    # upstream rows 0 and 1 (3x3 pieces -> no coarse term)
    np.testing.assert_allclose(mh[0, 1:], [0.01145, 0.22145, 0.26345], rtol=1e-4)
    np.testing.assert_allclose(mh[1, [0, 2, 3]], [0.06745, 0.17945, 0.22145], rtol=1e-4)
    np.testing.assert_allclose(mv[0, 1:], [0.12345, 0.08145, 0.12345], rtol=1e-4)
    np.testing.assert_allclose(mv[1, [0, 2, 3]], [0.20745, 0.03945, 0.08145], rtol=1e-4)


def test_progress_callback_reports_blocks():
    rng = np.random.default_rng(3)
    channels = rng.integers(0, 256, (5, 4, 4, 3)).astype(np.float32)
    events = []
    pairwise_dissimilarity_from_channels(
        channels, H, block_size=2, progress=lambda c, t: events.append((c, t))
    )
    assert events == [(2, 4), (4, 4), (4, 4)]


def test_one_pixel_pieces_only_use_boundary_term():
    a = Piece(np.full((1, 1, 3), 100, np.uint8), 0)
    b = Piece(np.full((1, 1, 3), 110, np.uint8), 1)
    # no inner line, no tangent, no coarse: 0.85 * 0.6 * huber(10/255)
    expected = 0.85 * 0.6 * huber(10 / 255)
    assert dissimilarity_measure(a, b, H) == pytest.approx(expected, rel=1e-6)
    m = pairwise_dissimilarity([a, b], H)
    assert m[0, 1] == pytest.approx(expected, rel=1e-6)


def test_two_pixel_pieces_have_normal_and_tangent_but_no_coarse():
    a = np.zeros((2, 2), np.uint8)
    b = np.zeros((2, 2), np.uint8)
    b[:, 0] = [0, 51]  # tangent diff 51 -> 0.1 ; boundary [0, 51]
    pa, pb = Piece(a, 0), Piece(b, 1)
    boundary = huber(51 / 255) / 2
    normal = (huber(0.0) + huber(51 / 510)) / 2  # (0-0) - (0-[0,51])
    tangent = huber(51 / 510)
    expected = 0.85 * (0.6 * boundary + 0.25 * normal + 0.15 * tangent)
    assert dissimilarity_measure(pa, pb, H) == pytest.approx(expected, rel=1e-6)
    assert pairwise_dissimilarity([pa, pb], H)[0, 1] == pytest.approx(expected, rel=1e-6)


def test_coarse_line_pools_pairs_and_keeps_odd_tail():
    line = np.array([[0.0], [2.0], [4.0], [8.0], [9.0]])
    np.testing.assert_array_equal(_coarse_line(line), [[1.0], [6.0], [9.0]])
    np.testing.assert_array_equal(_coarse_line(line[:4]), [[1.0], [6.0]])
    assert _coarse_line(line[:3]) is None


def test_validation_errors():
    pa, pb = hand_pair()
    with pytest.raises(ValueError):
        dissimilarity_measure(pa, pb, "diagonal")
    with pytest.raises(ValueError):
        pairwise_dissimilarity_from_channels(np.zeros((2, 3, 3, 3), np.float32), "diagonal")
    with pytest.raises(ValueError):
        pairwise_dissimilarity_from_channels(np.zeros((2, 3, 3, 3), np.float32), H, block_size=0)
    with pytest.raises(ValueError):
        pairwise_dissimilarity_from_channels(np.zeros((2, 3, 3, 2), np.float32), H)
    with pytest.raises(ValueError):
        pairwise_dissimilarity_from_channels(np.zeros((2, 3, 3), np.float32), H)
    assert pairwise_dissimilarity_from_channels(np.zeros((0, 3, 3, 3), np.float32), H).shape == (
        0,
        0,
    )
    small = Piece(np.zeros((2, 2, 3), np.uint8), 2)
    with pytest.raises(ValueError):
        dissimilarity_measure(pa, small, H)  # edge shapes differ
    with pytest.raises(ValueError):
        prepare_piece_channels([pa, small])
    with pytest.raises(ValueError):
        prepare_piece_channels([])


def test_one_and_two_pixel_pieces_vertically():
    a = Piece(np.full((1, 1, 3), 100, np.uint8), 0)
    b = Piece(np.full((1, 1, 3), 110, np.uint8), 1)
    expected = 0.85 * 0.6 * huber(10 / 255)
    assert dissimilarity_measure(a, b, V) == pytest.approx(expected, rel=1e-6)
    assert pairwise_dissimilarity([a, b], V)[0, 1] == pytest.approx(expected, rel=1e-6)
    # 2x2, vertical: a's bottom row 0, inner row 0; b's top row [0, 51], next row 0
    top = np.zeros((2, 2), np.uint8)
    bottom = np.zeros((2, 2), np.uint8)
    bottom[0, :] = [0, 51]
    pa, pb = Piece(top, 0), Piece(bottom, 1)
    boundary = huber(51 / 255) / 2
    normal = huber(51 / 510) / 2
    tangent = huber(51 / 510)
    expected = 0.85 * (0.6 * boundary + 0.25 * normal + 0.15 * tangent)
    assert dissimilarity_measure(pa, pb, V) == pytest.approx(expected, rel=1e-6)
    assert pairwise_dissimilarity([pa, pb], V)[0, 1] == pytest.approx(expected, rel=1e-6)
    # 3x3: the inner line is row 1 (not row 2)
    first = np.zeros((3, 3), np.uint8)
    second = np.zeros((3, 3), np.uint8)
    second[1, :] = 51
    pf, ps = Piece(first, 0), Piece(second, 1)
    # boundary 0, normal |0 - 51| / 510 everywhere, tangent 0, no coarse
    expected = 0.85 * 0.25 * huber(51 / 510)
    assert dissimilarity_measure(pf, ps, V) == pytest.approx(expected, rel=1e-6)
    assert pairwise_dissimilarity([pf, ps], V)[0, 1] == pytest.approx(expected, rel=1e-6)
    assert pairwise_dissimilarity([pf, ps], H)[0, 1] == pytest.approx(
        dissimilarity_measure(pf, ps, H), rel=1e-6
    )


def test_single_piece_and_single_channel():
    one = Piece(np.full((4, 4, 1), 9, np.uint8), 0)
    events = []
    m = pairwise_dissimilarity_from_channels(
        prepare_piece_channels([one]), H, progress=lambda c, t: events.append((c, t))
    )
    assert m.shape == (1, 1) and np.isinf(m[0, 0])
    assert events == []
    two = Piece(np.full((4, 4, 1), 60, np.uint8), 1)
    assert prepare_piece_channels([one, two]).shape == (2, 4, 4, 1)
    # constant pieces 9 vs 60: only boundary and coarse terms, |51| / 255 = 0.2
    expected = 0.85 * 0.6 * huber(0.2) + 0.15 * huber(0.2)
    assert dissimilarity_measure(one, two, H) == pytest.approx(expected, rel=1e-6)
    assert pairwise_dissimilarity([one, two], V)[0, 1] == pytest.approx(expected, rel=1e-6)
