"""Tests for ejps/jigsaw/ga.py (gaps kernel-growing crossover, ported).

All golden children / counters were produced by the original gaps (commit
5aa0a0d, Python 3.12) from the same parents and ``random.Random`` seeds; they pin
both the placement logic and the order of random draws.
"""

import hashlib
import random

import numpy as np
import pytest

from ejps.jigsaw.domain import (
    Arrangement,
    Direction,
    assemble_image,
    flatten_image,
)
from ejps.jigsaw.fitness import EdgeCostTable
from ejps.jigsaw.ga import Crossover, CrossoverStats, GeneticAlgorithm

STAT_FIELDS = (
    "candidate_pushes",
    "candidate_pops",
    "discarded_taken_positions",
    "reexpanded_placed_pieces",
    "placed_pieces",
    "shared_matches",
    "buddy_matches",
    "best_match_candidates",
    "best_match_calls",
    "best_match_scanned",
    "validity_checks",
    "preserved_blocks",
    "preserved_block_pieces",
)

# name: (H, W, {rng seed: (child ids, stats)}, child of (pa, pa) with Random(3))
CROSSOVER_GOLDEN = {
    "g44": (64, 64, {
        9: ([3, 0, 10, 2, 9, 4, 5, 6, 15, 8, 11, 7, 1, 12, 13, 14], (26, 26, 9, 5, 16, 1, 3, 22, 22, 70, 143, 1, 4)),
        10: ([14, 15, 1, 2, 10, 11, 5, 0, 12, 13, 9, 4, 3, 7, 6, 8], (30, 30, 9, 7, 16, 2, 3, 25, 25, 78, 161, 1, 2)),
    }, [4, 6, 5, 3, 8, 7, 11, 15, 1, 12, 13, 9, 2, 10, 0, 14]),
    "row": (16, 96, {
        9: ([2, 3, 1, 4, 5, 0], (2, 2, 0, 0, 6, 2, 0, 0, 0, 0, 2, 1, 4)),
        10: ([1, 2, 3, 5, 0, 4], (4, 4, 0, 1, 6, 1, 0, 3, 3, 6, 16, 1, 3)),
    }, [3, 5, 0, 4, 1, 2]),
    "g22": (32, 32, {
        9: ([0, 3, 2, 1], (4, 4, 1, 1, 4, 0, 0, 4, 4, 6, 18, 1, 2)),
        10: ([0, 3, 2, 1], (4, 4, 1, 1, 4, 0, 0, 4, 4, 6, 18, 1, 2)),
    }, [3, 0, 2, 1]),
    "g33": (48, 48, {
        9: ([0, 8, 1, 3, 6, 4, 5, 7, 2], (5, 5, 2, 0, 9, 1, 0, 4, 4, 12, 25, 1, 6)),
        10: ([5, 0, 6, 8, 1, 2, 3, 4, 7], (14, 14, 4, 3, 9, 1, 1, 12, 12, 33, 72, 1, 2)),
    }, [5, 6, 7, 4, 3, 0, 8, 1, 2]),
    "col": (80, 16, {
        9: ([0, 2, 1, 3, 4], (1, 1, 0, 0, 5, 0, 0, 1, 1, 2, 5, 1, 4)),
        10: ([3, 4, 0, 1, 2], (3, 3, 0, 0, 5, 2, 0, 1, 1, 1, 6, 1, 2)),
    }, [2, 3, 4, 0, 1]),
}  # fmt: skip

# (image, piece size) -> children for Random(9), Random(10), Random(11)
SPECIAL_GOLDEN = {
    # 1x2 layout: no block dimension fits -> random root piece
    "pair": [
        ([0, 1], (1, 1, 0, 0, 2, 0, 0, 1, 1, 1, 4, 0, 0)),
        ([1, 0], (1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0)),
        ([0, 1], (1, 1, 0, 0, 2, 0, 0, 1, 1, 1, 4, 0, 0)),
    ],
    # flat image: every confidence is 0 < MIN_BLOCK_CONFIDENCE -> random root piece
    "flat": [
        ([1, 0, 2, 3, 7, 4, 5, 6, 8], (20, 20, 4, 8, 9, 0, 1, 19, 19, 61, 120, 0, 0)),
        ([1, 0, 2, 3, 5, 4, 6, 7, 8], (20, 20, 4, 8, 9, 0, 1, 19, 19, 63, 122, 0, 0)),
        ([2, 0, 3, 4, 8, 1, 5, 6, 7], (18, 18, 4, 6, 9, 1, 0, 17, 17, 56, 108, 0, 0)),
    ],
    # grayscale noise, 3-pixel pieces
    "noise": [
        ([1, 10, 4, 7, 3, 5, 12, 14, 6, 11, 2, 8, 0, 13, 15, 9], (26, 26, 9, 5, 16, 0, 0, 26, 26, 67, 145, 1, 4)),
        ([11, 2, 8, 7, 0, 12, 15, 1, 10, 13, 9, 4, 6, 14, 3, 5], (27, 27, 9, 4, 16, 2, 1, 24, 24, 48, 124, 1, 2)),
        ([15, 1, 11, 7, 10, 8, 0, 3, 12, 13, 9, 4, 6, 14, 2, 5], (28, 28, 9, 7, 16, 1, 1, 26, 26, 81, 162, 1, 4)),
    ],
}  # fmt: skip

# crossover counters accumulated over upstream GeneticAlgorithm runs (see test below)
GA64_STATS = (6121, 6121, 2166, 817, 4480, 2667, 793, 2661, 2661, 11163, 23399, 280, 1342)


def smooth_image(h, w):
    """Synthetic smooth image (gradients + Gaussian blobs + texture); same as upstream golden."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.empty((h, w, 3))
    img[..., 0] = 30.0 + 150.0 * x / w
    img[..., 1] = 40.0 + 120.0 * y / h
    img[..., 2] = 60.0 + 60.0 * (x + y) / (h + w)
    blobs = [
        (0.25, 0.30, 0.12, (90.0, -30.0, 40.0)),
        (0.70, 0.65, 0.18, (-40.0, 70.0, 20.0)),
        (0.40, 0.80, 0.10, (30.0, 30.0, -50.0)),
        (0.85, 0.20, 0.14, (-20.0, -40.0, 80.0)),
    ]
    for cy, cx, s, amp in blobs:
        g = np.exp(-(((y / h - cy) ** 2) + ((x / w - cx) ** 2)) / (2.0 * s * s))
        for c in range(3):
            img[..., c] += amp[c] * g
    img += (6.0 * np.sin(x / 3.0) * np.cos(y / 4.0))[..., None]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def setup(img, piece_size):
    pieces, layout = flatten_image(img, piece_size, indexed=True)
    table = EdgeCostTable()
    table.analyze(pieces)
    pa = Arrangement.random(pieces, layout, random.Random(1))
    pb = Arrangement.random(pieces, layout, random.Random(2))
    return table, pa, pb


def run(pa, pb, table, seed):
    stats = CrossoverStats()
    crossover = Crossover(pa, pb, table, random.Random(seed), stats=stats)
    crossover.run()
    child = crossover.child()
    return [p.identifier for p in child.pieces], tuple(getattr(stats, f) for f in STAT_FIELDS)


def test_input_image_matches_upstream_golden_input():
    digest = hashlib.sha256(smooth_image(64, 64).tobytes()).hexdigest()
    assert digest == "7d35a3e44f5f456c6c31894d63ee0952c104e2f2619269a3f191113edec76298"


@pytest.mark.parametrize("name", sorted(CROSSOVER_GOLDEN))
def test_crossover_children_match_upstream(name):
    h, w, by_seed, same = CROSSOVER_GOLDEN[name]
    table, pa, pb = setup(smooth_image(h, w), 16)
    for seed, (child, stats) in by_seed.items():
        got_child, got_stats = run(pa, pb, table, seed)
        assert got_child == child
        assert dict(zip(STAT_FIELDS, got_stats, strict=True)) == dict(
            zip(STAT_FIELDS, stats, strict=True)
        )
        assert sorted(got_child) == list(range(len(child)))
    crossover = Crossover(pa, pa, table, random.Random(3))
    crossover.run()
    assert [p.identifier for p in crossover.child().pieces] == same


@pytest.mark.parametrize("name", sorted(SPECIAL_GOLDEN))
def test_crossover_random_root_and_grayscale_match_upstream(name):
    if name == "pair":
        args = (smooth_image(16, 32), 16)
    elif name == "flat":
        args = (np.full((48, 48, 3), 77, np.uint8), 16)
    else:
        args = (np.random.default_rng(3).integers(0, 256, (12, 12), dtype=np.uint8), 3)
    table, pa, pb = setup(*args)
    for seed, (child, stats) in zip((9, 10, 11), SPECIAL_GOLDEN[name], strict=True):
        assert run(pa, pb, table, seed) == (child, stats)


def test_child_pieces_are_taken_from_first_parent_objects():
    table, pa, pb = setup(smooth_image(48, 48), 16)
    crossover = Crossover(pa, pb, table, random.Random(9))
    crossover.run()
    child = crossover.child()
    assert child.layout == pa.layout
    for piece in child.pieces:
        assert piece is pa.piece_by_id(piece.identifier)


def test_child_before_run_is_incomplete():
    table, pa, pb = setup(smooth_image(32, 32), 16)
    with pytest.raises(RuntimeError):
        Crossover(pa, pb, table, random.Random(0)).child()


def test_stats_are_optional_and_do_not_change_the_child():
    table, pa, pb = setup(smooth_image(64, 64), 16)
    crossover = Crossover(pa, pb, table, random.Random(10))
    crossover.run()
    plain = [p.identifier for p in crossover.child().pieces]
    assert plain == CROSSOVER_GOLDEN["g44"][2][10][0]


def test_shared_and_buddy_lookups():
    table, pa, pb = setup(smooth_image(64, 64), 16)
    crossover = Crossover(pa, pb, table, random.Random(0))
    # parents' right neighbours of piece 5: pa -> 3, pb -> 14 (not shared)
    assert pa.edge(5, Direction.RIGHT) == 3 and pb.edge(5, Direction.RIGHT) == 14
    assert crossover._get_shared_piece(5, Direction.RIGHT) is None
    same = Crossover(pa, pa, table, random.Random(0))
    assert same._get_shared_piece(5, Direction.RIGHT) == 3
    assert same._get_shared_piece(4, Direction.RIGHT) is None  # right border of pa
    # 5's best buddy to the right is 6 (true neighbour) but neither parent has 5|6
    assert table.best_match(5, Direction.RIGHT) == 6
    assert table.best_match(6, Direction.LEFT) == 5
    assert crossover._get_buddy_piece(5, Direction.RIGHT) is None
    ordered = Arrangement(sorted(pa.pieces, key=lambda p: p.identifier), pa.layout)
    with_buddy = Crossover(ordered, pb, table, random.Random(0))
    assert with_buddy._get_buddy_piece(5, Direction.RIGHT) == 6
    assert with_buddy._get_buddy_piece(5, Direction.BOTTOM) == 9


def test_best_match_cursor_skips_used_pieces():
    table, pa, pb = setup(smooth_image(64, 64), 16)
    crossover = Crossover(pa, pb, table, random.Random(0))
    matches = table.matches(5, Direction.RIGHT)
    piece, priority = crossover._get_best_match_piece(5, Direction.RIGHT)
    assert piece == matches[0][0] == 6
    expected = matches[0][1] - 0.02 * table.confidence(5, Direction.RIGHT)
    assert priority == pytest.approx(expected, rel=1e-12)
    crossover._kernel[6] = (0, 0)
    crossover._kernel[matches[1][0]] = (0, 1)
    piece, priority = crossover._get_best_match_piece(5, Direction.RIGHT)
    assert piece == matches[2][0]
    assert priority == pytest.approx(
        matches[2][1] - 0.02 * table.confidence(5, Direction.RIGHT), rel=1e-12
    )
    assert crossover._match_cursors[(5, Direction.RIGHT)] == 2
    for other, _ in matches:
        crossover._kernel[other] = (9, 9)
    assert crossover._get_best_match_piece(5, Direction.RIGHT) == (None, None)
    assert crossover._match_cursors[(5, Direction.RIGHT)] == 15


def test_kernel_range_limits():
    table, pa, pb = setup(smooth_image(48, 48), 16)  # 3x3
    crossover = Crossover(pa, pb, table, random.Random(0))
    crossover._min_row, crossover._max_row = 0, 2
    crossover._min_column, crossover._max_column = -1, 0
    assert crossover._is_in_range((1, 1)) is True
    assert crossover._is_in_range((1, 2)) is False  # would span 4 columns
    assert crossover._is_in_range((1, -2)) is True  # spans 3 columns
    assert crossover._is_in_range((1, -3)) is False
    assert crossover._is_in_range((3, 0)) is False
    assert crossover._is_in_range((-1, 0)) is False
    crossover._min_row, crossover._max_row = 0, 1
    assert crossover._is_in_range((2, 0)) is True
    assert crossover._is_in_range((-1, 0)) is True
    crossover._update_kernel_boundaries((-1, -2))
    assert (crossover._min_row, crossover._max_row) == (-1, 1)
    assert (crossover._min_column, crossover._max_column) == (-2, 0)
    crossover._update_kernel_boundaries((3, 4))
    assert (crossover._min_row, crossover._max_row) == (-1, 3)
    assert (crossover._min_column, crossover._max_column) == (-2, 4)
    assert crossover._is_valid_piece(None) is False
    assert crossover._is_valid_piece(4) is True
    crossover._kernel[4] = (0, 0)
    assert crossover._is_valid_piece(4) is False


def test_available_boundaries_update_bounds_while_scanning():
    table, pa, pb = setup(smooth_image(32, 32), 16)  # 2x2
    crossover = Crossover(pa, pb, table, random.Random(0))
    crossover._kernel[0] = (0, 0)
    crossover._taken_positions.add((0, 0))
    # TOP widens the rows to 2, so BOTTOM no longer fits; RIGHT then blocks LEFT
    assert crossover._available_boundaries((0, 0)) == [
        (Direction.TOP, (-1, 0)),
        (Direction.RIGHT, (0, 1)),
    ]
    assert (crossover._min_row, crossover._max_row) == (-1, 0)
    assert (crossover._min_column, crossover._max_column) == (0, 1)
    crossover._min_row = crossover._max_row = 0
    crossover._min_column = crossover._max_column = 0
    crossover._taken_positions.add((0, 1))
    assert crossover._available_boundaries((0, 0)) == [
        (Direction.TOP, (-1, 0)),
        (Direction.LEFT, (0, -1)),
    ]
    for i, pos in enumerate([(0, 1), (1, 0), (1, 1)], start=1):
        crossover._kernel[i] = pos
    assert crossover._is_kernel_full() is True
    assert crossover._available_boundaries((0, 0)) == []
    assert crossover._kernel_piece_at((1, 0)) == 2
    with pytest.raises(RuntimeError):
        crossover._kernel_piece_at((5, 5))


def block_confidence(table, block):
    height = 1 + max(r for _, (r, _) in block)
    width = 1 + max(c for _, (_, c) in block)
    values = []
    for pid, (r, c) in block:
        if c + 1 < width:
            values.append(table.confidence(pid, Direction.RIGHT))
        if r + 1 < height:
            values.append(table.confidence(pid, Direction.BOTTOM))
    return sum(values) / len(values)


def test_select_parent_block():
    table, pa, pb = setup(smooth_image(64, 64), 16)
    pieces, confidence = Crossover(pa, pb, table, random.Random(9))._select_parent_block()
    # column 1 of the first parent, as a 4x1 block
    assert pieces == [(10, (0, 0)), (5, (1, 0)), (11, (2, 0)), (13, (3, 0))]
    assert [pa[r][1].identifier for r in range(4)] == [10, 5, 11, 13]
    assert confidence == pytest.approx(block_confidence(table, pieces), rel=1e-12)
    assert confidence == pytest.approx(0.9663394916936122, rel=1e-9)
    pieces, confidence = Crossover(pa, pb, table, random.Random(2))._select_parent_block()
    # rows 1..3 of the second parent (3x4)
    assert [pid for pid, _ in pieces] == [pb.pieces[i].identifier for i in range(4, 16)]
    assert [pos for _, pos in pieces][:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    assert confidence == pytest.approx(block_confidence(table, pieces), rel=1e-12)
    # a 1x2 layout has no block dimension (1 < h*w < 2 is empty)
    table2, pa2, pb2 = setup(smooth_image(16, 32), 16)
    assert Crossover(pa2, pb2, table2, random.Random(0))._select_parent_block() is None
    # a flat image has zero confidence everywhere -> every trial is rejected
    table3, pa3, pb3 = setup(np.full((48, 48, 3), 77, np.uint8), 16)
    assert Crossover(pa3, pb3, table3, random.Random(0))._select_parent_block() is None


def test_ga_crossover_counters_match_upstream():
    img = smooth_image(64, 64)
    pieces, layout = flatten_image(img, 16)
    perm = list(range(16))
    random.Random(1).shuffle(perm)
    shuffled = assemble_image([pieces[i] for i in perm], layout)
    ga = GeneticAlgorithm(
        shuffled, 16, population_size=30, generations=10, rng=random.Random(0), collect_stats=True
    )
    ga.solve()
    got = tuple(getattr(ga.crossover_stats, f) for f in STAT_FIELDS)
    assert dict(zip(STAT_FIELDS, got, strict=True)) == dict(
        zip(STAT_FIELDS, GA64_STATS, strict=True)
    )
