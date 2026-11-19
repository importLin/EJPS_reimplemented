"""Tests for ejps/jigsaw/domain.py (gaps domain model + transforms, ported).

Numbers marked "upstream" were produced by the original gaps (commit 5aa0a0d,
Python 3.12) on the same inputs.
"""

import math
import random

import numpy as np
import pytest

from ejps.jigsaw.domain import (
    Arrangement,
    Direction,
    EdgeAxis,
    Piece,
    PuzzleLayout,
    assemble_image,
    flatten_image,
)
from ejps.jigsaw.fitness import EdgeCostTable


def smooth64():
    """The 64x64 synthetic image also used for the upstream golden numbers."""
    h = w = 64
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


class TableLookup:
    """Plain callable cost lookup (no arrangement_cost): forces the generic path."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def __call__(self, ids, axis):
        self.calls += 1
        return self.table.cost(ids, axis)


@pytest.fixture(scope="module")
def puzzle():
    pieces, layout = flatten_image(smooth64(), 16, indexed=True)
    table = EdgeCostTable()
    table.analyze(pieces)
    return pieces, layout, table


def ids(arrangement):
    return [p.identifier for p in arrangement.pieces]


def fresh_score(arrangement, table):
    return Arrangement(list(arrangement.pieces), arrangement.layout).score(table)


# --------------------------------------------------------------------------- #
def test_direction_and_axis_enums():
    assert [d.value for d in Direction] == ["top", "right", "bottom", "left"]
    assert Direction.TOP.opposite is Direction.BOTTOM
    assert Direction.BOTTOM.opposite is Direction.TOP
    assert Direction.LEFT.opposite is Direction.RIGHT
    assert Direction.RIGHT.opposite is Direction.LEFT
    assert Direction("right") is Direction.RIGHT
    assert Direction.RIGHT == "right"
    assert EdgeAxis.HORIZONTAL == "horizontal"
    assert EdgeAxis.VERTICAL == "vertical"


def test_layout():
    layout = PuzzleLayout(3, 5, 16)
    assert layout.piece_count == 15
    for bad in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 2, 2)):
        with pytest.raises(ValueError):
            PuzzleLayout(*bad)
    assert PuzzleLayout(1, 1, 1).piece_count == 1


def test_piece_validation_and_copy():
    data = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    piece = Piece(data, 7)
    data[0, 0, 0] = 99
    assert piece.image[0, 0, 0] == 0  # copied
    assert piece.size == 2
    assert piece.shape == (2, 2, 3)
    assert piece[1, 1, 2] == 11
    assert Piece(np.zeros((3, 3), np.uint8), 0).shape == (3, 3)
    assert Piece(np.zeros((3, 3, 1), np.uint8), 0).shape == (3, 3, 1)
    with pytest.raises(ValueError):
        Piece(np.zeros(4, np.uint8), 0)
    with pytest.raises(ValueError):
        Piece(np.zeros((2, 2, 2), np.uint8), 0)
    with pytest.raises(ValueError):
        Piece(np.zeros((2, 3), np.uint8), 0)
    with pytest.raises(ValueError):
        Piece(np.zeros((2, 2), np.float32), 0)


def test_flatten_and_assemble_roundtrip():
    img = np.arange(6 * 9 * 3, dtype=np.uint8).reshape(6, 9, 3)
    pieces, layout = flatten_image(img, 3, indexed=True)
    assert layout == PuzzleLayout(2, 3, 3)
    assert [p.identifier for p in pieces] == list(range(6))
    np.testing.assert_array_equal(pieces[4].image, img[3:6, 3:6])  # row-major
    np.testing.assert_array_equal(assemble_image(pieces, layout), img)
    raw, _ = flatten_image(img, 3)
    assert isinstance(raw[0], np.ndarray)
    np.testing.assert_array_equal(raw[5], img[3:6, 6:9])
    np.testing.assert_array_equal(assemble_image(raw, layout), img)
    # incomplete edges are cropped
    cropped, layout2 = flatten_image(img, 4)
    assert layout2 == PuzzleLayout(1, 2, 4)
    np.testing.assert_array_equal(assemble_image(cropped, layout2), img[:4, :8])
    gray, layout3 = flatten_image(img[..., 0], 3)
    assert gray[0].shape == (3, 3)
    assert assemble_image(gray, layout3).dtype == np.uint8
    with pytest.raises(ValueError):
        flatten_image(img, 0)
    with pytest.raises(ValueError):
        flatten_image(img, 7)
    with pytest.raises(ValueError):
        flatten_image(np.zeros((6, 6, 2), np.uint8), 3)
    with pytest.raises(ValueError):
        flatten_image(np.zeros(6, np.uint8), 3)
    with pytest.raises(ValueError):
        assemble_image(pieces[:5], layout)


def test_arrangement_validation_and_access(puzzle):
    pieces, layout, _ = puzzle
    arr = Arrangement(pieces, layout)
    assert arr.pieces is not pieces
    assert all(a is b for a, b in zip(arr.pieces, pieces, strict=True))
    assert [p.identifier for p in arr[1]] == [4, 5, 6, 7]
    assert [p.identifier for p in arr[3]] == [12, 13, 14, 15]
    assert arr.piece_by_id(9) is pieces[9]
    with pytest.raises(ValueError):
        Arrangement(pieces[:15], layout)
    with pytest.raises(ValueError):
        Arrangement([pieces[0]] * 16, layout)
    shuffled = Arrangement.random(pieces, layout, random.Random(0))
    expected = list(range(16))
    random.Random(0).shuffle(expected)
    assert ids(shuffled) == expected
    assert [p.identifier for p in pieces] == list(range(16))  # input untouched


def test_score_formula_and_generic_lookup(puzzle):
    pieces, layout, table = puzzle
    arr = Arrangement(pieces, layout)
    # upstream identity score of this puzzle
    assert arr.score(table) == pytest.approx(0.9476862406767522, rel=1e-12)
    total = table.arrangement_cost(arr)
    assert arr.score(table) == pytest.approx(math.exp(-(total / 24) / 0.05), rel=1e-12)
    lookup = TableLookup(table)
    generic = Arrangement(pieces, layout)
    assert generic.score(lookup) == pytest.approx(arr.score(table), rel=1e-9)
    assert lookup.calls == 24  # 4*3 horizontal + 3*4 vertical seams
    generic.score(lookup)
    assert lookup.calls == 24  # cached


def test_score_with_constant_lookup_counts_seams():
    layout = PuzzleLayout(2, 3, 1)
    pieces = [Piece(np.zeros((1, 1), np.uint8), i) for i in range(6)]
    seen = []

    def lookup(pair, axis):
        seen.append((pair, axis))
        return 0.1 if axis is EdgeAxis.HORIZONTAL else 0.2

    arr = Arrangement(pieces, layout)
    # 4 horizontal * 0.1 + 3 vertical * 0.2 = 1.0 over 7 seams
    assert arr.score(lookup) == pytest.approx(math.exp(-(1.0 / 7) / 0.05), rel=1e-12)
    assert seen[:4] == [
        ((0, 1), EdgeAxis.HORIZONTAL),
        ((1, 2), EdgeAxis.HORIZONTAL),
        ((3, 4), EdgeAxis.HORIZONTAL),
        ((4, 5), EdgeAxis.HORIZONTAL),
    ]
    assert seen[4:] == [
        ((0, 3), EdgeAxis.VERTICAL),
        ((1, 4), EdgeAxis.VERTICAL),
        ((2, 5), EdgeAxis.VERTICAL),
    ]
    single = Arrangement([pieces[0]], PuzzleLayout(1, 1, 1))
    assert single.score(lookup) == 1.0  # no seams -> mean cost 0


def test_score_wraps_lookup_errors():
    layout = PuzzleLayout(1, 2, 1)
    pieces = [Piece(np.zeros((1, 1), np.uint8), i) for i in range(2)]

    class Bad:
        def __call__(self, ids, axis):
            return 0.0

        def arrangement_cost(self, arrangement):
            raise TypeError("boom")

    with pytest.raises(ValueError):
        Arrangement(pieces, layout).score(Bad())


def test_swap_updates_state_and_incremental_score(puzzle):
    pieces, layout, table = puzzle
    arr = Arrangement(pieces, layout)
    arr.score(table)
    arr.swap(0, 5, cost_lookup=table)
    assert ids(arr) == [5, 1, 2, 3, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert arr._cached_score == pytest.approx(0.6035841859458839, rel=1e-9)  # upstream
    assert arr._cached_score == pytest.approx(fresh_score(arr, table), rel=1e-9)
    assert arr.piece_by_id(0) is pieces[0] and arr.pieces[5] is pieces[0]
    assert arr.piece_by_id(5) is pieces[5] and arr.pieces[0] is pieces[5]
    arr.swap(3, 3, cost_lookup=table)  # no-op
    assert ids(arr)[3] == 3
    # without a cost lookup the cache is invalidated
    arr.swap(0, 5)
    assert arr._cached_score is None and arr._cached_total_cost is None
    assert ids(arr) == list(range(16))
    assert arr.score(table) == pytest.approx(0.9476862406767522, rel=1e-12)


def test_swap_with_generic_lookup_matches_table(puzzle):
    pieces, layout, table = puzzle
    arr = Arrangement(pieces, layout)
    lookup = TableLookup(table)
    arr.score(lookup)
    arr.swap(1, 14, cost_lookup=lookup)
    assert arr._cached_score == pytest.approx(fresh_score(arr, table), rel=1e-9)


def test_relocate_and_swap_blocks(puzzle):
    pieces, layout, table = puzzle
    arr = Arrangement(pieces, layout)
    arr.score(table)
    arr.swap(0, 5, cost_lookup=table)
    arr.relocate(2, 11, cost_lookup=table)
    assert ids(arr) == [5, 1, 3, 4, 0, 6, 7, 8, 9, 10, 11, 2, 12, 13, 14, 15]
    assert arr._cached_score == pytest.approx(0.29932689784134364, rel=1e-9)  # upstream
    assert all(arr.piece_by_id(i).identifier == i for i in range(16))
    assert [arr._piece_mapping[i] for i in (3, 2, 11)] == [2, 11, 10]
    arr.swap_blocks(1, 8, 3, cost_lookup=table)
    assert ids(arr) == [5, 9, 10, 11, 0, 6, 7, 8, 1, 3, 4, 2, 12, 13, 14, 15]
    assert arr._cached_score == pytest.approx(0.14587359364795047, rel=1e-9)  # upstream
    assert arr._cached_score == pytest.approx(fresh_score(arr, table), rel=1e-9)
    assert all(arr.pieces[arr._piece_mapping[i]].identifier == i for i in range(16))
    # reversed block order is the same exchange
    arr.swap_blocks(8, 1, 3, cost_lookup=table)
    assert ids(arr) == [5, 1, 3, 4, 0, 6, 7, 8, 9, 10, 11, 2, 12, 13, 14, 15]
    assert arr._cached_score == pytest.approx(0.29932689784134364, rel=1e-9)
    arr.relocate(11, 2, cost_lookup=table)  # inverse relocate
    assert ids(arr) == [5, 1, 2, 3, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert arr._cached_score == pytest.approx(0.6035841859458839, rel=1e-9)
    assert all(arr.pieces[arr._piece_mapping[i]].identifier == i for i in range(16))


def test_relocate_and_blocks_without_lookup_invalidate(puzzle):
    pieces, layout, table = puzzle
    arr = Arrangement(pieces, layout)
    arr.score(table)
    arr.relocate(0, 3)
    assert ids(arr)[:5] == [1, 2, 3, 0, 4]
    assert arr._cached_score is None
    arr.score(table)
    arr.swap_blocks(0, 4, 2)
    assert ids(arr)[:6] == [4, 5, 3, 0, 1, 2]
    assert arr._cached_total_cost is None
    arr.relocate(5, 5)
    arr.swap_blocks(2, 2, 1)
    assert ids(arr)[:6] == [4, 5, 3, 0, 1, 2]
    # no cached score -> moves with a lookup still leave the cache empty
    arr.relocate(0, 1, cost_lookup=table)
    assert arr._cached_score is None


def test_move_errors(puzzle):
    pieces, layout, _ = puzzle
    arr = Arrangement(pieces, layout)
    for args in ((-1, 3), (0, 16), (16, 0)):
        with pytest.raises(IndexError):
            arr.relocate(*args)
    with pytest.raises(ValueError):
        arr.swap_blocks(0, 4, 0)
    for args in ((0, 14, 3), (-1, 4, 1), (16, 0, 1), (13, 0, 4)):
        with pytest.raises(IndexError):
            arr.swap_blocks(*args)
    with pytest.raises(ValueError):
        arr.swap_blocks(0, 2, 3)  # overlap
    arr.swap_blocks(0, 3, 3)  # adjacent is fine
    assert ids(arr)[:6] == [3, 4, 5, 0, 1, 2]


def test_replace_positions(puzzle):
    pieces, layout, table = puzzle
    arr = Arrangement(pieces, layout)
    arr.score(table)
    window = (5, 6, 9, 10)
    new = (pieces[10], pieces[9], pieces[6], pieces[5])
    arr.replace_positions(window, new, cost_lookup=table)
    assert ids(arr)[4:12] == [4, 10, 9, 7, 8, 6, 5, 11]
    assert arr._cached_score == pytest.approx(fresh_score(arr, table), rel=1e-9)
    assert arr.piece_by_id(10) is pieces[10] and arr._piece_mapping[10] == 5
    # identical replacement is a no-op that keeps the cache
    cached = arr._cached_score
    arr.replace_positions(window, tuple(arr.pieces[i] for i in window), cost_lookup=table)
    assert arr._cached_score == cached
    arr.replace_positions(window, (pieces[5], pieces[6], pieces[9], pieces[10]))
    assert ids(arr) == list(range(16)) and arr._cached_score is None
    with pytest.raises(ValueError):
        arr.replace_positions((), ())
    with pytest.raises(ValueError):
        arr.replace_positions((1, 2), (pieces[1],))
    with pytest.raises(ValueError):
        arr.replace_positions((1, 1), (pieces[1], pieces[1]))
    with pytest.raises(IndexError):
        arr.replace_positions((1, 16), (pieces[1], pieces[2]))
    with pytest.raises(IndexError):
        arr.replace_positions((-1, 1), (pieces[1], pieces[2]))
    with pytest.raises(ValueError):
        arr.replace_positions((1, 2), (pieces[1], pieces[3]))


def test_edge_cache(puzzle):
    pieces, layout, _ = puzzle
    arr = Arrangement(pieces, layout)
    assert arr.edge(5, Direction.TOP) == 1
    assert arr.edge(5, Direction.RIGHT) == 6
    assert arr.edge(5, Direction.BOTTOM) == 9
    assert arr.edge(5, Direction.LEFT) == 4
    assert arr.edge(0, Direction.TOP) is None
    assert arr.edge(0, Direction.LEFT) is None
    assert arr.edge(3, Direction.RIGHT) is None
    assert arr.edge(2, Direction.RIGHT) == 3
    assert arr.edge(12, Direction.BOTTOM) is None
    assert arr.edge(8, Direction.BOTTOM) == 12
    assert arr.edge(15, Direction.TOP) == 11
    cache = arr._edge_cache
    arr._build_edge_cache()
    assert arr._edge_cache is cache  # built once
    arr.swap(5, 0)
    assert arr._edge_cache is None
    assert arr.edge(0, Direction.RIGHT) == 6
    assert arr.edge(5, Direction.RIGHT) == 1


def test_dataclass_contracts():
    import dataclasses

    layout = PuzzleLayout(2, 2, 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.rows = 3
    assert not hasattr(layout, "__dict__")
    assert not hasattr(Piece(np.zeros((2, 2), np.uint8), 0), "__dict__")
    pieces = [Piece(np.zeros((4, 4), np.uint8), i) for i in range(4)]
    arr = Arrangement(pieces, layout)
    assert not hasattr(arr, "__dict__")
    text = repr(arr)
    assert "_cached_score" not in text and "_piece_mapping" not in text
    assert "_edge_cache" not in text and "_cached_total_cost" not in text
    with pytest.raises(TypeError):
        Arrangement(pieces, layout, {})  # internal fields are not constructor arguments


def test_moves_at_the_ends_of_the_permutation(puzzle):
    pieces, layout, table = puzzle
    arr = Arrangement(pieces, layout)
    arr.score(table)
    arr.relocate(5, 0, cost_lookup=table)
    assert ids(arr)[:7] == [5, 0, 1, 2, 3, 4, 6]
    arr.relocate(0, 15, cost_lookup=table)
    assert ids(arr)[-3:] == [14, 15, 5]
    assert arr._cached_score == pytest.approx(fresh_score(arr, table), rel=1e-9)
    assert all(arr.pieces[arr._piece_mapping[i]].identifier == i for i in range(16))
    arr = Arrangement(pieces, layout)
    arr.score(table)
    arr.swap_blocks(0, 13, 3, cost_lookup=table)  # second block ends at the last piece
    assert ids(arr) == [13, 14, 15, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2]
    assert arr._cached_score == pytest.approx(fresh_score(arr, table), rel=1e-9)
    arr.swap_blocks(13, 0, 3, cost_lookup=table)  # reversed arguments, first at the end
    assert ids(arr) == list(range(16))
    arr.swap_blocks(4, 0, 2)
    assert ids(arr)[:6] == [4, 5, 2, 3, 0, 1]
    arr = Arrangement(pieces, layout)
    arr.replace_positions((0, 15), (pieces[15], pieces[0]))
    assert ids(arr)[0] == 15 and ids(arr)[15] == 0


def test_flatten_single_row_column_and_one_pixel_pieces():
    img = np.arange(6 * 3, dtype=np.uint8).reshape(6, 3)
    pieces, layout = flatten_image(img, 3, indexed=True)
    assert layout == PuzzleLayout(2, 1, 3)
    np.testing.assert_array_equal(pieces[1].image, img[3:6])
    pieces, layout = flatten_image(img.T.copy(), 3, indexed=True)
    assert layout == PuzzleLayout(1, 2, 3)
    pixels, layout = flatten_image(img, 1, indexed=True)
    assert layout == PuzzleLayout(6, 3, 1)
    assert [int(p.image[0, 0]) for p in pixels] == list(range(18))
