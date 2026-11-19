"""Tests for ejps/jigsaw/fitness.py (gaps EdgeCostTable + metrics, ported).

Numbers marked "upstream" were produced by the original gaps (commit 5aa0a0d,
Python 3.12) on the same inputs.
"""

import numpy as np
import pytest

from ejps.jigsaw.domain import (
    Arrangement,
    Direction,
    EdgeAxis,
    Piece,
    PuzzleLayout,
    flatten_image,
)
from ejps.jigsaw.fitness import (
    ArrangementMetrics,
    EdgeCostTable,
    evaluate_arrangement,
    pairwise_dissimilarity,
)

H, V = EdgeAxis.HORIZONTAL, EdgeAxis.VERTICAL

# upstream EdgeCostTable(smooth64 / 16).confidence(i, d) for i in 0..15, d in T,R,B,L
UPSTREAM_CONFIDENCE = [
    0.3960557504303941, 0.8439498213456661, 0.9810233403803487, 0.10716924633335147,
    0.19146633197298743, 0.9745322469519878, 0.9673996491983986, 0.9176426883568346,
    0.06989894804344017, 0.9489362408170319, 0.9557666887053253, 0.9707955101955367,
    0.33578662469942844, 0.09534425097651647, 0.9151840483600904, 0.923592571973848,
    0.9853756772903353, 0.8518780530449213, 0.9696851730977069, 0.12011825530653614,
    0.9757755111647057, 0.9693881148857075, 0.9174821938231594, 0.8940009080595126,
    0.9508109730688755, 0.9410065659410588, 0.8885040724902342, 0.9492976722141321,
    0.9237291195677567, 0.022384869532800635, 0.9387050065789079, 0.9247457204034225,
    0.9701892568864314, 0.9817129665404528, 0.9511830987369054, 0.14992271036789012,
    0.916086861670036, 0.9102358271451583, 0.9727434234900633, 0.9831885781404892,
    0.8900040982448421, 0.8903885762852765, 0.9926242040605456, 0.8869238702360522,
    0.9128924172033701, 0.3910517260049813, 0.9889120771971316, 0.8923532511834671,
    0.9495556646561205, 0.9643619388186961, 0.29435174334626835, 0.12270844332858526,
    0.9683070979023553, 0.9052192768804083, 0.13904098563744885, 0.9620389725829435,
    0.9902621234709399, 0.9086323591894934, 0.21743674375236824, 0.9247594233091048,
    0.9816691399353055, 0.2286301012671907, 0.21838294043717602, 0.9069647964216364,
]  # fmt: skip


def smooth64():
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


@pytest.fixture(scope="module")
def puzzle():
    pieces, layout = flatten_image(smooth64(), 16, indexed=True)
    table = EdgeCostTable()
    table.analyze(pieces)
    return pieces, layout, table


def test_costs_match_pairwise_and_upstream_total(puzzle):
    pieces, _, table = puzzle
    np.testing.assert_allclose(table._costs[:, :, 0], pairwise_dissimilarity(pieces, H), rtol=1e-6)
    np.testing.assert_allclose(table._costs[:, :, 1], pairwise_dissimilarity(pieces, V), rtol=1e-6)
    finite = table._costs[np.isfinite(table._costs)]
    assert finite.size == 16 * 15 * 2
    assert float(np.sum(finite, dtype=np.float64)) == pytest.approx(56.15612792852335, rel=1e-9)
    assert table.cost((3, 7), V) == pytest.approx(float(table._costs[3, 7, 1]), rel=0)
    assert table((3, 7), H) == pytest.approx(float(table._costs[3, 7, 0]), rel=0)
    assert table.cost((3, 7), H) != table.cost((7, 3), H)


def test_cost_errors(puzzle):
    _, _, table = puzzle
    with pytest.raises(KeyError):
        table.cost((2, 2), H)
    with pytest.raises(KeyError):
        table.cost((2, 99), H)
    with pytest.raises(KeyError):
        table.cost((99, 2), V)
    with pytest.raises(KeyError):
        table.cost((1, 2), "diagonal")
    with pytest.raises(KeyError):
        table.cost((1,), H)


def test_matches_are_sorted_by_directed_cost(puzzle):
    _, _, table = puzzle
    costs = table._costs
    for piece in (0, 5, 15):
        right = table.matches(piece, Direction.RIGHT)
        assert len(right) == 15 and piece not in [m for m, _ in right]
        assert [c for _, c in right] == sorted(c for _, c in right)
        assert right[0][1] == pytest.approx(float(np.min(costs[piece, :, 0])), rel=0)
        left = table.matches(piece, Direction.LEFT)
        assert left[0][1] == pytest.approx(float(np.min(costs[:, piece, 0])), rel=0)
        bottom = table.matches(piece, Direction.BOTTOM)
        assert bottom[0][1] == pytest.approx(float(np.min(costs[piece, :, 1])), rel=0)
        top = table.matches(piece, Direction.TOP)
        assert top[0][1] == pytest.approx(float(np.min(costs[:, piece, 1])), rel=0)
        assert table.best_match(piece, Direction.RIGHT) == right[0][0]
        assert table.best_cost(piece, Direction.TOP) == top[0][1]
    # on this smooth image the true neighbours are the best matches
    assert table.best_match(5, Direction.RIGHT) == 6
    assert table.best_match(5, Direction.BOTTOM) == 9
    assert table.best_match(6, Direction.LEFT) == 5
    assert table.best_match(9, Direction.TOP) == 5
    for direction, match in zip(Direction, table._best_matches[5], strict=True):
        assert direction is match


def test_confidence_formula_and_upstream_golden(puzzle):
    _, _, table = puzzle
    got = [table.confidence(i, d) for i in range(16) for d in Direction]
    np.testing.assert_allclose(got, UPSTREAM_CONFIDENCE, rtol=1e-9)
    m = table.matches(5, Direction.RIGHT)
    assert table.confidence(5, Direction.RIGHT) == pytest.approx(
        min(1.0, max(0.0, m[1][1] - m[0][1]) / max(abs(m[1][1]), 1e-6)), rel=1e-12
    )


def tiny_table(values):
    """Table over three 1x1 pieces whose matches are injected."""
    table = EdgeCostTable()
    table.analyze([Piece(np.full((1, 1), v, np.uint8), i) for i, v in enumerate(values)])
    return table


def test_confidence_edge_cases():
    table = tiny_table([0, 10, 20])
    table._best_matches[0][Direction.RIGHT] = [(1, 0.5), (2, 1.0)]
    assert table.confidence(0, Direction.RIGHT) == pytest.approx(0.5)
    table._best_matches[0][Direction.RIGHT] = [(1, 0.5), (2, 0.5)]
    assert table.confidence(0, Direction.RIGHT) == 0.0
    table._best_matches[0][Direction.RIGHT] = [(1, 0.0), (2, 0.0)]
    assert table.confidence(0, Direction.RIGHT) == 0.0
    table._best_matches[0][Direction.RIGHT] = [(1, 0.0), (2, 1e-7)]
    assert table.confidence(0, Direction.RIGHT) == pytest.approx(0.1)  # floor 1e-6
    table._best_matches[0][Direction.RIGHT] = [(1, 0.0), (2, 3.0)]
    assert table.confidence(0, Direction.RIGHT) == 1.0
    table._best_matches[0][Direction.RIGHT] = [(1, 2.0), (2, 1.0)]  # unsorted -> clipped
    assert table.confidence(0, Direction.RIGHT) == 0.0
    table._best_matches[0][Direction.RIGHT] = [(1, 0.5)]
    assert table.confidence(0, Direction.RIGHT) == 1.0
    table._best_matches[0][Direction.RIGHT] = []
    assert table.confidence(0, Direction.RIGHT) == 0.0
    with pytest.raises(ValueError):
        table.best_match(0, Direction.RIGHT)
    with pytest.raises(ValueError):
        table.best_cost(0, Direction.RIGHT)


def test_arrangement_cost_equals_seam_sum(puzzle):
    pieces, layout, table = puzzle
    rng = np.random.default_rng(0)
    order = rng.permutation(16)
    arr = Arrangement([pieces[i] for i in order], layout)
    grid = order.reshape(4, 4)
    expected = sum(table.cost((grid[r, c], grid[r, c + 1]), H) for r in range(4) for c in range(3))
    expected += sum(table.cost((grid[r, c], grid[r + 1, c]), V) for r in range(3) for c in range(4))
    assert table.arrangement_cost(arr) == pytest.approx(expected, rel=1e-9)
    # one row / one column layouts use only one axis
    row_pieces, row_layout = flatten_image(smooth64()[:16], 16, indexed=True)
    row_table = EdgeCostTable()
    row_table.analyze(row_pieces)
    row_arr = Arrangement(row_pieces, row_layout)
    assert row_table.arrangement_cost(row_arr) == pytest.approx(
        sum(row_table.cost((c, c + 1), H) for c in range(3)), rel=1e-9
    )
    col_pieces, col_layout = flatten_image(smooth64()[:, :16], 16, indexed=True)
    col_table = EdgeCostTable()
    col_table.analyze(col_pieces)
    assert col_table.arrangement_cost(Arrangement(col_pieces, col_layout)) == pytest.approx(
        sum(col_table.cost((r, r + 1), V) for r in range(3)), rel=1e-9
    )
    foreign = Arrangement([Piece(p.image, p.identifier + 100) for p in pieces], layout)
    with pytest.raises(KeyError):
        table.arrangement_cost(foreign)


def test_analyze_progress_and_validation():
    pieces, _ = flatten_image(smooth64()[:32, :32], 8, indexed=True)
    events = []
    table = EdgeCostTable()
    table.analyze(pieces, progress=lambda *e: events.append(e))
    assert events == [("analysis", 15, 15)]  # one block of 32 rows
    ref = EdgeCostTable()
    ref.analyze(pieces)
    np.testing.assert_array_equal(table._costs, ref._costs)
    with pytest.raises(ValueError):
        EdgeCostTable().analyze([pieces[0], pieces[0]])
    empty = EdgeCostTable()
    empty.analyze([])
    assert empty._costs.shape == (0, 0, 2)
    # re-analysing replaces the previous state
    table.analyze(pieces[:2])
    assert table._costs.shape == (2, 2, 2)
    assert len(table.matches(0, Direction.RIGHT)) == 1


def test_evaluate_arrangement():
    layout = PuzzleLayout(2, 3, 1)
    pieces = [Piece(np.zeros((1, 1), np.uint8), i) for i in range(6)]
    perfect = evaluate_arrangement(Arrangement(pieces, layout), range(6))
    assert perfect == ArrangementMetrics(1.0, 1.0, 1.0, 1.0, True)
    # positions 0,1,2,4 correct; 3 and 5 swapped
    arr = Arrangement([pieces[i] for i in (0, 1, 2, 5, 4, 3)], layout)
    m = evaluate_arrangement(arr, range(6))
    assert m.position_accuracy == pytest.approx(4 / 6)
    assert m.horizontal_adjacency_accuracy == pytest.approx(2 / 4)  # (0,1),(1,2)
    assert m.vertical_adjacency_accuracy == pytest.approx(1 / 3)  # (1,4)
    assert m.adjacency_accuracy == pytest.approx(3 / 7)
    assert m.exact is False
    # a single row has no vertical seams -> vertical accuracy 1.0
    row = Arrangement(pieces[:3][::-1], PuzzleLayout(1, 3, 1))
    m = evaluate_arrangement(row, [0, 1, 2])
    assert m.position_accuracy == pytest.approx(1 / 3)
    assert m.horizontal_adjacency_accuracy == 0.0
    assert m.vertical_adjacency_accuracy == 1.0
    assert m.adjacency_accuracy == 0.0
    single = evaluate_arrangement(Arrangement(pieces[:1], PuzzleLayout(1, 1, 1)), [0])
    assert single == ArrangementMetrics(1.0, 1.0, 1.0, 1.0, True)
    with pytest.raises(ValueError):
        evaluate_arrangement(arr, range(5))
    with pytest.raises(ValueError):
        evaluate_arrangement(arr, [0, 0, 1, 2, 3, 4])


def test_two_column_and_two_row_layouts_use_both_axes():
    pieces, layout = flatten_image(smooth64()[:32, :32], 16, indexed=True)  # 2x2
    table = EdgeCostTable()
    table.analyze(pieces)
    arr = Arrangement(pieces, layout)
    expected = (
        table.cost((0, 1), H)
        + table.cost((2, 3), H)
        + table.cost((0, 2), V)
        + table.cost((1, 3), V)
    )
    assert table.arrangement_cost(arr) == pytest.approx(expected, rel=1e-9)
    pieces, layout = flatten_image(smooth64()[:48, :32], 16, indexed=True)  # 3x2
    table.analyze(pieces)
    arr = Arrangement(pieces, layout)
    expected = sum(table.cost((2 * r, 2 * r + 1), H) for r in range(3))
    expected += sum(table.cost((i, i + 2), V) for i in range(4))
    assert table.arrangement_cost(arr) == pytest.approx(expected, rel=1e-9)


def test_metrics_are_frozen_and_evaluated_on_a_3x4_grid():
    import dataclasses

    m = ArrangementMetrics(1.0, 1.0, 1.0, 1.0, True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.exact = False
    layout = PuzzleLayout(3, 4, 1)
    pieces = [Piece(np.zeros((1, 1), np.uint8), i) for i in range(12)]
    # swap 6 <-> 7 and 9 <-> 11; positions 6, 7, 9, 11 wrong
    order = [0, 1, 2, 3, 4, 5, 7, 6, 8, 11, 10, 9]
    m = evaluate_arrangement(Arrangement([pieces[i] for i in order], layout), range(12))
    assert m.position_accuracy == pytest.approx(8 / 12)
    # horizontal seams (9): row0 3 ok; row1 (4,5) ok; row2 none (8|9, 9|10, 10|11 touch 9 or 11)
    assert m.horizontal_adjacency_accuracy == pytest.approx(4 / 9)
    # vertical seams (8): 0-4, 1-5 ok; 2-6, 3-7 bad; 4-8 ok; 5-9 bad; 6-10, 7-11 bad
    assert m.vertical_adjacency_accuracy == pytest.approx(3 / 8)
    assert m.adjacency_accuracy == pytest.approx(7 / 17)
    assert m.exact is False
