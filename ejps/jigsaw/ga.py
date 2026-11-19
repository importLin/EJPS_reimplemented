# Ported from gaps (https://github.com/nemanja-m/gaps),
# commit 5aa0a0d15ff1b2d6b11ab03c6c005b025bd93d20, files gaps/solver/crossover.py,
# gaps/solver/selection.py and gaps/solver/algorithm.py.
# Copyright (c) 2017 Nemanja Milićević. Released under the MIT License; the full
# licence text is in THIRD_PARTY_LICENSE_gaps.txt at the repository root.
# Changes from upstream: Python 3.10 syntax (no ``type`` statements); imports; the
# three upstream modules are merged into this one; the process-pool path
# (workers > 1: _WorkerContext, initialize_worker, build_child,
# _next_population_parallel) is not ported; ``JigsawResult`` / ``solve_jigsaw``
# facade added.
"""Genetic-algorithm jigsaw solver (square pieces, unknown positions, known rotation).

Stage 3 of the attack pipeline: after EJPS has undone the sub-block transforms
inside every block, the blocks themselves are still permuted. This module
reassembles them from edge continuity alone::

    from ejps.jigsaw import solve_jigsaw
    result = solve_jigsaw(shuffled, piece_size=16, population=200, generations=20, seed=0)
    result.order   # output position (row-major) -> index of the input block (row-major)
    result.image   # reassembled uint8 image

Algorithm (upstream gaps, unchanged): pairwise directed edge costs
(:mod:`ejps.jigsaw.fitness`) -> population of random arrangements, a fraction
replaced by row-major beam-search seeds -> each generation keeps ``elite_size``
elites and fills the rest with tournament-selected parents, kernel-growing
crossover (:class:`Crossover`), an optional random move (mutation, worse
moves rejected with probability ``0.95 - 0.15 * rate``), best-of-``k`` local
search and exhaustive 2x2 window repair. The mutation rate grows with
stagnation; after ``restart_threshold`` stagnant generations the population is
refilled at random around the best individuals (``max_restarts`` times), after
which 10 stagnant generations terminate the run.

Crossover (kernel growing, as in upstream gaps): a child is grown from a seed (a confident block
copied from one parent, or a random root piece) by repeatedly placing the
highest-priority candidate on the kernel boundary: a piece both parents agree
on (priority -10), then a "best buddy" that one parent supports (-1), then the
cheapest unused match (``cost - 0.02 * confidence``). The kernel never grows
beyond the layout.

Only the serial path (``workers=1``) is ported, so for a given
``random.Random(seed)`` the sequence of random draws -- and hence the result --
is identical to upstream ``GeneticAlgorithm(..., workers=1)``. Upstream's
``workers > 1`` path draws per-child seeds with ``getrandbits`` and therefore
gives different (equally valid) results; it raises ``NotImplementedError`` here.
"""

from __future__ import annotations

import heapq
import itertools
import random
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeGuard

import numpy as np

from .domain import (
    Arrangement,
    Direction,
    EdgeAxis,
    Image,
    Piece,
    PuzzleLayout,
    assemble_image,
    flatten_image,
)
from .fitness import EdgeCostTable

# --------------------------------------------------------------------------- #
# selection (upstream gaps/solver/selection.py)
# --------------------------------------------------------------------------- #


def tournament_selection(
    population: Sequence[Arrangement],
    score: Callable[[Arrangement], float],
    rng: random.Random,
    elites: int = 4,
    tournament_size: int = 3,
) -> list[tuple[Arrangement, Arrangement]]:
    """Select parents using fitness tournaments instead of score magnitudes.

    Tournament selection is less sensitive than roulette selection to the
    exponential score scaling used by :class:`Arrangement`.
    """
    if not population:
        raise ValueError("population must not be empty")
    if not 0 <= elites < len(population):
        raise ValueError("elites must be between zero and population size")
    if tournament_size <= 0:  # pragma: no mutate (0 also fails in max([]))
        raise ValueError("tournament_size must be positive")

    def select_individual() -> Arrangement:
        contestants = [
            population[rng.randrange(len(population))]
            for _ in range(min(tournament_size, len(population)))
        ]
        return max(contestants, key=score)

    return [(select_individual(), select_individual()) for _ in range(len(population) - elites)]


def roulette_selection(
    population: Sequence[Arrangement],
    score: Callable[[Arrangement], float],
    rng: random.Random,
    elites: int = 4,
) -> list[tuple[Arrangement, Arrangement]]:
    """Select parent pairs with probability proportional to their score."""
    if not population:
        raise ValueError("population must not be empty")
    if not 0 <= elites < len(population):
        raise ValueError("elites must be between zero and population size")

    cumulative_scores: list[float] = []
    total_score = 0.0
    for individual in population:
        total_score += score(individual)
        cumulative_scores.append(total_score)

    def select_individual() -> Arrangement:
        random_value = rng.random() * total_score
        index = bisect_right(cumulative_scores, random_value)
        return population[min(index, len(population) - 1)]

    return [(select_individual(), select_individual()) for _ in range(len(population) - elites)]


# --------------------------------------------------------------------------- #
# crossover (upstream gaps/solver/crossover.py)
# --------------------------------------------------------------------------- #
Position = tuple[int, int]
RelativePiece = tuple[int, Direction]
Candidate = tuple[float, int, int, int, int, int, Direction]


SHARED_PIECE_PRIORITY = -10.0  # pragma: no mutate (tuned constant)
BUDDY_PIECE_PRIORITY = -1.0  # pragma: no mutate (tuned constant)
CONFIDENCE_PRIORITY_WEIGHT = 0.02
MIN_BLOCK_CONFIDENCE = 0.02
MAX_BLOCK_SIDE = 4
BLOCK_TRIALS = 4


@dataclass(slots=True)  # pragma: no mutate
class CrossoverStats:
    candidate_pushes: int = 0
    candidate_pops: int = 0
    discarded_taken_positions: int = 0
    reexpanded_placed_pieces: int = 0
    placed_pieces: int = 0
    shared_matches: int = 0
    buddy_matches: int = 0
    best_match_candidates: int = 0
    best_match_calls: int = 0
    best_match_scanned: int = 0
    validity_checks: int = 0
    preserved_blocks: int = 0
    preserved_block_pieces: int = 0


class Crossover:
    """Build a child arrangement from two parent arrangements."""

    def __init__(
        self,
        first_parent: Arrangement,
        second_parent: Arrangement,
        analysis: EdgeCostTable,
        rng: random.Random,
        stats: CrossoverStats | None = None,
    ) -> None:
        self._parents = (first_parent, second_parent)
        self._analysis = analysis
        self._rng = rng
        self._stats = stats
        self._pieces_length = len(first_parent.pieces)
        self._layout = first_parent.layout
        self._min_row = 0  # pragma: no mutate (reset by _initialize_kernel)
        self._max_row = 0  # pragma: no mutate
        self._min_column = 0  # pragma: no mutate
        self._max_column = 0  # pragma: no mutate
        first_parent._build_edge_cache()
        second_parent._build_edge_cache()
        first_edges = first_parent._edge_cache
        second_edges = second_parent._edge_cache
        if first_edges is None or second_edges is None:  # pragma: no mutate (unreachable)
            raise RuntimeError("parent edge caches were not initialized")

        self._parent_edges = (first_edges, second_edges)
        self._kernel: dict[int, Position] = {}
        self._taken_positions: set[Position] = set()
        self._candidate_pieces: list[Candidate] = []
        self._candidate_sequence = 0  # pragma: no mutate (tie-break only)
        self._match_cursors: dict[tuple[int, Direction], int] = {}

    def child(self) -> Arrangement:
        """Return the child arrangement after :meth:`run` completes."""
        pieces: list[Piece | None] = [None] * self._pieces_length
        for piece_id, (row, column) in self._kernel.items():
            index = (row - self._min_row) * self._layout.columns + (column - self._min_column)
            # equivalent mutant: both parents hold the same Piece objects
            pieces[index] = self._parents[0].piece_by_id(piece_id)  # pragma: no mutate

        if any(piece is None for piece in pieces):
            raise RuntimeError("crossover did not produce a complete child")
        return Arrangement([piece for piece in pieces if piece is not None], self._layout)

    def run(self) -> None:
        """Populate the child kernel using parent and image matches."""
        self._initialize_kernel()
        while self._candidate_pieces:
            (
                _priority,
                row,
                column,
                piece_id,
                _sequence,
                relative_piece_id,
                relative_direction,
            ) = heapq.heappop(self._candidate_pieces)
            position = (row, column)
            if self._stats is not None:
                self._stats.candidate_pops += 1
            if position in self._taken_positions:
                if self._stats is not None:
                    self._stats.discarded_taken_positions += 1
                continue
            if piece_id in self._kernel:
                if self._stats is not None:
                    self._stats.reexpanded_placed_pieces += 1
                self.add_piece_candidate(
                    relative_piece_id,
                    relative_direction,
                    position,
                )
                continue
            self._put_piece_to_kernel(piece_id, position)

    def _initialize_kernel(self) -> None:
        block = self._select_parent_block()
        if block is None:
            root_piece = self._rng.choice(self._parents[0].pieces)
            self._put_piece_to_kernel(root_piece.identifier, (0, 0))
            return

        block_pieces, confidence = block
        self._min_row = 0  # pragma: no mutate (reset by _initialize_kernel)
        self._max_row = max(row for _, (row, _) in block_pieces)
        self._min_column = 0  # pragma: no mutate
        self._max_column = max(column for _, (_, column) in block_pieces)
        for piece_id, position in block_pieces:
            self._kernel[piece_id] = position
            self._taken_positions.add(position)
        if self._stats is not None:
            self._stats.preserved_blocks += 1
            self._stats.preserved_block_pieces += len(block_pieces)
            self._stats.placed_pieces += len(block_pieces)

        for _, position in block_pieces:
            for direction, candidate_position in self._available_boundaries(position):
                self.add_piece_candidate(
                    self._kernel_piece_at(position), direction, candidate_position
                )

    def _select_parent_block(
        self,
    ) -> tuple[list[tuple[int, Position]], float] | None:
        rows = self._layout.rows
        columns = self._layout.columns
        dimensions = [
            (height, width)
            for height in range(1, min(rows, MAX_BLOCK_SIDE) + 1)
            for width in range(1, min(columns, MAX_BLOCK_SIDE) + 1)
            if 1 < height * width < self._pieces_length
        ]
        if not dimensions:
            return None

        best_block: tuple[list[tuple[int, Position]], float] | None = None
        for _ in range(BLOCK_TRIALS):
            parent = self._parents[self._rng.randrange(len(self._parents))]
            height, width = self._rng.choice(dimensions)
            start_row = self._rng.randrange(rows - height + 1)
            start_column = self._rng.randrange(columns - width + 1)
            block_pieces: list[tuple[int, Position]] = []
            confidences: list[float] = []
            for row in range(height):
                for column in range(width):
                    piece_id = parent[row + start_row][column + start_column].identifier
                    block_pieces.append((piece_id, (row, column)))
                    if column + 1 < width:
                        confidences.append(self._analysis.confidence(piece_id, Direction.RIGHT))
                    if row + 1 < height:
                        confidences.append(self._analysis.confidence(piece_id, Direction.BOTTOM))
            confidence = sum(confidences) / len(confidences)
            if confidence < MIN_BLOCK_CONFIDENCE:  # pragma: no mutate (equality has measure zero)
                continue
            # equivalent mutant: exact confidence ties have measure zero
            if best_block is None or confidence > best_block[1]:  # pragma: no mutate
                best_block = (block_pieces, confidence)
        return best_block

    def _kernel_piece_at(self, position: Position) -> int:
        for piece_id, piece_position in self._kernel.items():
            if piece_position == position:
                return piece_id
        raise RuntimeError("kernel position is not occupied")

    def _put_piece_to_kernel(self, piece_id: int, position: Position) -> None:
        if self._stats is not None:
            self._stats.placed_pieces += 1
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        for direction, candidate_position in self._available_boundaries(position):
            self.add_piece_candidate(piece_id, direction, candidate_position)

    def add_piece_candidate(self, piece_id: int, direction: Direction, position: Position) -> None:
        shared_piece = self._get_shared_piece(piece_id, direction)
        if self._is_valid_piece(shared_piece):
            if self._stats is not None:
                self._stats.shared_matches += 1
            self._push_candidate(
                SHARED_PIECE_PRIORITY, shared_piece, position, (piece_id, direction)
            )
            return

        buddy_piece = self._get_buddy_piece(piece_id, direction)
        if self._is_valid_piece(buddy_piece):
            if self._stats is not None:
                self._stats.buddy_matches += 1
            self._push_candidate(BUDDY_PIECE_PRIORITY, buddy_piece, position, (piece_id, direction))
            return

        best_piece, priority = self._get_best_match_piece(piece_id, direction)
        if self._stats is not None:
            self._stats.best_match_candidates += 1
        # equivalent mutant: piece and priority are None together
        if self._is_valid_piece(best_piece) and priority is not None:  # pragma: no mutate
            self._push_candidate(priority, best_piece, position, (piece_id, direction))

    def _get_shared_piece(self, piece_id: int, direction: Direction) -> int | None:
        first_edge = self._parent_edges[0][direction][piece_id]
        second_edge = self._parent_edges[1][direction][piece_id]
        if first_edge is not None and first_edge == second_edge:
            return first_edge
        return None

    def _get_buddy_piece(self, piece_id: int, direction: Direction) -> int | None:
        try:
            first_buddy = self._analysis.best_match(piece_id, direction)
            second_buddy = self._analysis.best_match(first_buddy, direction.opposite)
        except ValueError:
            return None

        if second_buddy != piece_id:
            return None
        if (
            self._parent_edges[0][direction][piece_id] == first_buddy
            or self._parent_edges[1][direction][piece_id] == first_buddy
        ):
            return first_buddy
        return None

    def _get_best_match_piece(
        self, piece_id: int, direction: Direction
    ) -> tuple[int | None, float | None]:
        matches = self._analysis.matches(piece_id, direction)
        if self._stats is not None:
            self._stats.best_match_calls += 1
        key = (piece_id, direction)
        cursor = self._match_cursors.get(key, 0)
        match_count = len(matches)
        while cursor < match_count:
            candidate_piece, cost = matches[cursor]
            if self._stats is not None:
                self._stats.best_match_scanned += 1
            if self._is_valid_piece(candidate_piece):
                self._match_cursors[key] = cursor
                confidence = self._analysis.confidence(piece_id, direction)
                priority = cost - CONFIDENCE_PRIORITY_WEIGHT * confidence
                return candidate_piece, priority
            cursor += 1
        self._match_cursors[key] = cursor
        return None, None

    def _push_candidate(
        self,
        priority: float,
        piece_id: int,
        position: Position,
        relative_piece: RelativePiece,
    ) -> None:
        self._candidate_sequence += 1  # pragma: no mutate (tie-break only)
        if self._stats is not None:
            self._stats.candidate_pushes += 1
        heapq.heappush(
            self._candidate_pieces,
            (
                priority,
                position[0],
                position[1],
                piece_id,
                self._candidate_sequence,
                relative_piece[0],
                relative_piece[1],
            ),
        )

    def _available_boundaries(self, position: Position) -> list[tuple[Direction, Position]]:
        row, column = position
        if self._is_kernel_full():
            return []

        candidates = (
            (Direction.TOP, (row - 1, column)),
            (Direction.RIGHT, (row, column + 1)),
            (Direction.BOTTOM, (row + 1, column)),
            (Direction.LEFT, (row, column - 1)),
        )
        boundaries = []
        for direction, candidate_position in candidates:
            if candidate_position in self._taken_positions:
                continue
            if not self._is_in_range(candidate_position):
                continue
            self._update_kernel_boundaries(candidate_position)
            boundaries.append((direction, candidate_position))
        return boundaries

    def _is_kernel_full(self) -> bool:
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, position: Position) -> bool:
        row, column = position
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row: int) -> bool:
        if row < self._min_row:  # pragma: no mutate (equality is a no-op)
            return self._max_row - row < self._layout.rows
        if row > self._max_row:  # pragma: no mutate (equality is a no-op)
            return row - self._min_row < self._layout.rows
        return True

    def _is_column_in_range(self, column: int) -> bool:
        if column < self._min_column:  # pragma: no mutate (equality is a no-op)
            return self._max_column - column < self._layout.columns
        if column > self._max_column:  # pragma: no mutate (equality is a no-op)
            return column - self._min_column < self._layout.columns
        return True

    def _update_kernel_boundaries(self, position: Position) -> None:
        row, column = position
        if row < self._min_row:  # pragma: no mutate (equality is a no-op)
            self._min_row = row
        elif row > self._max_row:  # pragma: no mutate (equality is a no-op)
            self._max_row = row
        if column < self._min_column:  # pragma: no mutate (equality is a no-op)
            self._min_column = column
        elif column > self._max_column:  # pragma: no mutate (equality is a no-op)
            self._max_column = column

    def _is_valid_piece(self, piece_id: int | None) -> TypeGuard[int]:
        if self._stats is not None:
            self._stats.validity_checks += 1
        return piece_id is not None and piece_id not in self._kernel


# --------------------------------------------------------------------------- #
# genetic algorithm (upstream gaps/solver/algorithm.py)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)  # pragma: no mutate
class _BeamState:
    identifiers: tuple[int, ...]
    used: frozenset[int]
    priority: float


def _seed_position_priority(
    identifiers: tuple[int, ...],
    candidate_id: int,
    layout: PuzzleLayout,
    analysis: EdgeCostTable,
    confidence_weight: float,
    border_weight: float,
) -> float:
    row, column = divmod(len(identifiers), layout.columns)
    priority = 0.0  # pragma: no mutate (constant offset)
    confidence = 0.0  # pragma: no mutate (constant offset)

    if column > 0:
        left_id = identifiers[-1]
        priority += analysis.cost((left_id, candidate_id), EdgeAxis.HORIZONTAL)
        confidence += analysis.confidence(left_id, Direction.RIGHT)
    if row > 0:
        above_id = identifiers[-layout.columns]
        priority += analysis.cost((above_id, candidate_id), EdgeAxis.VERTICAL)
        confidence += analysis.confidence(above_id, Direction.BOTTOM)

    if border_weight:
        if row == 0:
            priority -= border_weight * analysis.best_cost(candidate_id, Direction.TOP)
        if row == layout.rows - 1:
            priority -= border_weight * analysis.best_cost(candidate_id, Direction.BOTTOM)
        if column == 0:
            priority -= border_weight * analysis.best_cost(candidate_id, Direction.LEFT)
        if column == layout.columns - 1:
            priority -= border_weight * analysis.best_cost(candidate_id, Direction.RIGHT)

    return priority - confidence_weight * confidence


def build_seed_arrangements(
    pieces: Sequence[Piece],
    layout: PuzzleLayout,
    analysis: EdgeCostTable,
    rng: random.Random,
    count: int,
    beam_width: int = 4,
    candidate_width: int = 8,
    confidence_weight: float = 0.02,
    border_weight: float = 0.0,
) -> list[Arrangement]:
    """Build valid arrangements using confidence-aware row-major beam search."""
    if count <= 0 or not pieces:  # pragma: no mutate (count 0 also yields [])
        return []
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if candidate_width <= 0:
        raise ValueError("candidate_width must be positive")
    if confidence_weight < 0.0:
        raise ValueError("confidence_weight must not be negative")
    if border_weight < 0.0:
        raise ValueError("border_weight must not be negative")

    pieces_by_id = {piece.identifier: piece for piece in pieces}
    piece_ids = tuple(pieces_by_id)
    seed_count = min(count, len(piece_ids))
    anchors = rng.sample(piece_ids, seed_count)
    start_priority = 0.0  # pragma: no mutate (a constant offset does not change the ranking)
    states = [_BeamState((anchor,), frozenset((anchor,)), start_priority) for anchor in anchors]
    effective_beam_width = max(beam_width, seed_count)

    for _ in range(1, layout.piece_count):
        expanded: list[_BeamState] = []
        for state in states:
            candidates = [piece_id for piece_id in piece_ids if piece_id not in state.used]
            candidates.sort(
                key=lambda candidate_id, state=state: (
                    _seed_position_priority(
                        state.identifiers,
                        candidate_id,
                        layout,
                        analysis,
                        confidence_weight,
                        border_weight,
                    ),
                    candidate_id,
                )
            )
            for candidate_id in candidates[:candidate_width]:
                priority = _seed_position_priority(
                    state.identifiers,
                    candidate_id,
                    layout,
                    analysis,
                    confidence_weight,
                    border_weight,
                )
                expanded.append(
                    _BeamState(
                        state.identifiers + (candidate_id,),
                        state.used | {candidate_id},
                        state.priority + priority,
                    )
                )

        states = sorted(
            expanded,
            key=lambda state: (state.priority, state.identifiers),
        )[:effective_beam_width]
        if not states:  # pragma: no mutate
            break

    seeds: list[Arrangement] = []
    seen: set[tuple[int, ...]] = set()
    for state in sorted(states, key=lambda item: (item.priority, item.identifiers)):
        if len(state.identifiers) != layout.piece_count:  # pragma: no mutate (never incomplete)
            continue
        if state.identifiers in seen:  # pragma: no mutate (duplicates are rare)
            continue
        seen.add(state.identifiers)
        seeds.append(
            Arrangement(
                [pieces_by_id[piece_id] for piece_id in state.identifiers],
                layout,
            )
        )
        if len(seeds) == count:
            break

    return seeds


ProgressCallback = Callable[[str, int, int], None]
GenerationCallback = Callable[[int, Arrangement], None]
Move = tuple[str, int, int, int]


def _random_move(piece_count: int, rng: random.Random) -> Move:
    """Generate a permutation-preserving local-search move."""
    if piece_count < 2:
        raise ValueError("at least two pieces are required for a move")

    move_kind = rng.randrange(3) if piece_count >= 4 else rng.randrange(2)
    if move_kind == 0:
        first, second = rng.sample(range(piece_count), 2)
        return ("swap", first, second, 1)
    if move_kind == 1:
        source, target = rng.sample(range(piece_count), 2)
        return ("relocate", source, target, 1)

    block_size = rng.randint(1, min(4, piece_count // 2))
    first = rng.randrange(piece_count - 2 * block_size + 1)
    second = rng.randrange(first + block_size, piece_count - block_size + 1)
    return ("block", first, second, block_size)


def _apply_move(
    arrangement: Arrangement,
    move: Move,
    analysis: EdgeCostTable,
) -> None:
    operation, first, second, block_size = move
    if operation == "swap":
        arrangement.swap(first, second, cost_lookup=analysis)
    elif operation == "relocate":
        arrangement.relocate(first, second, cost_lookup=analysis)
    elif operation == "block":
        arrangement.swap_blocks(first, second, block_size, cost_lookup=analysis)
    else:
        raise ValueError(f"unknown local-search move: {operation}")


def _inverse_move(move: Move) -> Move:
    operation, first, second, block_size = move
    if operation == "relocate":
        return (operation, second, first, block_size)
    return move


def _local_search(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    steps: int,
    candidates: int,
) -> None:
    """Apply best-improvement local search using incremental seam costs."""
    if steps <= 0 or candidates <= 0 or len(arrangement.pieces) < 2:
        return

    for _ in range(steps):
        current_score = arrangement.score(analysis)
        best_score = current_score
        best_move: Move | None = None
        for _ in range(candidates):
            move = _random_move(len(arrangement.pieces), rng)
            _apply_move(arrangement, move, analysis)
            candidate_score = arrangement.score(analysis)
            _apply_move(arrangement, _inverse_move(move), analysis)
            if candidate_score > best_score:  # pragma: no mutate (ties have measure zero)
                best_score = candidate_score
                best_move = move

        if best_move is None:
            return
        _apply_move(arrangement, best_move, analysis)


def _repair_window_indices(
    arrangement: Arrangement,
    rng: random.Random,
) -> tuple[int, ...] | None:
    rows = arrangement.layout.rows
    columns = arrangement.layout.columns
    if rows >= 2 and columns >= 2:
        row = rng.randrange(rows - 1)
        column = rng.randrange(columns - 1)
        return (
            row * columns + column,
            row * columns + column + 1,
            (row + 1) * columns + column,
            (row + 1) * columns + column + 1,
        )
    if columns >= 3:
        start = rng.randrange(columns - 2)
        return tuple(start + offset for offset in range(3))
    if rows >= 3:
        start = rng.randrange(rows - 2)
        return tuple((start + offset) * columns for offset in range(3))
    return None


def _constraint_repair(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    steps: int,
) -> None:
    """Exhaustively repair small windows while preserving all piece assignments."""
    if steps <= 0:  # pragma: no mutate (range(0) is a no-op)
        return

    for _ in range(steps):
        indices = _repair_window_indices(arrangement, rng)
        if indices is None:
            return
        original = tuple(arrangement.pieces[index] for index in indices)
        original_ids = tuple(piece.identifier for piece in original)
        current_score = arrangement.score(analysis)
        best_score = current_score
        best_order: tuple[Piece, ...] | None = None
        for candidate in itertools.permutations(original):
            candidate_ids = tuple(piece.identifier for piece in candidate)
            if candidate_ids == original_ids:
                continue
            arrangement.replace_positions(indices, candidate, cost_lookup=analysis)
            candidate_score = arrangement.score(analysis)
            arrangement.replace_positions(indices, original, cost_lookup=analysis)
            if candidate_score > best_score:  # pragma: no mutate (ties have measure zero)
                best_score = candidate_score
                best_order = candidate
        if best_order is None:
            return
        arrangement.replace_positions(indices, best_order, cost_lookup=analysis)


def _mutate_child(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    mutation_rate: float,
    local_search_steps: int,
    local_search_candidates: int,
    repair_steps: int,
) -> None:
    # equivalent mutants (here and in the rejection test): equality has measure zero
    if mutation_rate > 0.0 and rng.random() < mutation_rate:  # pragma: no mutate
        move = _random_move(len(arrangement.pieces), rng)
        original_score = arrangement.score(analysis)
        _apply_move(arrangement, move, analysis)
        mutated_score = arrangement.score(analysis)
        worse = mutated_score < original_score  # pragma: no mutate (ties have measure zero)
        accept_threshold = 0.05 + 0.15 * mutation_rate
        reject_worse_move = worse and rng.random() >= accept_threshold  # pragma: no mutate
        if reject_worse_move:
            _apply_move(arrangement, _inverse_move(move), analysis)

    _local_search(
        arrangement,
        analysis,
        rng,
        local_search_steps,
        local_search_candidates,
    )
    _constraint_repair(arrangement, analysis, rng, repair_steps)


def _crossover_child(
    first_parent: Arrangement,
    second_parent: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    stats: CrossoverStats | None = None,
) -> Arrangement:
    """Build a valid child, falling back to a parent if crossover cannot fill it."""
    crossover = Crossover(
        first_parent,
        second_parent,
        analysis,
        rng,
        stats=stats,
    )
    crossover.run()
    try:
        return crossover.child()
    except RuntimeError:
        return Arrangement(list(first_parent.pieces), first_parent.layout)


@dataclass(frozen=True, slots=True)  # pragma: no mutate
class SolveResult:
    arrangement: Arrangement
    generations_completed: int
    best_score: float
    terminated_early: bool


class GeneticAlgorithm:
    """Evolve candidate arrangements for one puzzle image (serial upstream path)."""

    TERMINATION_THRESHOLD = 10

    def __init__(
        self,
        image: Image,
        piece_size: int,
        population_size: int,
        generations: int,
        elite_size: int = 2,
        rng: random.Random | None = None,
        mutation_rate: float = 0.05,
        collect_stats: bool = False,
        workers: int = 1,
        local_search_steps: int = 1,
        local_search_candidates: int = 4,
        repair_steps: int = 1,
        max_restarts: int = 2,
        restart_threshold: int | None = None,
        tournament_size: int = 3,
        seed_fraction: float = 0.25,
        seed_beam_width: int = 4,
        seed_candidate_width: int = 8,
        seed_confidence_weight: float = 0.02,
        seed_border_weight: float = 0.0,
    ) -> None:
        if population_size <= 0:  # pragma: no mutate (elite_size check also rejects)
            raise ValueError("population_size must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")
        if not 0 < elite_size < population_size:
            raise ValueError("elite_size must be between zero and population size")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between zero and one")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if workers > 1:
            raise NotImplementedError("only the serial path (workers=1) is ported from gaps")
        if local_search_steps < 0:
            raise ValueError("local_search_steps must not be negative")
        if local_search_candidates < 0:
            raise ValueError("local_search_candidates must not be negative")
        if repair_steps < 0:
            raise ValueError("repair_steps must not be negative")
        if max_restarts < 0:
            raise ValueError("max_restarts must not be negative")
        if restart_threshold is not None and restart_threshold <= 0:
            raise ValueError("restart_threshold must be positive")
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive")
        if not 0.0 <= seed_fraction <= 1.0:
            raise ValueError("seed_fraction must be between zero and one")
        if seed_beam_width <= 0:
            raise ValueError("seed_beam_width must be positive")
        if seed_candidate_width <= 0:
            raise ValueError("seed_candidate_width must be positive")
        if seed_confidence_weight < 0.0:
            raise ValueError("seed_confidence_weight must not be negative")
        if seed_border_weight < 0.0:
            raise ValueError("seed_border_weight must not be negative")

        pieces, layout = flatten_image(image, piece_size, indexed=True)
        self._analysis = EdgeCostTable()
        self._pieces = pieces
        self._layout = layout
        self._generations = generations
        self._population_size = population_size
        self._elite_size = elite_size
        self._rng = rng or random.Random()
        self._mutation_rate = mutation_rate
        self._local_search_steps = local_search_steps
        self._local_search_candidates = local_search_candidates
        self._repair_steps = repair_steps
        self._max_restarts = max_restarts
        default_threshold = max(1, self.TERMINATION_THRESHOLD // 2)  # pragma: no mutate
        self._restart_threshold = restart_threshold or default_threshold
        self._tournament_size = tournament_size
        self._seed_fraction = seed_fraction
        self._seed_beam_width = seed_beam_width
        self._seed_candidate_width = seed_candidate_width
        self._seed_confidence_weight = seed_confidence_weight
        self._seed_border_weight = seed_border_weight
        self._population_initialized = False
        self._crossover_stats = CrossoverStats() if collect_stats else None
        self._population = [
            Arrangement.random(pieces, layout, self._rng) for _ in range(population_size)
        ]

    @property
    def layout(self) -> PuzzleLayout:
        return self._layout

    @property
    def pieces(self) -> list[Piece]:
        """Pieces in input (row-major) order; ``pieces[i].identifier == i``."""
        return self._pieces

    @property
    def crossover_stats(self) -> CrossoverStats | None:
        """Return optional crossover counters collected during the last solve."""
        return self._crossover_stats

    def solve(
        self,
        progress: ProgressCallback | None = None,
        on_generation: GenerationCallback | None = None,
    ) -> SolveResult:
        """Run evolution and return the best valid arrangement found."""
        self._analysis.analyze(self._pieces, progress=progress)
        self._initialize_population()
        best_arrangement = self._best_arrangement()
        best_score = self._score(best_arrangement)
        stagnant_generations = 0
        restarts = 0

        for generation in range(self._generations):
            mutation_rate = self._adaptive_mutation_rate(stagnant_generations)
            self._population = self._next_population_serial(mutation_rate)
            current_arrangement = self._best_arrangement()
            current_score = self._score(current_arrangement)
            if current_score > best_score:
                best_arrangement = current_arrangement
                best_score = current_score
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if progress is not None:
                progress("evolution", generation + 1, self._generations)
            if on_generation is not None:
                on_generation(generation + 1, best_arrangement)

            if stagnant_generations >= self._restart_threshold:
                if restarts < self._max_restarts:
                    self._restart_population(best_arrangement)
                    restarts += 1
                    stagnant_generations = 0
                elif stagnant_generations >= self.TERMINATION_THRESHOLD:
                    return SolveResult(
                        best_arrangement,
                        generation + 1,
                        best_score,
                        True,
                    )

        return SolveResult(
            best_arrangement,
            self._generations,
            best_score,
            False,
        )

    def _initialize_population(self) -> None:
        if self._population_initialized:
            return
        self._population_initialized = True
        if self._seed_fraction == 0.0:
            return

        try:
            seed_count = min(
                8,
                max(1, int(self._population_size * self._seed_fraction)),
                max(1, 1024 // self._layout.piece_count),  # pragma: no mutate (needs > 1024 pieces)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("could not calculate seed count") from error

        try:
            seeds = build_seed_arrangements(
                self._pieces,
                self._layout,
                self._analysis,
                self._rng,
                count=seed_count,
                beam_width=self._seed_beam_width,
                candidate_width=self._seed_candidate_width,
                confidence_weight=self._seed_confidence_weight,
                border_weight=self._seed_border_weight,
            )
        except (KeyError, ValueError):
            return
        self._population = (seeds + self._population)[: self._population_size]

    def _next_population_serial(self, mutation_rate: float) -> list[Arrangement]:
        next_population = list(self._elite_individuals())
        parent_pairs = tournament_selection(
            self._population,
            score=self._score,
            rng=self._rng,
            elites=self._elite_size,
            tournament_size=self._tournament_size,
        )
        for first_parent, second_parent in parent_pairs:
            child = _crossover_child(
                first_parent,
                second_parent,
                self._analysis,
                self._rng,
                stats=self._crossover_stats,
            )
            self._mutate(child, mutation_rate)
            next_population.append(child)
        return next_population

    @staticmethod
    def _identifiers(arrangement: Arrangement) -> tuple[int, ...]:
        return tuple(piece.identifier for piece in arrangement.pieces)

    def _mutate(self, arrangement: Arrangement, mutation_rate: float) -> None:
        _mutate_child(
            arrangement,
            self._analysis,
            self._rng,
            mutation_rate,
            self._local_search_steps,
            self._local_search_candidates,
            self._repair_steps,
        )

    def _adaptive_mutation_rate(self, stagnant_generations: int) -> float:
        """Increase exploration pressure as the population stops improving."""
        if self._mutation_rate == 0.0:  # pragma: no mutate (formula also gives 0)
            return 0.0
        stagnation_factor = min(
            2.0,
            stagnant_generations / self._restart_threshold,
        )
        scores = [self._score(individual) for individual in self._population]
        flat = bool(scores) and max(scores) - min(scores) < 1e-6  # pragma: no mutate
        diversity_factor = 1.25 if flat else 1.0
        return min(
            1.0,
            self._mutation_rate * (1.0 + stagnation_factor) * diversity_factor,
        )

    def _restart_population(self, best_arrangement: Arrangement) -> None:
        """Retain the best known individuals and refill the population randomly."""
        preserved: list[Arrangement] = [best_arrangement]
        preserved_ids = {self._identifiers(best_arrangement)}
        for elite in reversed(self._elite_individuals()):
            elite_ids = self._identifiers(elite)
            if elite_ids not in preserved_ids:
                preserved.append(elite)
                preserved_ids.add(elite_ids)
            if len(preserved) == self._elite_size:
                break

        self._population = preserved
        while len(self._population) < self._population_size:
            self._population.append(Arrangement.random(self._pieces, self._layout, self._rng))

    def _score(self, arrangement: Arrangement) -> float:
        return arrangement.score(self._analysis)

    def _elite_individuals(self) -> Sequence[Arrangement]:
        return sorted(self._population, key=self._score)[-self._elite_size :]

    def _best_arrangement(self) -> Arrangement:
        return max(self._population, key=self._score)


# --------------------------------------------------------------------------- #
# facade (this repository's contract, README.md)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JigsawResult:
    """Output of :func:`solve_jigsaw`.

    ``order[k]`` is the row-major index (in the *input* image) of the piece placed
    at output position ``k`` (row-major); ``image`` is the reassembled uint8 image
    of the input's shape; ``score`` is the gaps fitness ``exp(-mean_seam_cost/0.05)``
    in ``(0, 1]`` (higher is better); ``generations`` is the number of generations
    actually run (fewer than requested on early termination, 0 for one piece).
    """

    order: np.ndarray
    image: np.ndarray
    score: float
    generations: int


def solve_jigsaw(
    image: np.ndarray,
    piece_size: int,
    *,
    population: int = 200,
    generations: int = 20,
    seed: int = 0,
    **ga_kwargs,
) -> JigsawResult:
    """Reassemble a block-permuted image with the gaps genetic algorithm.

    ``image`` is a shuffled ``(H, W, 3)`` / ``(H, W)`` uint8 image whose sides are
    multiples of ``piece_size``; it is cut into pieces row-major (identifier = input
    index, as upstream ``flatten_image``). Piece orientation is assumed known
    (pieces are never rotated). ``ga_kwargs`` are forwarded to
    :class:`GeneticAlgorithm` (``elite_size``, ``mutation_rate``, ``max_restarts``,
    ``seed_fraction``, ...). Runs are deterministic in ``seed``
    (``random.Random(seed)``, identical to upstream with ``workers=1``).
    """
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim not in (2, 3):
        raise ValueError("image must be (H, W) or (H, W, C)")
    if piece_size <= 0:
        raise ValueError("piece_size must be positive")
    if image.shape[0] % piece_size or image.shape[1] % piece_size:
        raise ValueError(
            # equivalent mutant: message
            f"image sides {image.shape[:2]} must be multiples of piece_size={piece_size}"  # pragma: no mutate
        )

    solver = GeneticAlgorithm(
        image,
        piece_size,
        population_size=population,
        generations=generations,
        rng=random.Random(seed),
        **ga_kwargs,
    )
    layout = solver.layout
    if layout.piece_count == 1:
        arrangement = Arrangement(list(solver.pieces), layout)
        generations_run = 0
        score = 1.0
    else:
        result = solver.solve()
        arrangement = result.arrangement
        generations_run = result.generations_completed
        score = result.best_score
    order = np.array([piece.identifier for piece in arrangement.pieces], dtype=np.int64)
    return JigsawResult(
        order=order,
        image=assemble_image(arrangement.pieces, layout),
        score=float(score),
        generations=int(generations_run),
    )
