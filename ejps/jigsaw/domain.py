# Ported from gaps (https://github.com/nemanja-m/gaps),
# commit 5aa0a0d15ff1b2d6b11ab03c6c005b025bd93d20, files gaps/domain.py and
# gaps/imaging/transforms.py.
# Copyright (c) 2017 Nemanja Milićević. Released under the MIT License; the full
# licence text is in THIRD_PARTY_LICENSE_gaps.txt at the repository root.
# Changes from upstream: Python 3.10 syntax (no ``type`` statements, ``StrEnum``
# replaced by ``str, Enum``); transforms.py merged in (no overloads); no I/O.
"""Jigsaw domain model: pieces, layouts and candidate arrangements.

Part of the block-level jigsaw stage (stage 3) of the ciphertext-only attack.
A puzzle is an image cut into ``rows x columns`` square pieces of ``piece_size``
pixels. Pieces keep their row-major index in the *input* image as ``identifier``;
an :class:`Arrangement` is a permutation of those pieces, read row-major.

``Arrangement.score`` is ``exp(-mean_seam_cost / 0.05)`` and is cached; the move
methods (``swap``, ``relocate``, ``swap_blocks``, ``replace_positions``) update that
cache incrementally when given a cost lookup and invalidate it otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]

_FITNESS_TEMPERATURE = 0.05


class Direction(str, Enum):
    """Directions used to locate neighboring puzzle pieces."""

    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"

    @property
    def opposite(self) -> Direction:
        return _OPPOSITE_DIRECTIONS[self]


_OPPOSITE_DIRECTIONS = {
    Direction.TOP: Direction.BOTTOM,
    Direction.RIGHT: Direction.LEFT,
    Direction.BOTTOM: Direction.TOP,
    Direction.LEFT: Direction.RIGHT,
}


class EdgeAxis(str, Enum):
    """The axis along which two pieces are adjacent."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


CostLookup = Callable[[tuple[int, int], EdgeAxis], float]


@dataclass(frozen=True, slots=True)
class PuzzleLayout:
    """Dimensions shared by every arrangement of one puzzle."""

    rows: int
    columns: int
    piece_size: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0 or self.piece_size <= 0:
            raise ValueError("layout dimensions must be positive")

    @property
    def piece_count(self) -> int:
        return self.rows * self.columns


@dataclass(slots=True)
class Piece:
    """A square image fragment with a stable identifier."""

    image: Image
    identifier: int

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim not in (2, 3):
            raise ValueError("piece image must have two or three dimensions")
        if image.ndim == 3 and image.shape[2] not in (1, 3):
            raise ValueError("piece image must be grayscale or three-channel")
        if image.shape[0] != image.shape[1]:
            raise ValueError("piece image must be square")
        if image.dtype != np.uint8:
            raise ValueError("piece image must use uint8 pixels")
        self.image = image.copy()

    def __getitem__(self, index: Any) -> Any:
        return self.image[index]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape

    @property
    def size(self) -> int:
        return self.image.shape[0]  # pragma: no mutate (square)


@dataclass(slots=True)
class Arrangement:
    """A candidate arrangement of every piece in a puzzle."""

    pieces: list[Piece]
    layout: PuzzleLayout
    _piece_mapping: dict[int, int] = field(init=False, repr=False)
    _edge_cache: dict[Direction, dict[int, int | None]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_score: float | None = field(default=None, init=False, repr=False)
    _cached_total_cost: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.pieces = list(self.pieces)
        if len(self.pieces) != self.layout.piece_count:
            raise ValueError("piece count does not match the puzzle layout")
        identifiers = [piece.identifier for piece in self.pieces]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("piece identifiers must be unique")
        self._piece_mapping = {piece_id: index for index, piece_id in enumerate(identifiers)}

    @classmethod
    def random(
        cls,
        pieces: Sequence[Piece],
        layout: PuzzleLayout,
        rng,
    ) -> Arrangement:
        shuffled_pieces = list(pieces)
        rng.shuffle(shuffled_pieces)
        return cls(shuffled_pieces, layout)

    def __getitem__(self, row: int) -> list[Piece]:
        start = row * self.layout.columns
        return self.pieces[start : start + self.layout.columns]

    def score(self, cost_lookup: CostLookup) -> float:
        """Calculate and cache a normalized arrangement compatibility score."""
        if self._cached_score is None:
            arrangement_cost = getattr(cost_lookup, "arrangement_cost", None)
            if callable(arrangement_cost):
                score_lookup = cast(Callable[[Arrangement], float], arrangement_cost)
                try:
                    total_cost = score_lookup(self)
                except (TypeError, ValueError) as error:
                    raise ValueError("could not calculate arrangement score") from error
            else:
                total_cost = 0.0
                for row in range(self.layout.rows):
                    for column in range(self.layout.columns - 1):
                        ids = (
                            self[row][column].identifier,
                            self[row][column + 1].identifier,
                        )
                        total_cost += cost_lookup(ids, EdgeAxis.HORIZONTAL)
                for row in range(self.layout.rows - 1):
                    for column in range(self.layout.columns):
                        ids = (
                            self[row][column].identifier,
                            self[row + 1][column].identifier,
                        )
                        total_cost += cost_lookup(ids, EdgeAxis.VERTICAL)

            self._cached_total_cost = total_cost
            self._cached_score = self._score_from_total_cost(total_cost)
        return self._cached_score

    def _score_from_total_cost(self, total_cost: float) -> float:
        seam_count = (
            self.layout.rows * (self.layout.columns - 1)
            + (self.layout.rows - 1) * self.layout.columns
        )
        mean_cost = total_cost / seam_count if seam_count else 0.0
        return math.exp(-mean_cost / _FITNESS_TEMPERATURE)

    def _affected_seams(
        self, first_index: int, second_index: int
    ) -> set[tuple[EdgeAxis, int, int]]:
        seams: set[tuple[EdgeAxis, int, int]] = set()
        columns = self.layout.columns
        for index in (first_index, second_index):
            row, column = divmod(index, columns)
            if column > 0:
                seams.add((EdgeAxis.HORIZONTAL, row, column - 1))
            if column < columns - 1:
                seams.add((EdgeAxis.HORIZONTAL, row, column))
            if row > 0:
                seams.add((EdgeAxis.VERTICAL, row - 1, column))
            if row < self.layout.rows - 1:
                seams.add((EdgeAxis.VERTICAL, row, column))
        return seams

    def _seam_cost(
        self,
        seam: tuple[EdgeAxis, int, int],
        cost_lookup: CostLookup,
    ) -> float:
        axis, row, column = seam
        first_index = row * self.layout.columns + column
        second_index = (
            first_index + 1 if axis is EdgeAxis.HORIZONTAL else first_index + self.layout.columns
        )
        ids = (
            self.pieces[first_index].identifier,
            self.pieces[second_index].identifier,
        )
        return cost_lookup(ids, axis)

    def swap(
        self,
        first_index: int,
        second_index: int,
        cost_lookup: CostLookup | None = None,
    ) -> None:
        """Swap two pieces and update cached state when a cost lookup is given."""
        if first_index == second_index:
            return

        cached_total_cost = self._cached_total_cost
        affected_seams = (
            self._affected_seams(first_index, second_index)
            if cached_total_cost is not None and cost_lookup is not None  # pragma: no mutate
            else set()
        )
        old_local_cost = (
            sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            if cost_lookup is not None
            else 0.0  # pragma: no mutate
        )

        self.pieces[first_index], self.pieces[second_index] = (
            self.pieces[second_index],
            self.pieces[first_index],
        )
        self._piece_mapping[self.pieces[first_index].identifier] = first_index
        self._piece_mapping[self.pieces[second_index].identifier] = second_index
        self._edge_cache = None

        if cached_total_cost is not None and cost_lookup is not None:
            new_local_cost = sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            self._cached_total_cost = cached_total_cost - old_local_cost + new_local_cost
            self._cached_score = self._score_from_total_cost(self._cached_total_cost)
        else:
            self._cached_score = None
            self._cached_total_cost = None

    def relocate(
        self,
        source_index: int,
        target_index: int,
        cost_lookup: CostLookup | None = None,
    ) -> None:
        """Move one piece to another position while preserving the permutation."""
        if source_index == target_index:
            return
        piece_count = len(self.pieces)
        if not 0 <= source_index < piece_count or not 0 <= target_index < piece_count:
            raise IndexError("piece index out of range")

        cached_total_cost = self._cached_total_cost
        affected_seams = (
            self._affected_seams_for_range(source_index, target_index)
            if cached_total_cost is not None and cost_lookup is not None  # pragma: no mutate
            else set()
        )
        old_local_cost = (
            sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            if cost_lookup is not None
            else 0.0  # pragma: no mutate
        )

        piece = self.pieces.pop(source_index)
        self.pieces.insert(target_index, piece)
        start, stop = sorted((source_index, target_index))
        for index in range(start, stop + 1):
            self._piece_mapping[self.pieces[index].identifier] = index
        self._edge_cache = None

        if cached_total_cost is not None and cost_lookup is not None:
            new_local_cost = sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            self._cached_total_cost = cached_total_cost - old_local_cost + new_local_cost
            self._cached_score = self._score_from_total_cost(self._cached_total_cost)
        else:
            self._cached_score = None
            self._cached_total_cost = None

    def swap_blocks(
        self,
        first_start: int,
        second_start: int,
        block_size: int,
        cost_lookup: CostLookup | None = None,
    ) -> None:
        """Exchange two non-overlapping, equally sized contiguous blocks."""
        piece_count = len(self.pieces)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if (
            not 0 <= first_start < piece_count
            or not 0 <= second_start < piece_count
            or first_start + block_size > piece_count
            or second_start + block_size > piece_count
        ):
            raise IndexError("block index out of range")
        if first_start == second_start:
            return
        if first_start > second_start:  # pragma: no mutate (equal starts returned above)
            first_start, second_start = second_start, first_start
        if first_start + block_size > second_start:
            raise ValueError("blocks must not overlap")

        cached_total_cost = self._cached_total_cost
        affected_seams = (
            self._affected_seams_for_range(
                first_start,
                second_start + block_size - 1,
            )
            if cached_total_cost is not None and cost_lookup is not None  # pragma: no mutate
            else set()
        )
        old_local_cost = (
            sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            if cost_lookup is not None
            else 0.0  # pragma: no mutate
        )

        first_block = self.pieces[first_start : first_start + block_size]
        second_block = self.pieces[second_start : second_start + block_size]
        self.pieces[first_start : first_start + block_size] = second_block
        self.pieces[second_start : second_start + block_size] = first_block
        for index in range(first_start, second_start + block_size):
            self._piece_mapping[self.pieces[index].identifier] = index
        self._edge_cache = None

        if cached_total_cost is not None and cost_lookup is not None:
            new_local_cost = sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            self._cached_total_cost = cached_total_cost - old_local_cost + new_local_cost
            self._cached_score = self._score_from_total_cost(self._cached_total_cost)
        else:
            self._cached_score = None
            self._cached_total_cost = None

    def replace_positions(
        self,
        indices: Sequence[int],
        pieces: Sequence[Piece],
        cost_lookup: CostLookup | None = None,
    ) -> None:
        """Reorder a fixed set of positions without changing the permutation."""
        positions = tuple(indices)
        replacements = tuple(pieces)
        if len(positions) != len(replacements) or not positions:
            raise ValueError("positions and pieces must have the same non-zero length")
        if len(set(positions)) != len(positions):
            raise ValueError("positions must be unique")
        if any(index < 0 or index >= len(self.pieces) for index in positions):
            raise IndexError("piece index out of range")
        current_ids = {self.pieces[index].identifier for index in positions}
        replacement_ids = {piece.identifier for piece in replacements}
        if current_ids != replacement_ids:
            raise ValueError("replacement pieces must match the selected positions")
        if all(
            self.pieces[index] is piece
            for index, piece in zip(positions, replacements, strict=False)  # pragma: no mutate
        ):
            return

        cached_total_cost = self._cached_total_cost
        affected_seams = (
            {seam for index in positions for seam in self._affected_seams(index, index)}
            if cached_total_cost is not None and cost_lookup is not None  # pragma: no mutate
            else set()
        )
        old_local_cost = (
            sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            if cost_lookup is not None
            else 0.0  # pragma: no mutate
        )

        for index, piece in zip(positions, replacements, strict=False):  # pragma: no mutate
            self.pieces[index] = piece
            self._piece_mapping[piece.identifier] = index
        self._edge_cache = None

        if cached_total_cost is not None and cost_lookup is not None:
            new_local_cost = sum(self._seam_cost(seam, cost_lookup) for seam in affected_seams)
            self._cached_total_cost = cached_total_cost - old_local_cost + new_local_cost
            self._cached_score = self._score_from_total_cost(self._cached_total_cost)
        else:
            self._cached_score = None
            self._cached_total_cost = None

    def _affected_seams_for_range(
        self, first_index: int, second_index: int
    ) -> set[tuple[EdgeAxis, int, int]]:
        start, stop = sorted((first_index, second_index))
        seams: set[tuple[EdgeAxis, int, int]] = set()
        for index in range(start, stop + 1):
            seams.update(self._affected_seams(index, index))
        return seams

    def piece_by_id(self, identifier: int) -> Piece:
        return self.pieces[self._piece_mapping[identifier]]

    def _build_edge_cache(self) -> None:
        if self._edge_cache is not None:
            return
        edges = {direction: {} for direction in Direction}
        columns = self.layout.columns
        for piece_index, piece in enumerate(self.pieces):
            row, column = divmod(piece_index, columns)
            piece_id = piece.identifier
            edges[Direction.TOP][piece_id] = (
                self.pieces[piece_index - columns].identifier if row > 0 else None
            )
            edges[Direction.RIGHT][piece_id] = (
                self.pieces[piece_index + 1].identifier if column < columns - 1 else None
            )
            edges[Direction.BOTTOM][piece_id] = (
                self.pieces[piece_index + columns].identifier
                if row < self.layout.rows - 1
                else None
            )
            edges[Direction.LEFT][piece_id] = (
                self.pieces[piece_index - 1].identifier if column > 0 else None
            )
        self._edge_cache = edges

    def edge(self, piece_id: int, direction: Direction) -> int | None:
        if self._edge_cache is None:
            self._build_edge_cache()
        assert self._edge_cache is not None
        return self._edge_cache[direction][piece_id]


# --- from gaps/imaging/transforms.py ---------------------------------------------


def flatten_image(
    image: Image, piece_size: int, indexed: bool = False
) -> tuple[list[Image] | list[Piece], PuzzleLayout]:
    """Split an image into square pieces (row-major), cropping incomplete edges.

    With ``indexed=True`` the pieces are wrapped as :class:`Piece` whose
    ``identifier`` is the row-major position in ``image``.
    """
    if piece_size <= 0:
        raise ValueError("piece_size must be positive")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
        raise ValueError("image must be grayscale or three-channel color")

    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size
    if rows == 0 or columns == 0:
        raise ValueError("piece_size must fit within the image")

    layout = PuzzleLayout(rows, columns, piece_size)
    pieces = [
        image[
            row * piece_size : (row + 1) * piece_size,
            column * piece_size : (column + 1) * piece_size,
        ].copy()
        for row in range(rows)
        for column in range(columns)
    ]
    if indexed:
        return [Piece(piece, index) for index, piece in enumerate(pieces)], layout
    return pieces, layout


def assemble_image(pieces: list[Image] | list[Piece], layout: PuzzleLayout) -> Image:
    """Assemble a row-major sequence of pieces into an image."""
    if len(pieces) != layout.piece_count:
        raise ValueError("piece count does not match the puzzle layout")

    images = [piece.image if isinstance(piece, Piece) else piece for piece in pieces]
    rows = [
        np.hstack(images[row * layout.columns : (row + 1) * layout.columns])
        for row in range(layout.rows)
    ]
    return np.vstack(rows).astype(np.uint8, copy=False)  # pragma: no mutate
