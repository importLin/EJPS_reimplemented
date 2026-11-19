# Ported from gaps (https://github.com/nemanja-m/gaps),
# commit 5aa0a0d15ff1b2d6b11ab03c6c005b025bd93d20, files gaps/solver/fitness.py and
# gaps/solver/analysis.py.
# Copyright (c) 2017 Nemanja Milićević. Released under the MIT License; the full
# licence text is in THIRD_PARTY_LICENSE_gaps.txt at the repository root.
# Changes from upstream: imports only (behaviour and constants are unchanged); the
# two upstream modules are merged into this one.
"""Edge dissimilarity between puzzle pieces and the pairwise cost table.

Dissimilarity
-------------
For a seam between piece ``a`` (left / above) and piece ``b`` (right / below) the
cost is ``0.85 * (0.60 B + 0.25 N + 0.15 T) + 0.15 C`` where every term is a
normalized Huber mean (``beta = 0.01``) over pixels and channels:

* ``B``: boundary intensities, ``|a_edge - b_edge| / 255``;
* ``N``: normal gradients, ``|(a_edge - a_inner) - (b_inner - b_edge)| / 510``;
* ``T``: tangential gradients along the edge, ``|diff(a_edge) - diff(b_edge)| / 510``;
* ``C``: the boundary term on edges pooled by 2 (only for edges of >= 4 pixels).

The cost is directional (``cost(a, b) != cost(b, a)``); pairwise matrices have
``inf`` on the diagonal (a piece is never its own neighbour).

Cost table
----------
:class:`EdgeCostTable` caches the directed costs of every ordered piece pair on
both axes (``costs[i, j, 0]``: ``i`` left of ``j``; ``costs[i, j, 1]``: ``i``
above ``j``), the per-side match lists sorted by cost, and a confidence margin
``(second_best - best) / |second_best|`` clipped to ``[0, 1]``. It is also the
cost lookup passed to :meth:`Arrangement.score` and the move methods.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import SupportsFloat

import numpy as np

from .domain import Arrangement, Direction, EdgeAxis, Piece

_HUBER_BETA = 0.01
_BOUNDARY_WEIGHT = 0.60
_NORMAL_GRADIENT_WEIGHT = 0.25
_TANGENT_GRADIENT_WEIGHT = 0.15
_MULTISCALE_WEIGHT = 0.15
_PAIRWISE_BLOCK_SIZE = 32  # pragma: no mutate (memory bound only)


def _as_channels(image: np.ndarray) -> np.ndarray:
    """Return an image as float channels without changing its intensity."""
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)[..., np.newaxis]  # pragma: no mutate
    if image.ndim == 3 and image.shape[2] in (1, 3):
        return image.astype(np.float32, copy=False)
    raise ValueError("images must be grayscale or three-channel")


def _huber_mean(difference: np.ndarray, scale: float) -> float:
    """Return a normalized Huber loss for an array of pixel differences."""
    error = np.abs(difference) / scale
    loss = np.where(
        error <= _HUBER_BETA,  # pragma: no mutate (both branches equal 0.5*beta at the knot)
        0.5 * np.square(error) / _HUBER_BETA,
        error - 0.5 * _HUBER_BETA,
    )
    try:
        return float(np.mean(loss))
    except (TypeError, ValueError) as error:
        raise ValueError("could not calculate edge loss") from error


def _huber_mean_batch(difference: np.ndarray, scale: float) -> np.ndarray:
    """Calculate Huber means for a batch of pairwise pixel differences."""
    error = np.abs(difference) / scale
    loss = np.where(
        error <= _HUBER_BETA,  # pragma: no mutate (both branches equal 0.5*beta at the knot)
        0.5 * np.square(error) / _HUBER_BETA,
        error - 0.5 * _HUBER_BETA,
    )
    return np.mean(loss, axis=(-2, -1))


def _coarse_line(line: np.ndarray) -> np.ndarray | None:
    """Pool adjacent edge samples to provide a noise-resistant scale."""
    length = line.shape[-2]
    if length < 4:
        return None
    paired_length = length - length % 2
    coarse = (line[..., :paired_length:2, :] + line[..., 1:paired_length:2, :]) * 0.5
    if paired_length < length:
        coarse = np.concatenate((coarse, line[..., -1:, :]), axis=-2)
    return coarse


def _edge_lines(
    first_piece: Piece, second_piece: Piece, axis: EdgeAxis
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    first = _as_channels(first_piece.image)
    second = _as_channels(second_piece.image)

    if axis is EdgeAxis.HORIZONTAL:
        return (
            first[:, -1, :],
            first[:, -2, :] if first.shape[1] > 1 else None,
            second[:, 0, :],
            second[:, 1, :] if second.shape[1] > 1 else None,
        )
    if axis is EdgeAxis.VERTICAL:
        return (
            first[-1, :, :],
            first[-2, :, :] if first.shape[0] > 1 else None,
            second[0, :, :],
            second[1, :, :] if second.shape[0] > 1 else None,
        )
    raise ValueError(f"Unsupported edge orientation: {axis}")


def prepare_piece_channels(pieces: Sequence[Piece]) -> np.ndarray:
    """Convert all pieces to one contiguous float32 image array."""
    try:
        return np.stack([_as_channels(piece.image) for piece in pieces], axis=0)
    except (TypeError, ValueError) as error:
        raise ValueError("pieces must contain compatible numeric image data") from error


def pairwise_dissimilarity_from_channels(
    channels: np.ndarray,
    axis: EdgeAxis,
    block_size: int = _PAIRWISE_BLOCK_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Calculate directed pairwise edge costs in bounded vectorized blocks.

    ``costs[i, j]`` is the cost of piece ``i`` left of (HORIZONTAL) or above
    (VERTICAL) piece ``j``. ``block_size`` only bounds memory; it does not change
    the result.
    """
    if axis not in (EdgeAxis.HORIZONTAL, EdgeAxis.VERTICAL):
        raise ValueError(f"Unsupported edge orientation: {axis}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if channels.ndim != 4 or channels.shape[3] not in (1, 3):
        raise ValueError("pieces must contain compatible numeric image data")

    piece_count = channels.shape[0]
    if piece_count == 0:
        return np.empty((0, 0), dtype=np.float32)

    if axis is EdgeAxis.HORIZONTAL:
        first_edge = channels[:, :, -1, :]
        first_inner = channels[:, :, -2, :] if channels.shape[2] > 1 else None
        second_edge = channels[:, :, 0, :]
        second_inner = channels[:, :, 1, :] if channels.shape[2] > 1 else None
    else:
        first_edge = channels[:, -1, :, :]
        first_inner = channels[:, -2, :, :] if channels.shape[1] > 1 else None
        second_edge = channels[:, 0, :, :]
        second_inner = channels[:, 1, :, :] if channels.shape[1] > 1 else None

    first_normal = None if first_inner is None else first_edge - first_inner
    second_normal = None if second_inner is None else second_inner - second_edge
    first_tangent = np.diff(first_edge, axis=1) if first_edge.shape[1] > 1 else None
    second_tangent = np.diff(second_edge, axis=1) if second_edge.shape[1] > 1 else None
    first_coarse = _coarse_line(first_edge)
    second_coarse = _coarse_line(second_edge)
    costs = np.empty((piece_count, piece_count), dtype=np.float32)

    for start in range(0, piece_count, block_size):
        stop = min(start + block_size, piece_count)
        first_edge_block = first_edge[start:stop, np.newaxis, :, :]
        second_edge_block = second_edge[np.newaxis, :, :, :]

        boundary_cost = _huber_mean_batch(
            first_edge_block - second_edge_block,
            scale=255.0,
        )

        if first_normal is None or second_normal is None:  # pragma: no mutate (both or neither)
            normal_cost = 0.0
        else:
            first_normal_block = first_normal[start:stop, np.newaxis, :, :]
            second_normal_block = second_normal[np.newaxis, :, :, :]
            normal_cost = _huber_mean_batch(
                first_normal_block - second_normal_block,
                scale=510.0,
            )

        if first_tangent is None or second_tangent is None:  # pragma: no mutate (both or neither)
            tangent_cost = 0.0
        else:
            first_tangent_block = first_tangent[start:stop, np.newaxis, :, :]
            second_tangent_block = second_tangent[np.newaxis, :, :, :]
            tangent_cost = _huber_mean_batch(
                first_tangent_block - second_tangent_block,
                scale=510.0,
            )

        if first_coarse is None or second_coarse is None:  # pragma: no mutate (both or neither)
            coarse_cost = 0.0
        else:
            first_coarse_block = first_coarse[start:stop, np.newaxis, :, :]
            second_coarse_block = second_coarse[np.newaxis, :, :, :]
            coarse_cost = _huber_mean_batch(
                first_coarse_block - second_coarse_block,
                scale=255.0,
            )

        base_cost = (
            _BOUNDARY_WEIGHT * boundary_cost
            + _NORMAL_GRADIENT_WEIGHT * normal_cost
            + _TANGENT_GRADIENT_WEIGHT * tangent_cost
        )
        costs[start:stop] = (
            1.0 - _MULTISCALE_WEIGHT
        ) * base_cost + _MULTISCALE_WEIGHT * coarse_cost
        if progress is not None and piece_count > 1:
            progress(min(stop, piece_count - 1), piece_count - 1)

    np.fill_diagonal(costs, np.inf)
    return costs


def pairwise_dissimilarity(
    pieces: Sequence[Piece],
    axis: EdgeAxis,
    block_size: int = _PAIRWISE_BLOCK_SIZE,
) -> np.ndarray:
    """Calculate directed pairwise edge costs for a sequence of pieces."""
    return pairwise_dissimilarity_from_channels(
        prepare_piece_channels(pieces),
        axis,
        block_size=block_size,
    )


def dissimilarity_measure(
    first_piece: Piece,
    second_piece: Piece,
    axis: EdgeAxis = EdgeAxis.HORIZONTAL,
) -> float:
    """Calculate a robust, normalized compatibility cost for two edges.

    The cost combines boundary intensity, normal gradient, and tangential
    gradient differences.  Every term is averaged over pixels and channels,
    so grayscale images and color images have comparable score ranges.
    """
    if axis not in (EdgeAxis.HORIZONTAL, EdgeAxis.VERTICAL):
        raise ValueError(f"Unsupported edge orientation: {axis}")

    try:
        first_edge, first_inner, second_edge, second_inner = _edge_lines(
            first_piece, second_piece, axis
        )
        if first_edge.shape != second_edge.shape:
            raise ValueError("edge shapes do not match")

        boundary_cost = _huber_mean(first_edge - second_edge, scale=255.0)

        if first_inner is None or second_inner is None:  # pragma: no mutate (both or neither)
            normal_gradient_cost = 0.0
        else:
            normal_gradient_cost = _huber_mean(
                (first_edge - first_inner) - (second_inner - second_edge),
                scale=510.0,
            )

        if first_edge.shape[0] < 2:
            tangent_gradient_cost = 0.0
        else:
            tangent_gradient_cost = _huber_mean(
                np.diff(first_edge, axis=0) - np.diff(second_edge, axis=0),
                scale=510.0,
            )

        first_coarse = _coarse_line(first_edge)
        second_coarse = _coarse_line(second_edge)
        if first_coarse is None or second_coarse is None:  # pragma: no mutate (both or neither)
            coarse_cost = 0.0
        else:
            coarse_cost = _huber_mean(
                first_coarse - second_coarse,
                scale=255.0,
            )

        base_cost = (
            _BOUNDARY_WEIGHT * boundary_cost
            + _NORMAL_GRADIENT_WEIGHT * normal_gradient_cost
            + _TANGENT_GRADIENT_WEIGHT * tangent_gradient_cost
        )
        return float((1.0 - _MULTISCALE_WEIGHT) * base_cost + _MULTISCALE_WEIGHT * coarse_cost)
    except (TypeError, ValueError) as error:
        raise ValueError("pieces must contain compatible numeric image data") from error


# --------------------------------------------------------------------------- #
# pairwise cost table (upstream gaps/solver/analysis.py)
# --------------------------------------------------------------------------- #
_AXIS_INDEX = {
    EdgeAxis.HORIZONTAL: 0,
    EdgeAxis.VERTICAL: 1,
}


def _safe_float(value: SupportsFloat) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("cost table contains non-numeric values") from error


class EdgeCostTable:
    """Pairwise edge costs and best matches for one puzzle."""

    def __init__(self) -> None:
        self._costs = np.empty((0, 0, 2), dtype=np.float32)  # pragma: no mutate
        self._piece_ids = np.empty(0, dtype=np.intp)  # pragma: no mutate
        self._id_to_index: dict[int, int] = {}
        self._best_matches: dict[int, dict[Direction, list[tuple[int, float]]]] = {}

    def analyze(
        self,
        pieces: Sequence[Piece],
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Calculate and cache all pairwise edge comparisons."""
        piece_ids = [piece.identifier for piece in pieces]
        if len(set(piece_ids)) != len(piece_ids):
            raise ValueError("piece identifiers must be unique")

        self._piece_ids = np.asarray(piece_ids, dtype=np.intp)
        self._id_to_index = {piece_id: index for index, piece_id in enumerate(piece_ids)}
        self._best_matches = {
            piece_id: {direction: [] for direction in Direction} for piece_id in piece_ids
        }

        piece_count = len(pieces)
        if piece_count == 0:
            self._costs = np.empty((0, 0, 2), dtype=np.float32)  # pragma: no mutate
            return

        channels = prepare_piece_channels(pieces)
        costs = np.empty((piece_count, piece_count, 2), dtype=np.float32)
        # Upstream computes the horizontal costs as vertical costs of the transposed
        # pieces; the values are identical, only the memory layout differs.
        horizontal_channels = channels
        horizontal_axis = EdgeAxis.HORIZONTAL
        if channels.shape[2]:  # pragma: no mutate
            horizontal_channels = np.ascontiguousarray(channels.swapaxes(1, 2))
            horizontal_axis = EdgeAxis.VERTICAL
        costs[:, :, 0] = pairwise_dissimilarity_from_channels(
            horizontal_channels,
            horizontal_axis,
            block_size=3,  # pragma: no mutate  (memory bound only)
        )
        costs[:, :, 1] = pairwise_dissimilarity_from_channels(
            channels,
            EdgeAxis.VERTICAL,
            block_size=2 if progress is None else _PAIRWISE_BLOCK_SIZE,  # pragma: no mutate
            progress=(
                None
                if progress is None
                else lambda current, total: progress("analysis", current, total)
            ),
        )
        self._costs = costs

        ids = self._piece_ids.tolist()
        directional_costs = {
            Direction.RIGHT: np.ascontiguousarray(costs[:, :, 0]),
            Direction.LEFT: np.ascontiguousarray(costs[:, :, 0].T),
            Direction.BOTTOM: np.ascontiguousarray(costs[:, :, 1]),
            Direction.TOP: np.ascontiguousarray(costs[:, :, 1].T),
        }
        for direction, direction_costs in directional_costs.items():
            for index, piece_id in enumerate(ids):
                matches_by_direction = self._best_matches[piece_id]
                match_costs = direction_costs[index]
                order = np.argsort(match_costs, kind="stable")
                order = order[order != index]
                matches_by_direction[direction] = list(
                    zip(
                        self._piece_ids[order].tolist(),
                        match_costs[order].tolist(),
                        strict=False,  # pragma: no mutate
                    )
                )

    def __call__(self, ids: tuple[int, int], axis: EdgeAxis) -> float:
        return self.cost(ids, axis)

    def cost(self, ids: tuple[int, int], axis: EdgeAxis) -> float:
        try:
            first_index = self._id_to_index[ids[0]]
            second_index = self._id_to_index[ids[1]]
            axis_index = _AXIS_INDEX[axis]
        except (KeyError, IndexError) as error:
            raise KeyError(ids) from error
        if first_index == second_index:
            raise KeyError(ids)
        return _safe_float(self._costs[first_index, second_index, axis_index])

    def arrangement_cost(self, arrangement: Arrangement) -> float:
        """Return the total seam cost using direct dense-array indexing."""
        try:
            indices = np.fromiter(
                (self._id_to_index[piece.identifier] for piece in arrangement.pieces),
                dtype=np.intp,
                count=len(arrangement.pieces),
            )
        except KeyError as error:
            raise KeyError(error.args[0]) from error

        rows = arrangement.layout.rows
        columns = arrangement.layout.columns
        grid = indices.reshape(rows, columns)
        total_cost = 0.0
        try:
            if columns > 1:
                total_cost += _safe_float(
                    np.sum(
                        self._costs[grid[:, :-1], grid[:, 1:], 0],
                        dtype=np.float64,
                    )
                )
            if rows > 1:
                total_cost += _safe_float(
                    np.sum(
                        self._costs[grid[:-1, :], grid[1:, :], 1],
                        dtype=np.float64,
                    )
                )
        except (TypeError, ValueError) as error:
            raise ValueError("could not calculate arrangement cost") from error
        return total_cost

    def best_match(self, piece_id: int, direction: Direction) -> int:
        matches = self._best_matches[piece_id][direction]
        if not matches:
            raise ValueError(f"no best match available for piece {piece_id}")
        return matches[0][0]

    def matches(self, piece_id: int, direction: Direction) -> list[tuple[int, float]]:
        return self._best_matches[piece_id][direction]

    def best_cost(self, piece_id: int, direction: Direction) -> float:
        """Return the lowest available directed edge cost for a piece side."""
        matches = self.matches(piece_id, direction)
        if not matches:
            raise ValueError(f"no matches available for piece {piece_id}")
        return matches[0][1]

    def confidence(self, piece_id: int, direction: Direction) -> float:
        """Return a normalized margin between the best two edge matches."""
        matches = self.matches(piece_id, direction)
        if len(matches) < 2:
            return 1.0 if matches else 0.0
        best_cost = matches[0][1]
        second_cost = matches[1][1]
        margin = max(0.0, second_cost - best_cost)
        return min(1.0, margin / max(abs(second_cost), 1e-6))


@dataclass(frozen=True, slots=True)
class ArrangementMetrics:
    """Ground-truth quality metrics for a known piece ordering."""

    position_accuracy: float
    horizontal_adjacency_accuracy: float
    vertical_adjacency_accuracy: float
    adjacency_accuracy: float
    exact: bool


def evaluate_arrangement(
    arrangement: Arrangement,
    expected_identifiers: Sequence[int],
) -> ArrangementMetrics:
    """Compare an arrangement with expected row-major piece identifiers.

    Adjacency counts a seam as correct only when *both* pieces sit at their
    expected absolute positions (upstream definition).
    """
    expected = tuple(expected_identifiers)
    actual = tuple(piece.identifier for piece in arrangement.pieces)
    if len(expected) != arrangement.layout.piece_count:
        raise ValueError("expected identifiers do not match the puzzle layout")
    if len(set(expected)) != len(expected):
        raise ValueError("expected identifiers must be unique")
    if len(actual) != len(expected):
        raise ValueError("arrangement does not match the expected piece count")

    position_matches = sum(
        actual_id == expected_id
        for actual_id, expected_id in zip(actual, expected, strict=False)  # pragma: no mutate
    )
    # equivalent mutant: a layout always has at least one piece
    position_accuracy = position_matches / len(expected) if expected else 1.0  # pragma: no mutate

    columns = arrangement.layout.columns
    rows = arrangement.layout.rows
    horizontal_total = rows * max(0, columns - 1)
    vertical_total = max(0, rows - 1) * columns
    horizontal_matches = 0
    vertical_matches = 0
    for row in range(rows):
        row_start = row * columns
        for column in range(columns - 1):
            actual_index = row_start + column
            if (
                actual[actual_index] == expected[actual_index]
                and actual[actual_index + 1] == expected[actual_index + 1]
            ):
                horizontal_matches += 1
    for row in range(rows - 1):
        row_start = row * columns
        for column in range(columns):
            actual_index = row_start + column
            if (
                actual[actual_index] == expected[actual_index]
                and actual[actual_index + columns] == expected[actual_index + columns]
            ):
                vertical_matches += 1

    horizontal_accuracy = horizontal_matches / horizontal_total if horizontal_total else 1.0
    vertical_accuracy = vertical_matches / vertical_total if vertical_total else 1.0
    adjacency_total = horizontal_total + vertical_total
    adjacency_accuracy = (
        (horizontal_matches + vertical_matches) / adjacency_total if adjacency_total else 1.0
    )
    return ArrangementMetrics(
        position_accuracy=position_accuracy,
        horizontal_adjacency_accuracy=horizontal_accuracy,
        vertical_adjacency_accuracy=vertical_accuracy,
        adjacency_accuracy=adjacency_accuracy,
        exact=actual == expected,
    )
