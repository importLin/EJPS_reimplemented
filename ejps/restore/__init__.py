"""Stages 1-2: EJPS sub-block restoration (two solvers) and candidate selection.

Importing this package registers the solvers ``"edge_ref"`` and
``"layer_greedy"`` in :data:`ejps.geometry.EJPS_SOLVERS`.
"""

from .edge import EdgeRefSolver
from .layer_greedy import LayerGreedySolver
from .select import RankedCandidate, channel_correlations, rank_candidates, select_best

__all__ = [
    "EdgeRefSolver",
    "LayerGreedySolver",
    "RankedCandidate",
    "channel_correlations",
    "rank_candidates",
    "select_best",
]
