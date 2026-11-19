"""Stage 3: block-level jigsaw solver (a port of the gaps genetic algorithm, MIT).

See THIRD_PARTY_LICENSE_gaps.txt at the repository root.
"""

from .domain import Arrangement, Direction, EdgeAxis, Piece, PuzzleLayout
from .fitness import EdgeCostTable, dissimilarity_measure, pairwise_dissimilarity
from .ga import GeneticAlgorithm, JigsawResult, solve_jigsaw

__all__ = [
    "Arrangement",
    "Direction",
    "EdgeAxis",
    "EdgeCostTable",
    "GeneticAlgorithm",
    "JigsawResult",
    "Piece",
    "PuzzleLayout",
    "dissimilarity_measure",
    "pairwise_dissimilarity",
    "solve_jigsaw",
]
