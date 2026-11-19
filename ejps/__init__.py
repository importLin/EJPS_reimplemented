"""Ciphertext-only attack on block-wise EtC (encryption-then-compression) images.

Stages: key groups -> EJPS sub-block restoration -> candidate selection ->
block jigsaw -> evaluation. See README.md for the pipeline and the residual
ambiguities. Quick use::

    import ejps

    cipher, sigma = ejps.encrypt_image(plain, "common", seed=0, scramble=True)
    cfg = ejps.AttackConfig(solver="layer_greedy", jigsaw=True)
    result = ejps.attack_image(cipher, cfg)
    metrics = ejps.evaluate(result, plain, cfg, sigma)
"""

from .encrypt import EtcKey, decrypt_with_key, encrypt_frames, encrypt_image, generate_key
from .geometry import (
    EJPS_SOLVERS,
    KEY_GROUP_KINDS,
    KeyGroups,
    SubBlockCandidate,
    blocks_to_image,
    image_to_blocks,
    map_groups,
)
from .jigsaw import JigsawResult, solve_jigsaw
from .metrics import psnr, ssim, synthetic_natural_image
from .pipeline import (
    AttackConfig,
    AttackResult,
    attack_frames,
    attack_image,
    display_upright,
    evaluate,
)
from .restore import EdgeRefSolver, LayerGreedySolver, rank_candidates, select_best

__version__ = "0.1.0"

__all__ = [
    "EJPS_SOLVERS",
    "KEY_GROUP_KINDS",
    "AttackConfig",
    "AttackResult",
    "EdgeRefSolver",
    "EtcKey",
    "JigsawResult",
    "KeyGroups",
    "LayerGreedySolver",
    "SubBlockCandidate",
    "attack_frames",
    "attack_image",
    "blocks_to_image",
    "decrypt_with_key",
    "display_upright",
    "encrypt_frames",
    "encrypt_image",
    "evaluate",
    "generate_key",
    "image_to_blocks",
    "map_groups",
    "psnr",
    "rank_candidates",
    "select_best",
    "ssim",
    "solve_jigsaw",
    "synthetic_natural_image",
]
