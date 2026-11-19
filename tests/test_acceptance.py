"""End-to-end acceptance numbers (224 x 224, block scrambling, default GA settings).

These are the reference numbers quoted in README.md.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ejps import AttackConfig, attack_image, encrypt_image, evaluate, synthetic_natural_image

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "chelsea_224.png"
FIXED_LAYER = {"fix_fourth_flip": True, "one_plane_per_sub_block": True}
SOLVERS = [("edge_ref", {}), ("layer_greedy", FIXED_LAYER)]


def _run(plain, kind, solver, opts, jigsaw):
    cipher, sigma = encrypt_image(plain, kind, 0, scramble=True)
    cfg = AttackConfig(solver=solver, solver_options=opts, key_groups=kind, jigsaw=jigsaw)
    return evaluate(attack_image(cipher, cfg), plain, cfg, sigma)


@pytest.mark.parametrize("solver,opts", SOLVERS)
def test_common_key_synthetic(solver, opts):
    m = _run(synthetic_natural_image(), "common", solver, opts, jigsaw=True)
    assert m["ejps_global"]["sub_block_rate"] == 1.0
    assert m["ejps_global"]["block_rate"] == 1.0
    assert m["jigsaw_direct"] == 1.0 and m["jigsaw_neighbor"] == 1.0
    assert m["final"] == {"psnr": float("inf"), "ssim": 1.0}


@pytest.mark.parametrize("solver,opts", SOLVERS)
def test_independent_key_synthetic_resists(solver, opts):
    m = _run(synthetic_natural_image(), "independent", solver, opts, jigsaw=False)
    assert m["ejps_global"]["sub_block_rate"] < 0.05
    assert m["ejps_global"]["block_rate"] == 0.0
    assert m["final"]["psnr"] < 14.0


def test_example_image_is_the_documented_crop():
    img = np.asarray(Image.open(EXAMPLE))
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8


@pytest.mark.parametrize(
    "solver,opts,direct", [("edge_ref", {}, 0.95), ("layer_greedy", FIXED_LAYER, 0.98)]
)
def test_common_key_example_image(solver, opts, direct):
    plain = np.asarray(Image.open(EXAMPLE).convert("RGB"))
    m = _run(plain, "common", solver, opts, jigsaw=True)
    assert m["ejps_global"]["sub_block_rate"] == 1.0
    assert m["ejps_oracle"]["block_rate"] == 1.0
    assert m["jigsaw_direct"] >= direct
    assert m["jigsaw_neighbor"] >= 0.95
    assert m["final"]["psnr"] > 30.0 and m["final"]["ssim"] > 0.95
