"""Regenerate the golden fixtures in ``tests/fixtures/``.

    python tools/make_fixtures.py            # writes tests/fixtures/golden.{json,npz}
    python tools/make_fixtures.py --check    # recompute and compare, write nothing

The golden values pin the exact output of the attack (solvers, ranking, metrics,
pipeline numbers) on deterministic inputs: synthetic images / seeded noise,
encrypted with :mod:`ejps.encrypt`. The committed fixtures were produced by
the original implementation this package was refactored from (verified
bit-identical to it); this script recomputes them with ``ejps`` itself, so
``--check`` passing means the package still reproduces them exactly.

``build_golden(impl)`` takes the implementation as a namespace so that the
same input recipe can be run against another implementation of the same API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ejps.encrypt import encrypt_frames, encrypt_image  # noqa: E402
from ejps.geometry import image_to_blocks  # noqa: E402
from ejps.metrics import synthetic_natural_image  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"

# (plain seed, key seed, invert cipher channel 1, expected diagonal); 64 x 64 images
EDGE_CASES = (
    (0, 100, False, "ur"),
    (2, 102, False, "dl"),
    (5, 105, False, "dr"),
    (1, 77, True, "dl"),
)
LAYER_CASES = ("synth0", "synth3", "noise")
LAYER_FLAGS = {
    "default": {},
    "fix_fourth_flip": {"fix_fourth_flip": True},
    "one_plane_per_sub_block": {"one_plane_per_sub_block": True},
    "both": {"fix_fourth_flip": True, "one_plane_per_sub_block": True},
}
SELECT_SEEDS = (0, 1, 2, 3, 4, 5)


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def edge_input(seed: int, key: int, invert: bool) -> tuple[np.ndarray, np.ndarray]:
    """(plain blocks, cipher blocks) of a 64 x 64 synthetic image, common key."""
    plain = synthetic_natural_image(size=64, seed=seed)
    cipher, _ = encrypt_image(plain, "common", key)
    if invert:  # a global per-channel inversion: relative inversion between channels
        cipher[..., 1] = 255 - cipher[..., 1]
    return image_to_blocks(plain, 16), image_to_blocks(cipher, 16)


def noise_input() -> np.ndarray:
    """Cipher blocks of seeded 48 x 64 noise."""
    noise = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    return image_to_blocks(encrypt_image(noise, "common", 9)[0], 16)


def layer_input(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(plain blocks, cipher blocks) of a 224 x 224 input, common key."""
    if name == "synth0":
        plain, key = synthetic_natural_image(seed=0), 11
    elif name == "synth3":
        plain, key = synthetic_natural_image(seed=3), 44
    else:
        plain, key = np.random.default_rng(123).integers(0, 256, (224, 224, 3), np.uint8), 200
    return image_to_blocks(plain, 16), image_to_blocks(encrypt_image(plain, "common", key)[0], 16)


def select_input(seed: int) -> tuple[np.ndarray, np.ndarray]:
    plain = synthetic_natural_image(seed=seed)
    cipher, _ = encrypt_image(plain, "common", 500 + seed)
    return image_to_blocks(plain, 16), image_to_blocks(cipher, 16)


def metrics_input() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (24, 20, 3)).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-9, 10, a.shape), 0, 255).astype(np.uint8)
    return a, b


def per_position_input(t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = synthetic_natural_image(size=64, seed=0)
    plain = np.stack([np.roll(base, f, axis=1) for f in range(t)])
    enc, sigma = encrypt_frames(plain, "per_position", 0)
    return plain, enc, sigma


def build_golden(impl) -> tuple[dict, dict]:
    """Golden JSON values and arrays computed with ``impl`` (see :func:`ejps_impl`)."""
    js: dict = {}
    arr: dict = {}

    plane = np.random.default_rng(0).integers(0, 256, (8, 8)).astype(np.uint8)
    arr["plane_transforms"] = np.stack([impl.plane_transform(plane, i) for i in range(16)])

    # -- edge_ref ---------------------------------------------------------------
    js["edge"] = []
    for k, (seed, key, inv, _) in enumerate(EDGE_CASES):
        _, blocks = edge_input(seed, key, inv)
        cand = impl.EdgeRefSolver().solve(blocks, sub_block_size=8)[0]
        arr[f"edge_case{k}"] = cand.blocks
        js["edge"].append(
            {
                "input_sha": sha(blocks),
                "score": cand.score,
                "meta": cand.meta,
                "scramble_compat": impl.scramble_compat(impl.split_sub_blocks(blocks, 8)).tolist(),
            }
        )
    blocks = noise_input()
    js["edge_noise"] = {"input_sha": sha(blocks), "scores": {}}
    for stat in (None, 5, 2, 1):
        cand = impl.EdgeRefSolver(stat_blocks=stat).solve(blocks)[0]
        arr[f"edge_noise_stat{stat}"] = cand.blocks
        js["edge_noise"]["scores"][str(stat)] = cand.score

    # -- layer_greedy -----------------------------------------------------------
    js["layer"] = {}
    for name in LAYER_CASES:
        plain, blocks = layer_input(name)
        entry = {"input_sha": sha(blocks)}
        for flag, opts in LAYER_FLAGS.items():
            cands = impl.LayerGreedySolver(**opts).solve(blocks, sub_block_size=8)
            entry[flag] = {
                "sha": sha(np.stack([c.blocks for c in cands])),
                "score": cands[0].score,
                "layers": cands[0].meta["layers"],
                "equivalent": [
                    k for k, c in enumerate(cands) if impl.is_equivalent(c.blocks, plain)
                ],
            }
        js["layer"][name] = entry

    # -- candidate ranking ------------------------------------------------------
    js["select"] = {}
    for feature in ("pixel", "laplacian"):
        rows = []
        for seed in SELECT_SEEDS:
            plain, blocks = select_input(seed)
            ranked = impl.rank_candidates(impl.LayerGreedySolver().solve(blocks), feature=feature)
            rows.append(
                {
                    "input_sha": sha(blocks),
                    "top": [r.index for r in ranked[:3]],
                    "scores": [r.score for r in ranked[:3]],
                    "rho": list(ranked[0].rho),
                    "invert": list(ranked[0].invert),
                    "top_equivalent": bool(impl.is_equivalent(ranked[0].blocks, plain)),
                }
            )
        js["select"][feature] = rows
    _, blocks = edge_input(*EDGE_CASES[3][:3])
    ranked = impl.rank_candidates(impl.EdgeRefSolver().solve(blocks))
    js["select_edge_inverted"] = {"rho": list(ranked[0].rho), "invert": list(ranked[0].invert)}

    # -- metrics ----------------------------------------------------------------
    a, b = metrics_input()
    js["metrics"] = {
        "psnr": impl.psnr(a, b),
        "ssim": impl.ssim(a, b),
        "ssim_gray": impl.ssim(a[..., 0], b[..., 0]),
    }

    # -- pipeline ---------------------------------------------------------------
    js["pipeline"] = {}
    plain = synthetic_natural_image(size=64, seed=0)
    enc, sigma = encrypt_image(plain, "independent", 0)
    cfg = impl.AttackConfig(key_groups=impl.kind("independent"))
    m = impl.evaluate(impl.attack_frames(enc[None], cfg), plain, cfg, sigma)
    m.pop("timings")
    js["pipeline"]["independent_64"] = m
    for t in (4, 8):
        plain, enc, sigma = per_position_input(t)
        cfg = impl.AttackConfig(key_groups=impl.kind("per_position"))
        m = impl.evaluate(impl.attack_frames(enc, cfg), plain, cfg, sigma)
        m.pop("timings")
        js["pipeline"][f"per_position_{t}"] = m
    plain = synthetic_natural_image(size=64, seed=0)
    enc, sigma = encrypt_image(plain, "common", 0, scramble=True)
    cfg = impl.AttackConfig(jigsaw=True)
    res = impl.attack_frames(enc[None], cfg)
    m = impl.evaluate(res, plain, cfg, sigma)
    m.pop("timings")
    js["pipeline"]["common_scramble_64"] = {"metrics": m, "order": res.orders[0].tolist()}
    return js, arr


def ejps_impl() -> SimpleNamespace:
    from ejps import geometry, metrics, pipeline
    from ejps.restore import edge, layer_greedy, select

    return SimpleNamespace(
        plane_transform=geometry.plane_transform,
        split_sub_blocks=geometry.split_sub_blocks,
        EdgeRefSolver=edge.EdgeRefSolver,
        scramble_compat=edge.scramble_compat,
        LayerGreedySolver=layer_greedy.LayerGreedySolver,
        rank_candidates=select.rank_candidates,
        is_equivalent=metrics.is_equivalent,
        psnr=metrics.psnr,
        ssim=metrics.ssim,
        AttackConfig=pipeline.AttackConfig,
        attack_frames=pipeline.attack_frames,
        evaluate=pipeline.evaluate,
        kind=lambda k: k,
    )


def normalise(js: dict) -> dict:
    """JSON round trip (tuples -> lists, numpy scalars -> float)."""
    return json.loads(json.dumps(js, default=float))


def write(js: dict, arr: dict, out: Path = FIXTURES) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden.json").write_text(json.dumps(normalise(js), indent=1, default=float) + "\n")
    np.savez_compressed(out / "golden.npz", **arr)


def compare(js: dict, arr: dict, out: Path = FIXTURES) -> list[str]:
    """Differences between freshly computed values and the stored fixtures."""
    diffs = []
    if normalise(js) != json.loads((out / "golden.json").read_text()):
        diffs.append("golden.json")
    with np.load(out / "golden.npz") as stored:
        if set(stored.files) != set(arr):
            diffs.append("golden.npz keys")
        diffs += [k for k in arr if k in stored.files and not np.array_equal(stored[k], arr[k])]
    return diffs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="compare with the stored fixtures")
    a = p.parse_args(argv)
    js, arr = build_golden(ejps_impl())
    if a.check:
        diffs = compare(js, arr)
        print("fixtures match" if not diffs else f"MISMATCH: {diffs}")
        return 1 if diffs else 0
    write(js, arr)
    print(f"wrote {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
