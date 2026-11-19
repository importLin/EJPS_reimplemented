"""Command-line interface behind ``run_attack.py`` (argument parsing, file IO, printing).

All attack logic lives in :mod:`ejps.pipeline`; this module only reads images,
calls it and writes PNGs (plus ``metrics.json`` / ``sigma.npy`` on request).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .encrypt import encrypt_image
from .geometry import KEY_GROUP_KINDS, crop_to_blocks
from .pipeline import AttackConfig, attack_image, candidates_to_images, display_upright, evaluate

IMAGE_SUFFIXES = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".ppm")
SOLVERS = ("edge_ref", "layer_greedy")

EXAMPLES = """examples:
  # encrypt a plaintext (common key, block scrambling), attack it and score it
  python run_attack.py --simulate examples/chelsea_224.png --key-groups common --scramble
  # same, layer_greedy solver with both fixes enabled
  python run_attack.py --simulate examples/chelsea_224.png --scramble --solver layer_greedy \\
      --solver-option fix_fourth_flip=true --solver-option one_plane_per_sub_block=true
  # attack an existing ciphertext (optionally scored against the plaintext)
  python run_attack.py --cipher cipher.png --jigsaw [--plain plain.png --sigma sigma.npy]
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_attack.py",
        description="Ciphertext-only attack on block-wise EtC images: EJPS sub-block "
        "restoration -> candidate selection -> block jigsaw -> evaluation.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cipher", metavar="PATH", help="cipher image (or directory) to attack")
    src.add_argument(
        "--simulate",
        metavar="PATH",
        help="plaintext image (or directory): encrypt with ejps.encrypt, then attack and score",
    )
    p.add_argument(
        "--plain", metavar="PATH", help="--cipher mode: plaintext image (or directory) for scoring"
    )
    p.add_argument(
        "--sigma",
        metavar="PATH",
        help="--cipher mode: block scrambling .npy (cipher[i] = plain[sigma[i]]), or a directory "
        "of <image stem>.npy",
    )
    p.add_argument("--solver", choices=SOLVERS, default="edge_ref", help="EJPS solver")
    p.add_argument(
        "--solver-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="solver keyword argument, repeatable (e.g. fix_fourth_flip=true)",
    )
    p.add_argument(
        "--key-groups",
        choices=KEY_GROUP_KINDS,
        default="common",
        help="which blocks share one key (for a single image per_position == independent)",
    )
    p.add_argument("--block-size", type=int, default=16, help="block size B (default 16)")
    p.add_argument("--sub-block-size", type=int, default=8, help="sub-block size B/2 (default 8)")
    p.add_argument("--scramble", action="store_true", help="--simulate: also scramble blocks")
    p.add_argument(
        "--jigsaw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run the block jigsaw (default: on for --simulate --scramble or when --sigma "
        "is given, off otherwise)",
    )
    p.add_argument("--population", type=int, default=30, help="jigsaw GA population (30)")
    p.add_argument("--generations", type=int, default=20, help="jigsaw GA generations (20)")
    p.add_argument("--seed", type=int, default=0, help="key seed (--simulate) and GA seed")
    p.add_argument("--top-k", type=int, default=1, help="save the k best-ranked candidates")
    p.add_argument("--out", default="outputs", help="output directory (default ./outputs)")
    p.add_argument(
        "--save-metrics",
        action="store_true",
        help="also write metrics.json (and sigma.npy for --simulate)",
    )
    return p


def parse_value(text: str):
    """``true`` / ``false`` / int / float / string."""
    low = text.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_options(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"solver option {item!r} is not KEY=VALUE")
        out[key.strip()] = parse_value(value)
    return out


def list_images(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(q for q in path.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no images in {path}")
        return files
    if not path.is_file():
        raise FileNotFoundError(path)
    return [path]


def read_image(path: Path, block_size: int) -> np.ndarray:
    """RGB uint8, cropped (bottom / right) to multiples of ``block_size``."""
    img = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(crop_to_blocks(img, block_size))


def companion(arg: str | None, image: Path, suffix: str | None) -> Path | None:
    """The ``--plain`` / ``--sigma`` file for ``image``.

    A directory argument is matched by name: ``<dir>/<image name>`` for
    ``suffix=None``, else ``<dir>/<image stem><suffix>``; a file is used as is.
    """
    if arg is None:
        return None
    path = Path(arg)
    if not path.is_dir():
        return path
    return path / (image.name if suffix is None else image.stem + suffix)


def run_one(image: Path, out: Path, cfg: AttackConfig, args) -> dict:
    """Attack one image, write its PNGs into ``out``; returns the metrics (or {})."""
    sigma = None
    if args.simulate:
        plain = read_image(image, cfg.block_size)
        cipher, sigma = encrypt_image(
            plain,
            cfg.key_groups,
            args.seed,
            block_size=cfg.block_size,
            sub_block_size=cfg.sub_block_size,
            scramble=args.scramble,
        )
    else:
        cipher = read_image(image, cfg.block_size)
        plain_path = companion(args.plain, image, None)
        plain = None if plain_path is None else read_image(plain_path, cfg.block_size)
        sigma_path = companion(args.sigma, image, ".npy")
        sigma = None if sigma_path is None else np.load(sigma_path)

    res = attack_image(cipher, cfg)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cipher).save(out / "cipher.png")
    Image.fromarray(res.restored[0]).save(out / "restored.png")
    Image.fromarray(res.reconstruction[0]).save(out / "reconstruction.png")
    rows, cols = cipher.shape[0] // cfg.block_size, cipher.shape[1] // cfg.block_size
    for k, img in enumerate(candidates_to_images(res.top_candidates, rows, cols)):
        Image.fromarray(img).save(out / f"candidate_{k}.png")
    if plain is None:
        return {"timings": res.timings}

    # upright copies for figures (orientation only, colours as attacked; uses the plaintext)
    up = display_upright(res.restored[0], plain, cfg.block_size, sigma)
    Image.fromarray(up).save(out / "restored_upright.png")
    if res.orders is None or cipher.shape[0] == cipher.shape[1]:
        blk = None if res.orders is not None else cfg.block_size
        up = display_upright(res.reconstruction[0], plain, blk)
        Image.fromarray(up).save(out / "reconstruction_upright.png")
    metrics = evaluate(res, plain, cfg, sigma)
    if args.save_metrics:
        report = {"image": str(image), "config": dataclasses.asdict(cfg), "metrics": metrics}
        (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
        if sigma is not None and args.simulate:
            np.save(out / "sigma.npy", sigma)
    return metrics


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


TABLE_COLUMNS = (
    ("image", None),
    ("ejps_sub", ("ejps_global", "sub_block_rate")),
    ("ejps_block", ("ejps_global", "block_rate")),
    ("oracle_sub", ("ejps_oracle", "sub_block_rate")),
    ("jig_direct", ("jigsaw_direct",)),
    ("jig_neigh", ("jigsaw_neighbor",)),
    ("psnr", ("final", "psnr")),
    ("ssim", ("final", "ssim")),
    ("ejps_s", ("timings", "ejps_s")),
    ("jigsaw_s", ("timings", "jigsaw_s")),
)


def metrics_table(rows: list[tuple[str, dict]]) -> str:
    """A compact fixed-width table, one line per image."""

    def get(m: dict, path):
        for k in path:
            if not isinstance(m, dict) or k not in m:
                return None
            m = m[k]
        return m

    lines = [[name for name, _ in TABLE_COLUMNS]]
    for image, m in rows:
        lines.append([image] + [_fmt(get(m, path)) for _, path in TABLE_COLUMNS[1:]])
    widths = [max(len(r[i]) for r in lines) for i in range(len(TABLE_COLUMNS))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in lines
    )


def config_from_args(args) -> AttackConfig:
    jigsaw = args.jigsaw
    if jigsaw is None:
        jigsaw = bool((args.simulate and args.scramble) or (args.cipher and args.sigma))
    return AttackConfig(
        solver=args.solver,
        solver_options=parse_options(args.solver_option),
        key_groups=args.key_groups,
        block_size=args.block_size,
        sub_block_size=args.sub_block_size,
        jigsaw=jigsaw,
        jigsaw_population=args.population,
        jigsaw_generations=args.generations,
        seed=args.seed,
        keep_top=args.top_k,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.simulate and (args.plain or args.sigma):
        parser.error("--plain / --sigma only apply to --cipher mode")
    try:
        cfg = config_from_args(args)
    except ValueError as err:
        parser.error(str(err))
    src = Path(args.simulate or args.cipher)
    images = list_images(src)
    batch = src.is_dir()
    out = Path(args.out)
    rows = []
    for image in images:
        target = out / image.stem if batch else out
        rows.append((image.name, run_one(image, target, cfg, args)))
    print(metrics_table(rows))
    print(f"outputs -> {out}")
    return 0
