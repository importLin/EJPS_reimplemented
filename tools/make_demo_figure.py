"""Build the README demo figure ``docs/demo.png``.

    python tools/make_demo_figure.py [--image examples/chelsea_224.png] [--out docs/demo.png]

Two rows (one key for all blocks / an independent key per block), four columns:
plaintext | ciphertext | after EJPS | after jigsaw. The last two columns are turned
upright with the plaintext as an orientation oracle (display only; colours are
left exactly as the attack produced them).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ejps  # noqa: E402

ROWS = (("common", "One key for all blocks"), ("independent", "Independent key per block"))
COLUMNS = ("Plaintext", "Ciphertext", "After EJPS", "After jigsaw")
GAP = 8
FONT_SIZE = 18
BG = (255, 255, 255)


def _font(size: int):
    """Default PIL font at ``size`` px (falls back to the fixed bitmap font on old Pillow)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def run_row(plain: np.ndarray, key_groups: str) -> list[np.ndarray]:
    """Attack one setting; return the four images [plain, cipher, EJPS, jigsaw]."""
    cipher, sigma = ejps.encrypt_image(plain, key_groups, seed=0, scramble=True)
    cfg = ejps.AttackConfig(
        solver="layer_greedy",
        solver_options={"fix_fourth_flip": True, "one_plane_per_sub_block": True},
        key_groups=key_groups,
        jigsaw=True,
    )
    res = ejps.attack_image(cipher, cfg)
    ejps_up = ejps.display_upright(res.restored[0], plain, cfg.block_size, sigma)
    jig_up = ejps.display_upright(res.reconstruction[0], plain)
    return [plain, cipher, ejps_up, jig_up]


def compose(rows: list[list[np.ndarray]]) -> Image.Image:
    """Tile the rows with white gaps, column headers on top and row labels on the left."""
    font = _font(FONT_SIZE)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def size(text: str) -> tuple[int, int]:
        left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
        return right - left, bottom - top

    tile = rows[0][0].shape[0]
    label_w = max(size(label)[0] for _, label in ROWS) + 2 * GAP
    head_h = max(size(c)[1] for c in COLUMNS) + 2 * GAP
    width = label_w + len(COLUMNS) * (tile + GAP)
    height = head_h + len(rows) * (tile + GAP)
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(COLUMNS):
        x0 = label_w + c * (tile + GAP)
        w, h = size(name)
        draw.text((x0 + (tile - w) // 2, (head_h - h) // 2), name, fill=(0, 0, 0), font=font)
    for r, ((_, label), images) in enumerate(zip(ROWS, rows, strict=True)):
        y0 = head_h + r * (tile + GAP)
        w, h = size(label)
        draw.text((GAP, y0 + (tile - h) // 2), label, fill=(0, 0, 0), font=font)
        for c, img in enumerate(images):
            canvas.paste(Image.fromarray(img), (label_w + c * (tile + GAP), y0))
    return canvas


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", default="examples/chelsea_224.png")
    ap.add_argument("--out", default="docs/demo.png")
    args = ap.parse_args(argv)
    plain = np.asarray(Image.open(args.image).convert("RGB"))
    rows = [run_row(plain, kg) for kg, _ in ROWS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 256-colour palette keeps the noisy tiles under ~600 KB
    compose(rows).quantize(256, dither=Image.Dither.NONE).save(out, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
