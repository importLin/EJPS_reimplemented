# EJPS_reimplemented

A reimplementation of a **ciphertext-only attack on block-wise EtC
(encryption-then-compression) images**: extended jigsaw puzzle solver (EJPS)
sub-block restoration [1] followed by a block-level jigsaw solver ported from
gaps [2]. Pure NumPy / SciPy / Pillow, CPU only.

Given only a cipher image produced by block scrambling, sub-block shuffling,
rotation / flip, negative-positive transformation and colour channel shuffling,
the attack recovers the plaintext up to a small residual ambiguity whenever
many blocks share one key. When every block has its own key, it fails, which is
the point of that setting.

<p align="center"><code>cipher -> key groups -> EJPS restoration -> candidate selection -> block jigsaw -> evaluation</code></p>

## Install

```bash
git clone <this repository> && cd EJPS_reimplemented
pip install -e .            # or: pip install numpy scipy pillow, then run from the repo root
pip install -e ".[dev]"     # + pytest, ruff
```

Python >= 3.10. Dependencies: `numpy`, `scipy`, `pillow`.

## Quickstart

```bash
# encrypt the example image (one key for all blocks, block scrambling), attack it, score it
python run_attack.py --simulate examples/chelsea_224.png --key-groups common --scramble --jigsaw

# layer_greedy solver with both fixes enabled
python run_attack.py --simulate examples/chelsea_224.png --key-groups common --scramble \
    --solver layer_greedy --solver-option fix_fourth_flip=true \
    --solver-option one_plane_per_sub_block=true

# one independent key per block: the attack fails
python run_attack.py --simulate examples/chelsea_224.png --key-groups independent --scramble

# attack an existing cipher image (scoring needs the plaintext and, if blocks were
# scrambled, sigma with cipher_block[i] = plain_block[sigma[i]])
python run_attack.py --cipher cipher.png --jigsaw [--plain plain.png --sigma sigma.npy]

# a directory works too: one output subfolder per image
python run_attack.py --simulate my_images/ --scramble --out outputs/batch
```

The first command prints something like

```
image            ejps_sub  ejps_block  oracle_sub  jig_direct  jig_neigh  psnr     ssim    ejps_s  jigsaw_s
chelsea_224.png  1.0000    1.0000      1.0000      0.9541      0.9560     34.4546  0.9647  0.0706  2.1390
```

and writes PNGs to `./outputs` (`--out` to change):

| file | content |
|---|---|
| `cipher.png` | the attacked cipher image |
| `restored.png` | after EJPS + selection; blocks still at their cipher positions |
| `reconstruction.png` | after the block jigsaw (equal to `restored.png` without `--jigsaw`) |
| `candidate_<k>.png` | the `--top-k` best-ranked EJPS candidates |
| `*_upright.png` | (plaintext known) the same, turned upright for display only |

`metrics.json` (and `sigma.npy` in `--simulate` mode) is written only with
`--save-metrics`. Run `python run_attack.py --help` for all options
(`--block-size`, `--sub-block-size`, `--population`, `--generations`, `--seed`, ...).
The `*_upright.png` files use the plaintext to pick the display orientation;
they are an oracle for figures, never part of the attack.

## Python API

```python
import numpy as np
from PIL import Image
import ejps

plain = np.asarray(Image.open("examples/chelsea_224.png").convert("RGB"))
cipher, sigma = ejps.encrypt_image(plain, "common", seed=0, scramble=True)

cfg = ejps.AttackConfig(
    solver="layer_greedy",
    solver_options={"fix_fourth_flip": True, "one_plane_per_sub_block": True},
    key_groups="common",
    jigsaw=True,
)
result = ejps.attack_image(cipher, cfg)  # result.restored, result.reconstruction, result.orders
metrics = ejps.evaluate(result, plain, cfg, sigma)
print(metrics["ejps_global"]["sub_block_rate"], metrics["jigsaw_direct"])
```

Lower-level pieces: `ejps.EJPS_SOLVERS.build(name, **options).solve(blocks, sub_block_size=8)`
returns `SubBlockCandidate`s for one key group, `ejps.rank_candidates` /
`ejps.select_best` rank them, `ejps.solve_jigsaw(image, piece_size)` reassembles
permuted blocks, and `ejps.attack_frames` handles frame stacks `(T, H, W, 3)`.

## How the attack works

Notation: blocks of `B x B` pixels (default 16), each split into a 2x2 grid of
`S x S` sub-blocks (`S = B / 2`, default 8), sub-blocks `ul, ur, dl, dr`. A
*plane* is one colour channel of one sub-block.

**Stage 0 - key groups** (`ejps.geometry.KeyGroups`). The attack pools edge
statistics over all blocks that share one key, so the grouping decides its strength:

| `--key-groups` | setting | blocks per group |
|---|---|---|
| `common` | one key for every block of the image (Com-B/S) | all |
| `independent` | an independent key for every block (Ind-B/S) | 1 |
| `per_position` | frame stacks: one key per block position, reused by every frame | T (frames) |

**Stage 1 - EJPS sub-block restoration** [1] (`ejps.restore`). Inside every
block the sub-blocks are shuffled, turned, flipped, inverted and their channels
shuffled, all with the same key across the group. Two solvers undo this from
edge continuity alone:

- `edge_ref`: takes the cipher `ul` sub-block as reference, finds its diagonal
  partner (the sub-block whose edges match `ul` worst), then solves the three
  channels in turn: the 16 plane transforms (4 rotations x flip x inversion) and
  the source channel of `ur`, `dl` and finally `dr`, minimising the edge MSE
  summed over the group. One candidate.
- `layer_greedy`: treats the 12 planes of a block as jigsaw pieces and assembles
  three 2x2 layers greedily (anchor plane, best neighbour, orthogonal neighbour,
  fourth plane). The relative orientation of the second and third layer is
  unknown, so it emits 8 x 8 = 64 candidates. Options `fix_fourth_flip` and
  `one_plane_per_sub_block` (both off by default, which reproduces the original
  behaviour) fix two weaknesses of the original greedy search; see the module
  docstring.

**Stage 2 - candidate selection** (`ejps.restore.select`). In natural images the
colour channels of a block are strongly positively correlated. Every candidate
is scored by the mean per-block Pearson correlation of its channel pairs
(pixels, or their Laplacian with `select_feature="laplacian"`), after choosing
the channel inversions that maximise it; the best candidate is kept.

**Stage 3 - block jigsaw** [2] (`ejps.jigsaw`). If block positions were
scrambled, the restored blocks are reassembled by the gaps genetic algorithm
(pairwise edge dissimilarity, kernel-growing crossover, local search, restarts).
Only gaps' serial path is ported; for a given seed it reproduces upstream's
sequence of random draws. Pieces are never rotated (the residual rotation is
common to all blocks, so the whole image is assembled in that orientation).

**Stage 4 - evaluation** (`ejps.metrics`, `ejps.pipeline.evaluate`), when the
plaintext is known; see the metric definitions below.

### Residual ambiguity

Edge continuity only determines every sub-block *relative* to a reference, so
per key group the attack cannot recover

- one dihedral transform (4 rotations x flip) shared by all blocks of the group,
- the absolute order of the three colour channels (6),
- the inversion `255 - x` of each channel (2^3).

All quality numbers are therefore computed after aligning the reconstruction to
the plaintext with the best element of this 384-element group. With a common
key that is one element for the whole image (the image comes back rotated /
recoloured as a whole); with independent keys every block keeps its own
residual transform, which one global alignment cannot undo.

## The bundled encryptor (`ejps.encrypt`)

`--simulate` encrypts plaintexts with a standalone EtC-style encryptor written
from the published description of block scrambling-based EtC images (block
scrambling, block rotation and inversion, negative-positive transformation,
colour component shuffling), extended with the 2x2 sub-block layer that EJPS
targets. For block key `k`, cipher sub-block slot `j` is

```
x = plain_sub_block[k.sub_perm[j]]      # sub-block shuffle within the block
x = rot90(x, k.rot[j])                  # 0 / 90 / 180 / 270 degrees
x = fliplr(x) if k.flip[j]              # flip
x = 255 - x if k.nega[j]                # negative-positive transformation
x = x[..., k.chan[j]]                   # colour channel shuffle
```

Every op acts on a whole sub-block (all channels together). Optional block
scrambling then permutes the encrypted blocks, `cipher_block[i] =
enc(plain_block[sigma[i]])`; keys follow the plaintext position. Keys are drawn
deterministically from `--seed`; `ejps.encrypt.decrypt_with_key` inverts the
cipher. It is a test-data generator, not a secure cipher.

## Metrics

| key in `evaluate(...)` | meaning |
|---|---|
| `ejps_global.sub_block_rate` | fraction of sub-blocks identical to the plaintext sub-block at the same cipher position, after ONE residual alignment for the whole image (what the attacker actually gets; printed as `ejps_sub`) |
| `ejps_global.block_rate`, `.psnr`, `.ssim` | the same alignment: fraction of identical blocks, PSNR / SSIM |
| `ejps_oracle.*` | the same numbers with the best alignment per key group: an upper bound that forgives every residual ambiguity (printed as `oracle_sub`) |
| `jigsaw_direct` | fraction of output positions holding the correct block (best over the whole-image symmetries) |
| `jigsaw_neighbor` | fraction of true right / below neighbour pairs that are neighbours in the output |
| `final.psnr`, `final.ssim` | final reconstruction vs plaintext under one alignment per image (after a jigsaw: square images only) |
| `timings.ejps_s`, `timings.jigsaw_s` | wall-clock seconds of stages 1-2 and 3 |

PSNR uses a peak of 255; SSIM is the Gaussian-window SSIM (sigma 1.5),
averaged over channels.

## Reference numbers

224 x 224 images, B = 16, S = 8, block scrambling on, key seed 0, jigsaw with
the defaults (population 30, 20 generations, seed 0); `layer_greedy` with
`fix_fourth_flip=true one_plane_per_sub_block=true`. Timings: one CPU process
(Python 3.11, NumPy 2.4).

| image | key groups | solver | `ejps_sub` | `oracle_sub` | `jig_direct` | `jig_neigh` | PSNR (dB) | SSIM | EJPS (s) | jigsaw (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic | common | edge_ref | 1.0000 | 1.0000 | 1.0000 | 1.0000 | inf | 1.0000 | 0.09 | 2.12 |
| synthetic | common | layer_greedy | 1.0000 | 1.0000 | 1.0000 | 1.0000 | inf | 1.0000 | 0.18 | 2.12 |
| synthetic | independent | edge_ref | 0.0000 | 0.2487 | 0.0153 | 0.0357 | 13.86 | 0.4048 | 0.62 | 2.48 |
| synthetic | independent | layer_greedy | 0.0064 | 0.1378 | 0.0153 | 0.0220 | 14.11 | 0.4033 | 1.34 | 2.47 |
| chelsea | common | edge_ref | 1.0000 | 1.0000 | 0.9541 | 0.9560 | 34.45 | 0.9647 | 0.06 | 2.14 |
| chelsea | common | layer_greedy | 1.0000 | 1.0000 | 0.9847 | 0.9725 | 39.02 | 0.9870 | 0.18 | 2.09 |
| chelsea | independent | edge_ref | 0.0000 | 0.1518 | 0.0102 | 0.0137 | 11.60 | 0.1470 | 0.61 | 2.39 |
| chelsea | independent | layer_greedy | 0.0013 | 0.0931 | 0.0306 | 0.0110 | 11.58 | 0.1500 | 1.34 | 2.39 |

"synthetic" is `ejps.synthetic_natural_image()` (seed 0); "chelsea" is
`examples/chelsea_224.png`. With a common key the sub-blocks are restored
exactly by both solvers; the remaining error on the photograph comes from a few
misplaced blocks in the jigsaw stage. The two solvers' outputs differ only by
the residual transform, which changes the jigsaw's random trajectory. These
numbers are locked by `tests/test_acceptance.py`.

## Example image

`examples/chelsea_224.png` is a plain 224 x 224 crop (box `(128, 48, 352, 272)`)
of the CC0 "chelsea" image from scikit-image (photographer Stefan van der Walt);
see `examples/README.md`.

## Development

```bash
python -m pytest -q                  # about 30 s
ruff check . && ruff format --check .
python tools/make_fixtures.py --check
```

The solvers are refactors of the author's original notebook implementation (not
included). `tests/fixtures/golden.{json,npz}` hold golden outputs produced by
that original implementation on deterministic inputs (synthetic images and
seeded noise encrypted with `ejps.encrypt`); the tests require this package to
reproduce them bit for bit, and `tools/make_fixtures.py --check` recomputes them.

Layout:

```
run_attack.py          entry point (argparse -> ejps.cli)
ejps/geometry.py       blocks, sub-blocks, transform groups, edges, KeyGroups, solver registry
ejps/restore/          EJPS solvers (edge.py, layer_greedy.py) and candidate selection (select.py)
ejps/jigsaw/           gaps port: domain.py, fitness.py (dissimilarity + cost table), ga.py
ejps/metrics.py        residual alignment, PSNR / SSIM, jigsaw accuracy, synthetic test image
ejps/pipeline.py       AttackConfig, attack_image / attack_frames, evaluate, display_upright
ejps/encrypt.py        the bundled EtC encryptor
```

## Credits and license

MIT License, see `LICENSE`. The block jigsaw in `ejps/jigsaw/` is a port of
gaps by Nemanja Milićević [2] (MIT; notice in `THIRD_PARTY_LICENSE_gaps.txt`).

## References

[1] T. Chuman, K. Kurihara, and H. Kiya, "On the security of block scrambling-based
EtC systems against extended jigsaw puzzle solver attacks," *IEICE Transactions on
Information and Systems*, vol. E101.D, no. 1, pp. 37–44, 2018.

[2] Nemanja Milićević, *gaps*, https://github.com/nemanja-m/gaps (MIT License).

If you use this code, please cite [1] for the attack and credit [2] for the
jigsaw solver.
