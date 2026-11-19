"""Tests for ejps/cli.py and the run_attack.py entry point."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ejps import cli, encrypt_image, synthetic_natural_image

ROOT = Path(__file__).resolve().parents[1]


def _write(path, img):
    Image.fromarray(img).save(path)
    return path


def test_parse_value_and_options():
    assert cli.parse_value("true") is True and cli.parse_value("Off") is False
    assert cli.parse_value("3") == 3 and isinstance(cli.parse_value("3"), int)
    assert cli.parse_value("0.5") == 0.5 and cli.parse_value("abc") == "abc"
    assert cli.parse_options(["fix_fourth_flip=true", " stat_blocks = 5"]) == {
        "fix_fourth_flip": True,
        "stat_blocks": 5,
    }
    for bad in ("novalue", "=3"):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            cli.parse_options([bad])


def test_config_from_args_jigsaw_default():
    p = cli.build_parser()
    cfg = cli.config_from_args(p.parse_args(["--simulate", "x.png", "--scramble"]))
    assert cfg.jigsaw is True and cfg.solver == "edge_ref" and cfg.key_groups == "common"
    assert (cfg.jigsaw_population, cfg.jigsaw_generations, cfg.keep_top) == (30, 20, 1)
    assert cli.config_from_args(p.parse_args(["--simulate", "x.png"])).jigsaw is False
    assert cli.config_from_args(p.parse_args(["--cipher", "x.png"])).jigsaw is False
    a = p.parse_args(["--cipher", "x.png", "--sigma", "s.npy"])
    assert cli.config_from_args(a).jigsaw is True
    a = p.parse_args(["--simulate", "x.png", "--scramble", "--no-jigsaw", "--top-k", "3"])
    cfg = cli.config_from_args(a)
    assert cfg.jigsaw is False and cfg.keep_top == 3
    a = p.parse_args(
        ["--cipher", "x.png", "--solver", "layer_greedy", "--solver-option", "fix_fourth_flip=1"]
    )
    assert cli.config_from_args(a).solver_options == {"fix_fourth_flip": 1}
    with pytest.raises(SystemExit):
        p.parse_args([])
    with pytest.raises(SystemExit):
        p.parse_args(["--cipher", "a.png", "--simulate", "b.png"])


def test_metrics_table():
    m = {
        "ejps_global": {"sub_block_rate": 1.0, "block_rate": 0.5},
        "final": {"psnr": float("inf"), "ssim": 1.0},
    }
    lines = cli.metrics_table([("a.png", m), ("b.png", {})]).splitlines()
    assert lines[0].split()[:3] == ["image", "ejps_sub", "ejps_block"]
    assert lines[1].split() == [
        "a.png",
        "1.0000",
        "0.5000",
        "-",
        "-",
        "-",
        "inf",
        "1.0000",
        "-",
        "-",
    ]
    assert lines[2].split()[1:] == ["-"] * 9


def test_simulate_single_image(tmp_path, capsys):
    src = _write(tmp_path / "plain.png", synthetic_natural_image(size=64, seed=0))
    out = tmp_path / "out"
    rc = cli.main(
        ["--simulate", str(src), "--scramble", "--out", str(out), "--top-k", "2", "--save-metrics"]
    )
    assert rc == 0
    names = sorted(q.name for q in out.iterdir())
    assert names == [
        "candidate_0.png",
        "cipher.png",
        "metrics.json",
        "reconstruction.png",
        "reconstruction_upright.png",
        "restored.png",
        "restored_upright.png",
        "sigma.npy",
    ]
    text = capsys.readouterr().out
    assert text.splitlines()[1].split()[:6] == [
        "plain.png",
        "1.0000",
        "1.0000",
        "1.0000",
        "1.0000",
        "1.0000",
    ]
    report = json.loads((out / "metrics.json").read_text())
    assert report["metrics"]["jigsaw_direct"] == 1.0
    assert report["config"]["jigsaw"] is True
    cipher, sigma = encrypt_image(
        synthetic_natural_image(size=64, seed=0), "common", 0, scramble=True
    )
    np.testing.assert_array_equal(np.asarray(Image.open(out / "cipher.png")), cipher)
    np.testing.assert_array_equal(np.load(out / "sigma.npy"), sigma)
    up = np.asarray(Image.open(out / "reconstruction_upright.png"))
    assert up.shape == (64, 64, 3)


def test_cipher_mode_batch_with_plain_and_sigma(tmp_path, capsys):
    cdir, pdir, sdir = (tmp_path / d for d in ("c", "p", "s"))
    for d in (cdir, pdir, sdir):
        d.mkdir()
    for k in (1, 2):
        plain = synthetic_natural_image(size=48, seed=k)[:32]
        cipher, sigma = encrypt_image(plain, "common", k, scramble=True)
        _write(cdir / f"img{k}.png", cipher)
        _write(pdir / f"img{k}.png", plain)
        np.save(sdir / f"img{k}.npy", sigma)
    out = tmp_path / "out"
    args = ["--cipher", str(cdir), "--plain", str(pdir), "--sigma", str(sdir), "--out", str(out)]
    assert cli.main([*args, "--solver", "layer_greedy"]) == 0
    assert sorted(q.name for q in out.iterdir()) == ["img1", "img2"]
    files = sorted(q.name for q in (out / "img1").iterdir())
    assert "metrics.json" not in files and "sigma.npy" not in files
    assert "restored_upright.png" in files
    assert "reconstruction_upright.png" not in files  # 32 x 48 is not square after a jigsaw
    rows = capsys.readouterr().out.splitlines()
    assert [r.split()[0] for r in rows[1:3]] == ["img1.png", "img2.png"]
    assert [r.split()[1] for r in rows[1:3]] == ["1.0000", "1.0000"]  # EJPS sub-block rate
    assert [r.split()[4] for r in rows[1:3]] == ["1.0000", "1.0000"]  # jigsaw direct


def test_cipher_mode_without_plaintext(tmp_path, capsys):
    cipher, _ = encrypt_image(synthetic_natural_image(size=40, seed=2)[:32, :32], "common", 0)
    src = _write(tmp_path / "c.png", np.pad(cipher, ((0, 5), (0, 7), (0, 0))))  # gets cropped
    out = tmp_path / "out"
    assert cli.main(["--cipher", str(src), "--out", str(out)]) == 0
    assert sorted(q.name for q in out.iterdir()) == [
        "candidate_0.png",
        "cipher.png",
        "reconstruction.png",
        "restored.png",
    ]
    assert np.asarray(Image.open(out / "restored.png")).shape == (32, 32, 3)
    assert capsys.readouterr().out.splitlines()[1].split()[1:8] == ["-"] * 7


def test_errors(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--simulate", "x.png", "--plain", "y.png"])
    with pytest.raises(SystemExit):
        cli.main(["--simulate", "x.png", "--solver-option", "oops"])
    with pytest.raises(FileNotFoundError):
        cli.main(["--simulate", str(tmp_path / "missing.png")])
    with pytest.raises(FileNotFoundError, match="no images"):
        cli.main(["--simulate", str(tmp_path)])
    assert cli.companion(None, Path("a.png"), None) is None
    assert cli.companion(str(tmp_path), Path("x/a.png"), ".npy") == tmp_path / "a.npy"
    assert cli.companion(str(tmp_path), Path("x/a.png"), None) == tmp_path / "a.png"
    assert cli.companion("s.npy", Path("x/a.png"), ".npy") == Path("s.npy")


def test_entry_point_help_from_any_directory(tmp_path):
    res = subprocess.run(
        [sys.executable, str(ROOT / "run_attack.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.startswith("usage: run_attack.py")
    assert "--simulate PATH" in res.stdout and "--save-metrics" in res.stdout
