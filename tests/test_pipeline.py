"""Tests for ejps/pipeline.py (end-to-end on synthetic ciphertexts)."""

import dataclasses

import make_fixtures as MF
import numpy as np
import pytest

from ejps.encrypt import encrypt_frames
from ejps.geometry import KeyGroups, dihedral, image_to_blocks
from ejps.metrics import synthetic_natural_image
from ejps.pipeline import (
    AttackConfig,
    attack_frames,
    attack_image,
    candidates_to_images,
    display_upright,
    evaluate,
    final_image_metrics,
    restore_blocks,
    symmetric_truths,
    true_order,
)


def _case(kind="common", size=64, scramble=False, t=1, seed=0):
    if kind == "per_position":
        base = synthetic_natural_image(size=size, seed=seed)
        plain = np.stack([np.roll(base, f, axis=1) for f in range(t)])
    else:
        plain = np.stack([synthetic_natural_image(size=size, seed=seed + f) for f in range(t)])
    enc, sigma = encrypt_frames(plain, kind, seed, scramble=scramble)
    return plain, enc, sigma


def _no_timings(m):
    return {k: v for k, v in m.items() if k != "timings"}


def test_true_order_is_inverse_permutation():
    sigma = np.array([2, 0, 3, 1])
    assert true_order(sigma).tolist() == [1, 3, 0, 2]
    assert true_order(sigma)[sigma].tolist() == [0, 1, 2, 3]


def test_from_dict():
    cfg = AttackConfig.from_dict({"SOLVER": "layer_greedy", "jigsaw": True, "KEEP_TOP": 3})
    assert cfg.solver == "layer_greedy"
    assert cfg.jigsaw is True
    assert cfg.keep_top == 3
    assert cfg.block_size == 16
    with pytest.raises(ValueError, match="unknown AttackConfig keys"):
        AttackConfig.from_dict({"SOLVER": "edge_ref", "BOGUS": 1})


def test_defaults():
    cfg = AttackConfig()
    assert (cfg.solver, cfg.key_groups, cfg.block_size, cfg.sub_block_size) == (
        "edge_ref",
        "common",
        16,
        8,
    )
    assert (cfg.jigsaw, cfg.jigsaw_population, cfg.jigsaw_generations) == (False, 30, 20)
    assert (cfg.select_feature, cfg.resolve_inversion, cfg.seed, cfg.keep_top) == (
        "pixel",
        True,
        0,
        1,
    )


def test_symmetric_truths_square_and_rectangular():
    truth = np.arange(4)
    sq = symmetric_truths(truth, 2, 2)
    assert len(sq) == 8
    assert sq[0].tolist() == [0, 1, 2, 3]
    grid = np.arange(4).reshape(2, 2)
    assert [s.tolist() for s in sq] == [
        dihedral(grid, k, axes=(0, 1)).ravel().tolist() for k in range(8)
    ]
    rect = symmetric_truths(np.arange(6), 2, 3)
    assert len(rect) == 4  # 90/270-degree turns change the 2x3 shape
    assert rect[0].tolist() == list(range(6))
    permuted = symmetric_truths(np.array([3, 2, 1, 0]), 2, 2)
    assert permuted[0].tolist() == [3, 2, 1, 0]


@pytest.mark.parametrize("solver", ["edge_ref", "layer_greedy"])
def test_common_key_without_scramble_is_fully_restored(solver):
    plain, enc, sigma = _case()
    cfg = AttackConfig(solver=solver, keep_top=2)
    res = attack_image(enc[0], cfg)
    assert res.restored.shape == (1, 64, 64, 3)
    assert res.orders is None
    assert np.array_equal(res.reconstruction, res.restored)
    assert len(res.group_scores) == 1
    assert len(res.top_candidates) == (1 if solver == "edge_ref" else 2)
    m = evaluate(res, plain, cfg, sigma)
    for key in ("ejps_global", "ejps_oracle"):
        assert m[key]["block_rate"] == 1.0
        assert m[key]["sub_block_rate"] == 1.0
        assert m[key]["ssim"] == 1.0
    assert m["final"] == {"psnr": float("inf"), "ssim": 1.0}
    assert "jigsaw_direct" not in m
    assert set(m["timings"]) == {"ejps_s", "jigsaw_s"}


def test_common_key_with_scramble_is_solved_by_jigsaw(golden):
    plain, enc, sigma = _case(size=64, scramble=True)
    assert sigma.tolist() != list(range(16))
    cfg = AttackConfig(jigsaw=True)
    res = attack_frames(enc, cfg)
    assert len(res.orders) == 1
    m = evaluate(res, plain, cfg, sigma)
    want = golden["pipeline"]["common_scramble_64"]
    assert res.orders[0].tolist() == want["order"]
    assert MF.normalise(_no_timings(m)) == want["metrics"]
    assert m["ejps_global"]["sub_block_rate"] == 1.0
    assert m["jigsaw_direct"] == 1.0
    assert m["jigsaw_neighbor"] == 1.0
    assert m["final"]["ssim"] == 1.0
    # without the scrambling ground truth the EJPS stage looks wrong
    wrong = evaluate(res, plain, cfg, None)
    assert wrong["ejps_global"]["block_rate"] < 0.5


def test_independent_keys_resist(golden):
    plain, enc, sigma = _case("independent", size=64)
    cfg = AttackConfig(key_groups="independent")
    res = attack_image(enc[0], cfg)
    assert len(res.group_scores) == 16
    m = evaluate(res, plain, cfg, sigma)
    assert MF.normalise(_no_timings(m)) == golden["pipeline"]["independent_64"]
    assert m["ejps_global"]["block_rate"] == 0.0
    assert m["ejps_global"]["sub_block_rate"] == 0.0
    assert m["ejps_oracle"]["sub_block_rate"] == pytest.approx(0.28125)
    assert m["ejps_global"]["psnr"] == pytest.approx(14.9404, abs=1e-3)
    assert m["ejps_oracle"]["psnr"] > m["ejps_global"]["psnr"] + 10
    assert m["final"]["psnr"] < 20


def test_per_position_pools_frames(golden):
    plain4, enc4, s4 = MF.per_position_input(4)
    plain8, enc8, s8 = MF.per_position_input(8)
    np.testing.assert_array_equal(plain4, _case("per_position", t=4)[0])
    cfg = AttackConfig(key_groups="per_position")
    m4 = evaluate(attack_frames(enc4, cfg), plain4, cfg, s4)
    r8 = attack_frames(enc8, cfg)
    m8 = evaluate(r8, plain8, cfg, s8)
    assert len(r8.group_scores) == 16
    assert MF.normalise(_no_timings(m4)) == golden["pipeline"]["per_position_4"]
    assert MF.normalise(_no_timings(m8)) == golden["pipeline"]["per_position_8"]
    assert m4["ejps_oracle"]["sub_block_rate"] == pytest.approx(0.5625)
    assert m8["ejps_oracle"]["sub_block_rate"] == pytest.approx(0.734375)


def test_restore_blocks_rejects_mismatched_groups():
    _, enc, _ = _case()
    blocks = image_to_blocks(enc[0], 16)
    with pytest.raises(ValueError, match="key groups cover"):
        restore_blocks(blocks, KeyGroups.common(5), AttackConfig())


def test_attack_frames_validation():
    with pytest.raises(ValueError, match="uint8"):
        attack_frames(np.zeros((1, 32, 32, 3), np.float32), AttackConfig())
    with pytest.raises(ValueError, match="uint8"):
        attack_frames(np.zeros((32, 32, 3), np.uint8), AttackConfig())
    with pytest.raises(ValueError, match="multiple of block_size"):
        attack_frames(np.zeros((1, 40, 32, 3), np.uint8), AttackConfig())
    with pytest.raises(ValueError, match="multiple of block_size"):
        attack_frames(np.zeros((1, 32, 40, 3), np.uint8), AttackConfig())


def test_final_image_metrics_aligns_whole_frame():
    plain = synthetic_natural_image(size=32, seed=3)[None]
    turned = np.ascontiguousarray(255 - dihedral(plain[0], 1, axes=(0, 1))[..., ::-1])[None]
    m = final_image_metrics(turned, plain)
    assert m == {"psnr": float("inf"), "ssim": 1.0}
    noisy = np.clip(plain.astype(int) + 8, 0, 255).astype(np.uint8)
    assert final_image_metrics(noisy, plain)["psnr"] < 40


def test_candidates_to_images():
    blocks = np.zeros((4, 16, 16, 3), np.uint8)
    imgs = candidates_to_images([blocks, blocks[:2]], 2, 2)
    assert len(imgs) == 1
    assert imgs[0].shape == (32, 32, 3)


def test_rectangular_frames():
    base = synthetic_natural_image(size=64, seed=2)[:32]
    enc, sigma = encrypt_frames(base[None], "common", 0)
    cfg = dataclasses.replace(AttackConfig(), keep_top=1)
    m = evaluate(attack_frames(enc, cfg), base, cfg, sigma)
    assert m["final"]["ssim"] == 1.0  # no jigsaw: block-wise alignment works for any shape
    assert m["ejps_global"]["sub_block_rate"] == 1.0
    enc, sigma = encrypt_frames(base[None], "common", 0, scramble=True)
    cfg = dataclasses.replace(cfg, jigsaw=True)
    m = evaluate(attack_frames(enc, cfg), base, cfg, sigma)
    assert "final" not in m  # after a jigsaw only square frames are aligned
    assert m["jigsaw_direct"] == 1.0


def test_final_without_jigsaw_aligns_blocks_in_place():
    plain = synthetic_natural_image(size=32, seed=4)[None]
    blocks = image_to_blocks(plain[0], 16)
    turned = np.stack([dihedral(b, 2, axes=(0, 1)) for b in blocks])
    img = np.ascontiguousarray(
        turned.reshape(2, 2, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(32, 32, 3)
    )[None]
    assert final_image_metrics(img, plain, 16) == {"psnr": float("inf"), "ssim": 1.0}
    assert final_image_metrics(img, plain)["ssim"] < 1.0


def test_solver_options_are_passed_to_the_solver():
    cfg = AttackConfig.from_dict(
        {"solver": "layer_greedy", "solver_options": {"one_plane_per_sub_block": True}}
    )
    assert cfg.solver_options == {"one_plane_per_sub_block": True}
    assert AttackConfig().solver_options == {}
    plain, enc, sigma = _case()
    m = evaluate(attack_image(enc[0], cfg), plain, cfg, sigma)
    assert m["ejps_global"]["sub_block_rate"] == 1.0
    bad = dataclasses.replace(cfg, solver_options={"no_such_flag": True})
    with pytest.raises(TypeError):
        attack_image(enc[0], bad)


def test_display_upright_whole_image_turns_only_geometry():
    plain = synthetic_natural_image(size=32, seed=5)
    rec = np.ascontiguousarray(255 - dihedral(plain, 3, axes=(0, 1))[..., ::-1])
    up = display_upright(rec, plain)
    assert np.array_equal(up, 255 - plain[..., ::-1])  # upright, colours untouched


def test_display_upright_block_wise_with_sigma():
    plain, enc, sigma = _case(size=64, scramble=True)
    res = attack_image(enc[0], AttackConfig())
    up = display_upright(res.restored[0], plain[0], 16, sigma)
    want = image_to_blocks(plain[0], 16)[sigma]
    got = image_to_blocks(up, 16)
    # same geometry as the plaintext blocks: some fixed channel map makes them equal
    assert any(
        np.array_equal(np.where(inv, 255 - got[..., list(p)], got[..., list(p)]), want)
        for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        for inv in [np.array(v, bool) for v in np.ndindex(2, 2, 2)]
    )
    # the solver output itself was not upright
    raw = image_to_blocks(res.restored[0], 16)
    assert not np.array_equal(raw, got)
