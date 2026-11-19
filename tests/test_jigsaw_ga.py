"""Tests for ejps/jigsaw/ga.py (gaps genetic algorithm + solve_jigsaw facade).

Golden values marked "upstream" were produced by the original gaps (commit
5aa0a0d, Python 3.12, ``workers=1``) on the same inputs and ``random.Random``
seeds, so they pin the GA flow *and* the order of random draws.
"""

import hashlib
import math
import random

import numpy as np
import pytest

from ejps.jigsaw.domain import (
    Arrangement,
    Direction,
    PuzzleLayout,
    assemble_image,
    flatten_image,
)
from ejps.jigsaw.fitness import EdgeCostTable
from ejps.jigsaw.ga import (
    GeneticAlgorithm,
    JigsawResult,
    _apply_move,
    _constraint_repair,
    _crossover_child,
    _inverse_move,
    _local_search,
    _mutate_child,
    _random_move,
    _repair_window_indices,
    build_seed_arrangements,
    solve_jigsaw,
)

MOVES16 = [("swap", 9, 1, 1), ("block", 7, 11, 4), ("swap", 2, 0, 1), ("relocate", 9, 12, 1), ("swap", 7, 8, 1), ("block", 4, 13, 3), ("swap", 3, 4, 1), ("swap", 0, 13, 1), ("block", 4, 8, 3), ("swap", 9, 4, 1), ("block", 1, 13, 3), ("relocate", 12, 8, 1)]  # fmt: skip
MOVES3 = [("swap", 1, 0, 1), ("relocate", 1, 0, 1), ("swap", 0, 2, 1), ("relocate", 2, 1, 1), ("swap", 0, 1, 1), ("relocate", 0, 2, 1)]  # fmt: skip
MOVES4 = [("swap", 2, 0, 1), ("block", 0, 2, 2), ("swap", 0, 3, 1), ("relocate", 2, 0, 1), ("swap", 2, 1, 1), ("swap", 0, 1, 1), ("swap", 0, 2, 1), ("relocate", 2, 0, 1)]  # fmt: skip

# Applied to the parent pa = Arrangement.random(pieces, layout, Random(1)) of each puzzle:
# local = _local_search(Random(4), steps=3, candidates=5); repair = _constraint_repair(
# Random(4), steps=3); mutate = _mutate_child(Random(6), rate=0.9, 2, 4, 2);
# windows = 4 x _repair_window_indices(Random(11)).  (ids, score) -- upstream.
OPS_GOLDEN = {
    "g44": (64, 64, {
        "local": ([2, 10, 0, 14, 6, 5, 3, 7, 8, 11, 15, 1, 12, 13, 9, 4], 0.11582569203387907),
        "repair": ([2, 0, 5, 10, 6, 3, 15, 14, 7, 8, 11, 1, 12, 13, 9, 4], 0.1418478457598948),
        "mutate": ([2, 10, 0, 14, 6, 15, 3, 7, 8, 11, 5, 1, 12, 13, 9, 4], 0.15071395813596597),
        "windows": [(6, 7, 10, 11), (5, 6, 9, 10), (10, 11, 14, 15), (0, 1, 4, 5)],
    }),
    "row": (16, 96, {
        "local": ([2, 3, 4, 5, 0, 1], 0.4921883807840122),
        "repair": ([0, 2, 3, 4, 5, 1], 0.45859336187993105),
        "mutate": ([2, 3, 0, 1, 4, 5], 0.442491929138588),
        "windows": [(3, 4, 5), (3, 4, 5), (3, 4, 5), (1, 2, 3)],
    }),
    "g22": (32, 32, {
        "local": ([0, 1, 2, 3], 0.8490336613529137),
        "repair": ([0, 1, 2, 3], 0.8490336613529137),
        "mutate": ([0, 1, 2, 3], 0.8490336613529137),
        "windows": [(0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)],
    }),
    "g33": (48, 48, {
        "local": ([5, 6, 7, 4, 3, 0, 8, 1, 2], 0.12613717556392842),
        "repair": ([5, 3, 0, 4, 6, 7, 8, 1, 2], 0.13347817982340737),
        "mutate": ([5, 6, 7, 1, 0, 8, 4, 3, 2], 0.1470344837261579),
        "windows": [(4, 5, 7, 8), (3, 4, 6, 7), (1, 2, 4, 5), (0, 1, 3, 4)],
    }),
    "col": (80, 16, {
        "local": ([2, 3, 4, 0, 1], 0.4203753211580201),
        "repair": ([2, 3, 4, 0, 1], 0.4203753211580201),
        "mutate": ([0, 1, 2, 3, 4], 0.9596602983133091),
        "windows": [(1, 2, 3), (2, 3, 4), (1, 2, 3), (1, 2, 3)],
    }),
}  # fmt: skip

# GeneticAlgorithm(noise16, 4, rng=Random(5), collect_stats=True, **kwargs) -- upstream
RUNS_GOLDEN = {
    "early": (
        dict(population_size=6, generations=40, max_restarts=0, restart_threshold=1, mutation_rate=0.0),
        [1, 13, 2, 14, 0, 11, 8, 9, 5, 6, 7, 10, 12, 3, 4, 15], 0.00963550054298748, 11, True,
    ),
    "restart": (
        dict(population_size=6, generations=25, max_restarts=2, restart_threshold=2, elite_size=3),
        [1, 9, 2, 14, 0, 4, 11, 8, 5, 6, 7, 10, 12, 3, 13, 15], 0.009478859747131958, 21, True,
    ),
    "noseed": (
        dict(population_size=8, generations=6, seed_fraction=0.0, tournament_size=2),
        [1, 0, 4, 14, 13, 5, 6, 15, 11, 12, 3, 7, 9, 8, 2, 10], 0.008872632313364544, 6, False,
    ),
    "border": (
        dict(population_size=8, generations=6, seed_border_weight=0.5, seed_beam_width=2,
             seed_candidate_width=3, seed_confidence_weight=0.1, local_search_candidates=0,
             repair_steps=0),
        [9, 2, 14, 8, 4, 0, 1, 11, 13, 5, 6, 7, 15, 12, 3, 10], 0.00797084290759078, 6, False,
    ),
}  # fmt: skip
NOSEED_GEN_SCORES = [0.007727513097637165, 0.00800604280801259, 0.008394891180334514, 0.008394891180334514, 0.008558901287771706, 0.008872632313364544]  # fmt: skip
BORDER_GEN_SCORES = [0.007608399168776994, 0.007608399168776994, 0.007704625228583651, 0.007704625228583651, 0.007704625228583651, 0.00797084290759078]  # fmt: skip


def smooth_image(h, w):
    """Synthetic smooth image (gradients + Gaussian blobs + texture); same as upstream golden."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.empty((h, w, 3))
    img[..., 0] = 30.0 + 150.0 * x / w
    img[..., 1] = 40.0 + 120.0 * y / h
    img[..., 2] = 60.0 + 60.0 * (x + y) / (h + w)
    blobs = [
        (0.25, 0.30, 0.12, (90.0, -30.0, 40.0)),
        (0.70, 0.65, 0.18, (-40.0, 70.0, 20.0)),
        (0.40, 0.80, 0.10, (30.0, 30.0, -50.0)),
        (0.85, 0.20, 0.14, (-20.0, -40.0, 80.0)),
    ]
    for cy, cx, s, amp in blobs:
        g = np.exp(-(((y / h - cy) ** 2) + ((x / w - cx) ** 2)) / (2.0 * s * s))
        for c in range(3):
            img[..., c] += amp[c] * g
    img += (6.0 * np.sin(x / 3.0) * np.cos(y / 4.0))[..., None]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def shuffle_blocks(image, piece_size, seed):
    """Permute the blocks of ``image``; shuffled position k holds original block perm[k]."""
    pieces, layout = flatten_image(image, piece_size)
    perm = list(range(len(pieces)))
    random.Random(seed).shuffle(perm)
    return assemble_image([pieces[i] for i in perm], layout), perm


def inverse(perm):
    """The exact order solve_jigsaw must return for a shuffle by ``perm``."""
    inv = [0] * len(perm)
    for k, j in enumerate(perm):
        inv[j] = k
    return inv


def puzzle(h, w, piece_size=16, image=None):
    img = smooth_image(h, w) if image is None else image
    pieces, layout = flatten_image(img, piece_size, indexed=True)
    table = EdgeCostTable()
    table.analyze(pieces)
    return pieces, layout, table


def ids(arrangement):
    return [p.identifier for p in arrangement.pieces]


# --------------------------------------------------------------------------- #
# facade
# --------------------------------------------------------------------------- #
def test_input_images_match_upstream_golden_inputs():
    assert (
        hashlib.sha256(smooth_image(64, 64).tobytes()).hexdigest()
        == "7d35a3e44f5f456c6c31894d63ee0952c104e2f2619269a3f191113edec76298"
    )
    assert (
        hashlib.sha256(smooth_image(96, 96).tobytes()).hexdigest()
        == "b98e8df7b4bb5c729ef5cf100747f02eb9714f104e95805e91d9d9d77791efac"
    )


def test_solve_jigsaw_recovers_224px_image_exactly():
    original = smooth_image(224, 224)
    shuffled, perm = shuffle_blocks(original, 16, 3)
    assert perm != list(range(196))
    result = solve_jigsaw(shuffled, 16, population=30, generations=10, seed=0)
    assert isinstance(result, JigsawResult)
    assert result.order.shape == (196,)
    assert result.order.tolist() == inverse(perm)
    np.testing.assert_array_equal(result.image, original)
    assert result.image.dtype == np.uint8
    assert result.score == pytest.approx(0.9857671664275812, rel=1e-9)  # upstream
    assert result.generations == 10


def test_solve_jigsaw_is_deterministic_and_matches_upstream():
    original = smooth_image(96, 96)
    shuffled, perm = shuffle_blocks(original, 16, 2)
    first = solve_jigsaw(shuffled, 16, population=40, generations=15, seed=0)
    second = solve_jigsaw(shuffled, 16, population=40, generations=15, seed=0)
    assert first.order.tolist() == second.order.tolist() == inverse(perm)
    assert first.score == second.score == pytest.approx(0.9688829401212029, rel=1e-12)
    assert first.generations == second.generations == 15
    np.testing.assert_array_equal(first.image, original)


def test_ga_without_seeds_still_solves_and_terminates_early():
    # no beam-search seeds: the GA alone must find the answer (upstream: 22 generations)
    original = smooth_image(96, 96)
    shuffled, perm = shuffle_blocks(original, 16, 4)
    result = solve_jigsaw(shuffled, 16, population=60, generations=30, seed=0, seed_fraction=0.0)
    assert result.order.tolist() == inverse(perm)
    assert result.generations == 22
    assert result.score == pytest.approx(0.9688829401212029, rel=1e-12)


def test_solve_jigsaw_grayscale():
    original = smooth_image(80, 80)[..., 1].copy()
    shuffled, perm = shuffle_blocks(original, 16, 5)
    result = solve_jigsaw(shuffled, 16, population=20, generations=8, seed=0)
    assert result.order.tolist() == inverse(perm)
    assert result.image.shape == (80, 80)
    np.testing.assert_array_equal(result.image, original)
    assert result.score == pytest.approx(0.9609691352720503, rel=1e-12)  # upstream
    assert result.generations == 8


def test_solve_jigsaw_single_piece_and_errors():
    img = smooth_image(16, 16)
    result = solve_jigsaw(img, 16, population=4, generations=1)
    assert result.order.tolist() == [0]
    assert result.score == 1.0 and result.generations == 0
    np.testing.assert_array_equal(result.image, img)
    with pytest.raises(ValueError):
        solve_jigsaw(img.astype(np.float32), 16)
    with pytest.raises(ValueError):
        solve_jigsaw(img[..., 0][..., None, None], 16)
    with pytest.raises(ValueError):
        solve_jigsaw(img, 0)
    with pytest.raises(ValueError):
        solve_jigsaw(smooth_image(32, 40), 16)  # width not a multiple
    with pytest.raises(ValueError):
        solve_jigsaw(smooth_image(40, 32), 16)  # height not a multiple
    with pytest.raises(NotImplementedError):
        solve_jigsaw(smooth_image(32, 32), 16, population=4, workers=2)


def test_solve_jigsaw_forwards_ga_kwargs_and_seed():
    shuffled, _ = shuffle_blocks(
        np.random.default_rng(1).integers(0, 256, (16, 16, 3), np.uint8), 4, 0
    )
    kwargs, order, score, gens, early = RUNS_GOLDEN["early"]
    ga = GeneticAlgorithm(shuffled, 4, rng=random.Random(5), **kwargs).solve()
    result = solve_jigsaw(
        shuffled,
        4,
        population=kwargs["population_size"],
        generations=kwargs["generations"],
        seed=5,
        max_restarts=0,
        restart_threshold=1,
        mutation_rate=0.0,
    )
    assert result.order.tolist() == ids(ga.arrangement)
    assert result.generations == ga.generations_completed
    other = solve_jigsaw(shuffled, 4, population=6, generations=3, seed=6)
    assert other.generations == 3


# --------------------------------------------------------------------------- #
# GeneticAlgorithm
# --------------------------------------------------------------------------- #
def noise16():
    return np.random.default_rng(1).integers(0, 256, (16, 16, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(RUNS_GOLDEN))
def test_ga_runs_match_upstream(name):
    kwargs, order, score, gens, early = RUNS_GOLDEN[name]
    progress, scores = [], []
    ga = GeneticAlgorithm(noise16(), 4, rng=random.Random(5), collect_stats=True, **kwargs)
    result = ga.solve(
        progress=lambda *e: progress.append(e),
        on_generation=lambda g, a: scores.append((g, a.score(ga._analysis))),
    )
    assert ids(result.arrangement) == order
    assert result.best_score == pytest.approx(score, rel=1e-12)
    assert result.generations_completed == gens
    assert result.terminated_early is early
    assert progress[0] == ("analysis", 15, 15)
    assert progress[1:] == [("evolution", g, kwargs["generations"]) for g in range(1, gens + 1)]
    assert [g for g, _ in scores] == list(range(1, gens + 1))
    assert scores[-1][1] == pytest.approx(score, rel=1e-12)
    if name == "noseed":
        np.testing.assert_allclose([s for _, s in scores], NOSEED_GEN_SCORES, rtol=1e-12)
    if name == "border":
        np.testing.assert_allclose([s for _, s in scores], BORDER_GEN_SCORES, rtol=1e-12)
    if name == "restart":
        # best improved at generation 11 only (after the first restart)
        assert scores[9][1] == pytest.approx(0.008927666078754353, rel=1e-12)
        assert scores[10][1] == pytest.approx(0.009478859747131958, rel=1e-12)
    if name == "early":
        assert len({s for _, s in scores}) == 1


def test_ga_noise_puzzle_matches_upstream():
    noise = np.random.default_rng(0).integers(0, 256, (24, 24, 3), dtype=np.uint8)
    ga = GeneticAlgorithm(
        noise,
        4,
        population_size=12,
        generations=12,
        rng=random.Random(7),
        mutation_rate=0.3,
        local_search_steps=2,
        repair_steps=2,
        max_restarts=1,
        restart_threshold=2,
        collect_stats=True,
    )
    result = ga.solve()
    assert ids(result.arrangement) == [17, 14, 12, 33, 9, 7, 27, 19, 15, 22, 1, 25, 0, 3, 2, 24, 18, 29, 10, 23, 34, 30, 5, 4, 13, 35, 16, 28, 21, 6, 32, 8, 11, 31, 20, 26]  # fmt: skip
    assert result.best_score == pytest.approx(0.010224971043609175, rel=1e-12)
    assert result.generations_completed == 12 and result.terminated_early is False
    assert ga.crossover_stats.candidate_pushes == 7397
    assert ga.crossover_stats.validity_checks == 36909


def test_ga_best_score_is_consistent_with_a_fresh_table():
    result = GeneticAlgorithm(
        noise16(), 4, population_size=8, generations=4, rng=random.Random(31), mutation_rate=0.2
    ).solve()
    pieces, layout = flatten_image(noise16(), 4, indexed=True)
    table = EdgeCostTable()
    table.analyze(pieces)
    fresh = Arrangement(list(result.arrangement.pieces), layout).score(table)
    assert result.best_score == pytest.approx(fresh, rel=1e-9)
    assert sorted(ids(result.arrangement)) == list(range(16))


def test_ga_properties_and_population():
    ga = GeneticAlgorithm(noise16(), 4, population_size=5, generations=1, rng=random.Random(0))
    assert ga.layout == PuzzleLayout(4, 4, 4)
    assert [p.identifier for p in ga.pieces] == list(range(16))
    assert len(ga._population) == 5
    assert ga.crossover_stats is None
    assert ga._restart_threshold == 5  # TERMINATION_THRESHOLD // 2
    ref = random.Random(0)
    expected = list(range(16))
    ref.shuffle(expected)
    assert ids(ga._population[0]) == expected  # population drawn from the given rng
    ga._analysis.analyze(ga.pieces)
    ga._initialize_population()
    # seed count = min(8, max(1, int(5 * 0.25)), max(1, 1024 // 16)) = 1 seed in front
    assert len(ga._population) == 5
    assert ga._population_initialized is True
    seeded = ga._population[0]
    ga._initialize_population()  # idempotent
    assert ga._population[0] is seeded


def test_ga_seed_count_is_capped():
    ga = GeneticAlgorithm(noise16(), 4, population_size=60, generations=1, rng=random.Random(0))
    ga._analysis.analyze(ga.pieces)
    before = list(ga._population)
    ga._initialize_population()
    # min(8, int(60 * 0.25) = 15, 1024 // 16 = 64) = 8 seeds prepended, population kept at 60
    assert len(ga._population) == 60
    assert all(a is b for a, b in zip(ga._population[8:], before[:52], strict=True))
    assert not any(a is b for a in ga._population[:8] for b in before)
    noseed = GeneticAlgorithm(
        noise16(), 4, population_size=6, generations=1, rng=random.Random(0), seed_fraction=0.0
    )
    before = list(noseed._population)
    noseed._initialize_population()
    assert all(a is b for a, b in zip(noseed._population, before, strict=True))


def test_adaptive_mutation_rate():
    ga = GeneticAlgorithm(
        noise16(), 4, population_size=4, generations=1, rng=random.Random(0), restart_threshold=4
    )
    ga._analysis.analyze(ga.pieces)
    # diverse population: rate * (1 + min(2, s / 4))
    assert ga._adaptive_mutation_rate(0) == pytest.approx(0.05)
    assert ga._adaptive_mutation_rate(2) == pytest.approx(0.075)
    assert ga._adaptive_mutation_rate(8) == pytest.approx(0.15)
    assert ga._adaptive_mutation_rate(40) == pytest.approx(0.15)
    # identical scores -> x1.25 diversity factor
    ga._population = [ga._population[0]] * 4
    assert ga._adaptive_mutation_rate(4) == pytest.approx(0.05 * 2 * 1.25)
    ga._mutation_rate = 0.9
    assert ga._adaptive_mutation_rate(4) == 1.0  # capped
    ga._mutation_rate = 0.0
    assert ga._adaptive_mutation_rate(4) == 0.0


def test_restart_population_keeps_best_and_distinct_elites():
    ga = GeneticAlgorithm(
        noise16(), 4, population_size=6, generations=1, rng=random.Random(0), elite_size=3
    )
    ga._analysis.analyze(ga.pieces)
    ranked = sorted(ga._population, key=ga._score)
    best = ranked[-1]
    ga._restart_population(best)
    assert len(ga._population) == 6
    assert ga._population[0] is best
    assert ga._population[1] is ranked[-2]
    assert ga._population[2] is ranked[-3]
    assert all(p is not r for p in ga._population[3:] for r in ranked)
    # a duplicate of the best individual is not kept twice
    twin = Arrangement(list(best.pieces), best.layout)
    ga._population = [twin, ranked[0], ranked[1]] + ranked[2:5]
    ga._restart_population(best)
    kept = [ids(p) for p in ga._population[:3]]
    assert kept[0] == ids(best) and kept.count(ids(best)) == 1
    elite_size_one = GeneticAlgorithm(
        noise16(), 4, population_size=3, generations=1, rng=random.Random(0), elite_size=1
    )
    elite_size_one._analysis.analyze(elite_size_one.pieces)
    top = max(elite_size_one._population, key=elite_size_one._score)
    other = min(elite_size_one._population, key=elite_size_one._score)
    elite_size_one._restart_population(other)
    assert elite_size_one._population[0] is other
    # upstream checks the size only after appending: best + 1 elite are both kept
    assert elite_size_one._population[1] is top
    assert len(elite_size_one._population) == 3


def test_elites_and_best():
    ga = GeneticAlgorithm(
        noise16(), 4, population_size=6, generations=1, rng=random.Random(0), elite_size=2
    )
    ga._analysis.analyze(ga.pieces)
    ranked = sorted(ga._population, key=ga._score)
    assert list(ga._elite_individuals()) == ranked[-2:]
    assert ga._best_arrangement() is ranked[-1]
    nxt = ga._next_population_serial(0.0)
    assert len(nxt) == 6
    assert nxt[0] is ranked[-2] and nxt[1] is ranked[-1]
    assert all(sorted(ids(c)) == list(range(16)) for c in nxt)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(population_size=0),
        dict(generations=0),
        dict(elite_size=0),
        dict(elite_size=5),
        dict(mutation_rate=-0.1),
        dict(mutation_rate=1.1),
        dict(workers=0),
        dict(local_search_steps=-1),
        dict(local_search_candidates=-1),
        dict(repair_steps=-1),
        dict(max_restarts=-1),
        dict(restart_threshold=0),
        dict(tournament_size=0),
        dict(seed_fraction=-0.1),
        dict(seed_fraction=1.5),
        dict(seed_beam_width=0),
        dict(seed_candidate_width=0),
        dict(seed_confidence_weight=-1.0),
        dict(seed_border_weight=-1.0),
    ],
)
def test_ga_validation(kwargs):
    args = dict(population_size=5, generations=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        GeneticAlgorithm(noise16(), 4, **args)


def test_ga_accepts_boundary_values():
    ga = GeneticAlgorithm(
        noise16(),
        4,
        population_size=2,
        generations=1,
        elite_size=1,
        mutation_rate=1.0,
        local_search_steps=0,
        local_search_candidates=0,
        repair_steps=0,
        max_restarts=0,
        restart_threshold=1,
        seed_fraction=1.0,
        tournament_size=1,
        seed_beam_width=1,
        seed_candidate_width=1,
        seed_confidence_weight=0.0,
        seed_border_weight=0.0,
        rng=random.Random(0),
    )
    assert ga._restart_threshold == 1
    assert ga.solve().generations_completed == 1
    with pytest.raises(NotImplementedError):
        GeneticAlgorithm(noise16(), 4, population_size=5, generations=1, workers=3)


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #
def test_seed_arrangements_match_upstream():
    pieces, layout, table = puzzle(64, 64)
    seeds = build_seed_arrangements(pieces, layout, table, random.Random(3), count=3)
    assert [ids(s) for s in seeds] == [
        [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 6, 7, 4, 5, 2, 3],
        [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 7, 6],
    ]
    border = build_seed_arrangements(
        pieces, layout, table, random.Random(3), count=2, border_weight=0.5, candidate_width=3
    )
    assert [ids(s) for s in border] == [
        [9, 10, 11, 7, 13, 14, 15, 3, 8, 6, 5, 2, 12, 4, 0, 1],
        [9, 10, 11, 7, 13, 14, 15, 3, 8, 6, 5, 2, 12, 0, 1, 4],
    ]
    again = build_seed_arrangements(pieces, layout, table, random.Random(3), count=3)
    assert [ids(s) for s in again] == [ids(s) for s in seeds]
    assert all(s.pieces[0] is pieces[s.pieces[0].identifier] for s in seeds)


def test_seed_arrangements_limits_and_validation():
    pieces, layout, table = puzzle(32, 32)  # 4 pieces
    assert build_seed_arrangements(pieces, layout, table, random.Random(0), count=0) == []
    assert build_seed_arrangements([], layout, table, random.Random(0), count=3) == []
    many = build_seed_arrangements(pieces, layout, table, random.Random(0), count=10)
    assert 1 <= len(many) <= 4
    assert len({tuple(ids(s)) for s in many}) == len(many)
    one = build_seed_arrangements(
        pieces, layout, table, random.Random(0), count=1, beam_width=1, candidate_width=1
    )
    assert len(one) == 1
    rng = random.Random(0)
    first = rng.sample((0, 1, 2, 3), 1)[0]
    assert ids(one[0])[0] == first  # the anchor is drawn with rng.sample
    for bad in (
        dict(beam_width=0),
        dict(candidate_width=0),
        dict(confidence_weight=-0.1),
        dict(border_weight=-0.1),
    ):
        with pytest.raises(ValueError):
            build_seed_arrangements(pieces, layout, table, random.Random(0), count=2, **bad)


def test_seed_priority_prefers_true_neighbours():
    pieces, layout, table = puzzle(64, 64)
    # beam width 1, one candidate: greedy row-major completion from the anchor
    seeds = build_seed_arrangements(
        pieces, layout, table, random.Random(3), count=1, beam_width=1, candidate_width=1
    )
    order = ids(seeds[0])
    anchor = order[0]
    assert order[1] == table.best_match(anchor, Direction.RIGHT)


# --------------------------------------------------------------------------- #
# moves, local search, repair, mutation
# --------------------------------------------------------------------------- #
def test_random_moves_match_upstream():
    rng = random.Random(4)
    assert [_random_move(16, rng) for _ in range(12)] == MOVES16
    rng = random.Random(4)
    assert [_random_move(3, rng) for _ in range(6)] == MOVES3
    rng = random.Random(4)
    assert [_random_move(4, rng) for _ in range(8)] == MOVES4
    with pytest.raises(ValueError):
        _random_move(1, random.Random(0))
    rng = random.Random(0)
    moves = [_random_move(2, rng) for _ in range(30)]
    assert {m[0] for m in moves} == {"swap", "relocate"}
    assert all(sorted(m[1:3]) == [0, 1] and m[3] == 1 for m in moves)


def test_block_moves_never_overlap():
    rng = random.Random(1)
    for _ in range(300):
        op, first, second, size = _random_move(9, rng)
        if op == "block":
            assert 1 <= size <= 4
            assert first + size <= second and second + size <= 9


def test_apply_and_inverse_moves():
    pieces, layout, table = puzzle(64, 64)
    arr = Arrangement(list(pieces), layout)
    arr.score(table)
    for move in [("swap", 1, 6, 1), ("relocate", 2, 9, 1), ("block", 0, 8, 3)]:
        _apply_move(arr, move, table)
        assert ids(arr) != list(range(16))
        _apply_move(arr, _inverse_move(move), table)
        assert ids(arr) == list(range(16))
        assert arr.score(table) == pytest.approx(0.9476862406767522, rel=1e-9)
    assert _inverse_move(("relocate", 2, 9, 1)) == ("relocate", 9, 2, 1)
    assert _inverse_move(("swap", 2, 9, 1)) == ("swap", 2, 9, 1)
    assert _inverse_move(("block", 0, 8, 3)) == ("block", 0, 8, 3)
    _apply_move(arr, ("relocate", 0, 3, 1), table)
    assert ids(arr)[:4] == [1, 2, 3, 0]
    with pytest.raises(ValueError):
        _apply_move(arr, ("teleport", 0, 1, 1), table)


@pytest.mark.parametrize("name", sorted(OPS_GOLDEN))
def test_local_search_repair_mutate_match_upstream(name):
    h, w, golden = OPS_GOLDEN[name]
    pieces, layout, table = puzzle(h, w)
    pa = Arrangement.random(pieces, layout, random.Random(1))

    arr = Arrangement(list(pa.pieces), layout)
    _local_search(arr, table, random.Random(4), steps=3, candidates=5)
    assert ids(arr) == golden["local"][0]
    assert arr.score(table) == pytest.approx(golden["local"][1], rel=1e-12)
    assert arr.score(table) >= pa.score(table)

    arr = Arrangement(list(pa.pieces), layout)
    _constraint_repair(arr, table, random.Random(4), steps=3)
    assert ids(arr) == golden["repair"][0]
    assert arr.score(table) == pytest.approx(golden["repair"][1], rel=1e-12)
    assert arr.score(table) >= pa.score(table)

    arr = Arrangement(list(pa.pieces), layout)
    _mutate_child(arr, table, random.Random(6), 0.9, 2, 4, 2)
    assert ids(arr) == golden["mutate"][0]
    assert arr.score(table) == pytest.approx(golden["mutate"][1], rel=1e-12)

    rng = random.Random(11)
    assert [_repair_window_indices(pa, rng) for _ in range(4)] == golden["windows"]


def test_local_search_and_repair_noops():
    pieces, layout, table = puzzle(64, 64)
    identity = Arrangement(list(pieces), layout)
    for steps, candidates in ((0, 4), (3, 0)):
        rng = random.Random(0)
        _local_search(identity, table, rng, steps=steps, candidates=candidates)
        assert ids(identity) == list(range(16))
        assert rng.random() == random.Random(0).random()  # no draws
    rng = random.Random(0)
    _constraint_repair(identity, table, rng, steps=0)
    assert rng.random() == random.Random(0).random()
    # the solved arrangement cannot be improved: one window is drawn, then it stops
    rng = random.Random(0)
    _constraint_repair(identity, table, rng, steps=5)
    assert ids(identity) == list(range(16))
    ref = random.Random(0)
    ref.randrange(3)
    ref.randrange(3)
    assert rng.random() == ref.random()
    single = Arrangement([pieces[0]], PuzzleLayout(1, 1, 16))
    _local_search(single, table, random.Random(0), steps=2, candidates=2)
    assert _repair_window_indices(single, random.Random(0)) is None
    _constraint_repair(single, table, random.Random(0), steps=2)
    two = Arrangement(list(pieces[:2]), PuzzleLayout(1, 2, 16))
    assert _repair_window_indices(two, random.Random(0)) is None


def test_mutation_rejects_worse_moves_mostly():
    pieces, layout, table = puzzle(64, 64)
    kept = 0
    for seed in range(40):
        arr = Arrangement(list(pieces), layout)
        _mutate_child(arr, table, random.Random(seed), 1.0, 0, 0, 0)
        kept += ids(arr) != list(range(16))
    # every move worsens the solved puzzle; acceptance probability 0.05 + 0.15 = 0.2
    assert 2 <= kept <= 16
    arr = Arrangement(list(pieces), layout)
    rng = random.Random(0)
    _mutate_child(arr, table, rng, 0.0, 0, 0, 0)
    assert rng.random() == random.Random(0).random()  # rate 0: no draw at all


def test_crossover_child_falls_back_to_first_parent(monkeypatch):
    pieces, layout, table = puzzle(32, 32)
    pa = Arrangement.random(pieces, layout, random.Random(1))
    pb = Arrangement.random(pieces, layout, random.Random(2))
    from ejps.jigsaw import ga as jigsaw_ga

    def broken(self):
        raise RuntimeError("incomplete")

    monkeypatch.setattr(jigsaw_ga.Crossover, "child", broken)
    child = _crossover_child(pa, pb, table, random.Random(0))
    assert ids(child) == ids(pa) and child is not pa


def test_score_temperature():
    pieces, layout, table = puzzle(64, 64)
    arr = Arrangement(list(pieces), layout)
    total = table.arrangement_cost(arr)
    assert arr.score(table) == pytest.approx(math.exp(-total / 24 / 0.05), rel=1e-12)


def test_ga_defaults():
    ga = GeneticAlgorithm(noise16(), 4, population_size=5, generations=1)
    assert ga._elite_size == 2
    assert ga._mutation_rate == 0.05
    assert ga._local_search_steps == 1
    assert ga._local_search_candidates == 4
    assert ga._repair_steps == 1
    assert ga._max_restarts == 2
    assert ga._tournament_size == 3
    assert ga._seed_fraction == 0.25
    assert ga._seed_beam_width == 4
    assert ga._seed_candidate_width == 8
    assert ga._seed_confidence_weight == 0.02
    assert ga._seed_border_weight == 0.0
    assert GeneticAlgorithm.TERMINATION_THRESHOLD == 10


def test_facade_contract():
    import dataclasses
    import inspect

    params = inspect.signature(solve_jigsaw).parameters
    assert params["population"].default == 200
    assert params["generations"].default == 20
    assert params["seed"].default == 0
    result = JigsawResult(np.zeros(1, np.int64), np.zeros((1, 1), np.uint8), 1.0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.5
    # one-pixel pieces are allowed
    img = np.array([[0, 60], [120, 180]], np.uint8)
    out = solve_jigsaw(img[:, ::-1].copy(), 1, population=6, generations=2, seed=0)
    assert sorted(out.order.tolist()) == [0, 1, 2, 3]
    assert out.image.shape == (2, 2)


def test_seed_confidence_weight_zero_is_valid():
    pieces, layout, table = puzzle(32, 32)
    seeds = build_seed_arrangements(
        pieces, layout, table, random.Random(0), count=1, confidence_weight=0.0
    )
    assert len(seeds) == 1


def test_local_search_on_two_pieces_and_single_steps():
    pieces, layout, table = puzzle(16, 32)  # 1x2
    arr = Arrangement([pieces[1], pieces[0]], layout)
    _local_search(arr, table, random.Random(0), steps=1, candidates=4)
    assert ids(arr) == [0, 1]
    pieces, layout, table = puzzle(64, 64)
    arr = Arrangement.random(pieces, layout, random.Random(1))
    before = arr.score(table)
    _local_search(arr, table, random.Random(4), steps=1, candidates=5)
    assert arr.score(table) > before


def test_repair_windows_for_thin_layouts():
    for h, w, expected in ((16, 48, (0, 1, 2)), (48, 16, (0, 1, 2))):
        pieces, layout, _ = puzzle(h, w)
        arr = Arrangement(pieces, layout)
        assert _repair_window_indices(arr, random.Random(0)) == expected
    pieces, layout, _ = puzzle(80, 16)  # 5x1: windows (s, s+1, s+2) * columns
    arr = Arrangement(pieces, layout)
    rng = random.Random(3)
    windows = {_repair_window_indices(arr, rng) for _ in range(40)}
    assert windows == {(0, 1, 2), (1, 2, 3), (2, 3, 4)}
    pieces, layout, _ = puzzle(16, 80)  # 1x5
    arr = Arrangement(pieces, layout)
    rng = random.Random(3)
    assert {_repair_window_indices(arr, rng) for _ in range(40)} == windows


def test_restart_keeps_exactly_elite_size_when_best_is_an_elite_outsider():
    ga = GeneticAlgorithm(
        noise16(), 4, population_size=8, generations=1, rng=random.Random(0), elite_size=3
    )
    ga._analysis.analyze(ga.pieces)
    ranked = sorted(ga._population, key=ga._score)
    worst = ranked[0]
    ga._restart_population(worst)
    assert ga._population[0] is worst
    assert ga._population[1] is ranked[-1] and ga._population[2] is ranked[-2]
    assert all(p is not ranked[-3] for p in ga._population[3:])
    assert len(ga._population) == 8
