"""Tests for ejps/jigsaw/ga.py (gaps parent selection, ported).

Index pairs marked "upstream" were produced by the original gaps (commit
5aa0a0d, Python 3.12) with the same ``random.Random`` seeds and scores.
"""

import random

import pytest

from ejps.jigsaw.ga import roulette_selection, tournament_selection


class Item:
    def __init__(self, index, score):
        self.index = index
        self.value = score


def population(scores):
    return [Item(i, s) for i, s in enumerate(scores)]


def score(item):
    return item.value


def as_indices(pairs):
    return [[a.index, b.index] for a, b in pairs]


SIX = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SEVEN = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]


def test_tournament_matches_upstream():
    assert as_indices(tournament_selection(population(SIX), score, random.Random(5), elites=2)) == [
        [5, 5],
        [5, 5],
        [1, 3],
        [4, 4],
    ]
    pop = population(SEVEN)
    got = tournament_selection(pop, score, random.Random(8), elites=3, tournament_size=1)
    assert as_indices(got) == [[1, 2], [3, 1], [1, 5], [0, 0]]
    got = tournament_selection(pop, score, random.Random(8), elites=0, tournament_size=9)
    assert as_indices(got) == [[5, 4], [5, 4], [5, 5], [5, 5], [4, 5], [5, 4], [5, 4]]
    got = tournament_selection(pop, score, random.Random(8), elites=5, tournament_size=2)
    assert as_indices(got) == [[2, 3], [5, 0]]


def test_tournament_draws_exactly_k_contestants_per_parent():
    pop = population(SEVEN)
    rng = random.Random(8)
    pairs = tournament_selection(pop, score, rng, elites=4, tournament_size=2)
    assert len(pairs) == 3
    ref = random.Random(8)
    for a, b in pairs:
        first = [pop[ref.randrange(7)] for _ in range(2)]
        second = [pop[ref.randrange(7)] for _ in range(2)]
        assert a is max(first, key=score)
        assert b is max(second, key=score)
    assert rng.random() == ref.random()  # same number of draws
    # tournament size is capped at the population size
    rng = random.Random(1)
    tournament_selection(population([1.0, 2.0]), score, rng, elites=1, tournament_size=50)
    ref = random.Random(1)
    for _ in range(4):
        ref.randrange(2)
    assert rng.random() == ref.random()


def test_tournament_validation():
    pop = population(SIX)
    with pytest.raises(ValueError):
        tournament_selection([], score, random.Random(0))
    with pytest.raises(ValueError):
        tournament_selection(pop, score, random.Random(0), elites=6)
    with pytest.raises(ValueError):
        tournament_selection(pop, score, random.Random(0), elites=-1)
    with pytest.raises(ValueError):
        tournament_selection(pop, score, random.Random(0), tournament_size=0)
    assert len(tournament_selection(pop, score, random.Random(0), elites=5)) == 1
    assert len(tournament_selection(pop, score, random.Random(0), elites=0)) == 6


def test_roulette_matches_upstream():
    assert as_indices(roulette_selection(population(SIX), score, random.Random(5), elites=1)) == [
        [4, 5],
        [5, 5],
        [5, 5],
        [0, 3],
        [5, 4],
    ]
    got = roulette_selection(population(SEVEN), score, random.Random(8), elites=0)
    assert as_indices(got) == [[2, 6], [1, 5], [0, 2], [6, 2], [5, 4], [4, 4], [2, 5]]


class FixedRng:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


def test_roulette_is_proportional_to_score():
    # cumulative [1, 3, 6, 10]; draws * 10 -> 0, 0.999, 1.0, 2.9, 3.0, 9.99
    pop = population([1.0, 2.0, 3.0, 4.0])
    rng = FixedRng([0.0, 0.0999, 0.1, 0.29, 0.3, 0.999])
    got = roulette_selection(pop, score, rng, elites=1)
    assert as_indices(got) == [[0, 0], [1, 1], [2, 3]]
    # a draw at the very top is clamped to the last individual
    got = roulette_selection(population([1.0, 1.0]), score, FixedRng([1.0, 0.5]), elites=1)
    assert as_indices(got) == [[1, 1]]


def test_roulette_validation():
    with pytest.raises(ValueError):
        roulette_selection([], score, random.Random(0))
    with pytest.raises(ValueError):
        roulette_selection(population(SIX), score, random.Random(0), elites=6)
    with pytest.raises(ValueError):
        roulette_selection(population(SIX), score, random.Random(0), elites=-1)
    assert len(roulette_selection(population(SIX), score, random.Random(0))) == 2


def test_default_elites_is_four():
    assert len(tournament_selection(population(SIX), score, random.Random(0))) == 2
    assert len(tournament_selection(population(SIX * 2), score, random.Random(0))) == 8
    assert len(roulette_selection(population(SIX * 2), score, random.Random(0))) == 8
