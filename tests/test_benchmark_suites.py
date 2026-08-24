"""The published suites. These assert the project's own rules about them, so that a
suite edited later cannot quietly drop below the standard the results were published
under -- in particular the floor of ten seeds for anything stochastic.
"""

import dataclasses

import pytest

from gto_solver.benchmark.suites import (
    DEFAULT_SUITES,
    QUICK_SEEDS,
    SUITES,
    Suite,
    get_suite,
    quick,
    run_suite,
)
from gto_solver.games.kuhn import KuhnGame
from gto_solver.solvers.registry import ALGORITHMS, get_algorithm

SUITE_NAMES = sorted(SUITES)
MINIMUM_STOCHASTIC_SEEDS = 10


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_every_suite_names_registered_algorithms(name):
    for algorithm in get_suite(name).algorithms:
        assert algorithm in ALGORITHMS


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_every_suite_has_ascending_positive_checkpoints(name):
    checkpoints = get_suite(name).checkpoints
    assert checkpoints
    assert list(checkpoints) == sorted(checkpoints)
    assert checkpoints[0] > 0
    assert len(set(checkpoints)) == len(checkpoints)


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_a_suite_with_a_stochastic_algorithm_runs_at_least_ten_seeds(name):
    """This project's own floor for reporting a sampled result. A single run of MCCFR
    is a data point, not a result.
    """
    suite = get_suite(name)
    stochastic = [a for a in suite.algorithms if not get_algorithm(a).deterministic]
    if stochastic:
        assert len(suite.seeds) >= MINIMUM_STOCHASTIC_SEEDS, stochastic


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_seeds_are_distinct(name):
    seeds = get_suite(name).seeds
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_the_spread_chart_targets_a_stochastic_algorithm_in_the_suite(name):
    """A seed-spread chart of a deterministic variant would be one flat line drawn
    twenty times.
    """
    suite = get_suite(name)
    if suite.spread_algorithm is not None:
        assert suite.spread_algorithm in suite.algorithms
        assert not get_algorithm(suite.spread_algorithm).deterministic


def test_default_suites_exist():
    assert DEFAULT_SUITES
    for name in DEFAULT_SUITES:
        assert name in SUITES


def test_suite_keys_match_suite_names():
    for key, suite in SUITES.items():
        assert key == suite.name


def test_unknown_suite_lists_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        get_suite("kuhn_convergance")
    for name in SUITES:
        assert name in str(excinfo.value)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown suite kind"):
        Suite(
            name="bogus",
            title="",
            subtitle="",
            kind="vibes",
            make_game=KuhnGame,
            game_label="kuhn",
            algorithms=("vanilla",),
            checkpoints=(10,),
            seeds=(0,),
        )


# --- quick profile ---------------------------------------------------------


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_quick_shrinks_a_suite_and_says_what_it_cut(name):
    suite = get_suite(name)
    reduced, notes = quick(suite)
    assert reduced.checkpoints
    assert reduced.checkpoints[-1] <= suite.checkpoints[-1]
    assert len(reduced.seeds) <= QUICK_SEEDS
    assert len(notes) == 1
    assert "QUICK PROFILE" in notes[0]
    assert "not the published numbers" in notes[0]


def test_quick_never_empties_a_suite():
    """Every checkpoint being above the ceiling must still leave one to measure."""
    suite = dataclasses.replace(get_suite("kuhn_convergence"), checkpoints=(50_000, 100_000))
    reduced, _ = quick(suite)
    assert reduced.checkpoints == (50_000,)


def test_quick_leaves_the_original_suite_untouched():
    suite = get_suite("kuhn_convergence")
    before = (suite.checkpoints, suite.seeds)
    quick(suite)
    assert (suite.checkpoints, suite.seeds) == before


# --- running a suite -------------------------------------------------------


def test_running_a_convergence_suite_produces_one_run_per_algorithm():
    suite = dataclasses.replace(
        get_suite("kuhn_convergence"),
        algorithms=("vanilla", "mccfr"),
        checkpoints=(10, 50),
        seeds=(0, 1),
    )
    results = run_suite(suite, extra_notes=("a note",))
    assert results.suite == suite.name
    assert len(results.convergence) == 2
    assert results.wallclock == ()
    assert {run.algorithm for run in results.convergence} == {"vanilla", "mccfr"}
    assert all(run.game == suite.game_label for run in results.convergence)
    assert "a note" in results.all_notes()
    assert results.provenance["python"]


def test_running_a_wallclock_suite_routes_to_the_wallclock_runner():
    suite = dataclasses.replace(
        get_suite("leduc_wallclock"),
        make_game=KuhnGame,
        game_label="kuhn_poker",
        algorithms=("mccfr",),
        checkpoints=(0.05,),
        seeds=(0, 1),
    )
    results = run_suite(suite)
    assert results.convergence == ()
    assert len(results.wallclock) == 1
    assert results.wallclock[0].seeds == (0, 1)


def test_a_suite_run_carries_the_progress_callback_through():
    suite = dataclasses.replace(
        get_suite("kuhn_convergence"), algorithms=("mccfr",), checkpoints=(10,), seeds=(0, 1)
    )
    lines: list[str] = []
    run_suite(suite, progress=lines.append)
    assert len(lines) == 2
