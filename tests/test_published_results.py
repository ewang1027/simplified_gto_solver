"""The results files checked into `results/` are the numbers the README quotes, so
they get checked like anything else that is published.

The specific failure this guards against is the one the `--quick` profile creates: a
smoke run takes seconds and produces a file that looks exactly like the real thing.
Committing one would put numbers into the README that were measured over three seeds
and a thousand iterations, with nothing on the surface to say so.

Skips cleanly when `results/` is empty, so a fresh checkout that has not run the
benchmark yet is not a test failure.
"""

import math
from pathlib import Path

import pytest

from gto_solver.benchmark.results import BenchmarkResults
from gto_solver.benchmark.suites import SUITES
from gto_solver.solvers.registry import get_algorithm

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MINIMUM_STOCHASTIC_SEEDS = 10
PUBLISHED = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(not PUBLISHED, reason="no results have been published yet")


@pytest.fixture(scope="module", params=[p.name for p in PUBLISHED])
def published(request) -> BenchmarkResults:
    return BenchmarkResults.load(RESULTS_DIR / request.param)


def test_published_results_still_load(published):
    """A schema change that orphans the published files should fail here, not in
    whatever later phase next tries to compare against them.
    """
    assert published.suite
    assert published.convergence or published.wallclock


def test_published_results_are_not_a_quick_run(published):
    for note in published.all_notes():
        assert "QUICK PROFILE" not in note, f"{published.suite} was published from a smoke run"


def test_published_stochastic_runs_have_enough_seeds(published):
    for run in (*published.convergence, *published.wallclock):
        if not get_algorithm(run.algorithm).deterministic:
            assert len(run.seeds) >= MINIMUM_STOCHASTIC_SEEDS, run.algorithm


def test_published_deterministic_runs_have_exactly_one_seed(published):
    for run in (*published.convergence, *published.wallclock):
        if get_algorithm(run.algorithm).deterministic:
            assert len(run.seeds) == 1, run.algorithm


def test_published_results_were_measured_from_a_clean_tree(published):
    """A number measured against uncommitted edits cannot be reproduced from the
    commit it claims to come from.
    """
    git = published.provenance.get("git", {})
    assert git.get("commit"), published.suite
    assert git.get("dirty") is False, f"{published.suite} was measured with a dirty tree"


def test_published_suites_are_still_defined(published):
    """A results file for a suite nobody can run any more is a dead number."""
    assert published.suite in SUITES


def test_published_runs_match_their_suite_definition(published):
    """Guards against a results file that has drifted from the suite it names --
    fewer algorithms, or a shorter run than the definition says.
    """
    suite = SUITES[published.suite]
    runs = published.convergence or published.wallclock
    assert {run.algorithm for run in runs} == set(suite.algorithms)
    for run in runs:
        assert run.game == suite.game_label
    if published.convergence:
        for run in published.convergence:
            assert run.checkpoints == suite.iteration_checkpoints
    else:
        for run in published.wallclock:
            assert run.time_checkpoints == suite.checkpoints


def test_every_published_curve_is_finite_and_nonnegative(published):
    """Exploitability is >= 0 by construction; a nan or a negative would mean the
    metric, not the solver, is broken -- and it would still plot.
    """
    for run in (*published.convergence, *published.wallclock):
        for curve in run.exploitability_by_seed:
            for value in curve:
                assert math.isfinite(value), (run.algorithm, value)
                assert value >= 0.0, (run.algorithm, value)
