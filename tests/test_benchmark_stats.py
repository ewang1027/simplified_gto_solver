"""Aggregation: log-spaced checkpoints, the bootstrap, and the two kinds of band.

The distinction the whole module exists to hold is that a **seed envelope** and a
**confidence interval of the median** answer different questions, so the tests pin
the behaviour that distinguishes them: adding seeds must narrow the CI and must not
systematically narrow the envelope. Thresholds below come from running the check on
a fixed lognormal pool, not from guessing:

    n=10 vs n=2000, seed 12345:  CI width shrinks 11.8x, envelope width moves 0.665x

so "CI shrinks by more than 5x" and "envelope stays within 2.5x either way" both sit
a long way from the measured values.
"""

import math
from itertools import pairwise

import numpy as np
import pytest

from gto_solver.benchmark.stats import (
    DEFAULT_BOOTSTRAP_SEED,
    aggregate,
    bootstrap_ci,
    log_checkpoints,
)

KNOWN = [1.0, 2.0, 3.0, 4.0, 100.0]


@pytest.fixture(scope="module")
def pool() -> np.ndarray:
    """A fixed right-skewed sample, shaped like exploitability across seeds."""
    return np.random.default_rng(12345).lognormal(mean=-4.0, sigma=0.6, size=2000)


# --- log_checkpoints -------------------------------------------------------


def test_log_checkpoints_spans_the_range_and_ascends():
    points = log_checkpoints(10, 100_000, per_decade=3)
    assert points[0] == 10
    assert points[-1] == 100_000
    assert list(points) == sorted(points)
    assert len(set(points)) == len(points)


def test_log_checkpoints_are_roughly_log_spaced():
    points = log_checkpoints(10, 100_000, per_decade=3)
    ratios = [b / a for a, b in pairwise(points)]
    assert max(ratios) / min(ratios) < 1.2


def test_log_checkpoints_dedupes_collisions_at_the_low_end():
    """Rounding to integers can produce the same checkpoint twice; the result stays
    strictly ascending and simply gets shorter, which is why callers read it back.
    """
    points = log_checkpoints(1, 10, per_decade=8)
    assert list(points) == sorted(set(points))
    assert points[0] == 1 and points[-1] == 10


@pytest.mark.parametrize(
    "start, stop, per_decade",
    [(0, 100, 3), (-5, 100, 3), (100, 10, 3), (10, 100, 0)],
)
def test_log_checkpoints_rejects_nonsense(start, stop, per_decade):
    with pytest.raises(ValueError):
        log_checkpoints(start, stop, per_decade)


# --- bootstrap_ci ----------------------------------------------------------


def test_bootstrap_is_reproducible_for_a_given_seed(pool):
    assert bootstrap_ci(pool[:20], seed=4) == bootstrap_ci(pool[:20], seed=4)


def test_bootstrap_does_not_depend_on_its_seed_at_benchmark_sample_sizes(pool):
    """Measured property, and the reason the arbitrary default seed is defensible.

    At n <= 20 with 10,000 resamples the resample distribution of the median is
    discrete enough that its tail percentiles land on the same order statistics every
    time, so the interval is bit-identical across bootstrap seeds. If this ever
    starts failing, the published intervals have become seed-dependent and the
    default seed is quietly steering them.
    """
    for n in (3, 5, 10, 20):
        intervals = {bootstrap_ci(pool[:n], seed=s) for s in (1, 2, 7, 99, DEFAULT_BOOTSTRAP_SEED)}
        assert len(intervals) == 1, f"n={n} gave {len(intervals)} different intervals"


def test_bootstrap_ci_brackets_the_median(pool):
    for n in (5, 10, 20, 50):
        low, high = bootstrap_ci(pool[:n])
        assert low <= float(np.median(pool[:n])) <= high


def test_bootstrap_ci_stays_inside_the_sample_range(pool):
    """A percentile bootstrap resamples observed values, so it cannot invent one
    outside them -- unlike a normal approximation, which can put the lower end of a
    positive quantity below zero.
    """
    sample = pool[:20]
    low, high = bootstrap_ci(sample)
    assert sample.min() <= low <= high <= sample.max()


def test_bootstrap_ci_of_one_value_collapses_to_that_value():
    assert bootstrap_ci([0.25]) == (0.25, 0.25)


def test_bootstrap_ci_accepts_other_statistics(pool):
    low, high = bootstrap_ci(pool[:30], statistic=np.mean)
    assert low <= float(np.mean(pool[:30])) <= high


@pytest.mark.parametrize(
    "values, kwargs",
    [
        ([], {}),
        ([[1.0, 2.0], [3.0, 4.0]], {}),
        ([1.0, 2.0], {"confidence": 0.0}),
        ([1.0, 2.0], {"confidence": 1.0}),
    ],
)
def test_bootstrap_ci_rejects_nonsense(values, kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci(values, **kwargs)


# --- aggregate -------------------------------------------------------------


def test_aggregate_reports_the_textbook_statistics():
    agg = aggregate(KNOWN)
    assert agg.n == 5
    assert agg.median == 3.0
    assert agg.mean == pytest.approx(22.0)
    assert agg.std == pytest.approx(float(np.std(KNOWN, ddof=1)))
    assert (agg.minimum, agg.maximum) == (1.0, 100.0)
    assert agg.p_low == pytest.approx(1.4)
    assert agg.p_high == pytest.approx(61.6)


def test_aggregate_of_one_value_has_zero_spread():
    """One seed means no spread to estimate. numpy's sample std would be nan here,
    which would propagate into a results file as a null.
    """
    agg = aggregate([0.5])
    assert agg.n == 1
    assert agg.std == 0.0
    assert agg.median == agg.mean == agg.minimum == agg.maximum == 0.5
    assert agg.p_low == agg.p_high == agg.ci_low == agg.ci_high == 0.5


def test_more_seeds_narrow_the_confidence_interval_but_not_the_envelope(pool):
    """The load-bearing distinction, in one test.

    A confidence interval is about how well the seeds pin down the centre, so it
    shrinks as seeds are added. An envelope is about how much runs differ from each
    other, which is a property of the algorithm -- more seeds estimate it better but
    do not make it smaller. Swapping the two in a chart would make a noisy algorithm
    look like a precise one just by running it more.
    """
    few, many = aggregate(pool[:10]), aggregate(pool)
    ci_few, ci_many = few.ci_high - few.ci_low, many.ci_high - many.ci_low
    env_few, env_many = few.p_high - few.p_low, many.p_high - many.p_low

    assert ci_few / ci_many > 5.0, (ci_few, ci_many)
    assert 0.4 < env_few / env_many < 2.5, (env_few, env_many)


def test_aggregate_records_the_settings_that_produced_it():
    """A serialized band that does not say which percentiles it used can be
    reinterpreted later under different settings, silently.
    """
    agg = aggregate(KNOWN, envelope=(25.0, 75.0), confidence=0.80)
    assert agg.envelope == (25.0, 75.0)
    assert agg.confidence == 0.80
    assert agg.p_low == pytest.approx(2.0)
    assert agg.p_high == pytest.approx(4.0)


def test_spread_ratio_is_max_over_min():
    assert aggregate([2.0, 4.0, 8.0]).spread_ratio == pytest.approx(4.0)


def test_spread_ratio_is_infinite_when_a_run_hit_zero():
    """Exploitability really can be zero on a solved game, and a ratio against it is
    not a number a chart footnote should print.
    """
    assert math.isinf(aggregate([0.0, 1.0]).spread_ratio)


@pytest.mark.parametrize(
    "values, kwargs",
    [
        ([], {}),
        ([[1.0, 2.0]], {}),
        ([1.0, 2.0], {"envelope": (90.0, 10.0)}),
        ([1.0, 2.0], {"envelope": (-1.0, 90.0)}),
        ([1.0, 2.0], {"envelope": (10.0, 101.0)}),
    ],
)
def test_aggregate_rejects_nonsense(values, kwargs):
    with pytest.raises(ValueError):
        aggregate(values, **kwargs)
