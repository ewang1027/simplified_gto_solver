"""Aggregating a measurement repeated across seeds.

Two different questions get two different numbers here, and conflating them is the
easiest way to publish a band that means nothing:

* The **seed envelope** (`p_low` .. `p_high`) says how much individual runs differ
  from one another. It answers "if I run this once with a seed nobody has tried,
  what should I expect?" It does *not* shrink as seeds are added -- it is a property
  of the algorithm's variance, not of how hard we measured.
* The **bootstrap CI** (`ci_low` .. `ci_high`) says how well these N seeds pin down
  the central estimate. It *does* shrink as N grows.

Plots shade the envelope, tables quote the CI, and both are labelled as such
wherever they appear.

The central statistic is the **median**, not the mean. Exploitability across seeds is
positive and right-skewed and is read on a log axis; one unlucky seed drags a mean to
a place no individual run ever visited.
"""

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_ENVELOPE = (10.0, 90.0)
DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20250823


def log_checkpoints(start: int, stop: int, per_decade: int = 3) -> tuple[int, ...]:
    """Roughly log-spaced integer checkpoints from `start` to `stop`, inclusive.

    Convergence is read on a log-log axis, so evenly spaced iteration counts would
    crowd the right-hand end and leave the interesting early decade nearly empty.
    Rounding to integers can collide at the low end, so duplicates are dropped --
    the returned tuple can therefore be shorter than `per_decade` per decade, which
    is fine and is why callers should read the returned checkpoints rather than
    assume them.
    """
    if start < 1 or stop < start:
        raise ValueError(f"need 1 <= start <= stop, got start={start}, stop={stop}")
    if per_decade < 1:
        raise ValueError(f"per_decade must be >= 1, got {per_decade}")

    decades = math.log10(stop / start)
    count = max(2, round(decades * per_decade) + 1)
    raw = np.logspace(math.log10(start), math.log10(stop), num=count)
    points = sorted(set(np.rint(raw).astype(int).tolist()) | {start, stop})
    return tuple(point for point in points if start <= point <= stop)


def bootstrap_ci(
    values,
    statistic: Callable[..., np.ndarray] = np.median,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for `statistic` over `values`.

    `statistic` is called as `statistic(matrix, axis=1)`, so it must accept an
    `axis` argument -- `np.median` and `np.mean` do.

    Reproducible: the resampling draws from `np.random.default_rng(seed)`, so the
    published interval can be regenerated exactly. The interval is deliberately
    reported as a range and never converted into a p-value: at the ten to twenty
    seeds a benchmark run can afford, a percentile bootstrap is coarse, and saying
    so is cheaper than being caught pretending otherwise.

    Measured, because the obvious worry is that the arbitrary `seed` steers the
    published number: at n <= 20 with 10,000 resamples the interval comes out
    *bit-identical* across bootstrap seeds. The resample distribution of a median
    over so few points is discrete enough that its 2.5th and 97.5th percentiles land
    on the same order statistics every time. The wobble only appears further up --
    around 13% of the interval width at n = 50 -- which is still far above any seed
    count this project's benchmarks use. Pinned in tests/test_benchmark_stats.py.

    A single value has no spread to estimate, so the interval collapses to that
    value. That is the case for every deterministic variant, and it is honest: the
    uncertainty really is zero, because a rerun returns the same number.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError(f"bootstrap_ci expects a 1-D sample, got shape {sample.shape}")
    if sample.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if sample.size == 1:
        only = float(sample[0])
        return only, only

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(resamples, sample.size))
    stats = statistic(sample[draws], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(stats, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(low), float(high)


@dataclass(frozen=True)
class Aggregate:
    """Summary of one measurement across seeds.

    `p_low`/`p_high` are the seed envelope and `ci_low`/`ci_high` the bootstrap CI of
    the median; the two answer different questions (see the module docstring). The
    percentiles and confidence level that produced them are carried along so a
    serialized result cannot be reinterpreted under different settings later.
    """

    n: int
    median: float
    mean: float
    std: float
    minimum: float
    maximum: float
    p_low: float
    p_high: float
    ci_low: float
    ci_high: float
    envelope: tuple[float, float]
    confidence: float

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def spread_ratio(self) -> float:
        """max/min across seeds. On a log axis this is the readable measure of how
        much seed luck is worth; it is infinite if any run hit exactly zero.
        """
        if self.minimum <= 0.0:
            return math.inf
        return self.maximum / self.minimum


def aggregate(
    values,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Aggregate:
    """Summarize repeated measurements of one quantity across seeds."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError(f"aggregate expects a 1-D sample, got shape {sample.shape}")
    if sample.size == 0:
        raise ValueError("aggregate needs at least one value")
    low_pct, high_pct = envelope
    if not 0.0 <= low_pct <= high_pct <= 100.0:
        raise ValueError(f"envelope must be an ascending pair of percentiles, got {envelope}")

    ci_low, ci_high = bootstrap_ci(
        sample, statistic=np.median, confidence=confidence, resamples=resamples, seed=seed
    )
    return Aggregate(
        n=int(sample.size),
        median=float(np.median(sample)),
        mean=float(np.mean(sample)),
        # ddof=1 is the sample standard deviation; with one seed there is no spread
        # to estimate at all, and numpy would return nan rather than 0.
        std=float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0,
        minimum=float(sample.min()),
        maximum=float(sample.max()),
        p_low=float(np.percentile(sample, low_pct)),
        p_high=float(np.percentile(sample, high_pct)),
        ci_low=ci_low,
        ci_high=ci_high,
        envelope=(float(low_pct), float(high_pct)),
        confidence=float(confidence),
    )
