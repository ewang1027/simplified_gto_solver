"""Running a solver repeatedly and recording what it cost.

Every number this project publishes about convergence or speed comes through here.
That is the point rather than an accident: Phase 6 optimizes the hot loop, and a
"1.8x faster" claim only means something if the before and after were measured the
same way, on the same checkpoints, with the same things counted and the same things
excluded.

Three measurement rules are enforced here instead of being left to callers:

1. **Evaluating exploitability is never charged to the solver's clock.** Measuring
   costs 56 ms on Leduc against a 31 ms iteration -- charge it and the traversal
   that gets measured more often looks slower, which would make the exact-vs-sampled
   comparison a measurement artifact rather than a result.
2. **Deterministic variants are run once, not N times.** `FullTraversal` never reads
   the rng, so twenty seeds produce twenty identical curves and a band of width
   zero. The run is still labelled with how many seeds it *would* have used.
3. **Nothing is capped, reduced or skipped silently.** Anything the runner declines
   to do lands in `notes`, which is serialized with the results and printed by the
   benchmark script. A benchmark that quietly shrinks its own workload reads as a
   clean result later.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from gto_solver.benchmark.stats import Aggregate, aggregate
from gto_solver.games.base import Game
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.registry import AlgorithmSpec

GameFactory = Callable[[], Game]
Progress = Callable[[str], None]

# How much of the time remaining to a wall-clock checkpoint one training chunk aims
# to consume. Below 1.0 so a stale iterations/second estimate undershoots the
# checkpoint (costing an extra chunk) rather than sailing past it.
_CHUNK_FRACTION = 0.75
# Overshooting a wall-clock checkpoint by more than this fraction gets a note. It
# happens legitimately -- one Leduc iteration is 31 ms, so a 50 ms checkpoint cannot
# be hit closely -- but the reader should be told rather than left to assume the
# x-axis is exact.
_OVERSHOOT_TOLERANCE = 0.20


def _resolve_seeds(spec: AlgorithmSpec, seeds: Sequence[int]) -> tuple[tuple[int, ...], list[str]]:
    """Seeds actually worth running, plus any note explaining a reduction."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("need at least one seed")
    if spec.deterministic and len(seeds) > 1:
        note = (
            f"{spec.name}: ran 1 seed instead of {len(seeds)}. The traversal never reads "
            f"the rng, so every seed gives a bit-identical strategy; extra seeds would "
            f"widen no band. Verified by verify_determinism()."
        )
        return seeds[:1], [note]
    return seeds, []


@dataclass(frozen=True)
class ConvergenceRun:
    """Exploitability at each iteration checkpoint, for one variant on one game.

    `exploitability_by_seed` and `train_seconds_by_seed` are indexed [seed][checkpoint];
    `train_seconds_by_seed` is cumulative training time and excludes every
    exploitability evaluation.
    """

    algorithm: str
    label: str
    game: str
    deterministic: bool
    seeds: tuple[int, ...]
    checkpoints: tuple[int, ...]
    exploitability_by_seed: tuple[tuple[float, ...], ...]
    train_seconds_by_seed: tuple[tuple[float, ...], ...]
    info_sets: int
    requested_seeds: int
    notes: tuple[str, ...] = ()

    def values_at(self, index: int) -> tuple[float, ...]:
        """Every seed's exploitability at checkpoint `index`."""
        return tuple(curve[index] for curve in self.exploitability_by_seed)

    def aggregates(self, **kwargs) -> tuple[Aggregate, ...]:
        """One `Aggregate` per checkpoint, across seeds."""
        return tuple(
            aggregate(self.values_at(i), **kwargs) for i in range(len(self.checkpoints))
        )

    def median_curve(self) -> tuple[float, ...]:
        return tuple(float(np.median(self.values_at(i))) for i in range(len(self.checkpoints)))

    def iterations_per_second(self) -> float:
        """Training throughput at the longest checkpoint, median across seeds."""
        totals = [curve[-1] for curve in self.train_seconds_by_seed]
        return float(self.checkpoints[-1] / np.median(totals))

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "label": self.label,
            "game": self.game,
            "deterministic": self.deterministic,
            "seeds": list(self.seeds),
            "requested_seeds": self.requested_seeds,
            "checkpoints": list(self.checkpoints),
            "exploitability_by_seed": [list(curve) for curve in self.exploitability_by_seed],
            "train_seconds_by_seed": [list(curve) for curve in self.train_seconds_by_seed],
            "info_sets": self.info_sets,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConvergenceRun":
        return cls(
            algorithm=data["algorithm"],
            label=data["label"],
            game=data["game"],
            deterministic=data["deterministic"],
            seeds=tuple(data["seeds"]),
            checkpoints=tuple(data["checkpoints"]),
            exploitability_by_seed=tuple(tuple(c) for c in data["exploitability_by_seed"]),
            train_seconds_by_seed=tuple(tuple(c) for c in data["train_seconds_by_seed"]),
            info_sets=data["info_sets"],
            requested_seeds=data["requested_seeds"],
            notes=tuple(data["notes"]),
        )


@dataclass(frozen=True)
class WallclockRun:
    """Exploitability at each wall-clock training budget, for one variant on one game.

    The x-axis is `train_seconds_by_seed`, the time actually spent training, not the
    requested `time_checkpoints` -- a solver whose iteration is coarse relative to a
    checkpoint overshoots it, and the plot should show where the measurement really
    landed.
    """

    algorithm: str
    label: str
    game: str
    deterministic: bool
    seeds: tuple[int, ...]
    time_checkpoints: tuple[float, ...]
    train_seconds_by_seed: tuple[tuple[float, ...], ...]
    iterations_by_seed: tuple[tuple[int, ...], ...]
    exploitability_by_seed: tuple[tuple[float, ...], ...]
    info_sets: int
    requested_seeds: int
    notes: tuple[str, ...] = ()

    def values_at(self, index: int) -> tuple[float, ...]:
        return tuple(curve[index] for curve in self.exploitability_by_seed)

    def aggregates(self, **kwargs) -> tuple[Aggregate, ...]:
        return tuple(
            aggregate(self.values_at(i), **kwargs) for i in range(len(self.time_checkpoints))
        )

    def median_seconds(self) -> tuple[float, ...]:
        return tuple(
            float(np.median([curve[i] for curve in self.train_seconds_by_seed]))
            for i in range(len(self.time_checkpoints))
        )

    def median_iterations(self) -> tuple[float, ...]:
        return tuple(
            float(np.median([curve[i] for curve in self.iterations_by_seed]))
            for i in range(len(self.time_checkpoints))
        )

    def iterations_per_second(self) -> float:
        rates = [
            iters[-1] / seconds[-1]
            for iters, seconds in zip(self.iterations_by_seed, self.train_seconds_by_seed)
        ]
        return float(np.median(rates))

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "label": self.label,
            "game": self.game,
            "deterministic": self.deterministic,
            "seeds": list(self.seeds),
            "requested_seeds": self.requested_seeds,
            "time_checkpoints": list(self.time_checkpoints),
            "train_seconds_by_seed": [list(c) for c in self.train_seconds_by_seed],
            "iterations_by_seed": [list(c) for c in self.iterations_by_seed],
            "exploitability_by_seed": [list(c) for c in self.exploitability_by_seed],
            "info_sets": self.info_sets,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WallclockRun":
        return cls(
            algorithm=data["algorithm"],
            label=data["label"],
            game=data["game"],
            deterministic=data["deterministic"],
            seeds=tuple(data["seeds"]),
            time_checkpoints=tuple(data["time_checkpoints"]),
            train_seconds_by_seed=tuple(tuple(c) for c in data["train_seconds_by_seed"]),
            iterations_by_seed=tuple(tuple(c) for c in data["iterations_by_seed"]),
            exploitability_by_seed=tuple(tuple(c) for c in data["exploitability_by_seed"]),
            info_sets=data["info_sets"],
            requested_seeds=data["requested_seeds"],
            notes=tuple(data["notes"]),
        )


@dataclass(frozen=True)
class DeterminismReport:
    """Result of actually checking an `AlgorithmSpec.deterministic` claim."""

    algorithm: str
    claimed_deterministic: bool
    observed_identical: bool
    seeds: tuple[int, ...]
    iterations: int
    differing_info_sets: tuple[str, ...] = field(default=())

    @property
    def claim_holds(self) -> bool:
        return self.claimed_deterministic == self.observed_identical


def run_convergence(
    game_factory: GameFactory,
    spec: AlgorithmSpec,
    checkpoints: Sequence[int],
    seeds: Sequence[int] = (0,),
    game_label: str | None = None,
    progress: Progress | None = None,
) -> ConvergenceRun:
    """Train `spec` on the game, recording exploitability at each iteration checkpoint.

    Training is incremental -- one solver walks past every checkpoint in order rather
    than being retrained from scratch for each -- so the reported training seconds are
    cumulative and the curve costs one full run per seed, not one per checkpoint.
    """
    checkpoints = tuple(checkpoints)
    if not checkpoints:
        raise ValueError("need at least one checkpoint")
    if list(checkpoints) != sorted(checkpoints) or checkpoints[0] < 1:
        raise ValueError(f"checkpoints must be ascending positive integers, got {checkpoints}")

    used_seeds, notes = _resolve_seeds(spec, seeds)
    game = game_factory()
    label = game_label or game.name

    curves: list[tuple[float, ...]] = []
    timings: list[tuple[float, ...]] = []
    info_sets = 0

    for seed in used_seeds:
        game = game_factory()
        solver = spec.build(game, seed=seed)
        values: list[float] = []
        seconds: list[float] = []
        elapsed = 0.0
        for target in checkpoints:
            start = time.perf_counter()
            solver.train(target - solver.iterations)
            elapsed += time.perf_counter() - start
            # Deliberately outside the timed region: see rule 1 in the module docstring.
            values.append(exploitability(game, solver.average_strategy()))
            seconds.append(elapsed)
        curves.append(tuple(values))
        timings.append(tuple(seconds))
        info_sets = len(solver.store)
        if progress is not None:
            progress(
                f"  {spec.name:<22} {label:<24} seed {seed:>3}  "
                f"{checkpoints[-1]:>7d} it in {elapsed:6.2f}s  expl {values[-1]:.6f}"
            )

    return ConvergenceRun(
        algorithm=spec.name,
        label=spec.label,
        game=label,
        deterministic=spec.deterministic,
        seeds=used_seeds,
        checkpoints=checkpoints,
        exploitability_by_seed=tuple(curves),
        train_seconds_by_seed=tuple(timings),
        info_sets=info_sets,
        requested_seeds=len(tuple(seeds)),
        notes=tuple(notes),
    )


def run_wallclock(
    game_factory: GameFactory,
    spec: AlgorithmSpec,
    time_checkpoints: Sequence[float],
    seeds: Sequence[int] = (0,),
    game_label: str | None = None,
    max_chunk: int = 100_000,
    progress: Progress | None = None,
) -> WallclockRun:
    """Train `spec` under increasing wall-clock budgets, recording exploitability.

    This is the comparison exact traversal and sampling actually need: iterations are
    not a common currency between them, since one MCCFR iteration is a single sampled
    path and one exact iteration walks the whole tree. Seconds are.

    Iterations are run in chunks sized from the throughput observed so far, aiming at
    a fraction of the time remaining to the next checkpoint so a stale estimate
    undershoots rather than overshoots. Overshoot is still possible -- a single
    iteration can be coarser than the checkpoint spacing -- so the actual elapsed time
    is recorded per checkpoint and a note is emitted when it lands more than
    `_OVERSHOOT_TOLERANCE` past target.
    """
    time_checkpoints = tuple(float(t) for t in time_checkpoints)
    if not time_checkpoints:
        raise ValueError("need at least one time checkpoint")
    if list(time_checkpoints) != sorted(time_checkpoints) or time_checkpoints[0] <= 0:
        raise ValueError(f"time checkpoints must be ascending and positive, got {time_checkpoints}")

    used_seeds, notes = _resolve_seeds(spec, seeds)
    label = game_label or game_factory().name

    seconds_by_seed: list[tuple[float, ...]] = []
    iterations_by_seed: list[tuple[int, ...]] = []
    values_by_seed: list[tuple[float, ...]] = []
    overshoots: list[str] = []
    info_sets = 0

    for seed in used_seeds:
        game = game_factory()
        solver = spec.build(game, seed=seed)
        elapsed = 0.0
        iterations = 0
        seconds: list[float] = []
        counts: list[int] = []
        values: list[float] = []

        for target in time_checkpoints:
            while elapsed < target:
                remaining = target - elapsed
                if iterations == 0:
                    chunk = 1
                else:
                    rate = iterations / elapsed
                    chunk = max(1, min(int(rate * remaining * _CHUNK_FRACTION), max_chunk))
                start = time.perf_counter()
                solver.train(chunk)
                elapsed += time.perf_counter() - start
                iterations += chunk
            seconds.append(elapsed)
            counts.append(iterations)
            # Off the clock, as in run_convergence.
            values.append(exploitability(game, solver.average_strategy()))
            if elapsed > target * (1.0 + _OVERSHOOT_TOLERANCE):
                overshoots.append(
                    f"{spec.name} on {label} (seed {seed}): {target:g}s checkpoint measured at "
                    f"{elapsed:.3f}s ({elapsed / target:.2f}x target) -- one iteration is coarse "
                    f"relative to this budget. Plotted at the measured time, not the target."
                )

        seconds_by_seed.append(tuple(seconds))
        iterations_by_seed.append(tuple(counts))
        values_by_seed.append(tuple(values))
        info_sets = len(solver.store)
        if progress is not None:
            progress(
                f"  {spec.name:<22} {label:<24} seed {seed:>3}  "
                f"{counts[-1]:>9d} it in {seconds[-1]:6.2f}s  expl {values[-1]:.6f}"
            )

    return WallclockRun(
        algorithm=spec.name,
        label=spec.label,
        game=label,
        deterministic=spec.deterministic,
        seeds=used_seeds,
        time_checkpoints=time_checkpoints,
        train_seconds_by_seed=tuple(seconds_by_seed),
        iterations_by_seed=tuple(iterations_by_seed),
        exploitability_by_seed=tuple(values_by_seed),
        info_sets=info_sets,
        requested_seeds=len(tuple(seeds)),
        notes=tuple(notes + overshoots),
    )


def verify_determinism(
    game_factory: GameFactory,
    spec: AlgorithmSpec,
    iterations: int = 200,
    seeds: Sequence[int] = (0, 1, 2),
) -> DeterminismReport:
    """Check an `AlgorithmSpec.deterministic` claim by actually running the seeds.

    The benchmark skips extra seeds for variants claiming determinism, so the claim is
    load-bearing: if it were wrong, a genuinely stochastic variant would be published
    as a single run with no band. This trains the variant under each seed and compares
    the average strategies entry by entry, and it is what `tests/test_registry.py`
    calls -- for the stochastic variants too, where the expected answer is that the
    strategies *differ*.
    """
    seeds = tuple(seeds)
    if len(seeds) < 2:
        raise ValueError("determinism needs at least two seeds to compare")

    baseline: dict[str, np.ndarray] | None = None
    differing: set[str] = set()
    for seed in seeds:
        solver = spec.build(game_factory(), seed=seed)
        solver.train(iterations)
        strategy = solver.average_strategy()
        if baseline is None:
            baseline = strategy
            continue
        if strategy.keys() != baseline.keys():
            differing.update(set(strategy.keys()) ^ set(baseline.keys()))
            continue
        differing.update(
            key for key in strategy if not np.array_equal(strategy[key], baseline[key])
        )

    return DeterminismReport(
        algorithm=spec.name,
        claimed_deterministic=spec.deterministic,
        observed_identical=not differing,
        seeds=seeds,
        iterations=iterations,
        differing_info_sets=tuple(sorted(differing)),
    )
