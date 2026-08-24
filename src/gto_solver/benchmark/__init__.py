"""Benchmarking: repeated, seeded runs of a solver, aggregated with confidence bands
and serialized so a later phase can compare against them.

The measurement rules that make results comparable across phases live in
`runner.py`; the two-kinds-of-band distinction lives in `stats.py`; provenance and
before/after comparison live in `results.py`. `plots.py` needs the optional `viz`
extra and is not imported here, so `import gto_solver.benchmark` stays free of any
matplotlib dependency.
"""

from gto_solver.benchmark.results import (
    BenchmarkResults,
    ComparisonReport,
    RunComparison,
    compare,
    provenance,
)
from gto_solver.benchmark.runner import (
    ConvergenceRun,
    DeterminismReport,
    WallclockRun,
    run_convergence,
    run_wallclock,
    verify_determinism,
)
from gto_solver.benchmark.stats import Aggregate, aggregate, bootstrap_ci, log_checkpoints

__all__ = [
    "Aggregate",
    "BenchmarkResults",
    "ComparisonReport",
    "ConvergenceRun",
    "DeterminismReport",
    "RunComparison",
    "WallclockRun",
    "aggregate",
    "bootstrap_ci",
    "compare",
    "log_checkpoints",
    "provenance",
    "run_convergence",
    "run_wallclock",
    "verify_determinism",
]
