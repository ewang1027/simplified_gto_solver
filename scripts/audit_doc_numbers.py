"""Re-measure the machine-specific numbers the documents state, and print both.

    python scripts/audit_doc_numbers.py

`tests/test_docs.py` checks everything about the documents that can be checked without
running anything: paths exist, commands exist, benchmark tables regenerate. This covers
what is left — claims that are true of a machine at a moment, like how long an iteration
takes or how much the wall clock wobbles between identical runs.

It exists because Phase 10 found two such numbers already stale, and both had been
invalidated by *this project's own* Phase 6 optimization: speeding the solver up made
one iteration cheaper without making exploitability evaluation cheaper, so a sentence
comparing the two quietly stopped being true. Nothing failed, because nothing checked.

This prints rather than asserts. A timing that has drifted 20% on a different machine is
not a defect, and a test that failed on it would be turned off within a week; a script
you run when editing the docs is the right shape for it. Compare the columns and fix the
document, the way `scripts/verify_phase4.py` is used for the design doc's tables.
"""

import time

import numpy as np

from gto_solver.analysis.microstructure import GMParams
from gto_solver.benchmark.results import BenchmarkResults
from gto_solver.games.leduc import LeducGame
from gto_solver.games.registry import get_game
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.registry import ALGORITHMS, get_algorithm

RESULTS = "results"


def row(claim: str, documented: str, measured: str) -> None:
    print(f"  {claim:<46} doc: {documented:<22} measured: {measured}")


def main() -> int:
    print("\n=== structural, from the code ===")
    row("variants in the registry", "8", str(len(ALGORITHMS)))
    row(
        "of those, rule x traversal compositions",
        "7",
        str(sum(spec.composed for spec in ALGORITHMS.values())),
    )
    row("Glosten-Milgrom quote grid", "33", str(GMParams().num_quotes))
    for name, documented in (("kuhn", "12"), ("leduc", "288")):
        solver = get_algorithm("vanilla").build(get_game(name).create(), seed=0)
        solver.train(1)
        row(f"{name} info sets", documented, str(len(solver.store)))

    print("\n=== timing, on this machine ===")
    game = LeducGame()
    solver = get_algorithm("vanilla").build(game, seed=0)
    solver.train(20)  # warm the resolved-tree cache before timing anything
    start = time.perf_counter()
    solver.train(100)
    per_iteration = (time.perf_counter() - start) / 100
    strategy = solver.average_strategy()
    start = time.perf_counter()
    exploitability(game, strategy)
    per_measurement = time.perf_counter() - start
    row("one exact Leduc iteration", "11 ms", f"{per_iteration * 1000:.1f} ms")
    row("one Leduc exploitability evaluation", "58 ms", f"{per_measurement * 1000:.0f} ms")
    row(
        "measuring costs, in iterations",
        "5",
        f"{per_measurement / per_iteration:.1f}",
    )

    print("\n=== from the published results ===")
    wallclock = {
        run.algorithm: run
        for run in BenchmarkResults.load(f"{RESULTS}/leduc_wallclock.json").wallclock
    }
    last = len(wallclock["mccfr"].time_checkpoints) - 1
    counts = [curve[last] for curve in wallclock["mccfr"].iterations_by_seed]
    row(
        "wall-clock noise floor (relative sd)",
        "3.1%",
        f"{np.std(counts) / np.mean(counts):.1%}",
    )
    row(
        "iterations in 20s across seeds",
        "129,913-143,891",
        f"{min(counts):,}-{max(counts):,}",
    )
    ratio = wallclock["mccfr"].iterations_per_second() / wallclock["vanilla"].iterations_per_second()
    row("sampled/exact throughput ratio", "83x", f"{ratio:.0f}x")

    print(
        "\nDocumented values above are what docs/ARCHITECTURE.md states. Where a column\n"
        "disagrees, the document is what needs changing -- the disk is the measurement.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
