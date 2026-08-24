"""Run the published benchmark suites, write their results, and draw their charts.

    python scripts/benchmark.py                      # the default suites, full profile
    python scripts/benchmark.py --quick              # seconds instead of minutes
    python scripts/benchmark.py --suite gm_convergence
    python scripts/benchmark.py --list
    python scripts/benchmark.py --compare results/a.json results/b.json

Results land in `results/<suite>.json` and charts in `docs/images/<suite>.png`. The
markdown tables printed at the end are what the README's benchmark tables are pasted
from, so they can be regenerated rather than trusted.

`--compare` is the Phase 6 workflow: measure, optimize, measure again, and check that
throughput moved while the per-iteration convergence curves did not. It exits non-zero
if any curve changed, so it can gate an optimization.

Every suite prints a NOTES block. Anything the harness declined to do -- a reduced
seed count, a quick profile, a wall-clock checkpoint it overshot -- appears there
rather than being absorbed silently into the numbers.
"""

import argparse
import sys
from pathlib import Path

from gto_solver.benchmark.results import BenchmarkResults, compare
from gto_solver.benchmark.suites import DEFAULT_SUITES, SUITES, get_suite, quick, run_suite
from gto_solver.benchmark.tables import convergence_markdown, wallclock_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_IMAGE_DIR = REPO_ROOT / "docs" / "images"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def draw(results: BenchmarkResults, suite, image_dir: Path) -> list[str]:
    """Charts for one suite. Returns notes; missing matplotlib is a note, not a crash.

    A benchmark that refuses to record its numbers because a plotting library is
    absent would be a bad trade: the JSON is the result, the PNG is a rendering of it.
    """
    try:
        from gto_solver.benchmark.plots import plot_convergence, plot_seed_spread, plot_wallclock
    except ImportError as exc:
        return [f"charts skipped: {exc}"]

    written: list[str] = []
    if results.convergence:
        written.append(
            str(
                plot_convergence(
                    results.convergence,
                    image_dir / f"{suite.name}.png",
                    title=suite.title,
                    subtitle=suite.subtitle,
                )
            )
        )
    if results.wallclock:
        written.append(
            str(
                plot_wallclock(
                    results.wallclock,
                    image_dir / f"{suite.name}.png",
                    title=suite.title,
                    subtitle=suite.subtitle,
                )
            )
        )
    if suite.spread_algorithm:
        matches = [r for r in results.convergence if r.algorithm == suite.spread_algorithm]
        if matches and len(matches[0].seeds) > 1:
            written.append(
                str(plot_seed_spread(matches[0], image_dir / f"{suite.name}_seeds.png"))
            )
    for path in written:
        print(f"  wrote {path}")
    return []


def report(results: BenchmarkResults, extra_notes: list[str] | None = None) -> None:
    """Print the markdown tables and then, loudly, everything that was not done."""
    notes: list[str] = list(results.all_notes()) + list(extra_notes or [])
    if results.convergence:
        table, table_notes = convergence_markdown(results.convergence)
        notes += table_notes
        print(f"\n{table}")
    if results.wallclock:
        table, table_notes = wallclock_markdown(results.wallclock)
        notes += table_notes
        print(f"\n{table}")

    print("\nNOTES")
    if not notes:
        print("  (none -- nothing was capped, reduced or skipped)")
    for note in notes:
        print(f"  * {note}")


def run(names: list[str], is_quick: bool, results_dir: Path, image_dir: Path, plots: bool) -> int:
    for name in names:
        suite = get_suite(name)
        extra_notes: list[str] = []
        if is_quick:
            suite, extra_notes = quick(suite)

        rule(f"{suite.name}: {suite.title}")
        if extra_notes:
            for note in extra_notes:
                print(f"  !! {note}")
        print(
            f"  {len(suite.algorithms)} algorithms, up to {len(suite.seeds)} seeds, "
            f"checkpoints {suite.checkpoints[0]:g}..{suite.checkpoints[-1]:g}\n"
        )

        results = run_suite(suite, extra_notes=extra_notes, progress=print)
        path = results.save(results_dir / f"{suite.name}.json")
        print(f"\n  wrote {path}")
        drawing_notes = draw(results, suite, image_dir) if plots else ["charts skipped: --no-plots"]
        report(results, drawing_notes)
    return 0


def run_comparison(baseline_path: str, candidate_path: str) -> int:
    rule(f"compare: {baseline_path} -> {candidate_path}")
    report_ = compare(BenchmarkResults.load(baseline_path), BenchmarkResults.load(candidate_path))
    print(report_.format_table())
    print("\nNOTES")
    notes = list(report_.notes) + [n for c in report_.comparisons for n in c.notes]
    if not notes:
        print("  (none)")
    for note in notes:
        print(f"  * {note}")

    if report_.curves_unchanged:
        print("\nEvery comparable convergence curve is unchanged.")
        return 0
    print("\nA convergence curve CHANGED. Per-iteration behaviour is not supposed to move.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        choices=sorted(SUITES),
        help="suite to run; repeatable. Default: " + ", ".join(DEFAULT_SUITES),
    )
    parser.add_argument("--list", action="store_true", help="list the suites and exit")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="cheap smoke profile -- fewer seeds and checkpoints, NOT the published numbers",
    )
    parser.add_argument("--no-plots", action="store_true", help="write results but draw no charts")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="compare two results files instead of running anything",
    )
    args = parser.parse_args()

    if args.list:
        for name, suite in SUITES.items():
            default = " (default)" if name in DEFAULT_SUITES else ""
            print(f"{name}{default}\n    {suite.title}")
            print(
                f"    {suite.kind}, {len(suite.algorithms)} algorithms, {len(suite.seeds)} seeds, "
                f"{len(suite.checkpoints)} checkpoints"
            )
        return 0

    if args.compare:
        return run_comparison(*args.compare)

    return run(
        args.suites or list(DEFAULT_SUITES),
        args.quick,
        args.results_dir,
        args.image_dir,
        not args.no_plots,
    )


if __name__ == "__main__":
    sys.exit(main())
