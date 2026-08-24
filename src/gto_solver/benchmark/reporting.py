"""Running suites and printing what happened, shared by every entry point.

This lived inside `scripts/benchmark.py` until the CLI needed the same thing. Two
copies of "run a suite, save it, draw it, print the tables and then the notes" would
drift, and the half that drifted would be the notes -- which is precisely the half
whose job is to stop a reduced or capped run from reading like a complete one.

Charts are optional here in a specific sense: matplotlib missing is a *note*, not a
failure. The JSON is the result and the PNG is a rendering of it, so refusing to
record measured numbers because a plotting library is absent would be a bad trade.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from gto_solver.benchmark.results import BenchmarkResults, compare
from gto_solver.benchmark.suites import Suite, quick, run_suite
from gto_solver.benchmark.tables import convergence_markdown, wallclock_markdown

Printer = Callable[[str], None]


def rule(title: str, out: Printer = print) -> None:
    out(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def draw_charts(results: BenchmarkResults, suite: Suite, image_dir: Path, out: Printer = print):
    """Charts for one suite. Returns notes; a missing matplotlib is one of them."""
    try:
        from gto_solver.benchmark.plots import plot_convergence, plot_seed_spread, plot_wallclock
    except ImportError as exc:
        return [f"charts skipped: {exc}"]

    written: list[Path] = []
    if results.convergence:
        written.append(
            plot_convergence(
                results.convergence,
                image_dir / f"{suite.name}.png",
                title=suite.title,
                subtitle=suite.subtitle,
            )
        )
    if results.wallclock:
        written.append(
            plot_wallclock(
                results.wallclock,
                image_dir / f"{suite.name}.png",
                title=suite.title,
                subtitle=suite.subtitle,
            )
        )
    if suite.spread_algorithm:
        matches = [r for r in results.convergence if r.algorithm == suite.spread_algorithm]
        if matches and len(matches[0].seeds) > 1:
            written.append(plot_seed_spread(matches[0], image_dir / f"{suite.name}_seeds.png"))
    for path in written:
        out(f"  wrote {path}")
    return []


def print_report(
    results: BenchmarkResults, extra_notes: Sequence[str] = (), out: Printer = print
) -> None:
    """The markdown tables, and then -- loudly -- everything that was not done."""
    notes = list(results.all_notes()) + list(extra_notes)
    if results.convergence:
        table, table_notes = convergence_markdown(results.convergence)
        notes += table_notes
        out(f"\n{table}")
    if results.wallclock:
        table, table_notes = wallclock_markdown(results.wallclock)
        notes += table_notes
        out(f"\n{table}")

    out("\nNOTES")
    if not notes:
        out("  (none -- nothing was capped, reduced or skipped)")
    for note in notes:
        out(f"  * {note}")


def run_suites(
    suites: Sequence[Suite],
    is_quick: bool,
    results_dir: Path,
    image_dir: Path,
    plots: bool,
    out: Printer = print,
) -> list[BenchmarkResults]:
    """Run each suite, save it, chart it, and report it."""
    collected: list[BenchmarkResults] = []
    for suite in suites:
        extra_notes: list[str] = []
        if is_quick:
            suite, extra_notes = quick(suite)

        rule(f"{suite.name}: {suite.title}", out)
        for note in extra_notes:
            out(f"  !! {note}")
        out(
            f"  {len(suite.algorithms)} algorithms, up to {len(suite.seeds)} seeds, "
            f"checkpoints {suite.checkpoints[0]:g}..{suite.checkpoints[-1]:g}\n"
        )

        results = run_suite(suite, extra_notes=extra_notes, progress=out)
        out(f"\n  wrote {results.save(results_dir / f'{suite.name}.json')}")
        drawing_notes = (
            draw_charts(results, suite, image_dir, out) if plots else ["charts skipped by request"]
        )
        print_report(results, drawing_notes, out)
        collected.append(results)
    return collected


def print_comparison(baseline: str | Path, candidate: str | Path, out: Printer = print) -> bool:
    """Print a before/after report. Returns True when every comparable curve held."""
    rule(f"compare: {baseline} -> {candidate}", out)
    report = compare(BenchmarkResults.load(baseline), BenchmarkResults.load(candidate))
    out(report.format_table())
    out("\nNOTES")
    notes = list(report.notes) + [note for c in report.comparisons for note in c.notes]
    if not notes:
        out("  (none)")
    for note in notes:
        out(f"  * {note}")

    if report.curves_unchanged:
        out("\nEvery comparable convergence curve is unchanged.")
        return True
    out("\nA convergence curve CHANGED. Per-iteration behaviour is not supposed to move.")
    return False
