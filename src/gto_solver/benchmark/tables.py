"""Markdown tables for benchmark results.

The README's numbers are pasted from here, so they are regenerable rather than
transcribed -- the same reason `scripts/verify_phase4.py` exists. If a table in a
document disagrees with what this prints, the document is what is wrong.

Deterministic and stochastic variants get separate tables. Putting them in one would
either drop the seed envelope, which is the whole result for a sampled variant, or
print an empty envelope column for every exact one, which reads as missing data
rather than as zero variance.
"""

from collections.abc import Sequence

from gto_solver.benchmark.runner import ConvergenceRun, WallclockRun
from gto_solver.benchmark.stats import DEFAULT_ENVELOPE

# Below this many decade markers, showing every checkpoint beats showing a stub.
_MINIMUM_TABLE_ROWS = 3


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _rule(count: int, align: str = "---:") -> str:
    return "|" + "|".join([align] * count) + "|"


def _is_decade(value: int) -> bool:
    while value % 10 == 0 and value > 1:
        value //= 10
    return value == 1


def _select_checkpoints(
    checkpoints: Sequence[int], wanted: Sequence[int] | None
) -> tuple[tuple[int, ...], list[str]]:
    """Rows to print, and the note saying which measured points are not shown.

    A table that quietly shows five of thirteen checkpoints looks like the whole
    measurement, so the omission is stated and the reader is pointed at the file
    that has all of it.

    Decade markers are the default because a thirteen-row table nobody reads is
    worse than a five-row one everybody does. But a log grid need not contain many
    powers of ten -- the microstructure suite's eight checkpoints contain exactly
    one -- and a one-row table is not a table, so a selection that sparse is
    discarded in favour of showing everything.
    """
    if wanted is not None:
        missing = [c for c in wanted if c not in checkpoints]
        if missing:
            raise ValueError(f"checkpoints {missing} were not measured; have {list(checkpoints)}")
        rows = tuple(wanted)
    else:
        decades = tuple(c for c in checkpoints if _is_decade(c))
        rows = decades if len(decades) >= _MINIMUM_TABLE_ROWS else tuple(checkpoints)
    notes = []
    if len(rows) < len(checkpoints):
        notes.append(
            f"Table shows {len(rows)} of {len(checkpoints)} measured checkpoints "
            f"(decade markers). Every checkpoint is in the results file."
        )
    return rows, notes


def convergence_markdown(
    runs: Sequence[ConvergenceRun],
    checkpoints: Sequence[int] | None = None,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
) -> tuple[str, list[str]]:
    """Exploitability-vs-iterations tables, exact variants first, plus their notes."""
    if not runs:
        raise ValueError("convergence_markdown needs at least one run")
    reference = runs[0].checkpoints
    for run in runs:
        if run.checkpoints != reference:
            raise ValueError(
                f"runs measured at different checkpoints: {run.algorithm} has "
                f"{run.checkpoints}, expected {reference}"
            )
    rows, notes = _select_checkpoints(reference, checkpoints)
    index_of = {c: i for i, c in enumerate(reference)}

    exact = [r for r in runs if r.deterministic]
    sampled = [r for r in runs if not r.deterministic]
    blocks: list[str] = []

    if exact:
        header = ["Iterations", *(r.label for r in exact)]
        lines = [_row(header), _rule(len(header))]
        for checkpoint in rows:
            i = index_of[checkpoint]
            lines.append(
                _row([f"{checkpoint:,}", *(f"{r.exploitability_by_seed[0][i]:.6f}" for r in exact)])
            )
        blocks.append("\n".join(lines))

    if sampled:
        low, high = envelope
        header = ["Iterations"]
        for run in sampled:
            header += [f"{run.label} median", f"{low:g}-{high:g}% over {len(run.seeds)} seeds"]
        lines = [_row(header), _rule(len(header))]
        aggregates = {r.algorithm: r.aggregates(envelope=envelope) for r in sampled}
        for checkpoint in rows:
            i = index_of[checkpoint]
            cells = [f"{checkpoint:,}"]
            for run in sampled:
                agg = aggregates[run.algorithm][i]
                cells += [f"{agg.median:.6f}", f"{agg.p_low:.6f} - {agg.p_high:.6f}"]
            lines.append(_row(cells))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks), notes


def wallclock_markdown(
    runs: Sequence[WallclockRun],
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
) -> tuple[str, list[str]]:
    """Exploitability at each wall-clock budget, and what each variant got done."""
    if not runs:
        raise ValueError("wallclock_markdown needs at least one run")
    reference = runs[0].time_checkpoints
    for run in runs:
        if run.time_checkpoints != reference:
            raise ValueError(
                f"runs measured at different budgets: {run.algorithm} has "
                f"{run.time_checkpoints}, expected {reference}"
            )

    aggregates = {r.algorithm: r.aggregates(envelope=envelope) for r in runs}
    header = ["Budget", *(r.label for r in runs)]
    lines = [_row(header), _rule(len(header))]
    for i, budget in enumerate(reference):
        cells = [f"{budget:g}s"]
        for run in runs:
            cells.append(f"{aggregates[run.algorithm][i].median:.5f}")
        lines.append(_row(cells))
    exploitability_table = "\n".join(lines)

    header = ["Variant", "Iterations in the final budget", "Iterations/sec", "Seeds"]
    lines = [_row(header), "|---|---:|---:|---:|"]
    for run in runs:
        seeds = "deterministic" if run.deterministic else f"{len(run.seeds)}"
        lines.append(
            _row(
                [
                    run.label,
                    f"{round(run.median_iterations()[-1]):,}",
                    f"{run.iterations_per_second():,.0f}",
                    seeds,
                ]
            )
        )
    throughput_table = "\n".join(lines)

    low, high = envelope
    notes = [
        (
            f"Exploitability cells are medians across seeds; the {low:g}-{high:g}% envelopes "
            f"are in the results file and on the chart."
        ),
        "Budgets are training time only -- evaluating exploitability is not charged to it.",
    ]
    return f"{exploitability_table}\n\n{throughput_table}", notes
