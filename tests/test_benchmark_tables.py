"""Markdown tables. These are what the README's benchmark numbers are pasted from, so
a table that quietly shows a subset of what was measured would put a partial result
into the documentation looking like a complete one.
"""

import pytest

from gto_solver.benchmark.runner import run_convergence, run_wallclock
from gto_solver.benchmark.tables import convergence_markdown, wallclock_markdown
from gto_solver.games.kuhn import KuhnGame
from gto_solver.solvers.registry import get_algorithm

VANILLA = get_algorithm("vanilla")
CFR_PLUS = get_algorithm("cfr_plus")
MCCFR = get_algorithm("mccfr")


@pytest.fixture(scope="module")
def convergence_runs():
    checkpoints = (10, 50, 100, 500, 1000)
    return [
        run_convergence(KuhnGame, VANILLA, checkpoints, seeds=(0,)),
        run_convergence(KuhnGame, CFR_PLUS, checkpoints, seeds=(0,)),
        run_convergence(KuhnGame, MCCFR, checkpoints, seeds=(0, 1, 2)),
    ]


@pytest.fixture(scope="module")
def wallclock_runs():
    return [
        run_wallclock(KuhnGame, VANILLA, (0.05, 0.1), seeds=(0,)),
        run_wallclock(KuhnGame, MCCFR, (0.05, 0.1), seeds=(0, 1)),
    ]


# --- convergence tables ----------------------------------------------------


def test_exact_and_sampled_get_separate_tables(convergence_runs):
    """One table would either drop the sampled variant's envelope -- its whole
    result -- or print an empty envelope column for the exact ones, which reads as
    missing data rather than as zero variance.
    """
    table, _ = convergence_markdown(convergence_runs)
    blocks = table.split("\n\n")
    assert len(blocks) == 2
    assert "Vanilla CFR" in blocks[0] and "CFR+" in blocks[0]
    assert "MCCFR" in blocks[1] and "seeds" in blocks[1]


def test_sampled_table_carries_the_envelope(convergence_runs):
    _, sampled = convergence_markdown(convergence_runs)[0].split("\n\n")
    header = sampled.splitlines()[0]
    assert "median" in header
    assert "10-90% over 3 seeds" in header


def test_decade_rows_are_selected_and_the_omission_is_stated(convergence_runs):
    table, notes = convergence_markdown(convergence_runs)
    assert "| 10 |" in table
    assert "| 100 |" in table
    assert "| 1,000 |" in table
    assert "| 50 |" not in table
    assert any("3 of 5 measured checkpoints" in note for note in notes)


def test_showing_every_checkpoint_needs_no_note(convergence_runs):
    _, notes = convergence_markdown(convergence_runs, checkpoints=(10, 50, 100, 500, 1000))
    assert notes == []


def test_requesting_an_unmeasured_checkpoint_is_an_error(convergence_runs):
    with pytest.raises(ValueError, match="were not measured"):
        convergence_markdown(convergence_runs, checkpoints=(10, 42))


def test_runs_measured_at_different_checkpoints_cannot_share_a_table():
    """Rows are checkpoints, so mixing grids would silently align numbers that were
    measured at different iteration counts.
    """
    runs = [
        run_convergence(KuhnGame, VANILLA, (10, 100), seeds=(0,)),
        run_convergence(KuhnGame, CFR_PLUS, (10, 200), seeds=(0,)),
    ]
    with pytest.raises(ValueError, match="different checkpoints"):
        convergence_markdown(runs)


def test_convergence_markdown_needs_a_run():
    with pytest.raises(ValueError):
        convergence_markdown([])


def test_only_sampled_runs_still_produce_a_table(convergence_runs):
    table, _ = convergence_markdown([convergence_runs[2]])
    assert "MCCFR" in table
    assert "\n\n" not in table


# --- wall-clock tables -----------------------------------------------------


def test_wallclock_emits_an_exploitability_table_and_a_throughput_table(wallclock_runs):
    table, notes = wallclock_markdown(wallclock_runs)
    exploitability, throughput = table.split("\n\n")
    assert "| 0.05s |" in exploitability and "| 0.1s |" in exploitability
    assert "Iterations/sec" in throughput
    assert "deterministic" in throughput
    assert any("training time only" in note for note in notes)


def test_wallclock_rejects_mismatched_budgets():
    runs = [
        run_wallclock(KuhnGame, VANILLA, (0.05,), seeds=(0,)),
        run_wallclock(KuhnGame, MCCFR, (0.1,), seeds=(0,)),
    ]
    with pytest.raises(ValueError, match="different budgets"):
        wallclock_markdown(runs)


def test_wallclock_markdown_needs_a_run():
    with pytest.raises(ValueError):
        wallclock_markdown([])


# --- shape -----------------------------------------------------------------


def test_every_row_has_the_same_column_count(convergence_runs, wallclock_runs):
    for table, _ in (
        convergence_markdown(convergence_runs),
        wallclock_markdown(wallclock_runs),
    ):
        for block in table.split("\n\n"):
            widths = {line.count("|") for line in block.splitlines()}
            assert len(widths) == 1, block
