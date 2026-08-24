"""The measurement path: what gets timed, what gets skipped, and what gets said.

The rules under test are the ones that make two benchmark runs comparable across a
refactor, so they are checked as behaviour rather than trusted as documentation:
exploitability evaluation stays off the solver's clock, a deterministic variant is
run once, and every reduction the harness makes announces itself in `notes`.
"""

from itertools import pairwise

import pytest

from gto_solver.benchmark.runner import (
    ConvergenceRun,
    WallclockRun,
    run_convergence,
    run_wallclock,
    verify_determinism,
)
from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame
from gto_solver.solvers.registry import get_algorithm

VANILLA = get_algorithm("vanilla")
MCCFR = get_algorithm("mccfr")
CHECKPOINTS = (10, 100, 1000)


# --- run_convergence -------------------------------------------------------


def test_convergence_shape_matches_seeds_and_checkpoints():
    run = run_convergence(KuhnGame, MCCFR, CHECKPOINTS, seeds=(0, 1, 2))
    assert run.seeds == (0, 1, 2)
    assert run.checkpoints == CHECKPOINTS
    assert len(run.exploitability_by_seed) == 3
    assert all(len(curve) == len(CHECKPOINTS) for curve in run.exploitability_by_seed)
    assert len(run.train_seconds_by_seed) == 3
    assert run.info_sets == 12


def test_convergence_drives_exploitability_down():
    run = run_convergence(KuhnGame, VANILLA, (10, 1000), seeds=(0,))
    first, last = run.exploitability_by_seed[0]
    assert last < first / 5


def test_reported_time_is_cumulative_and_ascending():
    run = run_convergence(KuhnGame, VANILLA, CHECKPOINTS, seeds=(0,))
    seconds = run.train_seconds_by_seed[0]
    assert all(a <= b for a, b in pairwise(seconds))
    assert seconds[0] > 0.0


def test_exploitability_evaluation_is_not_charged_to_the_solver():
    """The rule that keeps the exact-vs-sampled comparison honest.

    Measuring Leduc exploitability costs more than a Leduc iteration, so if it were
    timed, a run measured at more checkpoints would report a lower iterations/sec for
    identical work. Same solver, same total iterations, different number of
    measurements: the throughput has to agree.
    """
    sparse = run_convergence(LeducGame, VANILLA, (20,), seeds=(0,))
    dense = run_convergence(LeducGame, VANILLA, (5, 10, 15, 20), seeds=(0,))
    ratio = dense.iterations_per_second() / sparse.iterations_per_second()
    assert 0.75 < ratio < 1.33, (sparse.iterations_per_second(), dense.iterations_per_second())


def test_deterministic_variant_runs_one_seed_and_says_so():
    run = run_convergence(KuhnGame, VANILLA, (10,), seeds=range(20))
    assert run.seeds == (0,)
    assert run.requested_seeds == 20
    assert len(run.notes) == 1
    assert "1 seed instead of 20" in run.notes[0]


def test_stochastic_variant_keeps_every_seed_and_stays_quiet():
    run = run_convergence(KuhnGame, MCCFR, (10,), seeds=range(4))
    assert run.seeds == (0, 1, 2, 3)
    assert run.requested_seeds == 4
    assert run.notes == ()


def test_seeds_actually_change_a_stochastic_run():
    run = run_convergence(KuhnGame, MCCFR, (200,), seeds=(0, 1, 2))
    finals = {curve[-1] for curve in run.exploitability_by_seed}
    assert len(finals) == 3


def test_median_curve_and_values_at_agree_with_the_raw_data():
    run = run_convergence(KuhnGame, MCCFR, (10, 100), seeds=(0, 1, 2))
    assert run.values_at(0) == tuple(curve[0] for curve in run.exploitability_by_seed)
    assert len(run.median_curve()) == 2
    assert min(run.values_at(1)) <= run.median_curve()[1] <= max(run.values_at(1))


def test_aggregates_line_up_with_the_checkpoints():
    run = run_convergence(KuhnGame, MCCFR, CHECKPOINTS, seeds=(0, 1, 2))
    aggregates = run.aggregates()
    assert len(aggregates) == len(CHECKPOINTS)
    assert all(agg.n == 3 for agg in aggregates)
    assert aggregates[0].median == pytest.approx(run.median_curve()[0])


@pytest.mark.parametrize("checkpoints", [(), (100, 10), (0, 10), (-5,)])
def test_convergence_rejects_bad_checkpoints(checkpoints):
    with pytest.raises(ValueError):
        run_convergence(KuhnGame, VANILLA, checkpoints, seeds=(0,))


def test_convergence_rejects_an_empty_seed_list():
    with pytest.raises(ValueError):
        run_convergence(KuhnGame, VANILLA, (10,), seeds=())


def test_progress_callback_fires_once_per_seed():
    lines: list[str] = []
    run_convergence(KuhnGame, MCCFR, (10,), seeds=(0, 1, 2), progress=lines.append)
    assert len(lines) == 3
    assert all("mccfr" in line for line in lines)


def test_game_label_defaults_to_the_game_name_and_can_be_overridden():
    assert run_convergence(KuhnGame, VANILLA, (10,)).game == "kuhn_poker"
    assert run_convergence(KuhnGame, VANILLA, (10,), game_label="toy").game == "toy"


# --- run_wallclock ---------------------------------------------------------


def test_wallclock_reaches_each_budget_with_more_iterations_each_time():
    run = run_wallclock(KuhnGame, MCCFR, (0.05, 0.15), seeds=(0, 1))
    for seconds, iterations in zip(run.train_seconds_by_seed, run.iterations_by_seed):
        assert seconds[0] >= 0.05
        assert seconds[1] >= 0.15
        assert iterations[1] > iterations[0]


def test_wallclock_measures_more_sampled_iterations_than_exact_ones():
    """The result the Leduc suite exists to quantify, in miniature: on the same
    budget, sampling completes far more iterations than exact traversal.
    """
    exact = run_wallclock(LeducGame, VANILLA, (0.3,), seeds=(0,))
    sampled = run_wallclock(LeducGame, MCCFR, (0.3,), seeds=(0,))
    assert sampled.iterations_per_second() > 10 * exact.iterations_per_second()


def test_wallclock_notes_a_checkpoint_it_had_to_overshoot():
    """One Leduc iteration is tens of milliseconds, so a 1 ms budget cannot be hit.

    The run still happens and is still reported -- at the time it actually took, with
    a note. Silently plotting it at the requested budget would put a point on the
    chart where no measurement was made.
    """
    run = run_wallclock(LeducGame, VANILLA, (0.001,), seeds=(0,))
    assert run.train_seconds_by_seed[0][0] > 0.001
    assert any("overshoot" in note or "coarse" in note for note in run.notes)


def test_wallclock_deterministic_variant_runs_one_seed_and_says_so():
    run = run_wallclock(KuhnGame, VANILLA, (0.05,), seeds=range(10))
    assert run.seeds == (0,)
    assert any("1 seed instead of 10" in note for note in run.notes)


def test_wallclock_medians_have_one_entry_per_budget():
    run = run_wallclock(KuhnGame, MCCFR, (0.05, 0.1), seeds=(0, 1, 2))
    assert len(run.median_seconds()) == 2
    assert len(run.median_iterations()) == 2
    assert len(run.aggregates()) == 2


@pytest.mark.parametrize("budgets", [(), (1.0, 0.5), (0.0,), (-1.0,)])
def test_wallclock_rejects_bad_budgets(budgets):
    with pytest.raises(ValueError):
        run_wallclock(KuhnGame, VANILLA, budgets, seeds=(0,))


# --- verify_determinism ----------------------------------------------------


def test_verify_determinism_confirms_an_exact_variant():
    report = verify_determinism(KuhnGame, VANILLA, iterations=50, seeds=(0, 1, 2))
    assert report.observed_identical
    assert report.differing_info_sets == ()
    assert report.claim_holds


def test_verify_determinism_catches_a_sampled_variant():
    report = verify_determinism(KuhnGame, MCCFR, iterations=50, seeds=(0, 1))
    assert not report.observed_identical
    assert report.differing_info_sets
    assert report.claim_holds


def test_verify_determinism_needs_two_seeds_to_compare():
    with pytest.raises(ValueError):
        verify_determinism(KuhnGame, VANILLA, seeds=(0,))


# --- serialization round trip ----------------------------------------------


def test_convergence_run_round_trips_through_a_dict():
    run = run_convergence(KuhnGame, MCCFR, (10, 100), seeds=(0, 1))
    assert ConvergenceRun.from_dict(run.to_dict()) == run


def test_wallclock_run_round_trips_through_a_dict():
    run = run_wallclock(KuhnGame, MCCFR, (0.05,), seeds=(0, 1))
    assert WallclockRun.from_dict(run.to_dict()) == run
