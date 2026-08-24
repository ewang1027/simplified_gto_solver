"""Charts. matplotlib is an optional dependency, so these skip without it.

Two properties are worth pinning beyond "a file appeared": the figures never import
pyplot (which carries global state and picks a GUI backend, the classic way a
headless CI run turns into a hang), and a seed-spread chart refuses to be drawn for
a run that has no seed spread to show.
"""

import sys

import pytest

pytest.importorskip("matplotlib", reason="charts need the optional viz extra")

from gto_solver.benchmark.plots import (
    plot_convergence,
    plot_seed_spread,
    plot_wallclock,
)
from gto_solver.benchmark.runner import run_convergence, run_wallclock
from gto_solver.games.kuhn import KuhnGame
from gto_solver.solvers.registry import get_algorithm

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
VANILLA = get_algorithm("vanilla")
CFR_PLUS = get_algorithm("cfr_plus")
MCCFR = get_algorithm("mccfr")


@pytest.fixture(scope="module")
def convergence_runs():
    checkpoints = (10, 100, 1000)
    return [
        run_convergence(KuhnGame, VANILLA, checkpoints, seeds=(0,)),
        run_convergence(KuhnGame, CFR_PLUS, checkpoints, seeds=(0,)),
        run_convergence(KuhnGame, MCCFR, checkpoints, seeds=(0, 1, 2, 3)),
    ]


@pytest.fixture(scope="module")
def wallclock_runs():
    return [
        run_wallclock(KuhnGame, VANILLA, (0.05, 0.1), seeds=(0,)),
        run_wallclock(KuhnGame, MCCFR, (0.05, 0.1), seeds=(0, 1)),
    ]


def assert_is_png(path):
    assert path.exists()
    assert path.read_bytes()[:8] == PNG_MAGIC
    assert path.stat().st_size > 5_000


def test_convergence_chart_is_written(tmp_path, convergence_runs):
    assert_is_png(plot_convergence(convergence_runs, tmp_path / "convergence.png"))


def test_wallclock_chart_is_written(tmp_path, wallclock_runs):
    assert_is_png(plot_wallclock(wallclock_runs, tmp_path / "wallclock.png"))


def test_seed_spread_chart_is_written(tmp_path, convergence_runs):
    assert_is_png(plot_seed_spread(convergence_runs[2], tmp_path / "seeds.png"))


def test_charts_create_their_directory(tmp_path, convergence_runs):
    assert_is_png(plot_convergence(convergence_runs, tmp_path / "deep" / "dir" / "c.png"))


def test_a_chart_of_only_deterministic_runs_needs_no_band(tmp_path, convergence_runs):
    assert_is_png(plot_convergence(convergence_runs[:2], tmp_path / "exact.png"))


def test_seed_spread_refuses_a_run_with_nothing_to_spread(tmp_path, convergence_runs):
    with pytest.raises(ValueError, match="several seeds"):
        plot_seed_spread(convergence_runs[0], tmp_path / "nope.png")


@pytest.mark.parametrize("plot", [plot_convergence, plot_wallclock])
def test_charts_reject_an_empty_run_list(tmp_path, plot):
    with pytest.raises(ValueError):
        plot([], tmp_path / "nothing.png")


def test_plotting_does_not_pull_in_pyplot(tmp_path, convergence_runs):
    """pyplot picks a backend on import and keeps global figure state; a bare Figure
    does neither, which is what makes these charts safe to draw in CI.
    """
    if "matplotlib.pyplot" in sys.modules:
        pytest.skip("something else in this session already imported pyplot")
    plot_convergence(convergence_runs, tmp_path / "no_pyplot.png")
    assert "matplotlib.pyplot" not in sys.modules


def test_annotation_can_be_turned_off(tmp_path, wallclock_runs):
    assert_is_png(
        plot_wallclock(wallclock_runs, tmp_path / "plain.png", annotate_iterations=False)
    )


def test_a_custom_envelope_is_accepted(tmp_path, convergence_runs):
    assert_is_png(
        plot_convergence(convergence_runs, tmp_path / "iqr.png", envelope=(25.0, 75.0))
    )
