"""The `gto` command line.

Two things are worth testing beyond "it runs". The CLI reads its games, algorithms
and suites from the registries, so `gto algorithms` cannot drift out of date -- and a
test that checks the listing against the registry is what keeps that true if someone
later hard-codes a list here.

And parameters that do not apply must be refused rather than dropped. `--mu 0.7
--game kuhn` is a caller who believes something false about the run; reporting a
spread for a mu the run never used would be worse than an error.
"""

import json

import pytest
from typer.testing import CliRunner

from gto_solver.benchmark.suites import SUITES
from gto_solver.cli import app
from gto_solver.games.registry import GAMES
from gto_solver.solvers.registry import ALGORITHMS

runner = CliRunner()


def run(*args: str):
    return runner.invoke(app, list(args))


# --- listings come from the registries -------------------------------------


def test_algorithms_lists_every_registered_variant():
    result = run("algorithms")
    assert result.exit_code == 0
    for name in ALGORITHMS:
        assert name in result.stdout


def test_algorithms_says_which_variants_are_sampled():
    result = run("algorithms")
    for spec in ALGORITHMS.values():
        line = next(ln for ln in result.stdout.splitlines() if ln.startswith(spec.name))
        assert ("deterministic" in line) == spec.deterministic, spec.name


def test_games_lists_every_registered_game_and_its_parameters():
    result = run("games")
    assert result.exit_code == 0
    for spec in GAMES.values():
        assert spec.name in result.stdout
    assert "mu" in result.stdout  # the only parameterized game


def test_benchmark_list_names_every_suite():
    result = run("benchmark", "--list")
    assert result.exit_code == 0
    for name in SUITES:
        assert name in result.stdout


# --- solve -----------------------------------------------------------------


def test_solve_reports_convergence_and_the_game_value():
    result = run("solve", "--game", "kuhn", "--iterations", "500", "--no-show-strategy")
    assert result.exit_code == 0
    assert "Exploitability" in result.stdout
    assert "Info sets discovered: 12" in result.stdout
    assert "Value to player 0" in result.stdout


def test_solve_drives_exploitability_down():
    """The CLI is not allowed to report a number the solver did not earn."""
    payload = json.loads(
        run("solve", "--game", "kuhn", "--iterations", "2000", "--json").stdout
    )
    curve = [point["exploitability"] for point in payload["convergence"]]
    assert curve[-1] < curve[0] / 5
    assert payload["exploitability"] == curve[-1]


def test_solve_reports_the_average_strategy_not_the_current_one():
    """Kuhn's equilibrium value is -1/18. The current regret-matching strategy
    oscillates and does not settle there; the average does, so landing near it is
    evidence the CLI evaluated the right object.
    """
    payload = json.loads(
        run("solve", "--game", "kuhn", "--iterations", "5000", "--json").stdout
    )
    assert payload["value_to_player_0"] == pytest.approx(-1 / 18, abs=0.01)


def test_solve_emits_valid_json_with_a_probability_distribution_per_info_set():
    payload = json.loads(run("solve", "--game", "kuhn", "--iterations", "200", "--json").stdout)
    assert payload["game"] == "kuhn_poker"
    assert len(payload["strategy"]) == payload["info_sets"] == 12
    for key, probs in payload["strategy"].items():
        assert sum(probs) == pytest.approx(1.0), key


def test_solve_accepts_a_game_parameter_that_applies():
    payload = json.loads(
        run("solve", "--game", "gm", "--mu", "0.7", "--iterations", "20", "--json").stdout
    )
    assert payload["parameters"] == {"mu": 0.7}
    assert payload["game"] == "glosten_milgrom"


def test_solve_refuses_a_parameter_the_game_does_not_take():
    result = run("solve", "--game", "kuhn", "--mu", "0.7", "--iterations", "10")
    assert result.exit_code == 2
    assert "does not take mu" in result.stderr


def test_solve_reports_the_seed_only_for_sampled_variants():
    """A seed printed beside a deterministic run reads as if it mattered."""
    sampled = json.loads(
        run("solve", "--algorithm", "mccfr", "--seed", "5", "--iterations", "50", "--json").stdout
    )
    exact = json.loads(
        run("solve", "--algorithm", "vanilla", "--seed", "5", "--iterations", "50", "--json").stdout
    )
    assert sampled["seed"] == 5
    assert exact["seed"] is None


def test_solve_truncates_a_long_strategy_listing_and_says_so():
    result = run("solve", "--game", "leduc", "--iterations", "5", "--max-info-sets", "4")
    assert result.exit_code == 0
    assert "more info sets" in result.stdout


@pytest.mark.parametrize(
    "args, fragment",
    [
        (("solve", "--game", "holdem"), "unknown game"),
        (("solve", "--algorithm", "cfr_ultra"), "unknown algorithm"),
        (("benchmark", "--suite", "nope"), "unknown suite"),
    ],
)
def test_unknown_names_fail_with_a_message_listing_the_known_ones(args, fragment):
    result = run(*args)
    assert result.exit_code == 2
    assert fragment in result.stderr


def test_an_unknown_name_error_is_not_wrapped_in_reprs_quotes():
    """`str(KeyError(...))` is the repr of its argument, so a message containing an
    apostrophe comes back double-quoted. Users should not have to read around that.
    """
    assert not run("solve", "--game", "holdem").stderr.strip().startswith('"')


# --- benchmark -------------------------------------------------------------


def test_benchmark_runs_a_quick_suite_and_labels_it_as_not_publishable(tmp_path):
    result = run(
        "benchmark",
        "--quick",
        "--suite",
        "kuhn_convergence",
        "--no-plots",
        "--results-dir",
        str(tmp_path),
        "--image-dir",
        str(tmp_path),
    )
    assert result.exit_code == 0
    assert (tmp_path / "kuhn_convergence.json").exists()
    assert "QUICK PROFILE" in result.stdout
    assert "not the published numbers" in result.stdout


def test_benchmark_compare_of_a_file_with_itself_passes(tmp_path):
    run(
        "benchmark", "--quick", "--suite", "kuhn_convergence", "--no-plots",
        "--results-dir", str(tmp_path), "--image-dir", str(tmp_path),
    )
    path = str(tmp_path / "kuhn_convergence.json")
    result = run("benchmark", "--compare", path, path)
    assert result.exit_code == 0
    assert "unchanged" in result.stdout
