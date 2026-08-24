"""Serialization, provenance, and the before/after comparison Phase 6 is built on.

The comparison makes a distinction that matters: a convergence curve is indexed by
iteration and must not move when the code is optimized, while a wall-clock curve is
indexed by seconds and is *supposed* to move. Checking both for equality would flag
every successful optimization as a regression; checking neither would let a changed
algorithm pass as a speedup.
"""

import dataclasses
import json

import pytest

from gto_solver.benchmark.results import (
    SCHEMA_VERSION,
    BenchmarkResults,
    compare,
    provenance,
)
from gto_solver.benchmark.runner import run_convergence, run_wallclock
from gto_solver.games.kuhn import KuhnGame
from gto_solver.solvers.registry import get_algorithm

VANILLA = get_algorithm("vanilla")
MCCFR = get_algorithm("mccfr")


@pytest.fixture(scope="module")
def results() -> BenchmarkResults:
    return BenchmarkResults(
        suite="test_suite",
        title="A tiny suite",
        provenance=provenance(),
        convergence=(
            run_convergence(KuhnGame, VANILLA, (10, 100), seeds=(0,)),
            run_convergence(KuhnGame, MCCFR, (10, 100), seeds=(0, 1)),
        ),
        wallclock=(run_wallclock(KuhnGame, MCCFR, (0.05,), seeds=(0, 1)),),
        notes=("a suite-level note",),
    )


# --- provenance ------------------------------------------------------------


def test_provenance_records_what_makes_two_files_comparable():
    got = provenance()
    for key in ("created_utc", "python", "numpy", "platform", "machine", "git"):
        assert key in got, key
    assert set(got["git"]) == {"commit", "dirty", "dirty_paths", "detail"}


def test_provenance_is_json_serializable():
    json.dumps(provenance())


# --- serialization ---------------------------------------------------------


def test_results_round_trip_through_json(tmp_path, results):
    path = results.save(tmp_path / "nested" / "results.json")
    assert path.exists()
    assert BenchmarkResults.load(path) == results


def test_saving_creates_the_directory(tmp_path, results):
    results.save(tmp_path / "a" / "b" / "c.json")
    assert (tmp_path / "a" / "b" / "c.json").exists()


def test_loading_a_different_schema_version_is_refused(tmp_path, results):
    """A silently-reinterpreted old file is worse than a missing one, because its
    numbers still look plausible.
    """
    data = results.to_dict()
    data["schema_version"] = SCHEMA_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema version"):
        BenchmarkResults.load(path)


def test_all_notes_surfaces_notes_from_inside_the_runs(results):
    collected = results.all_notes()
    assert "a suite-level note" in collected
    assert any("1 seed instead of" not in note for note in collected)
    assert len(collected) >= len(results.notes)


# --- compare ---------------------------------------------------------------


def test_comparing_a_file_with_itself_finds_no_change(results):
    report = compare(results, results)
    convergence = [c for c in report.comparisons if c.kind == "convergence"]
    assert len(convergence) == 2
    assert all(c.curves_match for c in convergence)
    assert all(c.speedup == pytest.approx(1.0) for c in convergence)
    assert report.curves_unchanged


def test_a_moved_convergence_curve_is_caught(results):
    """The check an optimization has to pass: same iterations, same numbers."""
    original = results.convergence[0]
    tampered = dataclasses.replace(
        original,
        exploitability_by_seed=((original.exploitability_by_seed[0][0] + 1e-6, 0.5),),
    )
    candidate = dataclasses.replace(results, convergence=(tampered, results.convergence[1]))

    report = compare(results, candidate)
    changed = [c for c in report.comparisons if c.algorithm == "vanilla"]
    assert changed[0].curves_match is False
    assert changed[0].max_exploitability_delta > 1e-9
    assert not report.curves_unchanged


def test_a_faster_run_reports_a_speedup(results):
    original = results.convergence[0]
    faster = dataclasses.replace(
        original,
        train_seconds_by_seed=tuple(
            tuple(second / 2 for second in curve) for curve in original.train_seconds_by_seed
        ),
    )
    candidate = dataclasses.replace(results, convergence=(faster, results.convergence[1]))

    report = compare(results, candidate)
    speedup = next(c.speedup for c in report.comparisons if c.algorithm == "vanilla")
    assert speedup == pytest.approx(2.0)


def test_wallclock_curves_are_compared_on_throughput_not_equality(results):
    """A wall-clock curve is meant to move when the code gets faster."""
    comparison = next(c for c in compare(results, results).comparisons if c.kind == "wallclock")
    assert comparison.curves_match is None
    assert comparison.max_exploitability_delta is None
    assert comparison.speedup == pytest.approx(1.0)


def test_a_run_present_in_only_one_file_is_reported_not_dropped(results):
    """The failure this harness exists to prevent: "no regressions" over a quietly
    smaller set of runs.
    """
    trimmed = dataclasses.replace(results, convergence=results.convergence[:1])
    report = compare(results, trimmed)
    assert any("baseline but not the candidate" in note for note in report.notes)
    assert len(report.comparisons) == 2  # one convergence run plus the wall-clock one


def test_mismatched_checkpoints_are_not_compared_and_say_so(results):
    original = results.convergence[0]
    shifted = dataclasses.replace(
        original,
        checkpoints=(10, 200),
    )
    candidate = dataclasses.replace(results, convergence=(shifted, results.convergence[1]))

    comparison = next(
        c for c in compare(results, candidate).comparisons if c.algorithm == "vanilla"
    )
    assert comparison.curves_match is None
    assert any("checkpoints differ" in note for note in comparison.notes)


def test_mismatched_seeds_fall_back_to_medians_and_say_so(results):
    original = results.convergence[1]
    fewer = dataclasses.replace(
        original,
        seeds=(0,),
        exploitability_by_seed=original.exploitability_by_seed[:1],
        train_seconds_by_seed=original.train_seconds_by_seed[:1],
    )
    candidate = dataclasses.replace(results, convergence=(results.convergence[0], fewer))

    comparison = next(c for c in compare(results, candidate).comparisons if c.algorithm == "mccfr")
    assert any("only the median curves" in note for note in comparison.notes)


def test_a_different_machine_invalidates_the_speed_comparison(results):
    other = dataclasses.replace(
        results, provenance={**results.provenance, "machine": "x86_64", "python": "3.11.9"}
    )
    report = compare(results, other)
    assert any("machine" in note for note in report.notes)
    assert any("python" in note for note in report.notes)


def test_a_dirty_tree_is_flagged(results):
    dirty = dataclasses.replace(
        results,
        provenance={
            **results.provenance,
            "git": {
                "commit": "abc1234",
                "dirty": True,
                "dirty_paths": ["src/gto_solver/solvers/traversal.py"],
                "detail": None,
            },
        },
    )
    report = compare(dirty, dirty)
    assert sum("dirty working tree" in note for note in report.notes) == 2
    assert all("traversal.py" in note for note in report.notes)


def test_tolerance_absorbs_floating_point_reassociation(results):
    """A vectorized rewrite can legitimately land a few ulps away; a changed
    algorithm cannot.
    """
    original = results.convergence[0]
    nudged = dataclasses.replace(
        original,
        exploitability_by_seed=tuple(
            tuple(value + 1e-12 for value in curve) for curve in original.exploitability_by_seed
        ),
    )
    candidate = dataclasses.replace(results, convergence=(nudged, results.convergence[1]))
    comparison = next(
        c for c in compare(results, candidate).comparisons if c.algorithm == "vanilla"
    )
    assert comparison.curves_match is True


def test_format_table_names_every_comparison(results):
    table = compare(results, results).format_table()
    for line in ("vanilla", "mccfr", "convergence", "wallclock", "speedup"):
        assert line in table
