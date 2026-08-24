"""Serializing benchmark results, and comparing two of them.

A result file carries its own provenance -- interpreter, numpy, machine, package
version, git commit and whether the tree was dirty -- because a timing number is
meaningless without the machine it was measured on, and a "reproduce this" that
points at a commit is worthless if the tree had uncommitted edits at the time.

`compare()` is what Phase 6 is built against. An optimization claim needs before and
after from the same measurement path, and it needs two *different* checks, because
the two kinds of run answer different questions:

* A **convergence** run is indexed by iteration count, so its curve is pure math. A
  faithful optimization must leave it unchanged -- if exploitability at 10,000
  iterations moved, the optimization changed the algorithm, whatever the clock says.
* A **wall-clock** run is indexed by seconds, so its curve is *supposed* to move: the
  whole point is more iterations per second. Comparing it for equality would flag
  every successful optimization as a regression, so it is compared on throughput.
"""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gto_solver.benchmark.runner import ConvergenceRun, WallclockRun

SCHEMA_VERSION = 1

# Absolute exploitability difference tolerated between two convergence curves that
# are supposed to be identical. Not zero: a refactor may legitimately reassociate
# floating-point sums (e.g. a vectorized regret update) and land a few ulps away.
DEFAULT_CURVE_TOLERANCE = 1e-9


def _git_description(repo: Path) -> dict:
    """Commit and dirty-flag for `repo`, or a reason it could not be determined.

    Tolerant on purpose: the package can be installed somewhere with no git at all,
    and a benchmark should still run and still say what it does not know.
    """

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                args, cwd=repo, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("git", "rev-parse", "--short", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "detail": "not a git checkout, or git unavailable"}
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "detail": None if not status else "tree had uncommitted changes when this was measured",
    }


def provenance() -> dict:
    """Everything needed to judge whether two result files are comparable."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            package_version = version("gto_solver")
        except PackageNotFoundError:
            package_version = None
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.10+
        package_version = None

    repo = Path(__file__).resolve().parents[3]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": package_version,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "git": _git_description(repo),
    }


@dataclass(frozen=True)
class BenchmarkResults:
    """One suite's runs, its provenance, and everything it declined to do."""

    suite: str
    title: str
    provenance: dict
    convergence: tuple[ConvergenceRun, ...] = ()
    wallclock: tuple[WallclockRun, ...] = ()
    notes: tuple[str, ...] = ()

    def all_notes(self) -> tuple[str, ...]:
        """Suite-level notes plus every note the individual runs recorded.

        Nothing a runner declined to do is allowed to stay buried one level down,
        which is how a capped benchmark ends up reading like a complete one.
        """
        collected = list(self.notes)
        for run in (*self.convergence, *self.wallclock):
            collected.extend(run.notes)
        return tuple(collected)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite": self.suite,
            "title": self.title,
            "provenance": self.provenance,
            "notes": list(self.notes),
            "convergence": [run.to_dict() for run in self.convergence],
            "wallclock": [run.to_dict() for run in self.wallclock],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResults":
        found = data.get("schema_version")
        if found != SCHEMA_VERSION:
            raise ValueError(
                f"results were written by schema version {found!r}, but this code reads "
                f"version {SCHEMA_VERSION}"
            )
        return cls(
            suite=data["suite"],
            title=data["title"],
            provenance=data["provenance"],
            convergence=tuple(ConvergenceRun.from_dict(d) for d in data["convergence"]),
            wallclock=tuple(WallclockRun.from_dict(d) for d in data["wallclock"]),
            notes=tuple(data["notes"]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResults":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class RunComparison:
    """One variant's before/after, on one game."""

    kind: str
    algorithm: str
    game: str
    baseline_rate: float
    candidate_rate: float
    speedup: float
    max_exploitability_delta: float | None
    curves_match: bool | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonReport:
    comparisons: tuple[RunComparison, ...]
    notes: tuple[str, ...] = ()

    @property
    def curves_unchanged(self) -> bool:
        """True when every comparable convergence curve survived unchanged.

        `None` results (wall-clock runs, mismatched checkpoints) are not evidence of
        a change and are excluded rather than counted as passes.
        """
        return all(c.curves_match is not False for c in self.comparisons)

    def format_table(self) -> str:
        header = (
            f"{'kind':<12} {'algorithm':<22} {'game':<24} {'before it/s':>12} "
            f"{'after it/s':>11} {'speedup':>8} {'max dexpl':>11} {'curves':>8}"
        )
        lines = [header, "-" * len(header)]
        for c in self.comparisons:
            delta = (
                "n/a"
                if c.max_exploitability_delta is None
                else f"{c.max_exploitability_delta:.3e}"
            )
            match = {None: "n/a", True: "same", False: "CHANGED"}[c.curves_match]
            lines.append(
                f"{c.kind:<12} {c.algorithm:<22} {c.game:<24} {c.baseline_rate:>12.1f} "
                f"{c.candidate_rate:>11.1f} {c.speedup:>7.2f}x {delta:>11} {match:>8}"
            )
        return "\n".join(lines)


def _provenance_notes(baseline: BenchmarkResults, candidate: BenchmarkResults) -> list[str]:
    """Flag the differences that make a *timing* comparison meaningless."""
    notes: list[str] = []
    for field_name in ("platform", "machine", "python", "numpy"):
        before = baseline.provenance.get(field_name)
        after = candidate.provenance.get(field_name)
        if before != after:
            notes.append(
                f"provenance mismatch on {field_name}: baseline {before!r} vs candidate "
                f"{after!r}. Speedups across different machines or interpreters are not "
                f"evidence of anything."
            )
    for name, results in (("baseline", baseline), ("candidate", candidate)):
        if results.provenance.get("git", {}).get("dirty"):
            notes.append(
                f"{name} was measured against a dirty working tree "
                f"(commit {results.provenance['git'].get('commit')}), so the code that "
                f"produced it is not fully recoverable from git."
            )
    return notes


def _curve_delta(before: ConvergenceRun, after: ConvergenceRun) -> tuple[float | None, list[str]]:
    """Largest absolute exploitability difference between two convergence curves.

    Compared per seed when the seed lists match, and on the median curve otherwise --
    with a note either way, since a median-only comparison is the weaker check.
    """
    if before.checkpoints != after.checkpoints:
        return None, [
            (
                f"{before.algorithm} on {before.game}: checkpoints differ "
                f"({before.checkpoints} vs {after.checkpoints}), so the curves were not compared."
            )
        ]
    if before.seeds == after.seeds:
        delta = np.abs(
            np.asarray(before.exploitability_by_seed) - np.asarray(after.exploitability_by_seed)
        )
        return float(delta.max()), []
    delta = np.abs(np.asarray(before.median_curve()) - np.asarray(after.median_curve()))
    return float(delta.max()), [
        (
            f"{before.algorithm} on {before.game}: seed lists differ "
            f"({list(before.seeds)} vs {list(after.seeds)}), so only the median curves were "
            f"compared -- a weaker check than the per-seed one."
        )
    ]


def compare(
    baseline: BenchmarkResults,
    candidate: BenchmarkResults,
    tolerance: float = DEFAULT_CURVE_TOLERANCE,
) -> ComparisonReport:
    """Before/after report for an optimization, per the split in the module docstring.

    Convergence runs are checked for an unchanged curve *and* a throughput ratio;
    wall-clock runs only for throughput, since their curve is meant to move.

    Runs present in only one of the two files are reported in `notes` rather than
    dropped: a "no regressions" verdict over a silently smaller set of runs is the
    exact failure this harness exists to prevent.
    """
    notes = _provenance_notes(baseline, candidate)
    comparisons: list[RunComparison] = []

    for kind, before_runs, after_runs in (
        ("convergence", baseline.convergence, candidate.convergence),
        ("wallclock", baseline.wallclock, candidate.wallclock),
    ):
        before_by_key = {(r.algorithm, r.game): r for r in before_runs}
        after_by_key = {(r.algorithm, r.game): r for r in after_runs}
        for key in sorted(before_by_key.keys() - after_by_key.keys()):
            notes.append(f"{kind}: {key[0]} on {key[1]} is in the baseline but not the candidate.")
        for key in sorted(after_by_key.keys() - before_by_key.keys()):
            notes.append(f"{kind}: {key[0]} on {key[1]} is in the candidate but not the baseline.")

        for key in sorted(before_by_key.keys() & after_by_key.keys()):
            before, after = before_by_key[key], after_by_key[key]
            run_notes: list[str] = []
            if kind == "convergence":
                delta, curve_notes = _curve_delta(before, after)
                run_notes.extend(curve_notes)
                curves_match = None if delta is None else delta <= tolerance
            else:
                delta, curves_match = None, None
            baseline_rate = before.iterations_per_second()
            candidate_rate = after.iterations_per_second()
            comparisons.append(
                RunComparison(
                    kind=kind,
                    algorithm=before.algorithm,
                    game=before.game,
                    baseline_rate=baseline_rate,
                    candidate_rate=candidate_rate,
                    speedup=candidate_rate / baseline_rate,
                    max_exploitability_delta=delta,
                    curves_match=curves_match,
                    notes=tuple(run_notes),
                )
            )

    return ComparisonReport(comparisons=tuple(comparisons), notes=tuple(notes))
