"""The benchmark suites this project publishes.

A suite is a game, a list of algorithm names, and the checkpoints to measure at. The
definitions live here rather than in the script so the published numbers have exactly
one source: the README, the build log and any later dashboard all describe the same
runs, and re-running the script regenerates them rather than approximating them.

Seed counts differ per suite on purpose, and the reason is cost, so it is written
down: twenty seeds of Kuhn MCCFR is a minute, twenty seeds of a twenty-second Leduc
budget is nearly seven. Both exceed the ten-seed floor this project holds stochastic
results to.

`quick()` shrinks a suite for smoke runs and returns the note saying it did. A quick
run's numbers are not the published ones, and the note travels with the results file
so nobody can mistake one for the other later.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from gto_solver.benchmark.results import BenchmarkResults, provenance
from gto_solver.benchmark.runner import (
    ConvergenceRun,
    WallclockRun,
    run_convergence,
    run_wallclock,
)
from gto_solver.benchmark.stats import log_checkpoints
from gto_solver.games.base import Game
from gto_solver.games.glosten_milgrom import GlostenMilgromGame
from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame
from gto_solver.solvers.registry import get_algorithm

QUICK_MAX_ITERATIONS = 1_000
QUICK_MAX_SECONDS = 0.5
QUICK_SEEDS = 3


@dataclass(frozen=True)
class Suite:
    """One published measurement: one game, several algorithms, one output file."""

    name: str
    title: str
    subtitle: str
    kind: str  # "convergence" | "wallclock"
    make_game: Callable[[], Game]
    game_label: str
    algorithms: tuple[str, ...]
    checkpoints: tuple[float, ...]
    seeds: tuple[int, ...]
    # Which run gets the per-seed spread figure. None means the suite draws none.
    spread_algorithm: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("convergence", "wallclock"):
            raise ValueError(f"unknown suite kind {self.kind!r}")

    @property
    def iteration_checkpoints(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.checkpoints)


def _gm_game() -> Game:
    """Glosten-Milgrom at the informed share the Phase 4 tables centre on."""
    return GlostenMilgromGame(mu=0.30)


SUITES: dict[str, Suite] = {
    suite.name: suite
    for suite in (
        Suite(
            name="kuhn_convergence",
            title="CFR variants on Kuhn poker",
            subtitle=(
                "Exploitability vs iterations. Equal iterations is the right axis for "
                "comparing update rules and the wrong one for comparing traversals -- see "
                "the wall-clock suite."
            ),
            kind="convergence",
            make_game=KuhnGame,
            game_label="kuhn_poker",
            algorithms=(
                "vanilla",
                "cfr_plus",
                "dcfr",
                "linear_cfr",
                "cfr_plus_alternating",
                "mccfr",
                "mccfr_plus",
            ),
            checkpoints=log_checkpoints(10, 100_000, per_decade=3),
            seeds=tuple(range(20)),
            spread_algorithm="mccfr",
        ),
        Suite(
            name="leduc_wallclock",
            title="Exact traversal vs sampling on Leduc Hold'em",
            subtitle=(
                "Exploitability vs training seconds. Iterations are not a common currency "
                "between the two: one sampled iteration is a single path, one exact "
                "iteration is the whole 288-info-set tree. Both CFR+ update schedules are "
                "here because CFR+ wins on Kuhn and loses here, and the schedule is the "
                "first explanation to rule out."
            ),
            kind="wallclock",
            make_game=LeducGame,
            game_label="leduc_poker",
            algorithms=(
                "vanilla",
                "cfr_plus",
                "cfr_plus_alternating",
                "mccfr",
                "mccfr_plus",
            ),
            checkpoints=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
            seeds=tuple(range(10)),
            spread_algorithm=None,
        ),
        Suite(
            name="kuhn_deep_cfr",
            title="Deep CFR against tabular CFR on Kuhn poker",
            subtitle=(
                "Scored by exploitability against a game solved exactly. Read the axis "
                "carefully: one Deep CFR iteration runs 30 sampled traversals per player "
                "to MCCFR's one, so equal iterations is NOT equal work -- normalized by "
                "traversals, MCCFR is 2-4x ahead of the curve shown here."
            ),
            kind="convergence",
            make_game=KuhnGame,
            game_label="kuhn_poker",
            algorithms=("vanilla", "mccfr", "deep_cfr"),
            checkpoints=log_checkpoints(5, 200, per_decade=3),
            seeds=tuple(range(10)),
            spread_algorithm="deep_cfr",
        ),
        Suite(
            name="leduc_convergence",
            title="Update rules on Leduc Hold'em, by iteration",
            subtitle=(
                "The wall-clock suite fits only ~600 exact iterations into 20 seconds, and a "
                "rule that merely starts slowly would look like a rule that loses. This is the "
                "same comparison on the axis the literature uses, run out to 5,000 iterations."
            ),
            kind="convergence",
            make_game=LeducGame,
            game_label="leduc_poker",
            algorithms=("vanilla", "cfr_plus", "cfr_plus_alternating", "mccfr"),
            checkpoints=log_checkpoints(10, 5_000, per_decade=3),
            seeds=tuple(range(20)),
            spread_algorithm="mccfr",
        ),
        Suite(
            name="gm_convergence",
            title="CFR on the Glosten-Milgrom market-making game (mu = 0.30)",
            subtitle=(
                "The microstructure game measured the same way as the poker ones. 298 info "
                "sets, 33 quotes at the maker's node. The update rules are comparable to "
                "each other on this axis; the sampled traversal is on it for reference "
                "only, since equal iterations is not a fair comparison against exact."
            ),
            kind="convergence",
            make_game=_gm_game,
            game_label="glosten_milgrom(mu=0.30)",
            algorithms=("vanilla", "cfr_plus", "cfr_plus_alternating", "mccfr"),
            checkpoints=log_checkpoints(10, 3_000, per_decade=3),
            seeds=tuple(range(20)),
            spread_algorithm="mccfr",
        ),
    )
}

# Run by default. The other two are opt-in on cost grounds, and the cost is the whole
# reason to say which is which: leduc_convergence walks the 288-info-set tree 5,000
# times per exact variant, and gm_convergence is the slowest game per iteration.
DEFAULT_SUITES: tuple[str, ...] = ("kuhn_convergence", "leduc_wallclock")


def get_suite(name: str) -> Suite:
    try:
        return SUITES[name]
    except KeyError:
        known = ", ".join(sorted(SUITES))
        raise KeyError(f"unknown suite {name!r}; known suites are: {known}") from None


def quick(suite: Suite) -> tuple[Suite, list[str]]:
    """A cheap version of `suite`, plus the note recording exactly what was cut.

    Used by `--quick` and by the tests, which need the whole pipeline exercised in
    seconds rather than minutes.
    """
    ceiling = QUICK_MAX_ITERATIONS if suite.kind == "convergence" else QUICK_MAX_SECONDS
    checkpoints = tuple(c for c in suite.checkpoints if c <= ceiling) or (suite.checkpoints[0],)
    seeds = suite.seeds[:QUICK_SEEDS]
    note = (
        f"QUICK PROFILE -- these are not the published numbers. Suite {suite.name!r} was cut "
        f"from {len(suite.checkpoints)} checkpoints (to {suite.checkpoints[-1]:g}) and "
        f"{len(suite.seeds)} seeds down to {len(checkpoints)} checkpoints (to "
        f"{checkpoints[-1]:g}) and {len(seeds)} seeds."
    )
    return replace(suite, checkpoints=checkpoints, seeds=seeds), [note]


def run_suite(
    suite: Suite,
    extra_notes: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> BenchmarkResults:
    """Execute every algorithm in `suite` and package the runs with their provenance."""
    convergence: list[ConvergenceRun] = []
    wallclock: list[WallclockRun] = []
    for name in suite.algorithms:
        spec = get_algorithm(name)
        if suite.kind == "convergence":
            convergence.append(
                run_convergence(
                    suite.make_game,
                    spec,
                    suite.iteration_checkpoints,
                    seeds=suite.seeds,
                    game_label=suite.game_label,
                    progress=progress,
                )
            )
        else:
            wallclock.append(
                run_wallclock(
                    suite.make_game,
                    spec,
                    suite.checkpoints,
                    seeds=suite.seeds,
                    game_label=suite.game_label,
                    progress=progress,
                )
            )
    return BenchmarkResults(
        suite=suite.name,
        title=suite.title,
        provenance=provenance(),
        convergence=tuple(convergence),
        wallclock=tuple(wallclock),
        notes=tuple(extra_notes),
    )
