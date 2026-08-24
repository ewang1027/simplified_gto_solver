"""`gto` — the command-line interface.

Everything here selects from the registries rather than keeping its own lists:
`games/registry.py` for games, `solvers/registry.py` for algorithms,
`benchmark/suites.py` for suites. Adding a variant or a game makes it appear in the
CLI with no change to this file, and the `algorithms` and `games` commands cannot
drift out of date because they read the same tables the solver does.

Two conventions worth stating, because both are easy to get subtly wrong:

* **What gets reported is the average strategy**, evaluated through
  `metrics/evaluation.py`. A solver's per-iteration return value is the *current*
  regret-matching strategy, which oscillates and does not settle on the game value;
  the average is the thing that converges.
* **Parameters that do not apply are an error, not a no-op.** `--mu 0.7 --game kuhn`
  raises. A spread reported for a mu the run never used would be worse than a refusal.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from gto_solver.benchmark.reporting import print_comparison, run_suites
from gto_solver.benchmark.suites import DEFAULT_SUITES, SUITES, get_suite
from gto_solver.games.registry import GAMES, get_game
from gto_solver.metrics.evaluation import expected_value
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.registry import ALGORITHMS, get_algorithm

app = typer.Typer(
    help="A from-scratch CFR solver for imperfect-information games.",
    no_args_is_help=True,
    add_completion=False,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_IMAGE_DIR = REPO_ROOT / "docs" / "images"


def _fail(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(code=2)


def _message(exc: Exception) -> str:
    """The message a registry lookup raised, without repr's quoting.

    `str(KeyError("no such game 'x'"))` is the *repr* of the argument, so a message
    containing an apostrophe comes back wrapped in double quotes. Users should not
    have to read around that.
    """
    return exc.args[0] if exc.args else str(exc)


def _checkpoints(iterations: int, count: int = 5) -> list[int]:
    """Roughly log-spaced points up to `iterations`, always including the last.

    A convergence table wants to show the decay, and evenly spaced rows on a
    quantity that falls like 1/sqrt(T) show almost nothing.
    """
    if iterations <= count:
        return list(range(1, iterations + 1))
    points = {iterations}
    for i in range(count):
        points.add(max(1, round(iterations ** ((i + 1) / count))))
    return sorted(points)


@app.command()
def solve(
    game: Annotated[str, typer.Option(help="Game to solve.")] = "kuhn",
    algorithm: Annotated[str, typer.Option(help="CFR variant.")] = "vanilla",
    iterations: Annotated[int, typer.Option(min=1, help="Training iterations.")] = 10_000,
    seed: Annotated[int, typer.Option(help="Seed; only sampled variants use it.")] = 0,
    mu: Annotated[float | None, typer.Option(help="Informed-trader share (gm only).")] = None,
    rounds: Annotated[int | None, typer.Option(help="Trading rounds (gm only).")] = None,
    show_strategy: Annotated[bool, typer.Option(help="Print the solved strategy.")] = True,
    max_info_sets: Annotated[
        int, typer.Option(min=1, help="Cap on strategy rows printed.")
    ] = 24,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of tables.")] = False,
) -> None:
    """Train a solver and report exploitability, game value and the strategy."""
    try:
        game_spec = get_game(game)
        algorithm_spec = get_algorithm(algorithm)
    except KeyError as exc:
        _fail(_message(exc))

    parameters = {name: value for name, value in (("mu", mu), ("rounds", rounds))
                  if value is not None}
    try:
        instance = game_spec.create(**parameters)
    except ValueError as exc:
        _fail(str(exc))

    solver = algorithm_spec.build(instance, seed=seed)
    checkpoints = _checkpoints(iterations)
    curve: list[tuple[int, float]] = []
    for target in checkpoints:
        solver.train(target - solver.iterations)
        curve.append((target, exploitability(instance, solver.average_strategy())))

    strategy = solver.average_strategy()
    value = expected_value(instance, strategy, 0)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "game": instance.name,
                    "algorithm": algorithm_spec.name,
                    "iterations": iterations,
                    "seed": seed if not algorithm_spec.deterministic else None,
                    "parameters": parameters,
                    "info_sets": len(solver.store),
                    "exploitability": curve[-1][1],
                    "value_to_player_0": value,
                    "convergence": [
                        {"iterations": i, "exploitability": e} for i, e in curve
                    ],
                    "strategy": {key: probs.tolist() for key, probs in strategy.items()},
                },
                indent=2,
            )
        )
        return

    typer.echo(f"\n{game_spec.label} — {algorithm_spec.label}")
    if parameters:
        typer.echo("  " + ", ".join(f"{k}={v}" for k, v in parameters.items()))
    typer.echo(
        "  seeded run" if not algorithm_spec.deterministic else "  deterministic (seed unused)"
    )

    typer.echo(f"\n{'Iterations':>12} {'Exploitability':>16}")
    typer.echo("-" * 29)
    for point, value_at in curve:
        typer.echo(f"{point:>12,} {value_at:>16.8f}")

    typer.echo(f"\nInfo sets discovered: {len(solver.store):,}")
    typer.echo(f"Value to player 0:    {value:+.6f}")
    typer.echo("Exploitability is >= 0 everywhere and 0 exactly at a Nash equilibrium.")

    if not show_strategy:
        return
    typer.echo(f"\n{'Info set':<16} Action probabilities")
    typer.echo("-" * 60)
    for index, (key, probs) in enumerate(sorted(strategy.items())):
        if index >= max_info_sets:
            typer.echo(
                f"... {len(strategy) - max_info_sets} more info sets "
                f"(raise --max-info-sets, or use --json)"
            )
            break
        formatted = "  ".join(f"{p:.4f}" for p in np.asarray(probs))
        typer.echo(f"{key:<16} {formatted}")


@app.command()
def algorithms() -> None:
    """List the CFR variants this build knows about."""
    typer.echo(f"{'name':<22} {'kind':<14} description")
    typer.echo("-" * 100)
    for spec in ALGORITHMS.values():
        kind = "deterministic" if spec.deterministic else "sampled"
        typer.echo(f"{spec.name:<22} {kind:<14} {spec.description}")


@app.command()
def games() -> None:
    """List the games this build knows about."""
    typer.echo(f"{'name':<8} {'parameters':<18} description")
    typer.echo("-" * 100)
    for spec in GAMES.values():
        parameters = ", ".join(spec.parameters) if spec.parameters else "-"
        typer.echo(f"{spec.name:<8} {parameters:<18} {spec.description}")


@app.command()
def benchmark(
    suite: Annotated[
        list[str] | None, typer.Option(help="Suite to run; repeatable.")
    ] = None,
    list_suites: Annotated[bool, typer.Option("--list", help="List suites and exit.")] = False,
    quick: Annotated[
        bool, typer.Option(help="Cheap smoke profile. NOT the published numbers.")
    ] = False,
    plots: Annotated[bool, typer.Option(help="Draw charts (needs the viz extra).")] = True,
    results_dir: Annotated[Path, typer.Option()] = DEFAULT_RESULTS_DIR,
    image_dir: Annotated[Path, typer.Option()] = DEFAULT_IMAGE_DIR,
    compare_files: Annotated[
        tuple[Path, Path] | None,
        typer.Option("--compare", help="Compare two results files instead of running."),
    ] = None,
) -> None:
    """Run the published benchmark suites, or compare two results files.

    `--compare` is the check an optimization has to pass: throughput may move,
    per-iteration convergence curves may not. It exits non-zero if one did.
    """
    if list_suites:
        for name, spec in SUITES.items():
            default = " (default)" if name in DEFAULT_SUITES else ""
            typer.echo(f"{name}{default}\n    {spec.title}")
            typer.echo(
                f"    {spec.kind}, {len(spec.algorithms)} algorithms, "
                f"{len(spec.seeds)} seeds, {len(spec.checkpoints)} checkpoints"
            )
        return

    if compare_files is not None:
        raise typer.Exit(code=0 if print_comparison(*compare_files) else 1)

    names = suite or list(DEFAULT_SUITES)
    try:
        specs = [get_suite(name) for name in names]
    except KeyError as exc:
        _fail(_message(exc))
    run_suites(specs, quick, results_dir, image_dir, plots)


@app.command()
def microstructure(
    mus: Annotated[
        list[float] | None, typer.Option("--mu", help="Informed shares to solve.")
    ] = None,
    iterations: Annotated[int, typer.Option(min=1)] = 400,
) -> None:
    """Solve the market-making game and compare against both benchmarks.

    Three numbers per informed share, and they answer different questions. The CFR
    maker maximizes profit; the brute-force optimum is an exhaustive grid search that
    shares no code with the solver, so agreement is evidence rather than a tautology;
    and Glosten-Milgrom's competitive maker earns zero profit by construction, which
    is why it quotes strictly tighter than either.
    """
    from gto_solver.analysis.microstructure import (
        GMParams,
        competitive_half_spread,
        strategic_half_spread,
    )

    shares = mus or [0.02, 0.10, 0.30, 0.50, 0.70]
    params = GMParams()
    spec = get_algorithm("cfr_plus")

    typer.echo(
        f"\n{'mu':>6} {'CFR spread':>12} {'brute force':>12} {'competitive':>12} "
        f"{'exploitability':>15} {'match':>7}"
    )
    typer.echo("-" * 70)
    agreed = True
    for mu in shares:
        if not 0.0 <= mu <= 1.0:
            _fail(f"mu must be a probability in [0, 1], got {mu}")
        game = get_game("gm").create(mu=mu)
        solver = spec.build(game, seed=0)
        solver.train(iterations)
        strategy = solver.average_strategy()

        keys = [key for key, probs in strategy.items() if len(probs) == params.num_quotes]
        quotes = params.quotes()
        solved = 2.0 * float(quotes[int(np.argmax(strategy[keys[0]]))])
        brute = 2.0 * strategic_half_spread(params, mu)
        competitive = 2.0 * competitive_half_spread(params, mu)
        matched = abs(solved - brute) < 1e-9
        agreed &= matched
        typer.echo(
            f"{mu:>6.2f} {solved:>12.4f} {brute:>12.4f} {competitive:>12.4f} "
            f"{exploitability(game, strategy):>15.6f} {'yes' if matched else 'NO':>7}"
        )

    typer.echo(
        "\nSpreads are full spreads (twice the half-spread). The strategic maker quotes "
        "wider\nthan the competitive one at every mu, and the spread widens with informed "
        "flow --\nadverse selection, recovered from self-play rather than assumed."
    )
    if not agreed:
        typer.secho(
            "\nThe solver did not match the brute-force optimum at every mu. More "
            "iterations may fix it; if not, something is wrong.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
