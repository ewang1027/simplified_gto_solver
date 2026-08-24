"""The Streamlit dashboard. Run it with `gto dashboard`.

Like the CLI, this selects from the registries and reuses the harness rather than
reimplementing anything: a live solve goes through `benchmark/runner.py` and produces
the same `ConvergenceRun` a benchmark does, which is then drawn by the same
`benchmark/plots.py` figure the README publishes. A dashboard that drew its own
lookalike chart from its own loop would eventually disagree with the documents, and
the disagreement would be invisible.

Two rules the layout follows, both inherited from earlier phases:

* **A shaded band is the seed envelope**, labelled as such, never the bootstrap CI.
  They answer different questions and swapping them makes a noisy algorithm look
  precise (see `benchmark/stats.py`).
* **Nothing expensive runs on a slider drag.** Published suites are read from
  `results/*.json` rather than re-measured -- a twenty-second Leduc budget is not an
  interactive latency -- and a live Leduc solve is put behind an explicit button,
  because it is seconds per run where Kuhn is milliseconds.

streamlit is an optional dependency (`pip install -e '.[dashboard]'`). Nothing else in
the package imports this module, so the solver, the CLI and the tests all run without it.
"""

from pathlib import Path

import numpy as np
import streamlit as st

from gto_solver.analysis.microstructure import (
    GMParams,
    competitive_half_spread,
    strategic_half_spread,
)
from gto_solver.benchmark.plots import figure_convergence, figure_seed_spread, figure_wallclock
from gto_solver.benchmark.results import BenchmarkResults
from gto_solver.benchmark.runner import run_convergence
from gto_solver.benchmark.stats import log_checkpoints
from gto_solver.benchmark.suites import SUITES
from gto_solver.benchmark.tables import convergence_markdown, wallclock_markdown
from gto_solver.games.registry import GAMES, get_game
from gto_solver.metrics.evaluation import expected_value
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.registry import ALGORITHMS, get_algorithm

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
# Games cheap enough to solve on every widget change. Leduc walks 9,451 nodes per
# iteration and goes behind a button instead.
INSTANT_GAMES = ("kuhn", "gm")


@st.cache_data(show_spinner=False)
def _solve(game: str, algorithm: str, iterations: int, seed: int, parameters: tuple):
    """One live solve, cached on its arguments so a rerun is free.

    Returns the harness's own `ConvergenceRun` plus the solved strategy, so the chart
    below is drawn by exactly the code that draws the published ones.
    """
    spec = get_game(game)
    kwargs = dict(parameters)
    algorithm_spec = get_algorithm(algorithm)
    checkpoints = log_checkpoints(1, iterations, per_decade=4)
    run = run_convergence(
        lambda: spec.create(**kwargs),
        algorithm_spec,
        checkpoints,
        seeds=(seed,),
        game_label=spec.label,
    )
    instance = spec.create(**kwargs)
    solver = algorithm_spec.build(instance, seed=seed)
    solver.train(iterations)
    strategy = solver.average_strategy()
    return run, strategy, expected_value(instance, strategy, 0), len(solver.store)


@st.cache_data(show_spinner=False)
def _load_results(name: str):
    path = RESULTS_DIR / f"{name}.json"
    return BenchmarkResults.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def _cfr_spread(mu: float, iterations: int) -> tuple[float, float]:
    """The solved full spread at `mu`, and the profile's exploitability."""
    game = get_game("gm").create(mu=mu)
    solver = get_algorithm("cfr_plus").build(game, seed=0)
    solver.train(iterations)
    strategy = solver.average_strategy()
    params = GMParams()
    keys = [key for key, probs in strategy.items() if len(probs) == params.num_quotes]
    quote = float(params.quotes()[int(np.argmax(strategy[keys[0]]))])
    return 2.0 * quote, exploitability(game, strategy)


def _solve_tab() -> None:
    st.subheader("Solve a game")
    st.caption(
        "Exploitability is >= 0 for every strategy profile and exactly 0 at a Nash "
        "equilibrium, so it measures convergence without needing to know the game's value "
        "in advance. What is reported below is the **average** strategy: a solver's "
        "per-iteration strategy oscillates and does not settle on the game value."
    )

    left, middle, right = st.columns(3)
    game = left.selectbox(
        "Game", list(GAMES), format_func=lambda name: GAMES[name].label, key="solve_game"
    )
    algorithm = middle.selectbox(
        "Algorithm",
        list(ALGORITHMS),
        format_func=lambda name: ALGORITHMS[name].label,
        key="solve_algorithm",
    )
    iterations = right.select_slider(
        "Iterations", options=[100, 300, 1_000, 3_000, 10_000], value=1_000, key="solve_iterations"
    )

    spec = get_algorithm(algorithm)
    seed = 0
    parameters: dict[str, float | int] = {}
    options = st.columns(3)
    if not spec.deterministic:
        seed = options[0].number_input("Seed", 0, 999, 0, key="solve_seed")
    else:
        options[0].caption("Deterministic — this traversal never reads the rng, so no seed.")
    if "mu" in GAMES[game].parameters:
        parameters["mu"] = options[1].slider(
            "mu (informed share)", 0.0, 0.95, 0.30, 0.05, key="solve_mu"
        )

    expensive = game not in INSTANT_GAMES
    if expensive:
        st.info(
            f"{GAMES[game].label} walks a 288-info-set tree every iteration — seconds, not "
            f"milliseconds. Solving is behind the button so it does not run on every "
            f"widget change."
        )
        if not st.button("Solve", type="primary", key="solve_run"):
            return

    with st.spinner("Solving…"):
        run, strategy, value, info_sets = _solve(
            game, algorithm, int(iterations), int(seed), tuple(sorted(parameters.items()))
        )

    columns = st.columns(3)
    columns[0].metric("Exploitability", f"{run.exploitability_by_seed[0][-1]:.6f}")
    columns[1].metric("Value to player 0", f"{value:+.6f}")
    columns[2].metric("Info sets", f"{info_sets:,}")

    st.pyplot(
        figure_convergence(
            [run],
            title=f"{GAMES[game].label} — {spec.label}",
            subtitle=f"{int(iterations):,} iterations, exploitability against iterations.",
        )
    )

    with st.expander(f"Solved strategy ({len(strategy)} info sets)"):
        st.dataframe(
            {
                "info set": list(strategy),
                "probabilities": [
                    "  ".join(f"{p:.4f}" for p in np.asarray(probs)) for probs in strategy.values()
                ],
            },
            width="stretch",
            hide_index=True,
        )


def _benchmarks_tab() -> None:
    st.subheader("Published benchmarks")
    st.caption(
        "Read from `results/*.json` rather than re-measured — these took about twenty "
        "minutes to produce. Every shaded band is the 10–90% **seed envelope**, how much "
        "individual runs differ, never a confidence interval."
    )

    available = [name for name in SUITES if (RESULTS_DIR / f"{name}.json").exists()]
    if not available:
        st.warning(
            "No results files found. Run `gto benchmark` to produce them, or "
            "`gto benchmark --quick` for a fast smoke run that is explicitly not publishable."
        )
        return

    name = st.selectbox(
        "Suite", available, format_func=lambda key: SUITES[key].title, key="bench_suite"
    )
    results = _load_results(name)
    suite = SUITES[name]
    st.caption(suite.subtitle)

    if results.convergence:
        st.pyplot(figure_convergence(results.convergence, suite.title, suite.subtitle))
        table, table_notes = convergence_markdown(results.convergence)
    else:
        st.pyplot(figure_wallclock(results.wallclock, suite.title, suite.subtitle))
        table, table_notes = wallclock_markdown(results.wallclock)
    st.markdown(table)

    spread = [r for r in results.convergence if r.algorithm == suite.spread_algorithm]
    if spread and len(spread[0].seeds) > 1:
        with st.expander("Seed variance: why one run is not a result"):
            st.pyplot(figure_seed_spread(spread[0]))

    notes = list(results.all_notes()) + list(table_notes)
    with st.expander(f"Notes ({len(notes)}) — anything capped, reduced or skipped"):
        for note in notes:
            st.markdown(f"- {note}")

    git = results.provenance.get("git", {})
    st.caption(
        f"Measured on {results.provenance.get('platform')} · Python "
        f"{results.provenance.get('python')} · numpy {results.provenance.get('numpy')} · "
        f"commit {git.get('commit')} · dirty tree: {git.get('dirty')}"
    )


def _microstructure_tab() -> None:
    st.subheader("Market making under adverse selection")
    st.caption(
        "The analytic curves are instant. The CFR point is solved live for the selected mu "
        "and cached — it is the same solver, run against a benchmark that shares no code "
        "with it, so agreement is evidence rather than a tautology."
    )

    left, right = st.columns(2)
    mu = left.slider("mu (informed share)", 0.02, 0.90, 0.30, 0.02, key="micro_mu")
    iterations = right.select_slider(
        "CFR iterations", options=[100, 200, 400, 800], value=400, key="micro_iterations"
    )

    params = GMParams()
    grid = np.linspace(0.02, 0.90, 45)
    strategic = [2.0 * strategic_half_spread(params, m) for m in grid]
    competitive = [2.0 * competitive_half_spread(params, m) for m in grid]

    with st.spinner("Solving…"):
        solved, exploit = _cfr_spread(float(mu), int(iterations))
    brute = 2.0 * strategic_half_spread(params, float(mu))

    columns = st.columns(4)
    columns[0].metric("CFR spread", f"{solved:.4f}")
    columns[1].metric("Brute-force optimum", f"{brute:.4f}", f"{solved - brute:+.4f}")
    competitive_now = 2.0 * competitive_half_spread(params, float(mu))
    columns[2].metric("Competitive (zero-profit)", f"{competitive_now:.4f}")
    columns[3].metric("CFR exploitability", f"{exploit:.6f}")

    st.line_chart(
        {
            "profit-maximizing maker (brute force)": strategic,
            "competitive maker (Glosten-Milgrom)": competitive,
        },
        x_label="mu (informed share)",
        y_label="full spread",
    )
    st.markdown(
        "The spread widens with informed flow — adverse selection, recovered from self-play "
        "rather than assumed. The two makers answer different questions and neither is "
        "presented as the other: Glosten–Milgrom's competitive maker earns zero expected "
        "profit by construction, so it always quotes tighter than one that maximizes."
    )


def main() -> None:
    st.set_page_config(page_title="CFR solver", page_icon="🃏", layout="wide")
    st.title("A from-scratch CFR solver")
    st.markdown(
        "Counterfactual regret minimization for two-player zero-sum imperfect-information "
        "games. Poker is the validation harness — Kuhn and Leduc have known equilibria, so "
        "they prove the solver is right. The destination is the market-making game."
    )
    solve_tab, benchmarks_tab, microstructure_tab = st.tabs(
        ["Solve", "Benchmarks", "Microstructure"]
    )
    with solve_tab:
        _solve_tab()
    with benchmarks_tab:
        _benchmarks_tab()
    with microstructure_tab:
        _microstructure_tab()


main()
