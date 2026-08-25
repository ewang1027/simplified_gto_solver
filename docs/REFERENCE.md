# Module reference

What each module holds and what it exposes. `docs/ARCHITECTURE.md` says *why* the code is
shaped this way; this is the map. Every module has a docstring that goes further than the
summaries here — they are the primary source, and several record measured constraints that
are not obvious from the code.

## games/

| Module | Holds |
|---|---|
| `games/base.py` | `GameState`, `Game`, the `CHANCE` sentinel |
| `games/kuhn.py` | Kuhn poker — 3 cards, 12 info sets |
| `games/leduc.py` | Leduc Hold'em — 6 cards, two betting rounds, 288 info sets |
| `games/glosten_milgrom.py` | Market making against possibly-informed flow, 298 info sets |
| `games/registry.py` | `GAMES`, `GameSpec`, `get_game` — names to games, and which parameters each takes |

`GameState` requires `is_terminal`, `is_chance`, `chance_outcomes`, `current_player`,
`legal_actions`, `apply`, `info_set_key` and `payout`. Two methods are optional:

- **`payouts(num_players)`** — the whole payoff vector in one call. The default calls
  `payout` per player; overriding it halves terminal work in a two-player zero-sum game.
- **`features()`** — a fixed-length encoding of the info set, or `None`. Only needed for
  function approximation. Glosten–Milgrom returns `None`.

States are immutable: `apply()` returns a new state. That is what lets `FullTraversal`
resolve a tree once and reuse it.

## solvers/

| Module | Holds |
|---|---|
| `solvers/base.py` | `CFRSolver`, `InfoSetStore`, `InfoSetRecord`, `RegretUpdateRule`, `Traversal`, `regret_matching` |
| `solvers/regret_rules.py` | `VanillaRegretMatching`, `CFRPlusRegretMatching`, `DiscountedRegretMatching`, `linear_cfr` |
| `solvers/traversal.py` | `FullTraversal` (exact, with the resolved-tree cache), `ExternalSamplingMCCFR`, `build_tree` |
| `solvers/registry.py` | `ALGORITHMS`, `AlgorithmSpec`, `get_algorithm` — the eight named variants |
| `solvers/deep_cfr.py` | `DeepCFRSolver`, `Reservoir`, `enumerate_info_sets` |

A solver is a game, a `RegretUpdateRule` and a `Traversal`. Seven of the eight registered
variants are combinations of those; `deep_cfr` is the exception and uses `AlgorithmSpec`'s
`make_solver` escape hatch, which a test asserts has exactly one user.

Anything with `train(n)`, `average_strategy()`, `iterations` and `store` can be measured by
the benchmark harness and graded by the metrics — which is how Deep CFR is compared against
the tabular solvers despite sharing none of their machinery.

## nn/

| Module | Holds |
|---|---|
| `nn/mlp.py` | `MLP` — a ReLU network in numpy with Adam and per-sample weights |

Written rather than imported: the project takes no deep-learning dependency for a
two-hidden-layer network. Every weight and bias is checked against central finite
differences in `tests/test_mlp.py`, because a gradient wrong by a constant factor still
trains and never shows up in a loss curve.

## metrics/

| Module | Holds |
|---|---|
| `metrics/exploitability.py` | `exploitability`, `best_response_value` |
| `metrics/evaluation.py` | `expected_value`, `action_probs` |

`exploitability` is the correctness metric: ≥ 0 everywhere, 0 exactly at a Nash equilibrium,
and it needs no published answer to compare against. It maximizes **once per information
set**, not per node — a per-node `max` would let the responder use hidden information.

`expected_value` answers the simpler question of what a profile is worth when everyone
follows it. Use it on the **average** strategy; a solver's per-iteration value oscillates.

## analysis/

| Module | Holds |
|---|---|
| `analysis/microstructure.py` | `GMParams`, `strategic_half_spread`, `competitive_half_spread`, `maker_profit`, `market_functions` |
| `analysis/kyle.py` | `KyleParams`, `solve_fixed_point`, `price_impact_regression`, `information_revealed` |

These are the independent benchmarks, and the rule that makes them worth anything is that
**they share no code with the solver they check**. `market_functions` is the viability test
that distinguishes a working market from a maker that has stopped trading — a distinction
that once produced a beautiful and entirely spurious result.

## benchmark/

| Module | Holds |
|---|---|
| `benchmark/stats.py` | `aggregate`, `bootstrap_ci`, `log_checkpoints`, `Aggregate` |
| `benchmark/runner.py` | `run_convergence`, `run_wallclock`, `verify_determinism`, `ConvergenceRun`, `WallclockRun` |
| `benchmark/results.py` | `BenchmarkResults`, `compare`, `provenance` |
| `benchmark/suites.py` | `SUITES`, `Suite`, `run_suite`, `quick` — the five published suites |
| `benchmark/tables.py` | `convergence_markdown`, `wallclock_markdown` |
| `benchmark/reporting.py` | `run_suites`, `print_report`, `print_comparison` — shared by the CLI |
| `benchmark/plots.py` | `figure_*` builders and `plot_*` savers (needs the `viz` extra) |

`stats.py` keeps two bands apart that are easy to confuse: the **seed envelope** says how
much runs differ and does not shrink with more seeds; the **bootstrap CI** says how well N
seeds pin the centre down and does. Charts shade the first, tables quote the second.

`compare()` is the before/after check an optimization must pass: convergence curves are
compared for **equality**, wall-clock curves for **throughput**, because only the second is
supposed to move.

`plots.py` is not imported from `benchmark/__init__.py`, so the package and the test suite
never require matplotlib.

## Entry points

| Module | Holds |
|---|---|
| `cli.py` | The `gto` command: `solve`, `algorithms`, `games`, `benchmark`, `dashboard`, `microstructure` |
| `dashboard.py` | The Streamlit app (needs the `dashboard` extra) |

Both read the registries rather than keeping their own lists, so a new game or variant
appears in them with no change. The dashboard goes further: a live solve produces the same
`ConvergenceRun` a benchmark does and is drawn by the same figure the README publishes.

## scripts/

| Script | Does |
|---|---|
| `scripts/verify_phase4.py` | Regenerates every number in `docs/phase4-microstructure-design.md` |
| `scripts/profile_hotloop.py` | Where training time goes, with a per-node cost |
| `scripts/audit_doc_numbers.py` | Re-measures the machine-specific numbers the documents state |

## tests/

One file per module, plus four that cut across:

| File | Checks |
|---|---|
| `tests/test_microstructure_gate.py` | Solver, game and independent benchmark all agree — the test the microstructure phase exists to pass |
| `tests/test_published_results.py` | The committed `results/*.json`: enough seeds, a clean tree, not accidentally a `--quick` run |
| `tests/test_docs.py` | Documents against the repository: paths exist, commands exist, tables regenerate |
| `tests/test_payout_vector.py` | Every `payouts()` override against `payout` at every terminal node of every game |
