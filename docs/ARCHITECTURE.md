# Architecture

Why the code is shaped the way it is, and how to extend it. For chronology, measured
results and the traps behind each phase, see `docs/BUILDLOG.md`; for the microstructure
modeling work see `docs/phase4-microstructure-design.md`.

Two ideas carry the whole project, one metric decides whether any of it is right, and one
distinction — which axis a comparison is on — turns out to be where every mistake was
actually made.

## The two abstractions

### 1. `GameState` — an extensive-form game node

`games/base.py`. A game supplies `is_terminal()`, `is_chance()`, `chance_outcomes()`,
`current_player()`, `legal_actions()`, `apply()`, `info_set_key()` and `payout(player)`.
States are immutable — `apply()` returns a new state — which is what later let the solver
resolve a tree once and reuse it.

Two decisions inside this are load-bearing:

**Chance is a node in the tree.** Dealing cards, drawing an asset value, drawing a trader
type — all are chance nodes the solver walks, weighted by outcome probability. The original
Kuhn script instead enumerated the six deals in its training loop with
`itertools.permutations`. That works for exactly one game. Leduc has two chance points
(private deal, then community card) and Glosten–Milgrom draws a value jointly with a trader
type, so loop enumeration cannot express either.

**Info-set keys define what a player cannot distinguish.** Two states a player cannot tell
apart *must* return equal keys. The whole solver depends on this, and it is worth testing
directly for every new game: Leduc keys on card *rank* only, since suits are strategically
irrelevant and keying on them would double the info-set count and split the strategy across
duplicates; the Glosten–Milgrom maker keys only on observed order flow, never on the asset
value or the trader type.

Two methods are **optional**, with defaults that keep every existing game working:

- `payouts(num_players)` returns the whole payoff vector in one call. The default calls
  `payout` per player; a two-player zero-sum game computes the same quantity both times, so
  overriding it halves the work at every terminal node. Get it wrong and the solver
  converges correctly to the equilibrium of a *different game* while every convergence test
  passes — hence `tests/test_payout_vector.py`, which checks the override against `payout`
  at every terminal node of every game.
- `features()` returns a fixed-length numeric encoding of the info set, or `None`. Tabular
  CFR needs only `info_set_key`, which says *whether* two states are the same; a function
  approximator needs to know how they are *related*, which a key cannot express. Returning
  `None` means the game does not support function approximation, and Deep CFR says so
  rather than inventing an encoding on the game's behalf.

### 2. `RegretUpdateRule` × `Traversal` — algorithms by composition

`solvers/base.py`. A CFR variant is one update rule composed with one traversal:

- **`RegretUpdateRule`** owns how cumulative regret becomes a strategy and how the
  accumulators update. Vanilla regret matching, CFR+'s non-negative flooring with linear
  averaging, and DCFR's discounting differ *only* here.
- **`Traversal`** owns how one iteration walks the tree. `FullTraversal` is exact;
  `ExternalSamplingMCCFR` samples chance and opponent nodes.

Seven of the eight variants in `solvers/registry.py` are combinations of these, not forked
files — including `mccfr_plus`, the CFR+ rule under external sampling, which neither paper
defines and which cost nothing to add. Any rule composes with any traversal.

Regret is stored per info set with a **per-info-set action count**. Kuhn has two actions
everywhere; Leduc has fold/call/raise or call/raise depending on the node; Glosten–Milgrom's
maker chooses among 33 quotes. A global `num_actions` constant — which the original script
had — cannot express that.

**`FullTraversal` resolves the tree once.** Terminality, payoffs, chance probabilities, the
acting player, info-set keys, action counts and each action's successor are all fixed by the
immutability of `GameState`, so they are computed once and reused. This is memoization, not
a second implementation, and results are bit-identical. The **sampled traversals
deliberately do not do it**: external sampling exists for trees too large to enumerate, and
materializing one would remove its reason to exist. A traversal that already visits every
node every iteration loses nothing by keeping it; one that walks a single path would lose
everything. The cache is keyed on game object *identity* — two `GlostenMilgromGame`s differ
only by a constructor argument, and a looser key would solve one and label the answer with
the other's parameters.

### Where the two abstractions stop

**Deep CFR does not compose from a rule and a traversal**, and that is a finding rather than
an oversight. Every tabular variant here differs only in how regret becomes a strategy or
how a traversal walks the tree. Deep CFR changes neither — it changes *what the regret store
is*, replacing the table with a network refitted every iteration.

So `solvers/deep_cfr.py` is its own module, `AlgorithmSpec` carries a `make_solver` escape
hatch and a `composed` flag, and a test asserts `deep_cfr` is the only uncomposed variant —
so a future one quietly bypassing the design fails rather than passing unnoticed. What is
still shared is the part that makes the comparison worth anything: the same
`average_strategy()` surface, so the same metric grades it and the same harness measures it.

## Exploitability is the correctness metric

`metrics/exploitability.py`. Exploitability is how much a best-responding opponent could win
against a strategy profile. It is ≥ 0 everywhere and exactly 0 at a Nash equilibrium.

The alternative — checking that the average game value matches Kuhn's known −1/18 — only
validates games somebody has already solved in closed form. The microstructure games have no
such constant, so that approach cannot carry the project.

**The subtle part:** a best responder cannot act differently in states it cannot
distinguish, so maximization happens **once per information set, not per tree node**. A
naive per-node `max` silently lets the responder use hidden information and reports an
exploitability that is too high. The implementation does two passes — collect every node of
each info set with its counterfactual reach, then take a single argmax per info set — with
`value` and `best_action_index` mutually recursive and the latter memoized.

Kuhn's `Q|cb` info set is the concrete case: its two member nodes have *opposite* locally
optimal actions, so per-node and per-info-set maximization genuinely disagree there.

`metrics/evaluation.py` answers the simpler question — what a profile is worth when everyone
follows it. Use it, not a solver's per-iteration return value, to evaluate a strategy: the
per-iteration value reflects the current regret-matching strategy, which oscillates, whereas
the **average** strategy is what converges.

## Three axes, and naming which one you are on

This is the single mistake this project has made most often, in three different disguises
across three phases. Comparisons here run on one of three axes and they are not
interchangeable.

**Iterations.** Pure math: a convergence curve indexed by iteration count is what a faithful
optimization must leave *unchanged*, which is why `compare()` checks it for equality. But an
"iteration" is not one thing. One exact iteration walks the whole tree; one external-sampling
iteration is a single sampled path per player; one Deep CFR iteration is thirty sampled
traversals per player. **Comparing update rules on this axis is fair. Comparing traversals on
it is a category error.**

**Seconds.** The only currency exact and sampled traversals share, and the axis that answers
"which should I actually run". But a wall-clock curve is *supposed* to move when the code
changes, so `compare()` checks it for throughput rather than equality — and it is a statement
about an implementation, not about an algorithm. Phase 6 sped exact traversal up 2.8× and
sampling by 9%, and the published Leduc exact-vs-sampled ratio fell from 211× to 83× with
nothing about either algorithm changing.

**Traversals.** Game-tree walks: the fair axis for comparing two *sampled* methods. Phase 9
needed it. At equal iterations Deep CFR appeared to beat MCCFR by 2–3×, but one Deep CFR
iteration does thirty times the sampling, and normalized by traversals MCCFR is 2.1–3.9×
ahead. The apparent win was entirely the axis.

The rule that falls out, and the reason this section exists: **name the axis before
comparing, and say why it is the right one for the question.** Every table in the README
does; the one place a caveat could not fit into a table, it went into the suite's subtitle so
it travels with the results file and prints on the chart.

## Measuring: what a benchmark number has to survive

`benchmark/`. Exploitability says whether the solver is right; this says whether a claim
about it is worth anything. The harness was built before Phase 6 optimized the hot loop,
because "2× faster" means nothing unless before and after were measured the same way.

**Two bands, two questions.** `stats.py` computes both and never lets them be confused:

- The **seed envelope** (10th–90th percentile across seeds) says how much individual runs
  differ. It is a property of the algorithm's variance, so it does *not* shrink as more
  seeds are added.
- The **bootstrap CI of the median** says how well N seeds pin the centre down. It *does*
  shrink as N grows.

Charts shade the envelope, tables quote the CI, and both say which they are. Swapping them
would make a noisy algorithm look precise just by running it more times, which is why the
distinction is pinned by a test rather than left to discipline.

The central statistic is the median, not the mean: exploitability across seeds is positive,
right-skewed and read on a log axis, where one unlucky seed drags a mean somewhere no run
actually visited.

**Three rules are enforced in code**, because each has a failure mode that produces a
plausible-looking wrong answer:

1. **Evaluating exploitability is never charged to the solver's clock.** Measuring costs
   58 ms on Leduc against an 11 ms iteration — five iterations' worth, and the ratio got
   *worse* with Phase 6, which sped the iteration up and left the measurement alone.
   Charge it and whichever configuration was
   measured at more checkpoints reports lower throughput for identical work — the
   exact-vs-sampled comparison would become an artifact of how often it was observed. Note
   the corollary: extracting a strategy is not charged either, and for Deep CFR that is a
   network fit, so its throughput is flattered relative to tabular. Compare it on iterations
   and traversals, never on throughput.
2. **Deterministic variants are run once.** `FullTraversal` never touches the rng, so every
   seed produces a bit-identical strategy. Twenty seeds would publish a band of width zero
   as though variance had been measured and found small, rather than being absent by
   construction. The flag asserting this is checked by `verify_determinism()` in both
   directions: exact variants must match bit for bit, sampled ones must not.
3. **Nothing is capped, reduced or skipped in silence.** A reduced seed count, a quick
   profile, a wall-clock checkpoint that had to be overshot — each lands in `notes`, which
   is serialized with the results, printed by the CLI, and rendered by the dashboard. A
   benchmark that quietly shrinks its own workload reads later as a clean result.

Results carry provenance — interpreter, numpy, machine, commit, and whether the tree was
dirty — because a timing number without the machine it ran on is not a measurement, and a
"reproduce this" pointing at a commit is worthless if the tree had uncommitted edits. The
dirty check deliberately ignores `results/` and `docs/images/`: the benchmark writes its own
outputs there as it runs, and counting them meant every suite after the first reported a tree
dirtied by the run in progress.

**Know the noise floor before claiming a speedup.** Ten seeds each training for exactly 20
seconds on this machine completed between 129,913 and 143,891 iterations — a 3.1% relative
standard deviation. A throughput difference smaller than that is machine drift.
`scripts/audit_doc_numbers.py` re-measures this and the other machine-specific numbers in
these documents, because two of them had already gone stale by Phase 10.

## Microstructure: modeling assumptions

Stated plainly because a reader will ask.

- CFR's guarantees and exploitability hold for **two-player zero-sum** games. The maker and
  the aggregate trader are therefore modeled as exactly two players. Real markets are
  neither two-player nor zero-sum.
- The CFR market maker is a **profit maximizer**, not Glosten–Milgrom's **competitive**
  zero-profit maker. Both benchmarks are computed and reported; neither is presented as the
  other. The strategic maker quotes strictly wider.
- Value, quote and reservation grids are discretized to keep the action set finite.
- The uninformed trader is mechanical, so it is modeled as a chance node rather than a
  player — and that chance node sits *after* the quote, which keeps the reservation out of
  the root and the tree small.
- **Kyle is not a CFR game at all.** Its maker is competitive, and a strategic one has no
  interior optimum. It is solved by fixed-point iteration in `analysis/kyle.py`. Choosing
  the algorithm to fit the model, rather than the reverse, is deliberate.

## Extending it

### Add a game

1. Implement `GameState` and `Game` in `games/`. Use `leduc.py` as the reference — it is the
   non-trivial one.
2. Make chance a node, and get the outcome probabilities right, especially with duplicate
   outcomes (Leduc's deck has two of each rank).
3. Make `payout(player)` exactly antisymmetric for two-player zero-sum. `exploitability`
   walks every terminal node asserting payouts sum to zero, so violations surface loudly.
   Optionally override `payouts()` — see above for the hazard.
4. Test info-set indistinguishability directly: states differing only in hidden information
   must produce equal keys.
5. Register it in `games/registry.py`, with the parameters it accepts. The CLI refuses a
   parameter a game does not take rather than ignoring it.
6. No solver change should be needed. **If you find yourself editing anything under
   `solvers/` or `metrics/` to make a game work, that is a finding about the abstraction —
   investigate it rather than working around it.** Leduc needed zero such changes, which is
   the evidence the interface is real.

### Add an algorithm variant

Add a `RegretUpdateRule` to `solvers/regret_rules.py` or a `Traversal` to
`solvers/traversal.py`, then register the combination in `solvers/registry.py`. Do not fork
the engine. New traversals must take all randomness from the passed-in `rng` so
`CFRSolver(..., seed=)` stays reproducible.

Set `deterministic` honestly: it decides whether the benchmark runs one seed or twenty, and
it is checked by actually running the seeds, so a wrong value fails a test rather than
silently publishing a stochastic variant as a single run.

If the variant genuinely does not fit rule × traversal, use `make_solver` and expect to
justify it — that escape hatch has exactly one user, and a test says so.

Validate with exploitability trending to zero, and report what you measure. Several variants
here (DCFR, alternating updates, Deep CFR) do *not* beat the baseline, and that is recorded
rather than tuned away.

### Add a benchmark

Analytical or brute-force benchmarks live in `analysis/`. The rule that makes them worth
anything: **a benchmark must share no code with the solver it checks.** The Glosten–Milgrom
benchmark is an independent exhaustive grid search, which is why the solver matching it is
evidence rather than a tautology.

*Performance* benchmarks are different work. Add a `Suite` to `benchmark/suites.py` — a game,
a list of algorithm names, and the checkpoints to measure at — and `gto benchmark` picks it
up. Anything stochastic gets at least ten seeds; a suite that breaks that rule fails a test.
Do not measure timings from a script of your own: everything published goes through
`benchmark/runner.py` so a later phase can compare against it.

## Layout

```
src/gto_solver/
├── games/          base.py, kuhn.py, leduc.py, glosten_milgrom.py, registry.py
├── solvers/        base.py, regret_rules.py, traversal.py, registry.py, deep_cfr.py
├── nn/             mlp.py — a small network in numpy, gradient-checked
├── analysis/       microstructure.py (GM benchmarks), kyle.py (fixed-point solver)
├── metrics/        exploitability.py, evaluation.py
├── benchmark/      stats.py, runner.py, results.py, suites.py, tables.py,
│                   reporting.py, plots.py
├── cli.py          the `gto` command line
└── dashboard.py    the Streamlit app (optional `dashboard` extra)
tests/              one file per module, plus the cross-cutting ones below
docs/               BUILDLOG.md, ARCHITECTURE.md, phase4-microstructure-design.md
results/            benchmark results as JSON, with provenance
scripts/            verify_phase4.py (regenerates the design doc's numbers),
                    profile_hotloop.py (where the training time goes)
```

**Everything with a user interface reads the registries rather than keeping its own lists.**
`cli.py` selects from `games/registry.py`, `solvers/registry.py` and `benchmark/suites.py`,
so adding a variant or a game makes it appear in the CLI with no change there, and
`gto algorithms` cannot drift out of date. `dashboard.py` goes one step further: a live solve
goes through `benchmark/runner.py` and produces the same `ConvergenceRun` a benchmark does,
drawn by the same `benchmark/plots.py` figure the README publishes — a dashboard drawing its
own lookalike would eventually disagree with the documents, invisibly. Running a suite lives
in `benchmark/reporting.py` rather than in either, because it was duplicated in a script
once and the half that drifts first is the notes.

**Optional dependencies stay optional.** `benchmark/plots.py` needs matplotlib (`viz`) and
`dashboard.py` needs streamlit (`dashboard`); neither is imported from a package `__init__`,
so importing `gto_solver` or running the test suite requires neither. Deep CFR needs no
extra at all — its network is `nn/mlp.py`.

Three test files are cross-cutting rather than per-module:

- `tests/test_microstructure_gate.py` checks the solver, the game and the independent
  benchmark all agree. It is the test the microstructure phase exists to pass.
- `tests/test_published_results.py` checks the committed `results/*.json` the way anything
  published gets checked — enough seeds, a clean tree, and not accidentally a `--quick` run.
- `tests/test_docs.py` checks these documents against the repository: every path they name
  exists, every command they tell you to run is real, and every benchmark table regenerates
  from the results file it came from. The prose is left to a reader; the checkable parts are
  checked.
