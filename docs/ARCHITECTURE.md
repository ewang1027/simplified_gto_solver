# Architecture

Why the code is shaped the way it is, and how to extend it. For chronology and status see
`BUILDLOG.md`; for the Phase 4 modeling work see `phase4-microstructure-design.md`.

## The two abstractions

Everything rests on two ideas. Get these and the rest follows.

### 1. `GameState` — an extensive-form game node

`games/base.py`. A game supplies: `is_terminal()`, `is_chance()`, `chance_outcomes()`,
`current_player()`, `legal_actions()`, `apply()`, `info_set_key()`, `payout(player)`.
States are immutable — `apply()` returns a new state.

Two design decisions inside this are load-bearing:

**Chance is a node in the tree.** Dealing cards, drawing an asset value, drawing a trader
type — all are chance nodes the solver walks, weighted by outcome probability. The original
Kuhn script instead enumerated the six deals in its training loop with
`itertools.permutations`. That works for exactly one game. Leduc has two chance points
(private deal, then community card) and Glosten–Milgrom draws a value jointly with a trader
type, so the loop-enumeration approach cannot express either.

**Info-set keys define what a player cannot distinguish.** Two states a player cannot tell
apart *must* return equal keys. This is the property the whole solver depends on, and it is
worth testing directly for every new game — Leduc keys on card *rank* only (suits are
strategically irrelevant, and keying on them would double the info-set count and split the
strategy across duplicates), and the Glosten–Milgrom maker keys only on observed order flow,
never on the asset value or trader type.

### 2. `RegretUpdateRule` × `Traversal` — algorithms by composition

`solvers/base.py`. A CFR variant is one update rule composed with one traversal:

- **`RegretUpdateRule`** owns how cumulative regret becomes a strategy and how accumulators
  update. Vanilla regret matching, CFR+'s non-negative flooring with linear averaging, and
  DCFR's discounting differ *only* here.
- **`Traversal`** owns how one iteration walks the tree. `FullTraversal` is exact;
  `ExternalSamplingMCCFR` samples chance and opponent nodes.

Four algorithms are therefore two rules × two traversals plus a flag, not four forked files.
Any rule composes with any traversal. Adding a fifth variant means adding one small class.

Regret is stored per info set with a **per-info-set action count**. Kuhn has two actions
everywhere; Leduc has fold/call/raise or call/raise depending on the node; Glosten–Milgrom's
maker chooses among 33 quotes. A global `num_actions` constant — which the original script
had — cannot express that.

## Exploitability is the correctness metric

`metrics/exploitability.py`. Exploitability is how much a best-responding opponent could win
against a strategy profile. It is ≥ 0 everywhere and exactly 0 at a Nash equilibrium.

The alternative — checking that the average game value matches Kuhn's known −1/18 — only
validates games somebody already solved in closed form. The microstructure games have no
such constant, so that approach cannot carry the project.

**The subtle part:** a best responder cannot act differently in states it cannot
distinguish, so maximization happens **once per information set, not per tree node**. A
naive per-node `max` silently lets the responder use hidden information and reports
exploitability that is too high. The implementation does two passes — collect every node of
each info set with its counterfactual reach, then take a single argmax per info set — with
`value` and `best_action_index` mutually recursive and the latter memoized.

Kuhn's `Q|cb` info set is the concrete case: its two member nodes have *opposite* locally
optimal actions, so per-node and per-info-set maximization genuinely disagree there.

`metrics/evaluation.py` answers the simpler question — what a profile is worth when everyone
follows it. Use it, not a solver's per-iteration return value, to evaluate a strategy: the
per-iteration value reflects the current regret-matching strategy, which oscillates, whereas
the **average** strategy is what converges.

## Measuring: what a benchmark number has to survive

`benchmark/`. Exploitability says whether the solver is right; this says whether a claim
about it is worth anything. The harness exists because Phase 6 optimizes the hot loop, and
"1.8x faster" means nothing unless before and after were measured the same way.

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
   56 ms on Leduc against a 31 ms iteration. Charge it and whichever configuration was
   measured at more checkpoints reports lower throughput for identical work — the
   exact-vs-sampled comparison would then be an artifact of how often it was observed.
2. **Deterministic variants are run once.** `FullTraversal` never touches the rng, so every
   seed produces a bit-identical strategy. Twenty seeds would publish a band of width zero
   as though variance had been measured and found small, rather than being absent by
   construction. The flag asserting this is checked by `verify_determinism()`, in both
   directions: exact variants must match bit for bit, sampled ones must not.
3. **Nothing is capped, reduced or skipped in silence.** A reduced seed count, a quick
   profile, a wall-clock checkpoint that had to be overshot — each lands in `notes`, which
   is serialized with the results and printed by the script. A benchmark that quietly
   shrinks its own workload reads later as a clean result.

**Iterations and seconds are different axes, on purpose.** A convergence run is indexed by
iteration count, so its curve is pure math and a faithful optimization must leave it
*unchanged*. A wall-clock run is indexed by seconds, so its curve is *supposed* to move.
`compare()` therefore checks the first for equality and the second for throughput; checking
both the same way would either flag every successful optimization as a regression or let a
changed algorithm pass as a speedup. Comparing iterations across traversals is the same
category error: one MCCFR iteration is a single sampled path, one exact iteration is the
whole tree.

Results carry provenance — interpreter, numpy, machine, commit, and whether the tree was
dirty — because a timing number without the machine it ran on is not a measurement, and a
"reproduce this" pointing at a commit is worthless if the tree had uncommitted edits.

## Microstructure: modeling assumptions

Stated plainly because a reader will ask.

- CFR's guarantees and exploitability hold for **two-player zero-sum** games. The maker and
  the aggregate trader are therefore modeled as exactly two players. Real markets are
  neither two-player nor zero-sum.
- The CFR market maker is a **profit maximizer**, not Glosten–Milgrom's **competitive**
  zero-profit maker. Both benchmarks are computed and reported; neither is presented as the
  other. The strategic maker quotes strictly wider.
- Value, quote, and reservation grids are discretized to keep the action set finite.
- The uninformed trader is mechanical, so it is modeled as a chance node rather than a
  player — and that chance node sits *after* the quote, which keeps the reservation out of
  the root and the tree small.
- **Kyle is not a CFR game at all.** Its maker is competitive, and a strategic one has no
  interior optimum. It is solved by fixed-point iteration in `analysis/kyle.py`. Choosing
  the algorithm to fit the model, rather than the reverse, is deliberate.

## Extending it

### Add a game

1. Implement `GameState` and `Game` in `src/gto_solver/games/`. Use `leduc.py` as the
   reference — it is the non-trivial one.
2. Make chance a node. Get the outcome probabilities right, especially with duplicate
   outcomes (Leduc's deck has two of each rank).
3. Make `payout(player)` exactly antisymmetric for two-player zero-sum. `exploitability`
   walks every terminal node asserting payouts sum to zero, so violations surface loudly.
4. Test info-set indistinguishability directly: states differing only in hidden information
   must produce equal keys.
5. No solver change should be needed. **If you find yourself editing anything under
   `solvers/` or `metrics/` to make a game work, that is a finding about the abstraction —
   investigate it rather than working around it.** Leduc needed zero such changes, which is
   the evidence the interface is real.

### Add an algorithm variant

Add a `RegretUpdateRule` to `solvers/regret_rules.py` or a `Traversal` to
`solvers/traversal.py`, then register the combination in `solvers/registry.py` so the
benchmark, and anything else that names a variant in a string, can find it. Do not fork the
engine. New traversals must take all randomness from the passed-in `rng` so
`CFRSolver(..., seed=)` stays reproducible.

Set `deterministic` on the registry entry honestly: it decides whether the benchmark runs
one seed or twenty, and it is checked by running the seeds, so a wrong value fails a test
rather than silently publishing a stochastic variant as a single run.

Validate with exploitability trending to zero — and report what you measure. Two variants
here (DCFR, alternating updates) do *not* beat the baseline on small games, and that is
recorded rather than tuned away.

### Add a benchmark

Analytical or brute-force benchmarks live in `src/gto_solver/analysis/`. The rule that makes
them worth anything: **a benchmark must share no code with the solver it checks.** The
Glosten–Milgrom benchmark is an independent exhaustive grid search, which is why the solver
matching it is evidence rather than a tautology.

*Performance* benchmarks are different work and live in `src/gto_solver/benchmark/`. Add a
`Suite` to `benchmark/suites.py` — a game, a list of algorithm names, and the checkpoints to
measure at — and `scripts/benchmark.py` picks it up. Anything stochastic gets at least ten
seeds; a suite that breaks that rule fails a test. Do not measure timings from a script of
your own: everything published goes through `benchmark/runner.py` so a later phase can
compare against it.

## Layout

```
src/gto_solver/
├── games/          base.py, kuhn.py, leduc.py, glosten_milgrom.py
├── solvers/        base.py, regret_rules.py, traversal.py, registry.py
├── analysis/       microstructure.py (GM benchmarks), kyle.py (fixed-point solver)
├── metrics/        exploitability.py, evaluation.py
└── benchmark/      stats.py, runner.py, results.py, suites.py, tables.py, plots.py
tests/              one file per module, plus test_microstructure_gate.py
docs/               BUILDLOG.md, ARCHITECTURE.md, phase4-microstructure-design.md
results/            benchmark results as JSON, with provenance
scripts/            verify_phase4.py (regenerates the design doc's numbers),
                    benchmark.py (runs the suites, writes results and charts)
main.py             Kuhn demo entry point (a real CLI is Phase 7)
```

`benchmark/plots.py` is the one module that needs an optional dependency (matplotlib, via
the `viz` extra). It is deliberately not imported from `benchmark/__init__.py`, so importing
the package — or running the tests — never requires it.

`tests/test_microstructure_gate.py` is the cross-cutting one: it checks the solver, the
game, and the independent benchmark all agree. That is the test the microstructure phase
exists to pass.
