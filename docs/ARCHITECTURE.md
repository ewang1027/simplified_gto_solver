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
`solvers/traversal.py`. Do not fork the engine. New traversals must take all randomness from
the passed-in `rng` so `CFRSolver(..., seed=)` stays reproducible.

Validate with exploitability trending to zero — and report what you measure. Two variants
here (DCFR, alternating updates) do *not* beat the baseline on small games, and that is
recorded rather than tuned away.

### Add a benchmark

Analytical or brute-force benchmarks live in `src/gto_solver/analysis/`. The rule that makes
them worth anything: **a benchmark must share no code with the solver it checks.** The
Glosten–Milgrom benchmark is an independent exhaustive grid search, which is why the solver
matching it is evidence rather than a tautology.

## Layout

```
src/gto_solver/
├── games/          base.py, kuhn.py, leduc.py, glosten_milgrom.py
├── solvers/        base.py, regret_rules.py, traversal.py
├── analysis/       microstructure.py (GM benchmarks), kyle.py (fixed-point solver)
└── metrics/        exploitability.py, evaluation.py
tests/              one file per module, plus test_microstructure_gate.py
docs/               BUILDLOG.md, ARCHITECTURE.md, phase4-microstructure-design.md
main.py             Kuhn demo entry point (a real CLI is Phase 7)
```

`tests/test_microstructure_gate.py` is the cross-cutting one: it checks the solver, the
game, and the independent benchmark all agree. That is the test the microstructure phase
exists to pass.
