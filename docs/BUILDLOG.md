# Build log

Running progress log so this project can be resumed cold. Newest phase last.
For *why* the code is shaped the way it is, see `ARCHITECTURE.md`. For the Phase 4
modeling work specifically, see `phase4-microstructure-design.md`.

## Resuming in one minute

```bash
cd ~/simplified_gto_solver
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest          # 230 tests, ~33s
ruff check .
python main.py  # Kuhn demo: exploitability + solved strategies
```

CI runs `ruff check .` then `pytest` on Python 3.11 and 3.12 for every push.

## What this project is

A from-scratch CFR solver for two-player zero-sum imperfect-information games, aimed at
market microstructure. Poker is the *validation harness* — Kuhn and Leduc have known
equilibria, so they prove the solver is right. The point is the microstructure work, where
a market maker quotes against possibly-informed order flow.

Correctness is measured by **exploitability**, never by comparing a game value to a known
constant. That choice is load-bearing: the microstructure games have no published value to
compare against.

## Status

Phases 1–4 of 10 complete. Phase 5 (benchmarking + plots) is next.

| Phase | Work | Status |
|---|---|---|
| 1 | Game/solver abstractions, exact CFR, exploitability, packaging + CI | done |
| 2 | CFR+, Discounted/Linear CFR, external-sampling MCCFR | done |
| 3 | Leduc Hold'em — validates the abstraction generalizes | done |
| 4 | Market microstructure: Glosten–Milgrom + Kyle | done |
| 5 | Multi-seed benchmarking, confidence bands, convergence plots | next |
| 6 | Performance engineering — profiling, optimized hot loop, published throughput | |
| 7 | CLI (typer) | |
| 8 | Interactive dashboard (Streamlit) | |
| 9 | Deep CFR — neural regret approximation vs tabular ground truth | |
| 10 | Architecture writeup and docs polish | |

Stretch, unscheduled: NFSP (Neural Fictitious Self-Play).

---

## Phase 1 — foundation (2026-08-21)

Rewrote a flat four-module Kuhn CFR script (`cfr.py`, `kuhn.py`, `main.py`, `test.py`) into a package built on two abstractions
(`GameState`, and `RegretUpdateRule` × `Traversal`). Added exploitability, pytest, ruff,
CI, and a `src/` layout.

Key changes from the original script:
- The card deal became a **chance node inside the tree** rather than deals enumerated by
  the training loop with `itertools.permutations`. Required for everything after.
- Regret is stored with a **per-info-set action count**, not a global `num_actions = 2`.
- Counterfactual regret uses the general per-player formulation instead of the original's
  `sign = 1 if player == 0 else -1` hack.

**Traps hit:** two tests I wrote were wrong, not the solver. Exploitability is *not*
monotonic per iteration (CFR bounds average regret at O(1/√T); there's a transient bump
around iteration 2). And the solver's per-iteration return value is the *current*
regret-matching strategy, which oscillates — the **average** strategy is what converges.
That second one is why `metrics/evaluation.py` exists.

## Phase 2 — algorithm breadth (2026-08-21)

CFR+, Discounted/Linear CFR, external-sampling MCCFR, and optional alternating updates.
All compose from the Phase 1 interfaces; no new engine.

**Measured results, kept as measured:**
- CFR+ beats vanilla at every horizon, matching the literature.
- **DCFR does not.** It wins early but is worse than vanilla around 10k iterations on
  Kuhn. Its published defaults (α=1.5, β=0, γ=2) were tuned on far larger games. This
  survived a correctness fix, so it is a real finding, not an implementation bug.
- Alternating updates show no consistent win on games this small.
- MCCFR needs roughly 10–15× more iterations for comparable accuracy on Kuhn.

**Correctness fix:** DCFR's discount order initially added the new regret and then
discounted everything. Brown & Sandholm discount the *prior* accumulator and add the
current iteration's contribution undiscounted. Fixed; conclusions unchanged.

## Phase 3 — Leduc Hold'em (2026-08-21)

The test of whether the abstraction generalizes or was merely Kuhn-shaped. Leduc has a
different deck, two betting rounds, two chance points, ties (Kuhn has none), and a
per-node action count that varies.

**It required zero changes to any solver or metrics file**, and runs on all four algorithm
variants unmodified. Info-set count is exactly 288, matching the published figure.

Also measured: on Leduc, MCCFR runs ~210× more iterations/sec than exact traversal, but
exact still wins on an equal 20-second budget (0.0233 vs 0.0360). The gap is far narrower
than on Kuhn — the expected trend as trees grow.

## Phase 4 — market microstructure (2026-08-22)

The centerpiece. Designed and verified numerically **before** writing game code, which was
worth it: two of the three obvious formulations are degenerate.

**Glosten–Milgrom is solved by CFR.** The solver recovers the profit-maximizing quote
exactly at all nine μ tested, exploitability ≈ 0.002, spread widening with informed flow.
The benchmark is an exhaustive grid search sharing no code with the solver.

**Kyle is deliberately not solved by CFR.** A strategic maker's PnL is `λσu² − S0/(4λ)`,
strictly increasing with no interior optimum, and exactly zero at the equilibrium λ — the
zero-profit condition, not an optimum. Kyle is genuinely not two-player zero-sum, so it
gets an iterated best-response fixed-point solver validated against five closed forms.

**Traps hit — read these before touching Phase 4 code:**
1. *Grid interiority is not a viability test.* A parameter sweep produced a beautiful
   result (spread growing 20× with μ) that was entirely spurious: maker PnL was exactly
   zero because it had stopped trading. That corner sits strictly *inside* the quote grid.
   Viability requires positive maker profit **and** both trader types actually trading —
   see `market_functions()`.
2. *Binary values are not universally degenerate.* An earlier claim that they were is
   wrong. Binary fails outright only when `|V| ≤ T/2`; above that the spread responds to μ
   but **saturates at `|V|`**. Both cases are pinned by tests.
3. *Single-round GM has a dominant-strategy trader*, so it is an optimization rather than a
   game. It is kept because it is what the benchmark validates. Multi-round is where the
   informed trader becomes genuinely strategic (trading reveals information).
4. *The solved maker strategy looks diffuse*, sitting at only ~0.24–0.33 probability on the
   best quote at the 150 iterations the gate test uses. The cause is a **flat payoff
   surface**: the second-best quote is worth between 5e-4 and 5e-3 less depending on μ, so
   there is very little regret pressure to concentrate. It does concentrate with more
   training (μ=0.7 reaches 0.94 by 4,000 iterations), and concentrates fastest exactly where
   the payoff gap is largest — which is the flatness explanation confirming itself. The
   argmax is correct throughout, which is what the gate checks. An earlier version of this
   note claimed the diffuseness was *not* convergence-related at all; that was wrong.

---

## Next up: Phase 5

Multi-seed benchmarking with confidence bands, and the convergence plots for the README.

Specifics that matter:
- MCCFR and any future stochastic solver must be reported over **≥10 seeds with confidence
  bands**, never a single run. Seeds are fixed and recorded via `CFRSolver(..., seed=)`.
- Charts should show the algorithm comparison (exploitability vs iterations, log-log) and
  the exact-vs-sampled wall-clock comparison.
- Do not silently truncate: if a benchmark caps iterations or drops a configuration, say so
  in the output.

Phase 6 (performance) depends on Phase 5's harness existing, because every optimization
claim needs before/after numbers from the same measurement path, plus a check that
exploitability curves are unchanged after optimizing.

## Conventions

- Commit prefixes: `feat:`, `fix:`, `update:`, `refactor:`, `docs:`, `chore:`, `test:`, `wip:`.
- Every phase ends green: `ruff check .` and `pytest` both clean, then commit and push.
- Report measured results as measured. Several findings here are negative (DCFR losing to
  vanilla, MCCFR losing on small trees, Kyle not fitting CFR) and they stay that way.
