# Build log

Running progress log so this project can be resumed cold. Newest phase last.
For *why* the code is shaped the way it is, see `ARCHITECTURE.md`. For the Phase 4
modeling work specifically, see `phase4-microstructure-design.md`.

## Resuming in one minute

```bash
cd ~/simplified_gto_solver
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest          # 419 tests, ~40s
ruff check .
python main.py  # Kuhn demo: exploitability + solved strategies

python scripts/benchmark.py --quick   # the benchmark harness, in seconds
python scripts/benchmark.py           # the published suites, about 13 minutes
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

Phases 1–5 of 10 complete. Phase 6 (performance engineering) is next.

| Phase | Work | Status |
|---|---|---|
| 1 | Game/solver abstractions, exact CFR, exploitability, packaging + CI | done |
| 2 | CFR+, Discounted/Linear CFR, external-sampling MCCFR | done |
| 3 | Leduc Hold'em — validates the abstraction generalizes | done |
| 4 | Market microstructure: Glosten–Milgrom + Kyle | done |
| 5 | Multi-seed benchmarking, confidence bands, convergence plots | done |
| 6 | Performance engineering — profiling, optimized hot loop, published throughput | next |
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
- CFR+ beats vanilla at every horizon, matching the literature. *(Phase 5 re-measured this
  on 13 checkpoints: still true on Kuhn at all 13 — and false on both other games, where
  CFR+ crosses over and ends behind. See Phase 5.)*
- **DCFR does not.** It wins early but is worse than vanilla around 10k iterations on
  Kuhn. Its published defaults (α=1.5, β=0, γ=2) were tuned on far larger games. This
  survived a correctness fix, so it is a real finding, not an implementation bug.
  *(Phase 5 refined ~~"wins early"~~: on a 13-checkpoint grid DCFR wins at 46, 100, 1,000
  and 21,544 iterations and loses at the other nine, so it is scattered rather than early.
  The headline — worse than vanilla overall, including at 100k — stands.)*
- Alternating updates show no consistent win on games this small. *(Phase 5, on larger
  games: not merely no win. Alternating CFR+ is worse than vanilla at every checkpoint on
  both Leduc and Glosten–Milgrom.)*
- ~~MCCFR needs roughly 10–15× more iterations for comparable accuracy on Kuhn.~~
  **Wrong — it needs 21.5×.** That figure came from a single run; Phase 5 measured the
  median over 20 seeds at two horizons, which agree on 21.5×.

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
than on Kuhn — the expected trend as trees grow. *(Phase 5 re-ran this over 10 seeds and six
budgets: it holds at every budget, and every individual seed loses to exact, not just the
median. Numbers moved slightly with the machine — 0.02125 against a sampled median of
0.03666 — and the throughput ratio measured 211×.)*

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

## Phase 5 — benchmarking (2026-08-23)

The harness landed first, ahead of the numbers, because Phase 6 has to measure against it:
an optimization claim is only worth something if before and after came from the same
measurement path.

What exists now, all green (`ruff check .`, 375 tests):

- `solvers/registry.py` — the seven named (regret rule x traversal) variants in one place,
  each carrying a `deterministic` flag. Includes `mccfr_plus`, the CFR+ rule composed with
  external sampling, which no paper defines and which the design gives away for free.
- `benchmark/stats.py` — seed envelope vs bootstrap CI, kept apart deliberately. The
  envelope says how much runs differ; the CI says how well N seeds pin the centre down.
- `benchmark/runner.py` — `run_convergence` (exploitability vs iterations) and
  `run_wallclock` (exploitability vs seconds), plus `verify_determinism`.
- `benchmark/results.py` — JSON with provenance, and `compare()` for Phase 6.
- `benchmark/{suites,tables,plots}.py` and `scripts/benchmark.py`.

**Three measurement rules are enforced in code, not in comments:**

1. *Evaluating exploitability is never charged to the solver's clock.* Measuring costs 56 ms
   on Leduc against a 31 ms iteration, so charging it would make whichever run was measured
   more often look slower — the exact-vs-sampled result would be a measurement artifact.
   Pinned by a test that runs identical work at 1 and at 4 checkpoints and requires the
   throughputs to agree.
2. *Deterministic variants run once, not N times.* `FullTraversal` never reads the rng, so
   twenty seeds give twenty identical curves and a band of width zero. The flag claiming
   this is verified by actually running the seeds, in both directions — exact variants must
   be bit-identical, sampled ones must not be.
3. *Nothing is capped, reduced or skipped in silence.* Every reduction lands in `notes`,
   which is serialized with the results and printed by the script.

**Measured while building it:** at n ≤ 20 seeds with 10,000 resamples, the bootstrap
interval is **bit-identical across bootstrap seeds** — the resample distribution of a median
over so few points is discrete enough that its tail percentiles land on the same order
statistics every time. Wobble appears around n = 50 (~13% of interval width). So the
arbitrary default bootstrap seed is not quietly steering any published interval, which was
the obvious thing to worry about. Pinned in `tests/test_benchmark_stats.py`.

### What the suites measured

Four suites, in `results/*.json`, each with the commit and machine it was measured on.

**The harness reproduced every previously published number exactly** — vanilla, CFR+, DCFR
and Linear CFR at 100/1k/10k/100k on Kuhn are identical to the Phase 2 table, and a repeat
run reproduced the sampled curves bit for bit as well, seeds being fixed. Only timings move
between runs.

**Kuhn, 13 checkpoints, 20 seeds for the sampled variants:**

- CFR+ beats vanilla at **13 of 13** checkpoints.
- **DCFR loses at 9 of 13**, including the last (0.000784 against 0.000633). Phase 2 recorded
  this as "wins early but worse around 10k"; on a finer grid it is not that tidy — DCFR wins
  at 46, 100, 1,000 and 21,544 iterations and loses everywhere else. ~~"wins early"~~ →
  *wins at four scattered checkpoints*.
- **MCCFR needs 21.5× more iterations, not the 10–15× recorded in Phase 2.** Two horizons
  agree on the factor: vanilla at 100 iterations is first matched by MCCFR's median at 2,154,
  and vanilla at 1,000 at 21,544. Vanilla at 10,000 is never matched inside 100,000 MCCFR
  iterations. The old figure came from a single run; this one is a median over 20 seeds.
- **Seed luck on Kuhn is worth 5.16×.** At 100,000 iterations the seeds span 0.000807 to
  0.004165 around a median of 0.002711. Every sampled figure published before this phase was
  a single run and could have been anywhere in that band.
- `mccfr_plus` (CFR+ rule under external sampling) helps from 46 to 1,000 iterations and
  trails after, but at 100,000 the two medians' 95% CIs overlap — [0.00203, 0.00317] against
  [0.00274, 0.00386] — so **20 seeds do not resolve which is better**, and neither ordering
  is claimed. It does carry less seed variance: 2.11× spread against 5.16×.

**Leduc, 10 seeds, six wall-clock budgets:**

- Exact traversal beats sampling at **every budget**, and not only on the median: every one
  of the ten MCCFR seeds at 20 s is worse than the single exact run (best sampled 0.02810
  against exact 0.02125). The Phase 3 claim survives the multi-seed treatment.
- Sampling completes **211× more iterations per second** and still loses.

**The finding that took the most work to trust: CFR+ wins on Kuhn and loses on both other
games.**

- On Leduc it trails vanilla at five of six wall-clock budgets and ties at 1 s (0.13548
  against 0.13583).
- The obvious explanation was the update schedule, since the published algorithm alternates
  and this suite did not. So `cfr_plus_alternating` was added to both non-Kuhn suites before
  anything was written down. **Alternating is worse still** — 5.36× vanilla at 5,000 Leduc
  iterations against simultaneous CFR+'s 2.17×.
- The second explanation was that 20 s only buys ~600 exact Leduc iterations, and a rule that
  merely starts slowly would look identical. So `leduc_convergence` was added, out to 5,000
  iterations on the axis the literature uses. **CFR+ leads for the first three checkpoints,
  crosses over between 47 and 103, and ends 2.2× behind.** The same shape appears on
  Glosten–Milgrom, crossing between 23 and 51 and ending 8.3× behind at 3,000.
- So it is neither the schedule nor a slow start. It is recorded as measured. It is also
  **unexplained**, and published CFR+ results run far past 5,000 iterations, so this is a
  candidate for a correctness review — not a general claim about CFR+.

**Glosten–Milgrom:** exact CFR reaches 0.000113 exploitability at 3,000 iterations over 298
info sets, at 183 it/s. The microstructure game goes through the same harness as the poker
ones, unchanged.

### Traps hit while building the harness

1. *The benchmark dirtied its own tree.* Each suite writes its results and charts into the
   repo as it finishes, so the first suite recorded a clean tree and every suite after it
   recorded a dirty one — dirtied by the run in progress. Found by the guard test written one
   commit earlier, on its first real run. Paths under `results/` and `docs/images/` are now
   excluded; a changed source file still counts. An earlier run was discarded outright rather
   than relaxing the check, because a benchmark measured against uncommitted edits cannot be
   reproduced from the commit it names.
2. *A `.strip()` shifted every recorded path one character left.* `git status --porcelain`
   puts two status columns and a space before each path, so the first line begins with a
   space whenever the index is clean — and stripping the output ate exactly that space.
   Caught by the test for the fix above.
3. *Decade-marker table rows collapsed a table to one row.* Selecting powers of ten keeps a
   thirteen-row Kuhn table readable, but the microstructure suite's eight log-spaced
   checkpoints contain exactly one power of ten. A selection that sparse now falls back to
   showing everything.

### Measured, and useful to Phase 6

- **The wall-clock noise floor on this machine is about 3%.** Ten MCCFR seeds each training
  for exactly 20 s completed 118,797 to 135,051 iterations — a 1.14× spread, 3.2% relative
  standard deviation (`mccfr_plus`, run later in the same session, spread only 1.03×). **A
  throughput claim smaller than that is not distinguishable from machine drift**, so Phase 6
  should not report one without repeating the measurement.
- **At n ≤ 20 seeds with 10,000 resamples the bootstrap interval is bit-identical across
  bootstrap seeds** — the resample distribution of a median over so few points is discrete
  enough that its tail percentiles land on the same order statistics every time. Wobble
  appears near n = 50 (~13% of interval width). The arbitrary default bootstrap seed is
  therefore not steering any published interval.
- Exact traversal on Leduc runs at ~31 it/s and sampling at ~6,500; on Kuhn, ~6,500 and
  ~30,000. Those are the numbers Phase 6 has to move.

---

## Next up: Phase 6

Performance engineering: profile the hot loop, optimize it, and publish before/after
throughput.

The workflow the harness was built for:

```bash
python scripts/benchmark.py                                   # baseline, on a clean tree
cp -r results results-baseline
# ... optimize ...
python scripts/benchmark.py
python scripts/benchmark.py --compare results-baseline/kuhn_convergence.json \
                                     results/kuhn_convergence.json
```

`--compare` exits non-zero if any per-iteration convergence curve moved. That is the check
that matters: throughput is allowed to change, and the curve is not. Wall-clock curves are
compared on throughput instead, since those are *supposed* to move.

Specifics that matter:
- Measure on a clean tree. The provenance check will say so if you did not, and a comparison
  across different machines or interpreters is flagged rather than silently reported.
- Beat 3%, or repeat the measurement (see the noise floor above).
- Every stochastic result stays at **≥10 seeds with bands**; the published-results test
  enforces it, and refuses a file produced by `--quick`.

## Conventions

- Commit prefixes: `feat:`, `fix:`, `update:`, `refactor:`, `docs:`, `chore:`, `test:`, `wip:`.
- Every phase ends green: `ruff check .` and `pytest` both clean, then commit and push.
- Report measured results as measured. Several findings here are negative (DCFR losing to
  vanilla, MCCFR losing on small trees, Kyle not fitting CFR) and they stay that way.
