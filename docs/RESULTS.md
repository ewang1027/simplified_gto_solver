# Results

Everything this project measured, organized by what it claims rather than by when it was
found. `docs/BUILDLOG.md` has the chronology and the traps; this has the findings.

Every number here comes from a file in `results/`, each measured from a clean tree and
carrying the commit, machine and seeds behind it. `tests/test_docs.py` checks that the
tables in `README.md` still regenerate from those files, and
`scripts/audit_doc_numbers.py` re-measures the machine-specific claims.

## How to read these numbers

**Correctness is exploitability**, not agreement with a known constant. Exploitability is
what a best-responding opponent could win against a strategy profile: ≥ 0 everywhere, and
exactly 0 at a Nash equilibrium. Kuhn's published −1/18 is checked too, but only as a
secondary sanity check — it cannot validate a game nobody has solved, which is every game
this project actually cares about.

**Every sampled result is a median over fixed seeds with a band**, never a single run. The
band is always the 10–90% *seed envelope* — how much individual runs differ — and never a
confidence interval, which is a narrower thing answering a different question. Deterministic
variants are run once, because their traversal never reads the rng and twenty seeds would be
twenty identical curves.

**Name the axis.** Iterations, seconds and traversals are three different axes and this
project got the distinction wrong three times before writing it down. See
`docs/ARCHITECTURE.md` for the full statement; the short version is that comparing update
*rules* on iterations is fair, comparing *traversals* on iterations is not, and any
wall-clock comparison is a statement about an implementation rather than an algorithm.

## Correctness

| What | How it was established | Result |
|---|---|---|
| CFR converges on Kuhn | Exploitability over 13 checkpoints to 100,000 iterations | 0.000633 |
| The value is right | Average strategy against the published −1/18 | matches to ±0.01 |
| Leduc info sets | Enumerated from the tree | exactly 288, the published figure |
| The abstraction generalizes | Leduc added *after* the solvers were written | zero changes to any solver or metrics file |
| Best response is per info set | Kuhn's `Q|cb`, whose two member nodes have opposite locally optimal actions | per-node and per-info-set maximization genuinely disagree there |
| Glosten–Milgrom is solved | Exhaustive grid search sharing no code with the solver | exact match at all 5 μ |
| Kyle is solved | Five closed forms, plus λ recovered independently by regression | exact; 0.038% regression error at 200k draws |

The gap in that table, found in Phase 10 and still open: **Leduc has no external reference at
all.** Its 26 tests check structure and that exploitability falls, but nothing compares its
solved strategy or game value to anything outside this repository.

## Algorithm comparisons

### CFR+ wins on Kuhn and loses on everything else

The most interesting result here, and the one that took the most work to trust.

| Game | Axis | CFR+ against vanilla |
|---|---|---|
| Kuhn | iterations, to 100,000 | **wins at all 13 checkpoints** (0.000614 vs 0.000633) |
| Leduc | iterations, to 5,000 | leads at 10, 22, 47; **loses from 103 on**, ending 2.2× behind |
| Leduc | wall-clock, to 20 s | loses at 5 of 6 budgets, ties at 1 s |
| Glosten–Milgrom | iterations, to 3,000 | leads at 10 and 23; **loses from 51 on**, ending 8.3× behind |

Two explanations were tested and ruled out before this was published:

- **Not the update schedule.** Published CFR+ alternates; this suite ran simultaneous
  updates. So `cfr_plus_alternating` was added to both non-Kuhn suites — and it is *worse
  still*, 5.4× behind vanilla at 5,000 Leduc iterations against simultaneous CFR+'s 2.2×.
- **Not a slow start.** The 20-second budget fits only ~600 exact Leduc iterations, and a
  rule that merely started slowly would look identical. `leduc_convergence` runs to 5,000 on
  the axis the literature uses; the gap widens rather than closing.

**Status: unexplained.** It wants a line-by-line review of `CFRPlusRegretMatching` against
Tammelin 2014, and an independent Leduc benchmark, not another measurement.

### DCFR loses to vanilla at its published defaults

At α=1.5, β=0, γ=2 on Kuhn, DCFR is worse than vanilla at **9 of 13 checkpoints**, including
the last (0.000784 against 0.000633). It wins at 46, 100, 1,000 and 21,544 iterations and
loses everywhere else — scattered, not "early". Those defaults were tuned on far larger
games. The result survived a correctness fix to the discount order, so it is a finding
rather than a bug.

### Sampling needs 21.5× more iterations, and still loses on Leduc

On Kuhn, MCCFR's median first reaches vanilla's 100-iteration exploitability at 2,154
iterations, and its 1,000-iteration value at 21,544 — **the same 21.5× factor at both
horizons**. Vanilla at 10,000 is never matched within 100,000 MCCFR iterations.

On Leduc at equal wall clock, exact traversal wins at every budget, and not only on the
median: **every one of the ten MCCFR seeds at 20 s is worse than the single exact run** (best
sampled 0.02869 against exact 0.01159). Sampling completes 83× more iterations per second
and still loses.

That 83× was 211× before Phase 6, which is the caveat: a wall-clock comparison between
traversals moves when either implementation changes.

### The CFR+ rule under sampling is not distinguishable at 20 seeds

`mccfr_plus` — the CFR+ rule composed with external sampling, which neither paper defines
and which the design gives away for free — beats plain MCCFR between 46 and 1,000 Kuhn
iterations and trails after. At 100,000 the two medians' 95% bootstrap CIs overlap
([0.00203, 0.00317] against [0.00274, 0.00386]), so **twenty seeds do not resolve which is
better** and neither ordering is claimed. It does carry visibly less seed variance: 2.11×
spread against 5.16×.

### Deep CFR loses to both, once the axis is read correctly

| | Exploitability at 200 iterations |
|---|---|
| Vanilla (exact) | 0.011947 |
| Deep CFR (median of 10 seeds) | 0.035639 |
| MCCFR (median of 10 seeds) | 0.067597 |

At equal iterations Deep CFR appears to beat MCCFR by 2–3×. **It does not.** One Deep CFR
iteration runs 30 sampled traversals per player where MCCFR runs one, so equal iterations is
not equal work. Normalized by traversals against MCCFR's own published curve, MCCFR is
2.1–3.9× ahead — at 200 Deep CFR iterations (12,000 traversals) Deep CFR sits at 0.0356
where MCCFR at 6,000 iterations is at 0.0103.

So Deep CFR loses to tabular CFR *and* to MCCFR here, which is the design working: it exists
for games whose info sets cannot be enumerated, and on twelve of them approximating a table
is worse than being one. It does win one thing — a narrower seed envelope than MCCFR, 1.36×
against 1.87×, because the network averages over many samples per info set instead of
counting visits to each separately.

### One run is not a result

At 100,000 iterations MCCFR's twenty Kuhn seeds span **0.000807 to 0.004165** around a median
of 0.002711 — the luckiest seed 5.16× better than the unluckiest. Every sampled number this
project published before Phase 5 was a single run, and could have landed anywhere in that
range.

## Performance

Phase 6 optimized the hot loop under one gate: **throughput may move, per-iteration
convergence curves may not.** They did not — maximum exploitability delta `0.000e+00` on
every run of every convergence suite, across 7 algorithms, 13 checkpoints and 20 seeds.

| Game | vanilla | cfr_plus | alternating | dcfr | linear | sampled |
|---|---:|---:|---:|---:|---:|---:|
| Kuhn | 2.03× | 1.98× | 1.99× | 1.73× | 1.79× | 0.97–1.03× |
| Leduc | 2.78× | 2.60× | 2.71× | — | — | 1.04–1.09× |
| Glosten–Milgrom | 3.14× | 3.16× | 3.26× | — | — | 1.05× |

Three changes, found by profiling rather than guessed: counterfactual reach was allocating an
array per decision node to drop one element from three; `payout` was called once per player
where a zero-sum game computes the same quantity both times; and the tree was re-derived
every iteration when a `GameState` is immutable and none of it can change.

**The speedup grows with the game** — 2.0× on Kuhn to 3.1× on Glosten–Milgrom — because the
more expensive a game's own methods are, the more there is to stop recomputing. **Sampling
barely moves**, by design: caching a tree would remove external sampling's reason to exist.

An independent confirmation fell out for free: of the seven committed chart PNGs, **six came
back byte-identical** after a full re-measurement on optimized code, and only the wall-clock
one — whose axis is seconds — changed.

The wall-clock noise floor on this machine is **3.1% relative standard deviation** (ten seeds
training for exactly 20 s completed 129,913 to 143,891 iterations). A throughput claim
smaller than that is machine drift.

## Market microstructure

**Glosten–Milgrom is solved by CFR.** The solver recovers the profit-maximizing quote exactly
at every μ tested, against an exhaustive grid search that shares no code with it:

| μ | CFR spread | Brute force | Competitive |
|---:|---:|---:|---:|
| 0.02 | 1.6250 | 1.6250 | 0.0306 |
| 0.30 | 1.8750 | 1.8750 | 0.4937 |
| 0.70 | 2.5000 | 2.5000 | 1.3143 |

The spread widens with informed flow — adverse selection, recovered from self-play rather
than assumed. Two makers appear on purpose: the competitive Glosten–Milgrom maker earns zero
profit by construction, and the CFR maker maximizes, so it quotes strictly wider. Neither is
presented as the other.

**Kyle is deliberately not solved by CFR.** A strategic maker's PnL is `λσu² − S0/(4λ)`,
strictly increasing with no interior optimum, and exactly zero at the equilibrium λ — the
zero-profit condition, not an optimum. Kyle is not two-player zero-sum, so it gets an
iterated best-response fixed-point solver validated against five closed forms. Choosing the
algorithm to fit the model rather than the reverse is the point.

Exact CFR drives the microstructure game to **0.000113** exploitability at 3,000 iterations
over its 298 info sets.

## Claims this project got wrong

Kept visible, because a corrected record is worth more than a clean one. Each was caught by a
later measurement rather than by review.

| Claim | What it actually was | Caught by |
|---|---|---|
| MCCFR needs 10–15× more iterations | **21.5×**, agreeing at two horizons | Phase 5, median over 20 seeds; the old figure was one run |
| DCFR "wins early, worse around 10k" | Wins at 4 scattered checkpoints of 13 | Phase 5, a finer grid |
| "Exact beats sampling at every budget" | True of *vanilla*; MCCFR beats alternating CFR+ from 5 s | Phase 6, re-reading the table |
| Deep CFR beats MCCFR by 2–3× | Loses by 2.1–3.9× once normalized by traversals | Phase 9, before publishing |
| "Measuring costs 56 ms against a 31 ms iteration" | 58 ms against **11 ms** — five iterations, not two | Phase 10; Phase 6's own optimization had invalidated it |
| Noise floor 118,797–135,051 iterations | 129,913–143,891 after re-measurement | Phase 10; the 3.1% conclusion held |
| Multi-round GM at R=2 is ~90k terminals | **166,617** — 80× the single-round game | Phase 10, checking the arithmetic |
| Binary asset values are universally degenerate | False; they fail only when `|V| ≤ T/2` | Phase 4 |
| The diffuse maker strategy is not convergence-related | False; it concentrates with more training | Phase 4 |

Two patterns in that list are worth naming. **Single runs were wrong twice** — both the MCCFR
factor and the DCFR shape came from one measurement and did not survive seeds and a finer
grid. And **an optimization can invalidate a document**: Phase 6 made an iteration cheaper
without making measurement cheaper, and a sentence comparing the two silently stopped being
true. Nothing failed, because nothing checked. That is why `tests/test_docs.py` and
`scripts/audit_doc_numbers.py` exist.

## Regenerating all of it

```bash
gto benchmark                                       # the two default suites, ~10 minutes
gto benchmark --suite gm_convergence --suite leduc_convergence --suite kuhn_deep_cfr
gto microstructure                                  # the spread table above
python scripts/verify_phase4.py                     # every number in the design doc
python scripts/audit_doc_numbers.py                 # the machine-specific claims
pytest                                              # 534 tests, including the doc checks
```

Results land in `results/` with provenance. `kuhn_deep_cfr.json` was measured at a later
commit than the other four, which is why its provenance differs; each file records its own.
