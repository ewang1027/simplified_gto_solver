# Simplified GTO Solver

A from-scratch **Counterfactual Regret Minimization (CFR)** engine for extensive-form games
with imperfect information, validated on poker and aimed at market microstructure.

Solving poker is the *means*, not the end. Kuhn and Leduc poker have known equilibria, so
they serve as a correctness harness for the solver. The destination is a **Glosten–Milgrom
market-making game**, where a market maker quotes against possibly-informed order flow —
the same asymmetric-information problem CFR solves, and the mechanism behind adverse
selection and bid-ask spreads.

## Status

Phases 1–5 of 10 complete: the engine, seven CFR variants, two poker games, the
market-microstructure centerpiece, and a multi-seed benchmarking harness. See
[Roadmap](#roadmap) for what's next.

## Market microstructure

A market maker quoting against possibly-informed order flow is playing the same
asymmetric-information game CFR solves. Two canonical models are implemented, and they
needed **different tools** — which is the most interesting result here.

### Glosten–Milgrom: solved by CFR

The solver recovers the profit-maximizing quote exactly. The benchmark is an exhaustive
search over the quote grid that shares no code with the solver, so agreement is evidence
rather than a tautology:

| μ (informed share) | CFR spread | Brute-force optimum | Competitive GM spread |
|---:|---:|---:|---:|
| 0.02 | 1.6250 | 1.6250 | 0.0306 |
| 0.10 | 1.6250 | 1.6250 | 0.1561 |
| 0.30 | 1.8750 | 1.8750 | 0.4937 |
| 0.50 | 2.1250 | 2.1250 | 0.8699 |
| 0.70 | 2.5000 | 2.5000 | 1.3143 |

Exact match at every μ, with exploitability ≈ 0.002. The spread widens as informed flow
rises — adverse selection, recovered from self-play rather than assumed.

Two market makers appear in that table on purpose. The **competitive** GM maker earns zero
expected profit by construction, while CFR solves for a **profit-maximizing** one, which
quotes strictly wider. They answer different questions and neither is presented as the
other.

### Kyle (1985): deliberately *not* solved by CFR

Kyle's market maker is also competitive, and checking the numbers shows why that matters:
a strategic maker's PnL is `λ·σu² − S0/(4λ)`, **strictly increasing in λ without bound**,
with no interior optimum. Its PnL is exactly zero at the equilibrium `λ = √S0/(2σu)` — the
zero-profit condition, not an optimum. Kyle is genuinely not a two-player zero-sum game, so
it gets an iterated best-response **fixed-point solver** instead, validated against five
closed-form results:

| Quantity | Closed form | Recovered |
|---|---|---|
| Price impact | `λ = √S0 / (2σu)` | exact, from 6 parameter settings |
| Trading intensity | `β = σu / √S0` | exact |
| Product | `λβ = 1/2` | exact |
| Information revealed | exactly **half** the private signal | 0.5000 |
| Informed profit | `σu·√S0 / 2` | exact |

λ is also recovered *independently* by regressing simulated price on order flow (0.038%
error at 200k draws), which shares no algebra with the fixed-point solver.

Choosing the right algorithm per model — and saying plainly why CFR is wrong for one of
them — is the point of building both.

## Results so far

**Algorithm comparison** (Kuhn poker, exploitability after N iterations):

| Iterations | Vanilla CFR | CFR+ | DCFR | Linear CFR |
|-----------:|------------:|-----:|-----:|-----------:|
| 100 | 0.023301 | 0.015560 | 0.016391 | 0.017999 |
| 1,000 | 0.006505 | 0.005135 | 0.004416 | 0.008694 |
| 10,000 | 0.001486 | 0.000951 | 0.002033 | 0.003232 |
| 100,000 | 0.000633 | 0.000614 | 0.000784 | 0.000593 |

CFR+ beats vanilla at every horizon, as the literature reports. **DCFR does not** — it wins
early but is worse than vanilla around 10k iterations. Its published defaults
(α=1.5, β=0, γ=2) were tuned on far larger games; on a 12-info-set tree the aggressive early
discounting trades away long-run precision. That result is left as measured rather than
tuned away.

**Exact vs. sampled** (Leduc Hold'em, equal 20s wall-clock budget):

| Traversal | Iterations | Exploitability |
|-----------|-----------:|---------------:|
| Exact CFR | 610 | 0.02333 |
| External-sampling MCCFR | 129,200 | 0.03599 |

MCCFR runs ~210× more iterations per second but each is far noisier. Exact traversal still
wins on Leduc — though by much less than on Kuhn, which is the expected trend: sampling pays
off once the tree is too large to walk exhaustively.

## Why exploitability, not "does it match −1/18"

Kuhn poker's equilibrium value is a published constant (−1/18 ≈ −0.0556), so it's tempting
to call a solver correct when its average game value lands there. That check doesn't
generalize — it can only validate games somebody has already solved in closed form, which
the market-making game is not.

So correctness here is measured by **exploitability**: how much a best-responding opponent
could win against the solved strategy. It's ≥ 0 for every strategy profile and exactly 0 at
a Nash equilibrium, and it requires no prior knowledge of the game's value.

```
 Iteration   Exploitability
---------------------------
        10       0.09114384
       100       0.02330064
      1000       0.00650489
     10000       0.00148590
```

The subtle part is that a best responder **cannot** act differently in states it cannot
tell apart, so the maximization runs once per *information set*, not per tree node — a
naive per-node `max` silently lets the responder use hidden information and reports an
exploitability that is too high. See `src/gto_solver/metrics/exploitability.py`.

## Architecture

Two abstractions carry the project:

**`GameState`** (`games/base.py`) — an extensive-form game node. Chance events (the deal,
and later the asset-value and trader-type draws) are *nodes inside the tree*, walked by the
solver, rather than deals enumerated by the training loop. This is what lets one solver run
unmodified on Kuhn, Leduc, and the market-making game.

**`RegretUpdateRule` + `Traversal`** (`solvers/base.py`) — every CFR variant is one update
rule composed with one traversal, so vanilla CFR, CFR+, DCFR, and MCCFR share a single
tree-walking engine and a single regret store instead of being four forked files.

```
src/gto_solver/
├── games/
│   ├── base.py            GameState / Game interfaces (incl. chance nodes)
│   ├── kuhn.py            Kuhn poker — 12 info sets
│   ├── leduc.py           Leduc Hold'em — 288 info sets, two betting rounds
│   └── glosten_milgrom.py Market making under adverse selection
├── solvers/
│   ├── base.py            InfoSetStore, RegretUpdateRule, Traversal, CFRSolver
│   ├── regret_rules.py    Vanilla / CFR+ / Discounted (and Linear) CFR
│   ├── traversal.py       FullTraversal (exact, optional alternating updates),
│   │                      ExternalSamplingMCCFR
│   └── registry.py        The named (rule x traversal) variants, in one place
├── analysis/
│   ├── microstructure.py  Competitive + strategic GM benchmarks
│   └── kyle.py            Kyle (1985) fixed-point solver
├── metrics/
│   ├── evaluation.py      Expected value of a fixed strategy profile
│   └── exploitability.py  Best-response computation
└── benchmark/
    ├── stats.py           Seed envelopes and bootstrap confidence intervals
    ├── runner.py          Multi-seed convergence and wall-clock measurement
    ├── results.py         Serialization with provenance, plus before/after compare()
    ├── suites.py          The published benchmark suites
    ├── tables.py          The markdown tables below, regenerated from results
    └── plots.py           Charts (needs the optional viz extra)
```

Any regret rule composes with any traversal, so the four algorithms above are combinations
rather than separate implementations.

**The abstraction was tested, not assumed.** Leduc Hold'em was added *after* the solvers
were written — different deck, two betting rounds, two chance points, ties (which Kuhn has
none of), and a variable action count per node. It required **zero changes to any solver or
metrics file**, and runs on all four algorithm variants unmodified. Its info-set count comes
out to exactly 288, matching the published figure.

Regret is tracked per information set with a **per-info-set action count**, not a global
one — Leduc and the market-making game have different numbers of legal actions at different
nodes.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

python main.py    # train on Kuhn poker, report exploitability + strategies
pytest            # correctness suite (375 tests, ~39s)
ruff check .
```

Benchmarks. Charts need the optional `viz` extra (`pip install -e '.[viz]'`); results are
written whether or not it is installed.

```bash
python scripts/benchmark.py           # the published suites -- about 12 minutes
python scripts/benchmark.py --quick   # a smoke run in seconds; NOT the published numbers
python scripts/benchmark.py --list    # what suites exist
python scripts/benchmark.py --compare results/before.json results/after.json
```

Results land in `results/*.json` with provenance (interpreter, numpy, machine, commit, and
whether the tree was dirty), and the markdown tables below are pasted from what the script
prints. `--compare` is the before/after check an optimization has to pass: throughput may
move, per-iteration convergence curves may not.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Why the code is shaped this way, and how to add a game, an algorithm, or a benchmark |
| [`docs/BUILDLOG.md`](docs/BUILDLOG.md) | Phase-by-phase progress log, findings, and the traps worth knowing before touching a phase |
| [`docs/phase4-microstructure-design.md`](docs/phase4-microstructure-design.md) | The microstructure modeling work, including the formulations that failed and why |

## Kuhn poker rules

- 3-card deck: J, Q, K — K beats Q beats J
- 2 players, each antes 1 chip
- Player 0 acts first: check or bet (1 chip)
- Player 1 responds: check/fold or bet/call
- If Player 0 checked and Player 1 bet, Player 0 gets one more action
- Showdown goes to the higher card

| History | Outcome |
|---------|---------|
| check → check | Showdown, winner takes ±1 |
| bet → fold | Bettor wins +1 |
| check → bet → fold | Bettor wins +1 |
| bet → call | Showdown, winner takes ±2 |
| check → bet → call | Showdown, winner takes ±2 |

## How CFR works

CFR is an iterative self-play algorithm. At each decision node it computes the
**counterfactual regret** of each action — how much better the player would have done by
taking it, weighted by the probability the *opponents and chance* reached that node. The
current strategy is proportional to positive cumulative regret; the **average** strategy
over all iterations converges to a Nash equilibrium.

Weighting by opponent-and-chance reach (rather than the acting player's own reach) is what
makes the regret "counterfactual", and it's why the average strategy — not the current one
— is the thing that converges.

## Solved Kuhn equilibrium

Info-set format is `Card|history`, where `b` = bet and `c` = check. No suffix means the
player is acting first. Eight of the twelve info sets are shown; `python main.py` prints
all of them.

| Info Set | Strategy | Intuition |
|----------|----------|-----------|
| Q (first) | always check | Q is a pure check-hand — no value in betting |
| K (first) | bet ~2/3 | Value bet with the best hand |
| J (first) | bet ~22% | Occasional bluff to balance K's betting range |
| J\|b | always fold | J never calls a bet — it's the weakest hand |
| Q\|b | call ~1/3 | Q is the bluff-catcher; the mix keeps the opponent indifferent |
| K\|b | always call | K never folds to a bet |
| K\|c | always bet | After a check, K always bets for value |
| J\|c | bet ~33% | J bluffs after a check to win otherwise-lost pots |

Kuhn poker has a one-parameter family of Nash equilibria, so exact bluffing frequencies
vary between runs — but the game value is always −1/18 at equilibrium.

## Roadmap

| Phase | Work | Status |
|-------|------|--------|
| 1 | Game/solver abstractions, exact CFR, exploitability, packaging + CI | done |
| 2 | CFR+, Discounted/Linear CFR, external-sampling MCCFR | done |
| 3 | Leduc Hold'em — validates the abstraction generalizes | done |
| 4 | **Market microstructure: Kyle (1985) and Glosten–Milgrom** — the centerpiece | done |
| 5 | Multi-seed benchmarking with confidence bands, convergence plots | done |
| 6 | Performance engineering — profiling, optimized hot loop, published throughput | next |
| 7 | CLI | |
| 8 | Interactive dashboard | |
| 9 | Deep CFR — neural regret approximation, scored against tabular ground truth | |
| 10 | Architecture writeup and docs | |

Phase 4 was the point of the project, and its results are above. The design was worked out
and checked numerically *before* any game code was written, which was worth it — two of the
three obvious formulations turn out to be degenerate. `docs/phase4-microstructure-design.md`
records what failed and why, including a parameter sweep that produced a beautiful-looking
result (spread growing 20× with μ) which was entirely spurious: the market maker had stopped
trading, a zero-profit corner sitting *inside* the search grid rather than on its edge.

## References

- Zinkevich et al., [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/file/08d98638c6a1f1b2c27c8acd1cf29a69-Paper.pdf) (NeurIPS 2007)
- Neller & Lanctot, [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
- Tammelin, [Solving Large Imperfect Information Games Using CFR+](https://arxiv.org/abs/1407.5042) (2014)
- Brown & Sandholm, [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040) (AAAI 2019)
- Glosten & Milgrom, *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders* (Journal of Financial Economics, 1985)
