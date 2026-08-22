# Simplified GTO Solver

A from-scratch **Counterfactual Regret Minimization (CFR)** engine for extensive-form games
with imperfect information, validated on poker and aimed at market microstructure.

Solving poker is the *means*, not the end. Kuhn and Leduc poker have known equilibria, so
they serve as a correctness harness for the solver. The destination is a **Glosten–Milgrom
market-making game**, where a market maker quotes against possibly-informed order flow —
the same asymmetric-information problem CFR solves, and the mechanism behind adverse
selection and bid-ask spreads.

## Status

Phase 1 of 10 complete. The engine, the game abstraction, and the correctness metric are in
place and tested on Kuhn poker. See [Roadmap](#roadmap) for what's next.

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
│   └── kuhn.py            Kuhn poker
├── solvers/
│   ├── base.py            InfoSetStore, RegretUpdateRule, Traversal, CFRSolver
│   ├── regret_rules.py    Vanilla regret matching
│   └── traversal.py       FullTraversal (exact CFR)
└── metrics/
    ├── evaluation.py      Expected value of a fixed strategy profile
    └── exploitability.py  Best-response computation
```

Regret is tracked per information set with a **per-info-set action count**, not a global
one — Leduc and the market-making game have different numbers of legal actions at different
nodes.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

python main.py    # train on Kuhn poker, report exploitability + strategies
pytest            # correctness suite
```

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
player is acting first.

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
| 2 | CFR+, Discounted/Linear CFR, external-sampling MCCFR | next |
| 3 | Leduc Hold'em — validates the abstraction generalizes | |
| 4 | **Glosten–Milgrom market-making game** — the centerpiece | |
| 5 | Multi-seed benchmarking with confidence bands, convergence plots | |
| 6 | Performance engineering — profiling, optimized hot loop, published throughput | |
| 7 | CLI | |
| 8 | Interactive dashboard | |
| 9 | Deep CFR — neural regret approximation, scored against tabular ground truth | |
| 10 | Architecture writeup and docs | |

Phase 4 is the point of the project: solve the market-making game and check the result
against the analytical Glosten–Milgrom quotes, confirming the solved spread widens as the
probability of informed flow rises.

## References

- Zinkevich et al., [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/file/08d98638c6a1f1b2c27c8acd1cf29a69-Paper.pdf) (NeurIPS 2007)
- Neller & Lanctot, [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
- Tammelin, [Solving Large Imperfect Information Games Using CFR+](https://arxiv.org/abs/1407.5042) (2014)
- Brown & Sandholm, [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040) (AAAI 2019)
- Glosten & Milgrom, *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders* (Journal of Financial Economics, 1985)
