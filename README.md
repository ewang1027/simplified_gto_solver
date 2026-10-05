# simplified_gto_solver

A small game-theory-optimal (GTO) solver: counterfactual regret minimization (CFR)
written from scratch in Python and numpy, run on Kuhn poker, Leduc hold'em and a small
Glosten-Milgrom market-making game. The games are small enough that every solved
strategy can be scored exactly.

## What's in it

- Kuhn poker (12 info sets), Leduc hold'em (288, the published count) and the
  market-making game, all behind one `GameState` interface with chance as tree nodes.
- CFR variants built as an update rule plus a traversal: vanilla, CFR+, discounted and
  linear CFR, alternating-update CFR+, and external-sampling MCCFR. Deep CFR is separate,
  with a small numpy MLP whose gradients are checked against finite differences.
- Exploitability as the measure of distance from equilibrium, with the best response
  taken per information set rather than per tree node
  (`src/gto_solver/metrics/exploitability.py`).
- Kyle (1985), solved as a best-response fixed point instead of with CFR, since its
  competitive market maker is not a player in a zero-sum game.
- A `gto` CLI, a benchmark runner that saves seeds and provenance to `results/*.json`,
  and an optional Streamlit dashboard.

## What has been checked

- Kuhn: vanilla CFR reaches exploitability 0.00063 after 100,000 iterations, and the
  average strategy's value is within 0.01 of the known −1/18 after 3,000
  (`tests/test_cfr_convergence.py`). On Leduc it reaches 0.0068 after 5,000.
- Kyle: the fixed point matches the closed forms (λβ = 1/2, half the private information
  revealed), and λ is recovered separately by regression on simulated order flow
  (`tests/test_kyle.py`).

## The market-making game

The maker posts a symmetric half-spread from a 33-level grid. With probability μ the
trader is informed (it knows the asset value, one of 9 levels, and buys, sells or
passes); otherwise it trades with a probability that falls as the spread widens. Payoffs
are zero-sum between maker and trader.

This is closer to an optimization than a rich game. All results use a single round,
where the maker has one information set and the informed trader's best response is
dominant, so solving it means finding the profit-maximizing spread on the grid. The
solver finds it at every tested μ, matching an exhaustive grid search that doesn't use
the solver (`gto microstructure`, `tests/test_microstructure_gate.py`):

| μ | solved spread | grid search | competitive (zero-profit) spread |
|---:|---:|---:|---:|
| 0.02 | 1.625 | 1.625 | 0.031 |
| 0.30 | 1.875 | 1.875 | 0.494 |
| 0.70 | 2.500 | 2.500 | 1.314 |

The solved maker is a profit-maximizing monopolist rather than Glosten and Milgrom's
competitive maker, which is why its spread is much wider. Multi-round versions
(`--rounds`) exist but haven't been studied.

## Known issue

`FullTraversal` in `src/gto_solver/solvers/traversal.py` updates regrets at every history
node it visits and re-reads the strategy partway through an iteration, instead of once per
information set per iteration. For CFR+ and DCFR that applies the regret floor and the
discount per visit, which is not the published algorithm, so the comparisons between
update rules in `docs/RESULTS.md` and `results/` shouldn't be relied on. Exploitability
is computed independently, so the numbers above do describe the strategies produced.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

gto solve                                  # Kuhn poker: exploitability and strategies
gto solve --game leduc --iterations 5000
gto microstructure                         # the spread table above
gto algorithms                             # list the variants
pytest                                     # correctness suite (531 tests, about a minute)
```

More detail: [`docs/RESULTS.md`](docs/RESULTS.md) (measurements),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (design) and
[`docs/REFERENCE.md`](docs/REFERENCE.md) (module map).

## References

- Zinkevich et al., [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/file/08d98638c6a1f1b2c27c8acd1cf29a69-Paper.pdf) (NeurIPS 2007)
- Neller & Lanctot, [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
- Tammelin, [Solving Large Imperfect Information Games Using CFR+](https://arxiv.org/abs/1407.5042) (2014)
- Brown & Sandholm, [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040) (AAAI 2019)
- Glosten & Milgrom, *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders* (JFE, 1985)
- Kyle, *Continuous Auctions and Insider Trading* (Econometrica, 1985)
