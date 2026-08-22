# Phase 4 design: market microstructure

Design and numerical verification done **before** implementation, because the modeling
is where this phase can quietly go wrong. Every quantitative claim below was checked
numerically; the scripts are described at the end.

## The core tension

CFR's convergence guarantees hold for **two-player zero-sum** games, and exploitability
is only defined there. Classic microstructure models assume a **competitive** market
maker earning **zero expected profit** — a market maker that is not optimizing at all.
Those are different objects, and the difference is not cosmetic: it decides whether a
model can be posed as a CFR game.

Both canonical models were checked against that constraint. They came out differently,
and that difference drives the design.

---

## Kyle (1985): not a zero-sum game — solved as a fixed point

### The model

`v ~ N(p0, S0)` is the asset value. Noise order flow is `u ~ N(0, su^2)`. An informed
trader observes `v` and submits quantity `x`. The market maker sees only total order
flow `y = x + u` and sets a price.

### Closed form (independently derived, then verified numerically)

Conjecture `x = beta*(v - p0)` and `p = p0 + lambda*y`.

The trader maximizes `E[(v - p)x] = (v - p0)x - lambda*x^2`, giving `beta = 1/(2*lambda)`.
The market maker prices at the Bayesian conditional expectation, giving
`lambda = beta*S0 / (beta^2*S0 + su^2)`. Solving the fixed point:

| Quantity | Closed form | Verified |
|---|---|---|
| Price impact | `lambda = sqrt(S0) / (2*su)` | yes |
| Trading intensity | `beta = su / sqrt(S0)` | yes |
| Product | `lambda*beta = 1/2` | yes |
| Residual variance | `Var(v\|y) = S0/2` — exactly **half** the private information is impounded in price | yes |
| Informed profit | `su*sqrt(S0)/2` | yes |

The fixed point is attracting: iterating the best-response map from `lambda = 3.7`
converges to `0.5` for `S0 = su^2 = 1`.

### Why it is not a CFR game

If the market maker is made strategic — picking `lambda` to maximize its own PnL against
the informed trader — its profit is **monotonically increasing in `lambda` without bound**:

| lambda | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 |
|---|---|---|---|---|---|---|---|
| MM PnL | −0.75 | **0.00** | 0.75 | 1.88 | 3.94 | 7.97 | 15.98 |

There is no interior optimum: a strategic Kyle market maker widens price impact forever.
Note where the equilibrium sits — MM PnL is exactly **0.0000 at `lambda = 0.5`**. Kyle's
`lambda` is pinned by the **zero-profit condition**, not by optimization. Kyle is
genuinely not a two-player zero-sum game, and forcing it into one would produce a
different model with no closed-form benchmark.

### Decision

Implement Kyle as an **iterated best-response fixed-point solver**, not as a CFR game,
and validate it against the five closed-form results above. Using the right algorithm for
the model — and saying plainly why CFR is the wrong tool here — is the honest outcome.

---

## Glosten–Milgrom: a viable zero-sum game, after fixing the value distribution

### Why the obvious formulation fails

With a **binary** value, `|V|` is a single number `c`. Once the half-spread reaches `c` the
informed never trade, so profit collapses to the noise-trader term `(1−mu)·s·(T−s)/T`,
which is maximized at `s = T/2` **for every `mu`**. The market maker parks there and the
spread carries no adverse-selection signal at all. Reproduced as a control at `c ≤ T/2`:

| mu | 0.05 | 0.20 | 0.50 | 0.80 |
|---|---|---|---|---|
| optimal half-spread | 0.8125 | 0.8125 | 0.8125 | 0.8125 |

A first version of this claim was too strong and testing caught it: when `c > T/2` the
binary spread *does* move with `mu`. But it still **saturates at `c`** and stops responding
once the informed are fully excluded — at `c = 1.25` the half-spread pins to 1.25 for every
`mu ≥ 0.4`. So binary fails either outright or by hitting a ceiling.

A multi-level value has no such ceiling: `P(|V| > s)` decays gradually, so adverse selection
trades off smoothly against lost noise flow at every spread.

### The fix

Use a **multi-level value distribution**. The informed trader then only trades when
`|V − mid|` exceeds the half-spread, so adverse selection decays smoothly in the spread
and trades off against lost noise-trader flow. Uninformed traders are price-elastic with
reservation `t ~ U[0, T]`, trading only if the half-spread is within it.

Market maker expected profit at half-spread `s`:

```
pi(s) = -mu * E[(|V| - s)^+]  +  (1 - mu) * s * (T - s)/T
         adverse selection        noise-trader capture
```

### The viability trap

Interiority in the search grid is **not** a sufficient check. A parameter sweep initially
reported configurations where the spread grew 19.7× with `mu` — all spurious. Those optima
had **MM PnL exactly 0.0000**: the market maker had shut down trading entirely, a
degenerate corner sitting strictly inside the grid rather than on its boundary.

A configuration is only valid if the market actually functions:
- MM expected profit `> 0`
- `P(uninformed trades) > 0`
- `P(informed trades) > 0` (adverse selection is live)

### Chosen parameters (all conditions verified across `mu`)

9-level value grid, `sigma = 1.0`, uninformed reservation `T = 1.6`, 33 quote levels.
These are the shipped defaults in `GMParams`:

| mu | strategic spread | competitive (GM) | MM PnL | P(uninf trades) |
|---|---|---|---|---|
| 0.02 | 1.6250 | 0.0306 | 0.3855 | 0.4922 |
| 0.10 | 1.6250 | 0.1561 | 0.3359 | 0.4922 |
| 0.20 | 1.7500 | 0.3204 | 0.2755 | 0.4531 |
| 0.30 | 1.8750 | 0.4937 | 0.2150 | 0.4141 |
| 0.50 | 2.1250 | 0.8699 | 0.1051 | 0.3359 |
| 0.70 | 2.5000 | 1.3143 | 0.0250 | 0.2188 |

The discrete-quote optimum tracks the continuous solution (1.61 → 2.50) closely.

The strategic spread is monotonically increasing in `mu`, strictly wider than the
competitive spread everywhere, and every point is a functioning market.

### An honest read of the result

The strategic spread rises only ~1.55× across the `mu` range while the competitive spread
rises ~43×. That is not a defect — it is economically sensible and worth stating plainly:
a monopolist's spread is set mostly by **demand elasticity**, with adverse selection as a
second-order adjustment, whereas the competitive spread is adverse selection and nothing
else. The two benchmarks together say more than either alone.

### Game-tree mapping

Single-round GM has a **dominant-strategy trader** (informed buys iff `V > ask`;
uninformed trades iff the spread is within reservation), which makes it an optimization
rather than a game. The trader only becomes strategic over **multiple rounds**, where
trading today reveals information and moves tomorrow's quotes — so the informed trader
must weigh immediate profit against information leakage. Multi-round is the design.

Mid is 0 and the maker posts a symmetric half-spread `s`, so ask = `+s`, bid = `−s`.

- **Root chance node**: draw `V` (9 levels) jointly with the trader type — informed with
  probability `mu`, else uninformed with a direction. `9 × (1 + 2) = 27` outcomes.
- **Player 0 (market maker)**: posts a half-spread from a 33-level grid each round. Info
  set = observed **order-flow history only** — never `V`, never the trader type. In a
  single round it therefore has exactly one info set.
- **Player 1 (informed trader)**: buy / sell / pass. Its info set includes `V` and the quote.
- **Uninformed trader**: mechanical, so it is *not* a player. It becomes a two-outcome
  chance node placed **after** the quote, trading with probability `(T − s)/T`.
- **Payout**: buy at the ask gives the maker `s − V`; sell at the bid gives `V + s`; no
  trade gives 0. The trader's payout is the exact negation, so the game is exactly
  zero-sum — which the existing `exploitability` guard checks at every terminal node.

Putting the uninformed reservation *after* the quote rather than in the root chance node is
what keeps this small: the reservation never enters the root, and its probability stays
exact rather than discretized. An earlier design drew the reservation as a root chance node
with discrete levels; that both inflated the tree and made the solved spread a coarse step
function, because the optimum simply pinned to a reservation level.

Single round is about `27 × 33 × 3 ≈ 2.7×10^3` terminal nodes — smaller than Leduc (5,520),
so exact traversal is comfortable. Rounds multiply that by roughly `33 × 3` each, so three
rounds reaches ~10^6, which is where **MCCFR earns its place** in this project rather than
existing only for completeness.

---

## Limitations

- The strategic market maker is a **monopolist**, not the competitive one GM assumes. Both
  benchmarks are reported; neither is presented as the other.
- Value, quote, and reservation grids are discretized to keep the action set finite.
- Real markets are neither two-player nor zero-sum; the aggregate trader is one player by
  construction so that CFR's guarantees and exploitability apply at all.
- Kyle is validated against closed form but is **not** solved by CFR, for the reason above.

## Verification scripts

Under the session scratchpad (`design_kyle/`, `design_gm/`): `verify_kyle.py` covers the
Kyle closed form, the attracting fixed point, and the unbounded-lambda degeneracy;
`verify_gm.py` and `tune_gm2.py` cover the binary-value control, the multi-level fix, the
viability conditions, and the parameter sweep. Their saved output is in `verify_output.txt`
alongside each.
