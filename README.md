# Simplified GTO Solver

A from-scratch implementation of **Counterfactual Regret Minimization (CFR)** applied to **Kuhn Poker** — a minimal poker variant used to study game theory and Nash equilibrium strategies.

## What it does

Runs vanilla CFR over the full Kuhn Poker game tree until strategies converge to a Nash equilibrium. After training, it prints each player's optimal (GTO) strategy for every possible information set, along with the average game value (which should converge to −1/18 ≈ −0.0556 for Player 0).

## Kuhn Poker rules

- 3-card deck: J (Jack), Q (Queen), K (King) — K beats Q beats J
- 2 players, each antes 1 chip
- Player 0 acts first: check or bet (1 chip)
- Player 1 responds: check/fold or bet/call
- If Player 0 checked and Player 1 bet, Player 0 gets one more action
- Showdown goes to the higher card

Possible game sequences:

| History | Outcome |
|---------|---------|
| check → check | Showdown, winner takes ±1 |
| bet → fold | Bettor wins +1 |
| check → bet → fold | Bettor wins +1 |
| bet → call | Showdown, winner takes ±2 |
| check → bet → call | Showdown, winner takes ±2 |

## How CFR works

CFR is an iterative self-play algorithm. Each iteration:

1. Traverse every possible card deal (6 permutations of 2 cards from 3)
2. At each decision node, compute the **counterfactual regret** — how much better the player could have done by taking a different action, given what the opponent played
3. Update the **cumulative regret** for each action at each information set
4. The current strategy is proportional to positive cumulative regret (if all regret is negative, play uniformly)

After many iterations, the **average strategy** across all iterations converges to a Nash equilibrium by the regret minimization theorem.

## Project structure

```
kuhn.py   — Game logic: KuhnState, payouts, information sets
cfr.py    — CFR algorithm: InformationSet, CFRSolver
main.py   — Entry point: trains the solver and prints results
test.py   — Unit tests for game logic
```

## Usage

```bash
python -m venv venv
source venv/bin/activate
pip install numpy

python test.py   # verify game logic
python main.py   # run the solver
```

## Example output

```
Training CFR solver on Kuhn Poker...
Iteration  10000 | Avg game value: -0.058233 (theory: -0.055556)
...
Iteration 100000 | Avg game value: -0.055772 (theory: -0.055556)

Final average game value: -0.055772 (theory: -0.055556)

=== Nash Equilibrium Strategies ===
Info Set       Check/Fold   Bet/Call
------------------------------------
J                  0.7800     0.2200
J|b                1.0000     0.0000
J|c                0.6682     0.3318
J|cb               1.0000     0.0000
K                  0.3370     0.6630
K|b                0.0000     1.0000
K|c                0.0000     1.0000
K|cb               0.0000     1.0000
Q                  1.0000     0.0000
Q|b                0.6641     0.3359
Q|c                1.0000     0.0000
Q|cb               0.4447     0.5553
```

**Reading the info sets:** the format is `Card|history` where `b` = bet and `c` = check. No history suffix means the player is acting first. For example, `Q|cb` means the player holds Q and the action history was check → bet (they are now deciding whether to call the raise).

## Interpreting the Nash equilibrium

| Info Set | Strategy | Intuition |
|----------|----------|-----------|
| Q (first) | always check | Q is a pure check-hand — no value in betting |
| K (first) | bet ~2/3 | Value bet with the best hand |
| J (first) | bet ~22% | Occasional bluff to balance K's betting range |
| J\|b | always fold | J never calls a bet — it's the weakest hand |
| Q\|b | call ~1/3 | Q is the bluff-catcher; mixed strategy keeps opponent indifferent |
| K\|b | always call | K never folds to a bet |
| K\|c | always bet | After a check, K always bets for value |
| J\|c | bet ~33% | J bluffs after opponent checks to win otherwise-lost pots |

Kuhn Poker has a one-parameter family of Nash equilibria, so the exact bluffing frequencies can vary — but the game value will always be −1/18 at equilibrium.

## References

- Zinkevich et al., [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/file/08d98638c6a1f1b2c27c8acd1cf29a69-Paper.pdf) (NeurIPS 2007)
- Neller & Lanctot, [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
