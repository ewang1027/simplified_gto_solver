"""Regenerate every number in docs/phase4-microstructure-design.md.

The design doc's tables were originally produced by throwaway scripts that were
never committed, which made them impossible to spot-check later. This script
replaces them: run it and compare against the doc.

    python scripts/verify_phase4.py
"""

import numpy as np

from gto_solver.analysis.kyle import (
    KyleParams,
    information_revealed,
    price_impact_regression,
    solve_fixed_point,
    strategic_maker_pnl,
)
from gto_solver.analysis.microstructure import (
    GMParams,
    competitive_half_spread,
    maker_profit,
    market_functions,
    strategic_half_spread,
    uninformed_trade_prob,
)
from gto_solver.games.base import GameState
from gto_solver.games.glosten_milgrom import GlostenMilgromGame

MUS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def kyle_section() -> None:
    rule("KYLE: closed form vs solved fixed point")
    print(f"{'sigma_v':>8} {'sigma_u':>8} {'lambda':>10} {'closed':>10} {'beta':>10} {'closed':>10}")
    for sv, su in [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0), (0.1, 5.0), (5.0, 0.1)]:
        p = KyleParams(sigma_v=sv, sigma_u=su)
        r = solve_fixed_point(p)
        print(
            f"{sv:>8.1f} {su:>8.1f} {r.lam:>10.6f} {p.equilibrium_lambda:>10.6f} "
            f"{r.beta:>10.6f} {p.equilibrium_beta:>10.6f}"
        )

    p = KyleParams()
    print(f"\ninformation revealed: {information_revealed(p):.4f}  (theory 0.5)")
    print(f"lambda*beta:          {p.equilibrium_lambda * p.equilibrium_beta:.4f}  (theory 0.5)")
    lam_hat = price_impact_regression(p)
    print(
        f"regression lambda:    {lam_hat:.6f} vs closed {p.equilibrium_lambda:.6f}  "
        f"(rel err {abs(lam_hat - p.equilibrium_lambda) / p.equilibrium_lambda:.2%})"
    )

    rule("KYLE: a strategic maker has no interior optimum")
    print(f"{'lambda':>8} {'maker PnL':>12}")
    for lam in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        print(f"{lam:>8.2f} {strategic_maker_pnl(p, lam):>12.5f}")
    print("\nPnL = lambda*su^2 - S0/(4*lambda): strictly increasing, zero at equilibrium.")


def gm_section() -> None:
    params = GMParams()
    rule("GLOSTEN-MILGROM: chosen parameters (spread = 2 x half-spread)")
    print(
        f"{'mu':>5} {'spread':>9} {'competitive':>12} {'MM PnL':>9} "
        f"{'P(uninf)':>9} {'functions':>10}"
    )
    for mu in MUS:
        s = strategic_half_spread(params, mu)
        print(
            f"{mu:>5.2f} {2 * s:>9.4f} {2 * competitive_half_spread(params, mu):>12.4f} "
            f"{maker_profit(params, mu, s):>9.4f} {uninformed_trade_prob(params, s):>9.4f} "
            f"{market_functions(params, mu)!s:>10}"
        )

    rule("GLOSTEN-MILGROM: binary value is degenerate when |V| <= T/2")
    for c in [0.5, 1.25]:
        binary = GMParams(num_values=2, sigma=c / 2.5)
        spreads = [strategic_half_spread(binary, mu) for mu in MUS]
        label = "constant" if len(set(spreads)) == 1 else f"saturates at {max(spreads):.4f}"
        print(f"  |V| = {c:.2f} (T/2 = {params.reservation_max / 2:.2f}): {label}")
        print("     " + " ".join(f"{s:.4f}" for s in spreads))

    rule("GLOSTEN-MILGROM: payoff surface is flat near the optimum")
    print(f"{'mu':>5} {'best - 2nd best':>16}")
    for mu in [0.02, 0.1, 0.3, 0.5, 0.7]:
        profits = np.sort([maker_profit(params, mu, s) for s in params.quotes()])[::-1]
        print(f"{mu:>5.2f} {profits[0] - profits[1]:>16.5f}")


def count_terminals(state: GameState) -> int:
    if state.is_terminal():
        return 1
    if state.is_chance():
        return sum(count_terminals(state.apply(o)) for o, _ in state.chance_outcomes())
    return sum(count_terminals(state.apply(a)) for a in state.legal_actions())


def tree_section() -> None:
    rule("GLOSTEN-MILGROM: tree size")
    print(f"{'rounds':>7} {'terminal nodes':>16} {'closed form':>14}")
    for rounds in [1, 2]:
        game = GlostenMilgromGame(mu=0.3, params=GMParams(), num_rounds=rounds)
        actual = count_terminals(game.new_initial_state())
        closed = 9 * 99**rounds + 18 * 66**rounds
        print(f"{rounds:>7} {actual:>16,} {closed:>14,}")
    for rounds in [3]:
        closed = 9 * 99**rounds + 18 * 66**rounds
        print(f"{rounds:>7} {'(not walked)':>16} {closed:>14,}")
    print("\n9*99^R + 18*66^R: the informed branch has 3 flow outcomes per round,")
    print("each uninformed branch only 2 (its direction is fixed at the root).")


if __name__ == "__main__":
    kyle_section()
    gm_section()
    tree_section()
