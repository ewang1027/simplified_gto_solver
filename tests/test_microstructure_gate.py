"""The Phase 4 gate: CFR must reproduce an independently-computed result.

This is the test the whole microstructure phase exists to pass. The solver, the
game, and the benchmark are written separately; here they have to agree. The
benchmark in analysis/microstructure.py is an exhaustive search over the quote
grid and shares no code with the CFR solver, so agreement is real evidence rather
than a tautology.
"""

import numpy as np
import pytest

from gto_solver.analysis.microstructure import (
    GMParams,
    competitive_half_spread,
    maker_profit,
    strategic_half_spread,
)
from gto_solver.games.glosten_milgrom import GlostenMilgromGame
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.base import CFRSolver
from gto_solver.solvers.regret_rules import CFRPlusRegretMatching
from gto_solver.solvers.traversal import FullTraversal

ITERATIONS = 150


def solve(mu: float, params: GMParams, iterations: int = ITERATIONS):
    game = GlostenMilgromGame(mu=mu, params=params, num_rounds=1)
    solver = CFRSolver(game, CFRPlusRegretMatching(), FullTraversal())
    solver.train(iterations)
    return game, solver.average_strategy()


def maker_quote_probs(strategy: dict[str, np.ndarray], params: GMParams) -> np.ndarray:
    """The maker has exactly one info set in a single-round game."""
    keys = [k for k, v in strategy.items() if len(v) == params.num_quotes]
    assert len(keys) == 1, keys
    return strategy[keys[0]]


@pytest.mark.parametrize("mu", [0.05, 0.3, 0.7])
def test_cfr_recovers_the_brute_force_optimal_spread(mu):
    params = GMParams()
    _, strategy = solve(mu, params)
    probs = maker_quote_probs(strategy, params)
    solved = float(params.quotes()[int(np.argmax(probs))])
    assert solved == pytest.approx(strategic_half_spread(params, mu), abs=1e-9)


@pytest.mark.parametrize("mu", [0.05, 0.5])
def test_solved_profile_is_near_equilibrium(mu):
    params = GMParams()
    game, strategy = solve(mu, params)
    assert exploitability(game, strategy) < 0.02


def test_solved_spread_widens_with_adverse_selection():
    """The economic result: more informed flow means a wider quoted spread."""
    params = GMParams()
    spreads = []
    for mu in [0.05, 0.3, 0.7]:
        _, strategy = solve(mu, params)
        probs = maker_quote_probs(strategy, params)
        spreads.append(float(params.quotes()[int(np.argmax(probs))]))
    assert spreads == sorted(spreads), spreads
    assert spreads[-1] > spreads[0]


def test_strategic_maker_quotes_wider_than_the_competitive_benchmark():
    """A profit-maximizing maker is not a Glosten-Milgrom maker.

    GM's maker earns zero expected profit by construction; this one maximizes, so
    it quotes strictly wider. Keeping both benchmarks visible is the point.
    """
    params = GMParams()
    for mu in [0.05, 0.3, 0.7]:
        _, strategy = solve(mu, params)
        probs = maker_quote_probs(strategy, params)
        solved = float(params.quotes()[int(np.argmax(probs))])
        assert solved > competitive_half_spread(params, mu)


def test_optimum_is_flat_so_spread_mass_is_expected():
    """Why the solved maker strategy is not a spike on one quote.

    Adjacent quotes are worth almost exactly the same, so there is very little
    regret pressure to concentrate. The argmax is still correct -- this test
    documents that the diffuse strategy is a property of the payoff surface, not
    a convergence failure.
    """
    params = GMParams()
    for mu in [0.1, 0.5]:
        profits = np.array([maker_profit(params, mu, s) for s in params.quotes()])
        ranked = np.sort(profits)[::-1]
        assert ranked[0] - ranked[1] < 5e-3
