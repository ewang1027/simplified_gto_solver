"""The Phase 1 rigor gate: CFR must drive exploitability toward zero.

This is the real convergence test. Checking the game value against Kuhn's known
-1/18 is kept only as a secondary sanity check, since that approach cannot
validate a game whose equilibrium value isn't already published.
"""

import pytest

from gto_solver.games.kuhn import KuhnGame
from gto_solver.metrics.evaluation import expected_value
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.base import CFRSolver
from gto_solver.solvers.regret_rules import VanillaRegretMatching
from gto_solver.solvers.traversal import FullTraversal

KUHN_GAME_VALUE = -1 / 18
KUHN_INFO_SETS = 12


def make_solver(seed: int = 0) -> CFRSolver:
    return CFRSolver(KuhnGame(), VanillaRegretMatching(), FullTraversal(), seed=seed)


def test_exploitability_decreases_with_training():
    """CFR bounds *average* regret at O(1/sqrt(T)); it does not promise a decrease
    at every single step, and there is a transient bump in the first few
    iterations. So this checks decay on a coarse schedule past that transient.
    """
    game = KuhnGame()
    solver = make_solver()
    measurements = []
    for checkpoint in [50, 250, 1000, 4000]:
        solver.train(checkpoint - solver.iterations)
        measurements.append(exploitability(game, solver.average_strategy()))

    assert measurements == sorted(measurements, reverse=True), measurements
    assert measurements[-1] < measurements[0] / 5


def test_converges_to_nash():
    game = KuhnGame()
    solver = make_solver()
    solver.train(3000)
    assert exploitability(game, solver.average_strategy()) < 0.01


def test_discovers_every_info_set():
    solver = make_solver()
    solver.train(1)
    assert len(solver.store) == KUHN_INFO_SETS


def test_game_value_matches_theory():
    """Secondary sanity check against Kuhn's published equilibrium value.

    Evaluated on the average strategy, which is what converges -- the current
    regret-matching strategy oscillates and does not settle on the game value.
    """
    game = KuhnGame()
    solver = make_solver()
    solver.train(3000)
    strategy = solver.average_strategy()

    assert expected_value(game, strategy, 0) == pytest.approx(KUHN_GAME_VALUE, abs=0.01)
    assert expected_value(game, strategy, 0) + expected_value(game, strategy, 1) == pytest.approx(
        0.0
    )


def test_strategies_are_probability_distributions():
    solver = make_solver()
    solver.train(100)
    for key, probs in solver.average_strategy().items():
        assert probs.sum() == pytest.approx(1.0), key
        assert (probs >= 0).all(), key
