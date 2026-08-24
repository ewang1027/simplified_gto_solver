"""`payouts()` must agree with `payout()` at every terminal node of every game.

The tree walk asks for the whole vector once instead of once per player, because a
two-player zero-sum game computes the same quantity both times. That is a performance
change with a correctness hazard attached: a game that overrides `payouts()` and gets
a sign or an order wrong would produce a solver that trains happily on wrong payoffs,
and every convergence test would still pass, because the solver would be converging
correctly to the equilibrium of a different game.

So this walks every terminal node of every game and checks the two against each other
directly, rather than trusting that a wrong override would surface somewhere else.
"""

import pytest

from gto_solver.games.base import Game, GameState
from gto_solver.games.glosten_milgrom import GlostenMilgromGame
from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame

GAMES = [
    pytest.param(KuhnGame(), id="kuhn"),
    pytest.param(LeducGame(), id="leduc"),
    pytest.param(GlostenMilgromGame(mu=0.30), id="glosten_milgrom"),
]


def terminals(state: GameState):
    if state.is_terminal():
        yield state
        return
    moves = (
        [outcome for outcome, _ in state.chance_outcomes()]
        if state.is_chance()
        else state.legal_actions()
    )
    for move in moves:
        yield from terminals(state.apply(move))


@pytest.mark.parametrize("game", GAMES)
def test_payouts_agrees_with_payout_at_every_terminal(game: Game):
    checked = 0
    for state in terminals(game.new_initial_state()):
        vector = state.payouts(game.num_players)
        assert len(vector) == game.num_players
        for player in range(game.num_players):
            assert vector[player] == state.payout(player), (state.info_set_key(), player)
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("game", GAMES)
def test_payouts_is_still_zero_sum(game: Game):
    """The property exploitability relies on, checked through the new accessor."""
    for state in terminals(game.new_initial_state()):
        assert sum(state.payouts(game.num_players)) == pytest.approx(0.0, abs=1e-9)


def test_the_default_implementation_is_used_when_a_game_does_not_override():
    """A game that implements only `payout` still works, since the base class fills
    the vector in. This is what keeps `payouts()` an optional optimization rather
    than a new required method on the interface.
    """

    class Minimal(GameState):
        def is_terminal(self):
            return True

        def is_chance(self):
            return False

        def chance_outcomes(self):
            raise AssertionError("terminal")

        def current_player(self):
            raise AssertionError("terminal")

        def legal_actions(self):
            raise AssertionError("terminal")

        def apply(self, action):
            raise AssertionError("terminal")

        def info_set_key(self):
            return "terminal"

        def payout(self, player):
            return 3.0 if player == 0 else -3.0

    assert Minimal().payouts(2) == [3.0, -3.0]
