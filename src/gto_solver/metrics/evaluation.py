"""Evaluating a fixed strategy profile by exact game-tree traversal.

`exploitability` asks what a best responder could win; this module answers the
simpler question of what a profile is actually worth when everyone follows it.
"""

import numpy as np

from gto_solver.games.base import Game, GameState


def action_probs(state: GameState, strategy: dict[str, np.ndarray]) -> np.ndarray:
    """Probability vector the fixed `strategy` assigns at `state`.

    Falls back to uniform when the info set has no entry (e.g. never visited
    during training).
    """
    actions = state.legal_actions()
    probs = strategy.get(state.info_set_key())
    if probs is None:
        return np.full(len(actions), 1.0 / len(actions))
    if len(probs) != len(actions):
        raise ValueError(
            f"strategy for info set {state.info_set_key()!r} has {len(probs)} entries "
            f"but the state offers {len(actions)} legal actions"
        )
    return probs


def expected_value(game: Game, strategy: dict[str, np.ndarray], player: int) -> float:
    """Expected payoff for `player` when every player follows `strategy`.

    Unlike a solver's per-iteration return value, which reflects the current
    regret-matching strategy and oscillates, this evaluates the average strategy
    profile -- the one that converges to equilibrium.
    """
    if not 0 <= player < game.num_players:
        raise ValueError(f"player={player} is out of range for {game.num_players} players")

    def value(state: GameState) -> float:
        if state.is_terminal():
            return state.payout(player)
        if state.is_chance():
            return sum(
                prob * value(state.apply(outcome)) for outcome, prob in state.chance_outcomes()
            )
        return sum(
            prob * value(state.apply(action))
            for action, prob in zip(state.legal_actions(), action_probs(state, strategy))
        )

    return value(game.new_initial_state())
