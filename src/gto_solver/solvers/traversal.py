"""Tree-walking strategies. FullTraversal is exact CFR: every iteration walks the
entire game tree. ExternalSamplingMCCFR is added later as a sibling class.
"""

import numpy as np

from gto_solver.games.base import Game, GameState
from gto_solver.solvers.base import InfoSetStore, RegretUpdateRule, Traversal


class FullTraversal(Traversal):
    """Exact (non-sampled) CFR: one iteration is one full recursive walk."""

    def run_iteration(
        self,
        game: Game,
        store: InfoSetStore,
        rule: RegretUpdateRule,
        iteration: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        num_players = game.num_players
        reach = np.ones(num_players + 1, dtype=np.float64)
        return _walk(game.new_initial_state(), reach, num_players, store, rule, iteration)


def _walk(
    state: GameState,
    reach: np.ndarray,
    num_players: int,
    store: InfoSetStore,
    rule: RegretUpdateRule,
    iteration: int,
) -> np.ndarray:
    """Recursively evaluate `state`, updating regret/strategy accumulators along the
    way. Returns the per-player expected payoff vector for `state`.

    `reach` has length num_players + 1: reach[i] is player i's contribution to
    reaching `state`, and reach[-1] is chance's contribution.
    """
    if state.is_terminal():
        return np.array([state.payout(p) for p in range(num_players)], dtype=np.float64)

    if state.is_chance():
        value = np.zeros(num_players, dtype=np.float64)
        for outcome, prob in state.chance_outcomes():
            child_reach = reach.copy()
            child_reach[-1] *= prob
            child_state = state.apply(outcome)
            value += prob * _walk(
                child_state, child_reach, num_players, store, rule, iteration
            )
        return value

    player = state.current_player()
    actions = state.legal_actions()
    record = store.get(state.info_set_key(), len(actions))
    strategy = rule.strategy(record)

    action_values = np.empty((len(actions), num_players), dtype=np.float64)
    for i, action in enumerate(actions):
        child_reach = reach.copy()
        child_reach[player] *= strategy[i]
        child_state = state.apply(action)
        action_values[i] = _walk(
            child_state, child_reach, num_players, store, rule, iteration
        )

    node_value = strategy @ action_values

    # Counterfactual reach for `player`: probability of reaching this node under
    # everyone else's play (opponents and chance), excluding player's own reach.
    cf_reach = np.prod(np.delete(reach, player))

    regrets = cf_reach * (action_values[:, player] - node_value[player])
    rule.accumulate_regret(record, regrets, iteration)
    rule.accumulate_strategy(record, strategy, reach[player], iteration)

    return node_value
