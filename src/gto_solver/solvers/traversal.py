"""Tree-walking strategies. FullTraversal is exact CFR: every iteration walks the
entire game tree. ExternalSamplingMCCFR samples opponents and chance instead, to
scale to games too large to walk exactly.
"""

import numpy as np

from gto_solver.games.base import Game, GameState
from gto_solver.solvers.base import InfoSetStore, RegretUpdateRule, Traversal


class FullTraversal(Traversal):
    """Exact (non-sampled) CFR: one iteration is one full recursive walk.

    alternating=True switches to alternating updates, as published CFR+ uses:
    iteration t updates only player (t - 1) % num_players's accumulators. Other
    players' current strategies still drive the walk (and are read to compute
    reach and action values), but neither their regret nor their strategy sum is
    touched that iteration. Default is False so existing (simultaneous-update)
    behavior is unchanged.
    """

    def __init__(self, alternating: bool = False):
        self.alternating = alternating

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
        update_player = (iteration - 1) % num_players if self.alternating else None
        return _walk(
            game.new_initial_state(), reach, num_players, store, rule, iteration, update_player
        )


def _walk(
    state: GameState,
    reach: np.ndarray,
    num_players: int,
    store: InfoSetStore,
    rule: RegretUpdateRule,
    iteration: int,
    update_player: int | None = None,
) -> np.ndarray:
    """Recursively evaluate `state`, updating regret/strategy accumulators along the
    way. Returns the per-player expected payoff vector for `state`.

    `reach` has length num_players + 1: reach[i] is player i's contribution to
    reaching `state`, and reach[-1] is chance's contribution. `update_player`
    restricts accumulation to one player's info sets (alternating updates); None
    updates everyone, i.e. plain simultaneous-update CFR.
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
                child_state, child_reach, num_players, store, rule, iteration, update_player
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
            child_state, child_reach, num_players, store, rule, iteration, update_player
        )

    node_value = strategy @ action_values

    if update_player is None or player == update_player:
        # Counterfactual reach for `player`: probability of reaching this node under
        # everyone else's play (opponents and chance), excluding player's own reach.
        cf_reach = np.prod(np.delete(reach, player))
        regrets = cf_reach * (action_values[:, player] - node_value[player])
        rule.accumulate_regret(record, regrets, iteration)
        rule.accumulate_strategy(record, strategy, reach[player], iteration)

    return node_value


class ExternalSamplingMCCFR(Traversal):
    """External-sampling MCCFR (Lanctot et al. 2009).

    One traversal per player per iteration, each with that player as
    "traverser": the traverser's own actions are all explored exactly, while
    every other player's action and every chance outcome is a single sample from
    the current strategy/chance distribution. That's what keeps one traversal
    cheaper than walking the whole tree, at the cost of a noisier per-iteration
    update.
    """

    def run_iteration(
        self,
        game: Game,
        store: InfoSetStore,
        rule: RegretUpdateRule,
        iteration: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        num_players = game.num_players
        payoffs = np.zeros(num_players, dtype=np.float64)
        for traverser in range(num_players):
            value = _sample_walk(
                game.new_initial_state(), num_players, store, rule, iteration, rng, traverser
            )
            payoffs[traverser] = value[traverser]
        return payoffs


def _sample_walk(
    state: GameState,
    num_players: int,
    store: InfoSetStore,
    rule: RegretUpdateRule,
    iteration: int,
    rng: np.random.Generator,
    traverser: int,
) -> np.ndarray:
    """One external-sampling traversal on behalf of `traverser`. Returns a
    per-player payoff estimate: exact in expectation over the traverser's own
    actions, single-sample (but unbiased) over everyone else's actions and chance.
    """
    if state.is_terminal():
        return np.array([state.payout(p) for p in range(num_players)], dtype=np.float64)

    if state.is_chance():
        outcomes, probs = zip(*state.chance_outcomes())
        outcome = outcomes[rng.choice(len(outcomes), p=probs)]
        return _sample_walk(
            state.apply(outcome), num_players, store, rule, iteration, rng, traverser
        )

    player = state.current_player()
    actions = state.legal_actions()
    record = store.get(state.info_set_key(), len(actions))
    strategy = rule.strategy(record)

    if player == traverser:
        action_values = np.empty((len(actions), num_players), dtype=np.float64)
        for i, action in enumerate(actions):
            action_values[i] = _sample_walk(
                state.apply(action), num_players, store, rule, iteration, rng, traverser
            )
        node_value = strategy @ action_values
        # No explicit reach weighting: sampling opponents/chance on-policy with
        # ratio 1 already makes this an unbiased estimate of the counterfactual
        # regret, unlike FullTraversal's exact cf_reach multiplier.
        regrets = action_values[:, player] - node_value[player]
        rule.accumulate_regret(record, regrets, iteration)
        return node_value

    # Opponent node: sample one action on-policy and accumulate its strategy sum
    # here with weight 1 -- again, sampling is what makes this unbiased.
    action_idx = rng.choice(len(actions), p=strategy)
    rule.accumulate_strategy(record, strategy, 1.0, iteration)
    return _sample_walk(
        state.apply(actions[action_idx]), num_players, store, rule, iteration, rng, traverser
    )
