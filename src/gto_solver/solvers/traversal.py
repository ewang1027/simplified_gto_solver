"""Tree-walking strategies. FullTraversal is exact CFR: every iteration walks the
entire game tree. ExternalSamplingMCCFR samples opponents and chance instead, to
scale to games too large to walk exactly.
"""

import numpy as np

from gto_solver.games.base import Game, GameState
from gto_solver.solvers.base import InfoSetStore, RegretUpdateRule, Traversal


class _TreeNode:
    """One pre-resolved node of a game tree.

    Everything a `GameState` can tell the solver about *structure* is the same on
    every iteration: which nodes are terminal, what the payoffs there are, the chance
    probabilities, whose turn it is, the info-set key, how many actions there are, and
    which node each action leads to. Only the strategy changes between iterations. A
    node here holds all of that, resolved once.

    Three shapes, distinguished without a tag so the hot path is two attribute loads:
    a terminal has `payouts`, a chance node has `chance_probs`, and a decision node
    has neither.
    """

    __slots__ = ("chance_probs", "children", "info_key", "num_actions", "payouts", "player")

    def __init__(self, payouts=None, chance_probs=None, children=(), player=-1, info_key="",
                 num_actions=0):
        self.payouts = payouts
        self.chance_probs = chance_probs
        self.children = children
        self.player = player
        self.info_key = info_key
        self.num_actions = num_actions


def build_tree(state: GameState, num_players: int) -> _TreeNode:
    """Resolve a whole game tree into `_TreeNode`s, once.

    A terminal node's payoff vector is stored as an array and handed back to callers
    by reference on every visit, so **nothing may mutate a vector returned by the
    walk**. Everything downstream builds new arrays (`value += prob * child` and
    `action_values[i] = child` both write elsewhere), and `test_tree_cache.py` pins
    the invariant by training twice and requiring identical results.
    """
    if state.is_terminal():
        return _TreeNode(payouts=np.array(state.payouts(num_players), dtype=np.float64))
    if state.is_chance():
        outcomes = state.chance_outcomes()
        return _TreeNode(
            chance_probs=[prob for _, prob in outcomes],
            children=[build_tree(state.apply(outcome), num_players) for outcome, _ in outcomes],
        )
    actions = state.legal_actions()
    return _TreeNode(
        player=state.current_player(),
        info_key=state.info_set_key(),
        num_actions=len(actions),
        children=[build_tree(state.apply(action), num_players) for action in actions],
    )


class FullTraversal(Traversal):
    """Exact (non-sampled) CFR: one iteration is one full recursive walk.

    alternating=True switches to alternating updates, as published CFR+ uses:
    iteration t updates only player (t - 1) % num_players's accumulators. Other
    players' current strategies still drive the walk (and are read to compute
    reach and action values), but neither their regret nor their strategy sum is
    touched that iteration. Default is False so existing (simultaneous-update)
    behavior is unchanged.

    cache_tree=True (the default) resolves the game tree once and walks the resolved
    copy on every later iteration, instead of re-deriving identical structure every
    time. On Leduc that removed 1.4M `apply()` calls, 567k info-set-key string builds
    and 850k payoff computations *per 150 iterations*. The arithmetic is untouched, so
    results are bit-identical.

    The cost is memory proportional to the whole tree — which is why **the sampled
    traversals deliberately do not do this.** External sampling exists to handle trees
    too large to enumerate, and materializing one would take that away. A traversal
    that already walks every node every iteration loses nothing by holding onto it;
    one that walks a single path would lose its entire reason to exist.
    """

    def __init__(self, alternating: bool = False, cache_tree: bool = True):
        self.alternating = alternating
        self.cache_tree = cache_tree
        self._tree: _TreeNode | None = None
        self._tree_game: Game | None = None

    def _resolved_tree(self, game: Game) -> _TreeNode:
        """The cached tree, rebuilt if this traversal is handed a different game.

        Identity, not equality: two games of the same class can carry different
        parameters (Glosten-Milgrom's mu), and a tree silently reused across them
        would solve the wrong game.
        """
        if self._tree is None or self._tree_game is not game:
            self._tree = build_tree(game.new_initial_state(), game.num_players)
            self._tree_game = game
        return self._tree

    def run_iteration(
        self,
        game: Game,
        store: InfoSetStore,
        rule: RegretUpdateRule,
        iteration: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        num_players = game.num_players
        # A plain list, not an ndarray: it is copied once per action at every
        # decision node, and copying a three-element list is far cheaper than
        # copying an array. Nothing here needs vectorized arithmetic.
        reach = [1.0] * (num_players + 1)
        update_player = (iteration - 1) % num_players if self.alternating else None
        if not self.cache_tree:
            return _walk(
                game.new_initial_state(), reach, num_players, store, rule, iteration, update_player
            )
        return _walk_resolved(
            self._resolved_tree(game), reach, num_players, store, rule, iteration, update_player
        )


def _walk_resolved(
    node: _TreeNode,
    reach: list[float],
    num_players: int,
    store: InfoSetStore,
    rule: RegretUpdateRule,
    iteration: int,
    update_player: int | None = None,
) -> np.ndarray:
    """`_walk` over a pre-resolved tree. Same arithmetic, same order, same results —
    it only stops asking the game questions whose answers cannot have changed.
    """
    payouts = node.payouts
    if payouts is not None:
        return payouts

    probs = node.chance_probs
    if probs is not None:
        value = np.zeros(num_players, dtype=np.float64)
        for child, prob in zip(node.children, probs):
            child_reach = reach.copy()
            child_reach[-1] *= prob
            value += prob * _walk_resolved(
                child, child_reach, num_players, store, rule, iteration, update_player
            )
        return value

    player = node.player
    record = store.get(node.info_key, node.num_actions)
    strategy = rule.strategy(record)

    action_values = np.empty((node.num_actions, num_players), dtype=np.float64)
    for i, child in enumerate(node.children):
        child_reach = reach.copy()
        child_reach[player] *= strategy[i]
        action_values[i] = _walk_resolved(
            child, child_reach, num_players, store, rule, iteration, update_player
        )

    node_value = strategy @ action_values

    if update_player is None or player == update_player:
        cf_reach = _counterfactual_reach(reach, player)
        regrets = cf_reach * (action_values[:, player] - node_value[player])
        rule.accumulate_regret(record, regrets, iteration)
        rule.accumulate_strategy(record, strategy, reach[player], iteration)

    return node_value


def _counterfactual_reach(reach: list[float], player: int) -> float:
    """Probability of reaching this node under everyone else's play — opponents and
    chance — with `player`'s own contribution left out. That omission is what makes
    the regret "counterfactual".

    Written as a loop rather than `np.prod(np.delete(reach, player))`, which was 14%
    of a Leduc training run on its own: `delete` allocates a new array for every
    decision node visited, to drop one element from a list of three. The arithmetic
    is identical — `delete` preserves order and `prod` multiplies left to right — so
    the result is bit-for-bit what it was.
    """
    product = 1.0
    for index, value in enumerate(reach):
        if index != player:
            product *= value
    return product


def _walk(
    state: GameState,
    reach: list[float],
    num_players: int,
    store: InfoSetStore,
    rule: RegretUpdateRule,
    iteration: int,
    update_player: int | None = None,
) -> np.ndarray:
    """Recursively evaluate `state`, updating regret/strategy accumulators along the
    way. Returns the per-player expected payoff vector for `state`.

    `reach` has length num_players + 1: reach[i] is player i's contribution to
    reaching `state`, and reach[-1] is chance's contribution. It is a plain list;
    see run_iteration. `update_player`
    restricts accumulation to one player's info sets (alternating updates); None
    updates everyone, i.e. plain simultaneous-update CFR.
    """
    if state.is_terminal():
        # payouts() rather than a payout() per player: a two-player zero-sum game
        # computes the same quantity both times, so asking once halves the work at
        # every terminal node, and terminals are most of the tree.
        return np.array(state.payouts(num_players), dtype=np.float64)

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
        cf_reach = _counterfactual_reach(reach, player)
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
        # payouts() rather than a payout() per player: a two-player zero-sum game
        # computes the same quantity both times, so asking once halves the work at
        # every terminal node, and terminals are most of the tree.
        return np.array(state.payouts(num_players), dtype=np.float64)

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
