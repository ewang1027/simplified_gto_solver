"""Deep CFR (Brown et al. 2019): counterfactual regret with a learned value function.

**This does not compose from `RegretUpdateRule` x `Traversal`, and that is a finding
rather than an oversight.** Those two abstractions carry every tabular variant in this
package — vanilla, CFR+, DCFR, Linear, alternating, external sampling — because each
differs only in how cumulative regret becomes a strategy or how one iteration walks the
tree. Deep CFR changes neither. It changes *what the regret store is*: a network fitted
to sampled regrets rather than a table of them, retrained from scratch every iteration.
No composition of a rule and a traversal expresses that, so this is its own module.

What it does share is the part that makes the comparison worth anything: it produces a
`dict[str, np.ndarray]` average strategy like every other solver here, so
`metrics/exploitability.py` scores it against the same yardstick, and it exposes the
same `train()` / `average_strategy()` / `iterations` / `store` surface, so the Phase 5
benchmark harness measures it with the same seeds, checkpoints and bands.

The algorithm, per iteration and per player:

1. Run `traversals` external-sampling walks, collecting (info set, instantaneous
   counterfactual regret) into a reservoir, and (info set, strategy) into a shared
   strategy reservoir.
2. Refit that player's value network **from scratch** on its whole reservoir, with each
   sample weighted by the iteration that produced it — later iterations reflect a
   better strategy and should count more.
3. The current strategy at an info set is regret matching on the network's output.

At the end, a policy network is fitted to the strategy reservoir; its output is the
average strategy. That last step is what makes this Deep CFR rather than "CFR with a
network in the middle": the average strategy is never stored in a table at all.

Expect it to lose to tabular CFR on games this small, and read that as the design
working. Deep CFR exists for games whose info sets cannot be enumerated; on twelve of
them, approximating a table is strictly worse than being one.
"""

from collections.abc import Callable

import numpy as np

from gto_solver.games.base import Game, GameState
from gto_solver.nn import MLP
from gto_solver.solvers.base import regret_matching


class Reservoir:
    """Fixed-capacity uniform sample of an unbounded stream (Vitter's algorithm R).

    Deep CFR's memories must stay a uniform sample of *all* samples ever collected, not
    the most recent ones: a sliding window would fit the network to whatever the last
    few iterations happened to visit, and the early iterations are where the strategy
    is most wrong and therefore most informative about regret.
    """

    def __init__(self, capacity: int, rng: np.random.Generator):
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._rng = rng
        self._items: list = []
        self.seen = 0

    def add(self, item) -> None:
        self.seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        index = int(self._rng.integers(0, self.seen))
        if index < self.capacity:
            self._items[index] = item

    def __len__(self) -> int:
        return len(self._items)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(features, targets, weights) stacked, ready for `MLP.fit`."""
        features = np.stack([item[0] for item in self._items])
        targets = np.stack([item[1] for item in self._items])
        weights = np.array([item[2] for item in self._items], dtype=np.float64)
        return features, targets, weights


def enumerate_info_sets(game: Game) -> dict[str, tuple[np.ndarray, int]]:
    """Every info set in the game, with its features and action count.

    Used to materialize a network's strategy as the dict `exploitability()` consumes.
    Enumerating the tree is exactly what Deep CFR is built to avoid, so this is an
    evaluation-only convenience: it is what makes scoring against tabular ground truth
    possible on these games, and it is the step that would not exist on a real one.
    """
    found: dict[str, tuple[np.ndarray, int]] = {}

    def walk(state: GameState) -> None:
        if state.is_terminal():
            return
        if state.is_chance():
            for outcome, _ in state.chance_outcomes():
                walk(state.apply(outcome))
            return
        key = state.info_set_key()
        if key not in found:
            features = state.features()
            if features is None:
                raise ValueError(
                    f"game {state.__class__.__name__} does not implement features(), so it "
                    f"cannot be solved by function approximation. Implement GameState."
                    f"features() to return a fixed-length encoding of the info set."
                )
            found[key] = (features, len(state.legal_actions()))
        for action in state.legal_actions():
            walk(state.apply(action))

    walk(game.new_initial_state())
    return found


class DeepCFRSolver:
    """Deep CFR, exposing the same surface as `CFRSolver` so the harness can measure it.

    `value_hidden` and `policy_hidden` are deliberately small. These games have twelve
    and 288 info sets; a large network would memorize them and the comparison would say
    nothing about approximation.
    """

    def __init__(
        self,
        game: Game,
        seed: int = 0,
        traversals: int = 30,
        value_hidden: tuple[int, ...] = (32, 32),
        policy_hidden: tuple[int, ...] = (32, 32),
        value_epochs: int = 20,
        policy_epochs: int = 60,
        memory: int = 20_000,
        learning_rate: float = 5e-3,
    ):
        self.game = game
        self.seed = seed
        self.traversals = traversals
        self.value_epochs = value_epochs
        self.policy_epochs = policy_epochs
        self.rng = np.random.default_rng(seed)
        self.iterations = 0

        self.info_sets = enumerate_info_sets(game)
        self.num_actions = max(count for _, count in self.info_sets.values())
        feature_size = len(next(iter(self.info_sets.values()))[0])

        self.value_networks = [
            MLP([feature_size, *value_hidden, self.num_actions],
                seed=seed * 977 + player, learning_rate=learning_rate)
            for player in range(game.num_players)
        ]
        self.policy_network = MLP(
            [feature_size, *policy_hidden, self.num_actions],
            seed=seed * 977 + 91,
            learning_rate=learning_rate,
        )
        self.value_memory = [
            Reservoir(memory, self.rng) for _ in range(game.num_players)
        ]
        self.strategy_memory = Reservoir(memory, self.rng)
        # Named `store` so `len(solver.store)` means the same thing it does for the
        # tabular solvers: how many info sets this run has actually touched.
        self.store: dict[str, None] = {}

    # --- strategy from the value network -----------------------------------

    def _strategy(self, player: int, features: np.ndarray, num_actions: int) -> np.ndarray:
        """Regret matching on predicted regret, restricted to the legal actions.

        The network has a fixed output width — the widest action count in the game — so
        nodes with fewer actions read a prefix. Slicing rather than masking keeps the
        prefix meaning the same thing at every node, which is what lets one network
        serve nodes of different arity at all.
        """
        predicted = self.value_networks[player].predict(features[None, :])[0, :num_actions]
        return regret_matching(predicted)

    # --- one external-sampling traversal -----------------------------------

    def _traverse(self, state: GameState, traverser: int, iteration: int) -> float:
        if state.is_terminal():
            return state.payout(traverser)
        if state.is_chance():
            outcomes, probabilities = zip(*state.chance_outcomes())
            outcome = outcomes[self.rng.choice(len(outcomes), p=probabilities)]
            return self._traverse(state.apply(outcome), traverser, iteration)

        player = state.current_player()
        actions = state.legal_actions()
        key = state.info_set_key()
        features = state.features()
        self.store.setdefault(key, None)
        strategy = self._strategy(player, features, len(actions))

        if player == traverser:
            values = np.array(
                [self._traverse(state.apply(a), traverser, iteration) for a in actions]
            )
            node_value = float(strategy @ values)
            regrets = np.zeros(self.num_actions, dtype=np.float64)
            regrets[: len(actions)] = values - node_value
            self.value_memory[player].add((features, regrets, float(iteration)))
            return node_value

        # Opponent node: record what it would have done, then sample one action.
        padded = np.zeros(self.num_actions, dtype=np.float64)
        padded[: len(actions)] = strategy
        self.strategy_memory.add((features, padded, float(iteration)))
        choice = self.rng.choice(len(actions), p=strategy)
        return self._traverse(state.apply(actions[choice]), traverser, iteration)

    # --- the training loop -------------------------------------------------

    def train(self, num_iterations: int) -> None:
        for _ in range(num_iterations):
            self.iterations += 1
            for player in range(self.game.num_players):
                for _ in range(self.traversals):
                    self._traverse(self.game.new_initial_state(), player, self.iterations)
                memory = self.value_memory[player]
                if len(memory) == 0:
                    continue
                features, targets, weights = memory.arrays()
                # From scratch, not fine-tuned: the regret targets move as the strategy
                # does, and a carried-over network fits the previous iteration's first.
                self.value_networks[player].reset(seed=self.seed * 977 + player)
                self.value_networks[player].fit(
                    features, targets, weights,
                    epochs=self.value_epochs, rng=self.rng,
                )

    def average_strategy(self) -> dict[str, np.ndarray]:
        """The policy network's strategy at every info set.

        Fitted here rather than incrementally, because the strategy reservoir keeps
        growing and the network should see all of it. Falls back to uniform before any
        training, which is what an untrained solver's average strategy is.
        """
        if len(self.strategy_memory) == 0:
            return {
                key: np.full(count, 1.0 / count)
                for key, (_, count) in sorted(self.info_sets.items())
            }

        features, targets, weights = self.strategy_memory.arrays()
        self.policy_network.reset(seed=self.seed * 977 + 91)
        self.policy_network.fit(
            features, targets, weights, epochs=self.policy_epochs, rng=self.rng
        )

        strategy: dict[str, np.ndarray] = {}
        keys = sorted(self.info_sets)
        matrix = np.stack([self.info_sets[key][0] for key in keys])
        predictions = self.policy_network.predict(matrix)
        for row, key in enumerate(keys):
            count = self.info_sets[key][1]
            # The policy net is fitted to probability vectors, so its output is close
            # to one already; clipping and renormalizing is what makes it exactly one.
            positive = np.maximum(predictions[row, :count], 0.0)
            total = positive.sum()
            strategy[key] = (
                positive / total if total > 0 else np.full(count, 1.0 / count)
            )
        return strategy


def make_deep_cfr(**kwargs) -> Callable[[Game, int], DeepCFRSolver]:
    """A builder for `AlgorithmSpec.make_solver`."""

    def build(game: Game, seed: int) -> DeepCFRSolver:
        return DeepCFRSolver(game, seed=seed, **kwargs)

    return build
