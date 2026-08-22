"""Core CFR machinery: regret storage, the update-rule/traversal interfaces, and the
solver that composes them.

Every CFR variant in this package is one RegretUpdateRule plus one Traversal, so the
tree-walking code and the regret store are written once.
"""

from abc import ABC, abstractmethod

import numpy as np

from gto_solver.games.base import Game


class InfoSetRecord:
    """Regret and strategy accumulators for a single information set.

    Action count is per-info-set, not global: Leduc and the market-making game have
    different numbers of legal actions at different nodes.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.cumulative_regret = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(self.num_actions, 1.0 / self.num_actions)


class InfoSetStore:
    """Lazily-populated map from info-set key to its accumulators."""

    def __init__(self):
        self._records: dict[str, InfoSetRecord] = {}

    def get(self, key: str, num_actions: int) -> InfoSetRecord:
        record = self._records.get(key)
        if record is None:
            record = InfoSetRecord(num_actions)
            self._records[key] = record
        return record

    def average_strategy(self) -> dict[str, np.ndarray]:
        return {key: r.average_strategy() for key, r in sorted(self._records.items())}

    def __len__(self) -> int:
        return len(self._records)


class RegretUpdateRule(ABC):
    """How cumulative regret becomes a strategy, and how accumulators are updated.

    This is the only thing that differs between vanilla CFR, CFR+, and DCFR.
    """

    @abstractmethod
    def strategy(self, record: InfoSetRecord) -> np.ndarray:
        """Current strategy from cumulative regret (regret matching)."""

    @abstractmethod
    def accumulate_regret(
        self, record: InfoSetRecord, action_regrets: np.ndarray, iteration: int
    ) -> None: ...

    @abstractmethod
    def accumulate_strategy(
        self,
        record: InfoSetRecord,
        strategy: np.ndarray,
        reach_prob: float,
        iteration: int,
    ) -> None: ...


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Strategy proportional to positive regret; uniform when there is none."""
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    return np.full(len(regrets), 1.0 / len(regrets))


class Traversal(ABC):
    """How one training iteration walks the tree."""

    @abstractmethod
    def run_iteration(
        self,
        game: Game,
        store: InfoSetStore,
        rule: RegretUpdateRule,
        iteration: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Update accumulators in place. Returns expected payoffs per player."""


class CFRSolver:
    """Composes a traversal with a regret update rule to solve a game."""

    def __init__(
        self,
        game: Game,
        rule: RegretUpdateRule,
        traversal: Traversal,
        seed: int = 0,
    ):
        self.game = game
        self.rule = rule
        self.traversal = traversal
        self.store = InfoSetStore()
        self.rng = np.random.default_rng(seed)
        self.iterations = 0

    def train(self, num_iterations: int) -> None:
        for _ in range(num_iterations):
            self.iterations += 1
            self.traversal.run_iteration(
                self.game, self.store, self.rule, self.iterations, self.rng
            )

    def average_strategy(self) -> dict[str, np.ndarray]:
        return self.store.average_strategy()
