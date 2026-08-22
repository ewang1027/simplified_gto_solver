"""Regret update rules: how cumulative regret becomes a strategy, and how the
regret/strategy accumulators are updated each iteration.

Vanilla CFR lives here; CFR+ and DCFR are added later as sibling classes.
"""

import numpy as np

from gto_solver.solvers.base import InfoSetRecord, RegretUpdateRule, regret_matching


class VanillaRegretMatching(RegretUpdateRule):
    """Unweighted cumulative regret and strategy sums (vanilla CFR)."""

    def strategy(self, record: InfoSetRecord) -> np.ndarray:
        return regret_matching(record.cumulative_regret)

    def accumulate_regret(
        self, record: InfoSetRecord, action_regrets: np.ndarray, iteration: int
    ) -> None:
        record.cumulative_regret += action_regrets

    def accumulate_strategy(
        self,
        record: InfoSetRecord,
        strategy: np.ndarray,
        reach_prob: float,
        iteration: int,
    ) -> None:
        record.strategy_sum += reach_prob * strategy
