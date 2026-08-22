"""Regret update rules: how cumulative regret becomes a strategy, and how the
regret/strategy accumulators are updated each iteration.

Vanilla CFR, CFR+, and Discounted/Linear CFR all live here as sibling classes.
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


class CFRPlusRegretMatching(RegretUpdateRule):
    """CFR+ (Tammelin 2014).

    Regret-matching+ floors cumulative regret at zero so it never goes
    arbitrarily negative (unlike vanilla), which lets a recently-bad action
    recover instantly instead of "digging out of a hole". Strategy averaging is
    weighted linearly by iteration number, since later iterations are closer to
    equilibrium and should count more than early, noisy ones.
    """

    def strategy(self, record: InfoSetRecord) -> np.ndarray:
        return regret_matching(record.cumulative_regret)

    def accumulate_regret(
        self, record: InfoSetRecord, action_regrets: np.ndarray, iteration: int
    ) -> None:
        record.cumulative_regret = np.maximum(record.cumulative_regret + action_regrets, 0.0)

    def accumulate_strategy(
        self,
        record: InfoSetRecord,
        strategy: np.ndarray,
        reach_prob: float,
        iteration: int,
    ) -> None:
        record.strategy_sum += iteration * reach_prob * strategy


class DiscountedRegretMatching(RegretUpdateRule):
    """Discounted CFR (Brown & Sandholm, AAAI 2019).

    Positive regret, negative regret, and the strategy sum each get their own
    discount, so early (noisy) iterations fade as training progresses. Linear CFR
    is the special case alpha=beta=gamma=1 -- see linear_cfr() below. Defaults are
    the paper's recommended DCFR settings.

    Following the paper, the discount applies to what was accumulated BEFORE this
    iteration; the current iteration's contribution is then added undiscounted:

        R^t = R^{t-1} * d(t-1) + r^t

    where d uses alpha when R^{t-1} is positive and beta when it isn't.
    """

    def __init__(self, alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def strategy(self, record: InfoSetRecord) -> np.ndarray:
        return regret_matching(record.cumulative_regret)

    def accumulate_regret(
        self, record: InfoSetRecord, action_regrets: np.ndarray, iteration: int
    ) -> None:
        prior = record.cumulative_regret
        elapsed = iteration - 1
        pos_discount = elapsed**self.alpha / (elapsed**self.alpha + 1)
        # beta=0 makes elapsed**0 == 1 for every iteration, so neg_discount is
        # always 1/2 -- that's the paper's recommended setting, not a bug.
        neg_discount = elapsed**self.beta / (elapsed**self.beta + 1)
        discounted = np.where(prior > 0, prior * pos_discount, prior * neg_discount)
        record.cumulative_regret = discounted + action_regrets

    def accumulate_strategy(
        self,
        record: InfoSetRecord,
        strategy: np.ndarray,
        reach_prob: float,
        iteration: int,
    ) -> None:
        record.strategy_sum *= ((iteration - 1) / iteration) ** self.gamma
        record.strategy_sum += reach_prob * strategy


def linear_cfr() -> DiscountedRegretMatching:
    """Linear CFR: DCFR with alpha=beta=gamma=1, i.e. every accumulator weighted
    linearly by iteration number.
    """
    return DiscountedRegretMatching(1.0, 1.0, 1.0)
