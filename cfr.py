from kuhn import KuhnState, Pass, Bet, cards, num_actions
import numpy as np
from itertools import permutations


class InformationSet:
    """Tracks regret and strategy sum for one information set."""

    def __init__(self):
        self.cumulative_regret = np.zeros(num_actions, dtype=np.float64)
        self.strat_sum = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self):
        positive_regret = np.maximum(self.cumulative_regret, 0)
        total = positive_regret.sum()
        if total > 0:
            return positive_regret / total
        return np.ones(num_actions) / num_actions

    def get_average_strat(self):
        total = self.strat_sum.sum()
        if total > 0:
            return self.strat_sum / total
        return np.ones(num_actions) / num_actions


class CFRSolver:
    """Vanilla CFR solver for Kuhn Poker."""

    def __init__(self):
        self.info_sets: dict[str, InformationSet] = {}
        self.iterations = 0

    def _get_info_set(self, key: str) -> InformationSet:
        if key not in self.info_sets:
            self.info_sets[key] = InformationSet()
        return self.info_sets[key]

    def cfr(self, state: KuhnState, reach_probs: np.ndarray) -> float:
        """
        Recursive CFR traversal. Returns expected value for Player 0.

        reach_probs[i] = probability player i plays to reach this node.
        """
        if state.is_terminal():
            return state.payout()

        player = state.current_player()
        opponent = 1 - player
        info_set = self._get_info_set(state.info_set())
        actions = state.get_actions()

        strategy = info_set.get_strategy()
        # Accumulate weighted strategy for average-strategy computation
        info_set.strat_sum += reach_probs[player] * strategy

        action_values = np.zeros(num_actions, dtype=np.float64)
        node_value = 0.0

        for i, action in enumerate(actions):
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_values[i] = self.cfr(state.use_action(action), new_reach)
            node_value += strategy[i] * action_values[i]

        # P0 maximizes, P1 minimizes P0's value, so negate regret at P1's nodes
        sign = 1 if player == 0 else -1
        for i in range(num_actions):
            info_set.cumulative_regret[i] += reach_probs[opponent] * sign * (action_values[i] - node_value)

        return node_value

    def train(self, num_iterations: int, print_every: int = 1000) -> float:
        """
        Run CFR for num_iterations. Each iteration cycles through all 6
        possible card deals so every information set is updated every iteration.
        Returns the average game value for Player 0.
        """
        all_deals = list(permutations(range(len(cards)), 2))
        cumulative_game_value = 0.0

        for t in range(1, num_iterations + 1):
            iteration_value = 0.0
            for deal in all_deals:
                state = KuhnState(cards=list(deal))
                iteration_value += self.cfr(state, np.ones(2))
            avg_value = iteration_value / len(all_deals)
            cumulative_game_value += avg_value
            self.iterations = t

            if t % print_every == 0:
                running_avg = cumulative_game_value / t
                print(f"Iteration {t:>6d} | Avg game value: {running_avg:+.6f} (theory: -0.055556)")

        return cumulative_game_value / num_iterations

    def get_strategies(self) -> dict[str, np.ndarray]:
        """Return the average strategy for every information set."""
        return {key: self.info_sets[key].get_average_strat() for key in sorted(self.info_sets)}

    def print_strategies(self):
        print("\n=== Nash Equilibrium Strategies ===")
        print(f"{'Info Set':<12} {'Check/Fold':>12} {'Bet/Call':>10}")
        print("-" * 36)
        for key, avg in self.get_strategies().items():
            print(f"{key:<12} {avg[Pass]:>12.4f} {avg[Bet]:>10.4f}")
