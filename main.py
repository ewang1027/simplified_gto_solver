"""Train vanilla CFR on Kuhn poker and report convergence by exploitability."""

from gto_solver.games.kuhn import BET, PASS, KuhnGame
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.base import CFRSolver
from gto_solver.solvers.regret_rules import VanillaRegretMatching
from gto_solver.solvers.traversal import FullTraversal

KUHN_GAME_VALUE = -1 / 18


def main() -> None:
    game = KuhnGame()
    solver = CFRSolver(game, VanillaRegretMatching(), FullTraversal())

    print("Training vanilla CFR on Kuhn poker...\n")
    print(f"{'Iteration':>10} {'Exploitability':>16}")
    print("-" * 27)

    checkpoints = [10, 100, 1000, 10_000]
    trained = 0
    for target in checkpoints:
        solver.train(target - trained)
        trained = target
        print(f"{trained:>10d} {exploitability(game, solver.average_strategy()):>16.8f}")

    strategy = solver.average_strategy()
    print("\nExploitability -> 0 means the strategy profile is a Nash equilibrium.")
    print(f"Info sets discovered: {len(solver.store)}")

    print("\n=== Nash Equilibrium Strategies ===")
    print(f"{'Info Set':<12} {'Check/Fold':>12} {'Bet/Call':>10}")
    print("-" * 36)
    for key, probs in strategy.items():
        print(f"{key:<12} {probs[PASS]:>12.4f} {probs[BET]:>10.4f}")


if __name__ == "__main__":
    main()
