from cfr import CFRSolver

if __name__ == "__main__":
    solver = CFRSolver()
    print("Training CFR solver on Kuhn Poker...")
    game_value = solver.train(num_iterations=100000, print_every=10000)
    print(f"\nFinal average game value: {game_value:+.6f} (theory: -0.055556)")
    solver.print_strategies()
