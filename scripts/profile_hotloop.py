"""Profile the CFR hot loop, so Phase 6 optimizes what is actually expensive.

    python scripts/profile_hotloop.py            # Leduc, exact traversal
    python scripts/profile_hotloop.py --game kuhn --algorithm mccfr

Prints cProfile's cumulative and total-time rankings, plus a per-node cost derived
from the number of tree nodes actually walked. The point of the per-node figure is
that "40% of time in regret_matching" means nothing without knowing how many times a
node is visited: the tree walk visits every node of every iteration, so a microsecond
there is multiplied by hundreds of thousands.

Profiling overhead is real (cProfile roughly doubles the runtime of a call-heavy
workload like this one), so the ratios here are the signal and the absolute times are
not. Timings that get published come from scripts/benchmark.py, which does not profile.
"""

import argparse
import cProfile
import io
import pstats
import time

from gto_solver.games.glosten_milgrom import GlostenMilgromGame
from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame
from gto_solver.solvers.registry import get_algorithm

GAMES = {
    "kuhn": KuhnGame,
    "leduc": LeducGame,
    "gm": lambda: GlostenMilgromGame(mu=0.30),
}


def count_nodes(state) -> tuple[int, int]:
    """(decision nodes, total nodes) in the whole tree, walked once."""
    if state.is_terminal():
        return 0, 1
    if state.is_chance():
        decisions, total = 0, 1
        for outcome, _ in state.chance_outcomes():
            d, t = count_nodes(state.apply(outcome))
            decisions += d
            total += t
        return decisions, total
    decisions, total = 1, 1
    for action in state.legal_actions():
        d, t = count_nodes(state.apply(action))
        decisions += d
        total += t
    return decisions, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", choices=sorted(GAMES), default="leduc")
    parser.add_argument("--algorithm", default="vanilla")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()

    game = GAMES[args.game]()
    spec = get_algorithm(args.algorithm)
    decisions, total = count_nodes(game.new_initial_state())

    print(f"{game.name} / {spec.name}: {args.iterations} iterations")
    print(f"tree: {decisions:,} decision nodes, {total:,} nodes total")

    unprofiled = spec.build(GAMES[args.game](), seed=0)
    start = time.perf_counter()
    unprofiled.train(args.iterations)
    elapsed = time.perf_counter() - start
    print(
        f"unprofiled: {elapsed:.3f}s for {args.iterations} iterations "
        f"({args.iterations / elapsed:,.0f} it/s, "
        f"{elapsed / args.iterations / total * 1e9:,.0f} ns per node visit)\n"
    )

    solver = spec.build(GAMES[args.game](), seed=0)
    profiler = cProfile.Profile()
    profiler.enable()
    solver.train(args.iterations)
    profiler.disable()

    for sort, title in (("tottime", "by total time in the function itself"),
                        ("cumtime", "by cumulative time")):
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(sort).print_stats(args.top)
        print(f"--- {title} " + "-" * (60 - len(title)))
        body = stream.getvalue().split("ncalls", 1)
        print("ncalls" + body[1] if len(body) > 1 else stream.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
