"""The resolved-tree cache in FullTraversal.

The cache answers structural questions once instead of every iteration, which is only
sound because a `GameState` is immutable and a game tree is therefore the same shape
every time. These tests pin the three ways that could go wrong:

* the cached walk disagreeing with the uncached one,
* a terminal's payoff vector being handed out by reference and then mutated, which
  would corrupt the tree silently and permanently,
* one traversal being reused across two *different* games and solving the first one
  twice.

The last is not hypothetical: two `GlostenMilgromGame`s differ only in a constructor
argument, so a cache keyed on anything looser than object identity would happily
solve mu=0.30 and report it as mu=0.70.
"""

import numpy as np
import pytest

from gto_solver.games.glosten_milgrom import GlostenMilgromGame
from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame
from gto_solver.solvers.base import CFRSolver
from gto_solver.solvers.regret_rules import CFRPlusRegretMatching, VanillaRegretMatching
from gto_solver.solvers.traversal import FullTraversal, build_tree

GAMES = [
    pytest.param(KuhnGame, id="kuhn"),
    pytest.param(LeducGame, id="leduc"),
    pytest.param(lambda: GlostenMilgromGame(mu=0.30), id="glosten_milgrom"),
]


def strategies(game_factory, rule, iterations, cache_tree):
    solver = CFRSolver(game_factory(), rule(), FullTraversal(cache_tree=cache_tree))
    solver.train(iterations)
    return solver.average_strategy()


def assert_identical(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert np.array_equal(left[key], right[key]), key


# --- equivalence -----------------------------------------------------------


@pytest.mark.parametrize("game_factory", GAMES)
def test_cached_and_uncached_agree_bit_for_bit(game_factory):
    """The whole claim of the optimization, on every game."""
    iterations = 60
    assert_identical(
        strategies(game_factory, VanillaRegretMatching, iterations, cache_tree=True),
        strategies(game_factory, VanillaRegretMatching, iterations, cache_tree=False),
    )


def test_cached_and_uncached_agree_under_alternating_updates():
    """Alternating updates touch a different branch of the walk, so they get their
    own check rather than riding on the simultaneous one.
    """
    def run(cache_tree):
        solver = CFRSolver(
            LeducGame(),
            CFRPlusRegretMatching(),
            FullTraversal(alternating=True, cache_tree=cache_tree),
        )
        solver.train(40)
        return solver.average_strategy()

    assert_identical(run(True), run(False))


# --- the shared payoff vector must never be mutated ------------------------


def test_training_in_two_calls_matches_training_in_one():
    """A terminal's payoff vector is handed to the walk by reference on every visit.
    If anything downstream wrote through it, the tree would be corrupted for all
    later iterations, and the damage would show up as a split run diverging from a
    single one.
    """
    split = CFRSolver(LeducGame(), VanillaRegretMatching(), FullTraversal())
    split.train(30)
    split.train(30)

    single = CFRSolver(LeducGame(), VanillaRegretMatching(), FullTraversal())
    single.train(60)

    assert_identical(split.average_strategy(), single.average_strategy())


def test_terminal_payoff_vectors_survive_training():
    """Checked directly, not only through its symptoms."""
    game = LeducGame()
    tree = build_tree(game.new_initial_state(), game.num_players)

    def collect(node, out):
        if node.payouts is not None:
            out.append(node.payouts.copy())
            return
        for child in node.children:
            collect(child, out)

    before: list[np.ndarray] = []
    collect(tree, before)

    traversal = FullTraversal()
    solver = CFRSolver(game, VanillaRegretMatching(), traversal)
    solver.train(25)

    after: list[np.ndarray] = []
    collect(traversal._resolved_tree(game), after)
    assert len(before) == len(after)
    for original, current in zip(before, after):
        assert np.array_equal(original, current)


# --- the cache must not outlive the game it was built from -----------------


def test_a_traversal_reused_on_a_different_game_rebuilds_its_tree():
    """Two Glosten-Milgrom games differ only by a constructor argument. Reusing the
    first one's tree would solve mu=0.30 and label the answer mu=0.70.
    """
    traversal = FullTraversal()

    low = GlostenMilgromGame(mu=0.05)
    solver_low = CFRSolver(low, VanillaRegretMatching(), traversal)
    solver_low.train(40)
    reused = solver_low.average_strategy()

    high = GlostenMilgromGame(mu=0.70)
    solver_high = CFRSolver(high, VanillaRegretMatching(), traversal)
    solver_high.train(40)
    rebuilt = solver_high.average_strategy()

    fresh = CFRSolver(GlostenMilgromGame(mu=0.70), VanillaRegretMatching(), FullTraversal())
    fresh.train(40)

    assert_identical(rebuilt, fresh.average_strategy())
    assert any(not np.array_equal(rebuilt[key], reused[key]) for key in rebuilt)


def test_the_tree_is_built_once_and_then_reused():
    game = LeducGame()
    traversal = FullTraversal()
    solver = CFRSolver(game, VanillaRegretMatching(), traversal)
    solver.train(1)
    first = traversal._tree
    solver.train(3)
    assert traversal._tree is first


# --- the resolved tree describes the game it came from ---------------------


@pytest.mark.parametrize(
    "game_factory, expected_info_sets",
    [
        pytest.param(KuhnGame, 12, id="kuhn"),
        pytest.param(LeducGame, 288, id="leduc"),
    ],
)
def test_the_resolved_tree_finds_every_info_set(game_factory, expected_info_sets):
    game = game_factory()
    tree = build_tree(game.new_initial_state(), game.num_players)

    keys: set[str] = set()

    def collect(node):
        if node.payouts is None and node.chance_probs is None:
            keys.add(node.info_key)
        for child in node.children:
            collect(child)

    collect(tree)
    assert len(keys) == expected_info_sets


def test_a_terminal_node_carries_a_payoff_vector_and_nothing_else():
    game = KuhnGame()
    tree = build_tree(game.new_initial_state(), game.num_players)

    def first_terminal(node):
        if node.payouts is not None:
            return node
        for child in node.children:
            found = first_terminal(child)
            if found is not None:
                return found
        return None

    terminal = first_terminal(tree)
    assert terminal is not None
    assert len(terminal.payouts) == 2
    assert terminal.payouts.sum() == pytest.approx(0.0)
    assert terminal.children == ()
