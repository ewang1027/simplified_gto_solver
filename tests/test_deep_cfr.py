"""Deep CFR: the reservoir, the featurization contract, and convergence.

The phase this belongs to exists for a comparison, not for the network, so the tests
that matter are the ones that would let a *wrong* comparison through:

* a featurization that leaks hidden information would train a network on knowledge the
  player does not have, and the resulting strategy would look good against
  exploitability while being unimplementable;
* a reservoir that is not uniform would fit the value network to whichever iterations
  happened to be recent;
* and a solver that does not actually learn would still produce a plausible-looking
  exploitability, because a uniform strategy on Kuhn is only 0.46 away from one.
"""

import numpy as np
import pytest

from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.deep_cfr import DeepCFRSolver, Reservoir, enumerate_info_sets

# --- the featurization contract --------------------------------------------


@pytest.mark.parametrize(
    "game, expected_info_sets", [(KuhnGame(), 12), (LeducGame(), 288)]
)
def test_states_sharing_an_info_set_share_their_features(game, expected_info_sets):
    """The property the whole encoding rests on. Two states a player cannot tell
    apart must encode identically, or the network is being handed the hidden
    information that makes the game a game.
    """
    seen: dict[str, np.ndarray] = {}
    checked = 0

    def walk(state):
        nonlocal checked
        if state.is_terminal():
            return
        if state.is_chance():
            for outcome, _ in state.chance_outcomes():
                walk(state.apply(outcome))
            return
        key = state.info_set_key()
        features = state.features()
        if key in seen:
            assert np.array_equal(seen[key], features), key
            checked += 1
        else:
            seen[key] = features
        for action in state.legal_actions():
            walk(state.apply(action))

    walk(game.new_initial_state())
    assert len(seen) == expected_info_sets
    assert checked > 0, "no info set had two member states, so nothing was compared"


@pytest.mark.parametrize("game", [KuhnGame(), LeducGame()])
def test_features_are_fixed_length_and_finite(game):
    sizes = {len(features) for features, _ in enumerate_info_sets(game).values()}
    assert len(sizes) == 1
    for features, _ in enumerate_info_sets(game).values():
        assert np.isfinite(features).all()


def test_different_info_sets_mostly_get_different_features():
    """A collision means two distinguishable situations look identical to the network,
    which caps how well any amount of training can do.
    """
    encodings = {tuple(features) for features, _ in enumerate_info_sets(LeducGame()).values()}
    assert len(encodings) == 288


def test_a_game_without_features_is_refused_with_an_explanation():
    """Glosten-Milgrom implements no featurization, so Deep CFR should say so rather
    than invent an encoding on the game's behalf.
    """
    from gto_solver.games.glosten_milgrom import GlostenMilgromGame

    with pytest.raises(ValueError, match="features()"):
        DeepCFRSolver(GlostenMilgromGame(mu=0.3))


# --- the reservoir ---------------------------------------------------------


def test_the_reservoir_holds_everything_below_capacity():
    reservoir = Reservoir(10, np.random.default_rng(0))
    for i in range(7):
        reservoir.add(i)
    assert len(reservoir) == 7
    assert reservoir.seen == 7


def test_the_reservoir_stays_at_capacity_and_keeps_counting():
    reservoir = Reservoir(10, np.random.default_rng(0))
    for i in range(1000):
        reservoir.add(i)
    assert len(reservoir) == 10
    assert reservoir.seen == 1000


def test_the_reservoir_samples_uniformly_not_recently():
    """A sliding window would fit the value network to whatever the last few
    iterations happened to visit. Over many trials each position of the stream should
    survive about equally often; a recency bias shows up as the tail dominating.
    """
    capacity, stream, trials = 10, 200, 400
    counts = np.zeros(stream)
    for trial in range(trials):
        reservoir = Reservoir(capacity, np.random.default_rng(trial))
        for i in range(stream):
            reservoir.add(i)
        counts[list(reservoir._items)] += 1

    expected = trials * capacity / stream
    first_half, second_half = counts[: stream // 2].sum(), counts[stream // 2 :].sum()
    assert 0.85 < first_half / second_half < 1.18, (first_half, second_half)
    assert counts.max() < 4 * expected


def test_a_reservoir_needs_positive_capacity():
    with pytest.raises(ValueError):
        Reservoir(0, np.random.default_rng(0))


# --- the solver ------------------------------------------------------------


def test_it_exposes_the_same_surface_as_the_tabular_solvers():
    """This is what lets the Phase 5 harness measure it with the same seeds and
    checkpoints, and the same exploitability metric, as everything else.
    """
    solver = DeepCFRSolver(KuhnGame(), seed=0, traversals=5)
    assert solver.iterations == 0
    solver.train(2)
    assert solver.iterations == 2
    assert len(solver.store) == 12
    strategy = solver.average_strategy()
    assert set(strategy) == set(enumerate_info_sets(KuhnGame()))


def test_every_strategy_it_reports_is_a_probability_distribution():
    solver = DeepCFRSolver(KuhnGame(), seed=1, traversals=5)
    solver.train(3)
    for key, probs in solver.average_strategy().items():
        assert probs.sum() == pytest.approx(1.0), key
        assert (probs >= 0).all(), key


def test_an_untrained_solver_reports_a_uniform_strategy():
    """Not an empty dict and not a network's uninitialized output: uniform is what an
    average strategy over zero iterations actually is.
    """
    for probs in DeepCFRSolver(KuhnGame(), seed=0).average_strategy().values():
        assert probs == pytest.approx(np.full(len(probs), 1.0 / len(probs)))


def test_it_learns_a_strategy_far_better_than_uniform():
    """The check that a plausible-looking number is really learning. Uniform play on
    Kuhn is exploitable for about 0.46; anything near that means the network is not
    fitting regret at all.
    """
    game = KuhnGame()
    uniform = exploitability(game, DeepCFRSolver(game, seed=0).average_strategy())
    solver = DeepCFRSolver(game, seed=0, traversals=30)
    solver.train(60)
    trained = exploitability(game, solver.average_strategy())
    assert uniform > 0.3
    assert trained < uniform / 5


def test_the_same_seed_gives_the_same_run_and_a_different_one_does_not():
    """It is registered as stochastic, so both halves need to hold: reproducible
    under a fixed seed, and genuinely different across seeds.
    """
    def run(seed):
        solver = DeepCFRSolver(KuhnGame(), seed=seed, traversals=5)
        solver.train(3)
        return solver.average_strategy()

    first, again, other = run(4), run(4), run(5)
    for key in first:
        assert np.array_equal(first[key], again[key]), key
    assert any(not np.array_equal(first[key], other[key]) for key in first)


def test_it_is_worse_than_tabular_cfr_at_equal_iterations():
    """The result the phase exists to report, and not a failure: Deep CFR approximates
    a table that these games are small enough to just hold. It exists for games whose
    info sets cannot be enumerated -- which is also why enumerate_info_sets() is used
    only to score it, never to solve.
    """
    from gto_solver.solvers.base import CFRSolver
    from gto_solver.solvers.regret_rules import VanillaRegretMatching
    from gto_solver.solvers.traversal import FullTraversal

    game = KuhnGame()
    tabular = CFRSolver(game, VanillaRegretMatching(), FullTraversal())
    tabular.train(100)
    neural = DeepCFRSolver(game, seed=0, traversals=30)
    neural.train(100)

    assert exploitability(game, tabular.average_strategy()) < exploitability(
        game, neural.average_strategy()
    )
