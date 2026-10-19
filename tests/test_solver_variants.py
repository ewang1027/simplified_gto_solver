"""Phase 2 solver variants: CFR+, Discounted/Linear CFR, alternating updates, and
external-sampling MCCFR. Same rigor gate as Phase 1's vanilla CFR test -- drive
exploitability toward zero on Kuhn -- applied to every variant.

Thresholds below are set from measured convergence (see the Phase 2 scratchpad
convergence script), with comfortable margin, not guessed:
  - vanilla/CFR+/DCFR/LinearCFR/alternating are all under 0.005 by iteration
    3000; 0.01 leaves >50% margin.
  - MCCFR needs far more iterations for comparable accuracy since each iteration
    is a single sample, not an exact expectation: seeds 0-2 are all under 0.005
    by 30000 iterations, so 0.02 leaves a wide margin while still being a real
    convergence check.
"""

from collections import Counter

import pytest

from gto_solver.games.glosten_milgrom import GlostenMilgromGame
from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.base import CFRSolver, RegretUpdateRule
from gto_solver.solvers.regret_rules import (
    CFRPlusRegretMatching,
    DiscountedRegretMatching,
    VanillaRegretMatching,
    linear_cfr,
)
from gto_solver.solvers.traversal import ExternalSamplingMCCFR, FullTraversal

KUHN_INFO_SETS = 12


def make_solver(rule, traversal, seed: int = 0) -> CFRSolver:
    return CFRSolver(KuhnGame(), rule, traversal, seed=seed)


EXACT_VARIANTS = [
    pytest.param(VanillaRegretMatching, FullTraversal, id="vanilla"),
    pytest.param(CFRPlusRegretMatching, FullTraversal, id="cfr_plus"),
    pytest.param(DiscountedRegretMatching, FullTraversal, id="dcfr"),
    pytest.param(linear_cfr, FullTraversal, id="linear_cfr"),
]

ALL_VARIANTS = EXACT_VARIANTS + [
    pytest.param(
        CFRPlusRegretMatching, lambda: FullTraversal(alternating=True), id="cfr_plus_alternating"
    ),
    pytest.param(VanillaRegretMatching, ExternalSamplingMCCFR, id="mccfr"),
]


# --- Convergence: every variant drives exploitability toward zero ----------


@pytest.mark.parametrize("rule_factory, traversal_factory", EXACT_VARIANTS)
def test_exact_variants_converge(rule_factory, traversal_factory):
    game = KuhnGame()
    solver = make_solver(rule_factory(), traversal_factory())
    solver.train(3000)
    assert exploitability(game, solver.average_strategy()) < 0.01


def test_alternating_cfr_plus_converges():
    """Published CFR+ uses alternating updates: iteration t updates only one
    player, so each player gets roughly half as many real updates by iteration
    3000 as the simultaneous variants above -- still comfortably converges.
    """
    game = KuhnGame()
    solver = make_solver(CFRPlusRegretMatching(), FullTraversal(alternating=True))
    solver.train(3000)
    assert exploitability(game, solver.average_strategy()) < 0.01


def test_alternating_vanilla_also_converges():
    """Alternating is a Traversal-level flag, independent of the regret rule."""
    game = KuhnGame()
    solver = make_solver(VanillaRegretMatching(), FullTraversal(alternating=True))
    solver.train(3000)
    assert exploitability(game, solver.average_strategy()) < 0.01


def test_mccfr_converges_below_looser_threshold():
    game = KuhnGame()
    solver = make_solver(VanillaRegretMatching(), ExternalSamplingMCCFR())
    solver.train(30000)
    assert exploitability(game, solver.average_strategy()) < 0.02


# --- Structural correctness: every variant finds all 12 info sets and ------
# --- produces valid probability distributions -------------------------------


@pytest.mark.parametrize("rule_factory, traversal_factory", ALL_VARIANTS)
def test_discovers_every_info_set_and_valid_distributions(rule_factory, traversal_factory):
    solver = make_solver(rule_factory(), traversal_factory())
    solver.train(200)
    assert len(solver.store) == KUHN_INFO_SETS
    for key, probs in solver.average_strategy().items():
        assert probs.sum() == pytest.approx(1.0), key
        assert (probs >= 0).all(), key


# --- MCCFR reproducibility: seeded rng, no bare np.random calls ------------


def test_mccfr_same_seed_gives_identical_strategy():
    solver_a = make_solver(VanillaRegretMatching(), ExternalSamplingMCCFR(), seed=7)
    solver_b = make_solver(VanillaRegretMatching(), ExternalSamplingMCCFR(), seed=7)
    solver_a.train(500)
    solver_b.train(500)

    strategy_a = solver_a.average_strategy()
    strategy_b = solver_b.average_strategy()
    assert strategy_a.keys() == strategy_b.keys()
    for key in strategy_a:
        assert (strategy_a[key] == strategy_b[key]).all(), key


def test_mccfr_different_seeds_give_different_strategies():
    solver_a = make_solver(VanillaRegretMatching(), ExternalSamplingMCCFR(), seed=1)
    solver_b = make_solver(VanillaRegretMatching(), ExternalSamplingMCCFR(), seed=2)
    solver_a.train(500)
    solver_b.train(500)

    strategy_a = solver_a.average_strategy()
    strategy_b = solver_b.average_strategy()
    assert any(not (strategy_a[key] == strategy_b[key]).all() for key in strategy_a)


# --- Updates apply once per info set per iteration -------------------------


class RecordingRule(RegretUpdateRule):
    """Wraps a real rule and logs every read and update, in order."""

    def __init__(self, inner: RegretUpdateRule):
        self.inner = inner
        self.events: list[tuple[str, int]] = []

    def strategy(self, record):
        self.events.append(("read", id(record)))
        return self.inner.strategy(record)

    def accumulate_regret(self, record, action_regrets, iteration):
        self.events.append(("regret", id(record)))
        self.inner.accumulate_regret(record, action_regrets, iteration)

    def accumulate_strategy(self, record, strategy, reach_prob, iteration):
        self.events.append(("strategy", id(record)))
        self.inner.accumulate_strategy(record, strategy, reach_prob, iteration)


@pytest.mark.parametrize(
    "rule_factory", [VanillaRegretMatching, CFRPlusRegretMatching, DiscountedRegretMatching]
)
@pytest.mark.parametrize("alternating", [False, True], ids=["simultaneous", "alternating"])
def test_full_traversal_updates_each_info_set_once_under_a_fixed_profile(
    rule_factory, alternating
):
    """Every Kuhn info set is reached through two histories, one per opponent card.
    Each must still be updated exactly once per iteration, and only after the walk
    has finished reading strategies, so the whole iteration uses one profile.
    """
    rule = RecordingRule(rule_factory())
    solver = CFRSolver(KuhnGame(), rule, FullTraversal(alternating=alternating))
    for _ in range(4):
        rule.events.clear()
        solver.train(1)
        first_update = next(i for i, (kind, _) in enumerate(rule.events) if kind != "read")
        assert all(kind == "read" for kind, _ in rule.events[:first_update])
        assert all(kind != "read" for kind, _ in rule.events[first_update:])

        updated = KUHN_INFO_SETS // 2 if alternating else KUHN_INFO_SETS
        for kind in ("regret", "strategy"):
            counts = Counter(record for k, record in rule.events if k == kind)
            assert len(counts) == updated
            assert set(counts.values()) == {1}, kind


def test_mccfr_updates_each_info_set_at_most_once_per_iteration():
    """External sampling reaches an info set more than once per traversal only without
    perfect recall. Two-round Glosten-Milgrom is such a game: the maker's key is the
    order flow alone, so its second-round info set is reached from every first-round
    quote.
    """
    rule = RecordingRule(CFRPlusRegretMatching())
    game = GlostenMilgromGame(mu=0.30, num_rounds=2)
    solver = CFRSolver(game, rule, ExternalSamplingMCCFR(), seed=0)
    for _ in range(3):
        rule.events.clear()
        solver.train(1)
        for kind in ("regret", "strategy"):
            counts = Counter(record for k, record in rule.events if k == kind)
            assert counts and max(counts.values()) == 1, kind


def test_cfr_plus_beats_vanilla_on_leduc():
    """CFR+ is expected to converge much faster than vanilla CFR on Leduc. When the
    floor was applied at every history instead of once per info set, CFR+ was behind
    vanilla from about 100 iterations on.
    """
    game = LeducGame()
    results = {}
    for name, rule in (("vanilla", VanillaRegretMatching()), ("cfr_plus", CFRPlusRegretMatching())):
        solver = CFRSolver(game, rule, FullTraversal())
        solver.train(100)
        results[name] = exploitability(game, solver.average_strategy())
    assert results["cfr_plus"] < results["vanilla"] / 2, results
