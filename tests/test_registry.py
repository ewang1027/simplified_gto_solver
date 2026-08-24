"""The algorithm registry, and the determinism claim it makes.

`AlgorithmSpec.deterministic` is load-bearing rather than decorative: the benchmark
runs a variant claiming determinism exactly once, so a wrong claim would publish a
genuinely stochastic variant as a single run with no confidence band and no sign
anything was missing. So it is checked by running the seeds, in both directions --
the exact variants must produce bit-identical strategies, and the sampled ones must
not.
"""

import pytest

from gto_solver.benchmark.runner import verify_determinism
from gto_solver.games.kuhn import KuhnGame
from gto_solver.solvers.registry import (
    ALGORITHMS,
    EXACT_ALGORITHMS,
    SAMPLED_ALGORITHMS,
    algorithm_names,
    get_algorithm,
)

ALL_NAMES = sorted(ALGORITHMS)
# Deep CFR retrains a network every iteration, so the 150-iteration determinism sweep
# below would cost tens of seconds for it alone. It gets its own cheaper check in
# tests/test_deep_cfr.py rather than slowing every run of this file.
COMPOSED_NAMES = sorted(name for name, spec in ALGORITHMS.items() if spec.composed)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_spec_builds_and_trains(name):
    solver = get_algorithm(name).build(KuhnGame(), seed=0)
    solver.train(2 if not get_algorithm(name).composed else 20)
    assert len(solver.store) == 12
    for key, probs in solver.average_strategy().items():
        assert probs.sum() == pytest.approx(1.0), key


@pytest.mark.parametrize("name", COMPOSED_NAMES)
def test_determinism_claim_is_true(name):
    """The claim, checked by running it -- in whichever direction it was made."""
    report = verify_determinism(KuhnGame, get_algorithm(name), iterations=150, seeds=(0, 1, 7))
    assert report.claim_holds, (
        f"{name} claims deterministic={report.claimed_deterministic} but running seeds "
        f"{report.seeds} produced identical strategies = {report.observed_identical} "
        f"(differing info sets: {report.differing_info_sets})"
    )


def test_a_spec_needs_exactly_one_way_to_build_itself():
    """Either a rule and a traversal, or a make_solver -- never both, and never
    neither. Both would leave it ambiguous which one `build` honours.
    """
    from gto_solver.solvers.registry import AlgorithmSpec

    with pytest.raises(ValueError, match="either a rule and a traversal"):
        AlgorithmSpec(name="x", label="x", description="", deterministic=True)
    with pytest.raises(ValueError, match="not both"):
        AlgorithmSpec(
            name="x",
            label="x",
            description="",
            deterministic=True,
            make_rule=lambda: None,
            make_traversal=lambda: None,
            make_solver=lambda game, seed: None,
        )


def test_only_deep_cfr_is_uncomposed():
    """The two abstractions carry every tabular variant; this records the one place
    they do not reach, so a future variant quietly bypassing them is visible.
    """
    uncomposed = {name for name, spec in ALGORITHMS.items() if not spec.composed}
    assert uncomposed == {"deep_cfr"}


def test_exact_variants_are_the_deterministic_ones():
    """The two groupings are complementary, and neither is empty."""
    assert set(EXACT_ALGORITHMS) | set(SAMPLED_ALGORITHMS) == set(ALGORITHMS)
    assert not set(EXACT_ALGORITHMS) & set(SAMPLED_ALGORITHMS)
    assert EXACT_ALGORITHMS and SAMPLED_ALGORITHMS


def test_registry_keys_match_spec_names():
    for key, spec in ALGORITHMS.items():
        assert key == spec.name


def test_labels_are_unique():
    """Labels are what a plot legend and a results table show; duplicates would make
    two different variants indistinguishable in exactly the places people read.
    """
    labels = [spec.label for spec in ALGORITHMS.values()]
    assert len(set(labels)) == len(labels)


def test_algorithm_names_preserves_registration_order():
    assert algorithm_names() == tuple(ALGORITHMS)


def test_unknown_algorithm_lists_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        get_algorithm("cfr_ultra")
    message = str(excinfo.value)
    assert "cfr_ultra" in message
    for name in ALGORITHMS:
        assert name in message


def test_build_returns_independent_solvers():
    spec = get_algorithm("vanilla")
    first, second = spec.build(KuhnGame()), spec.build(KuhnGame())
    first.train(10)
    assert second.iterations == 0
    assert first.rule is not second.rule
    assert first.traversal is not second.traversal
