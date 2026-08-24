"""Named algorithm configurations: every (regret rule x traversal) combination this
package ships, in one place.

A CFR variant here is a composition rather than a class, which is the point of the
design -- but it means "which variants exist" is otherwise scattered across whatever
tuple a test or a script happened to write last. The benchmark suite needs to name a
variant in a string (and so will the Phase 7 CLI and the Phase 8 dashboard), so the
combinations live here and are looked up by name.

`deterministic` is the load-bearing field. `FullTraversal` never touches the rng, so
every exact variant returns a bit-identical strategy for any seed; only the sampled
traversals actually consume randomness. The benchmark uses that flag to run exact
variants once instead of publishing a confidence band of width zero across twenty
identical runs.

That flag is a claim about the code, not a comment, so `tests/test_registry.py`
checks it the hard way: it trains each variant under different seeds and compares
the resulting strategies entry by entry, both for the variants claiming determinism
and for the ones claiming the opposite.
"""

from collections.abc import Callable
from dataclasses import dataclass

from gto_solver.games.base import Game
from gto_solver.solvers.base import CFRSolver, RegretUpdateRule, Traversal
from gto_solver.solvers.regret_rules import (
    CFRPlusRegretMatching,
    DiscountedRegretMatching,
    VanillaRegretMatching,
    linear_cfr,
)
from gto_solver.solvers.traversal import ExternalSamplingMCCFR, FullTraversal


def _deep_cfr(game: Game, seed: int):
    """Imported lazily so `solvers.registry` does not drag in the nn package for the
    six variants that have nothing to do with it.
    """
    from gto_solver.solvers.deep_cfr import DeepCFRSolver

    return DeepCFRSolver(game, seed=seed)


@dataclass(frozen=True)
class AlgorithmSpec:
    """One named CFR variant: how to build it, and how to report it.

    `label` is what appears in a plot legend or a results table; `name` is the
    lookup key and the string that ends up in serialized results.

    Most variants are a regret rule composed with a traversal, which is the design
    this package is built around. `make_solver` is the escape hatch for one that is
    not: Deep CFR replaces the regret *store* with a fitted network, which no pairing
    of a rule and a traversal expresses. `composed` records which kind a spec is, so
    callers can tell "another combination" from "a different sort of thing" — the
    tests use it to keep an expensive variant out of the cheap parametrized sweeps.
    """

    name: str
    label: str
    description: str
    deterministic: bool
    make_rule: Callable[[], RegretUpdateRule] | None = None
    make_traversal: Callable[[], Traversal] | None = None
    make_solver: Callable[[Game, int], object] | None = None

    def __post_init__(self) -> None:
        composed = self.make_rule is not None and self.make_traversal is not None
        if composed == (self.make_solver is not None):
            raise ValueError(
                f"{self.name!r} needs either a rule and a traversal, or a make_solver, "
                f"and not both"
            )

    @property
    def composed(self) -> bool:
        """True when this variant is a regret rule composed with a traversal."""
        return self.make_solver is None

    def build(self, game: Game, seed: int = 0):
        """A fresh solver for `game`. Fresh rule and traversal objects too, since a
        rule may carry per-run state in future variants.
        """
        if self.make_solver is not None:
            return self.make_solver(game, seed)
        return CFRSolver(game, self.make_rule(), self.make_traversal(), seed=seed)


ALGORITHMS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            name="vanilla",
            label="Vanilla CFR",
            description="Regret matching on unweighted cumulative regret (Zinkevich et al. 2007).",
            make_rule=VanillaRegretMatching,
            make_traversal=FullTraversal,
            deterministic=True,
        ),
        AlgorithmSpec(
            name="cfr_plus",
            label="CFR+",
            description="Regret-matching+ with linear strategy averaging (Tammelin 2014).",
            make_rule=CFRPlusRegretMatching,
            make_traversal=FullTraversal,
            deterministic=True,
        ),
        AlgorithmSpec(
            name="dcfr",
            label="DCFR",
            description="Discounted CFR at the paper's defaults, alpha=1.5, beta=0, gamma=2.",
            make_rule=DiscountedRegretMatching,
            make_traversal=FullTraversal,
            deterministic=True,
        ),
        AlgorithmSpec(
            name="linear_cfr",
            label="Linear CFR",
            description="Discounted CFR at alpha=beta=gamma=1, i.e. linear weighting.",
            make_rule=linear_cfr,
            make_traversal=FullTraversal,
            deterministic=True,
        ),
        AlgorithmSpec(
            name="cfr_plus_alternating",
            label="CFR+ (alternating)",
            description="CFR+ with alternating updates, as the published algorithm uses.",
            make_rule=CFRPlusRegretMatching,
            make_traversal=lambda: FullTraversal(alternating=True),
            deterministic=True,
        ),
        AlgorithmSpec(
            name="mccfr",
            label="MCCFR (ext. sampling)",
            description="External-sampling MCCFR on vanilla regret matching (Lanctot et al. 2009).",
            make_rule=VanillaRegretMatching,
            make_traversal=ExternalSamplingMCCFR,
            deterministic=False,
        ),
        AlgorithmSpec(
            name="deep_cfr",
            label="Deep CFR",
            description=(
                "Counterfactual regret with a learned value function (Brown et al. 2019). "
                "The only variant here that is not a regret rule composed with a traversal: "
                "it replaces the regret table with a network refitted every iteration, and "
                "needs the game to implement GameState.features()."
            ),
            deterministic=False,
            make_solver=_deep_cfr,
        ),
        AlgorithmSpec(
            name="mccfr_plus",
            label="MCCFR + CFR+ rule",
            description=(
                "External sampling composed with the CFR+ regret rule -- a combination "
                "neither paper defines, available for free because rules and traversals "
                "are independent here."
            ),
            make_rule=CFRPlusRegretMatching,
            make_traversal=ExternalSamplingMCCFR,
            deterministic=False,
        ),
    )
}

# Convenience groupings. Sorted-by-registration order is meaningful: it is the order
# results tables and plot legends use.
EXACT_ALGORITHMS: tuple[str, ...] = tuple(
    name for name, spec in ALGORITHMS.items() if spec.deterministic
)
SAMPLED_ALGORITHMS: tuple[str, ...] = tuple(
    name for name, spec in ALGORITHMS.items() if not spec.deterministic
)


def get_algorithm(name: str) -> AlgorithmSpec:
    """Look up a variant by name, with a message that lists the alternatives."""
    try:
        return ALGORITHMS[name]
    except KeyError:
        known = ", ".join(sorted(ALGORITHMS))
        raise KeyError(f"unknown algorithm {name!r}; known algorithms are: {known}") from None


def algorithm_names() -> tuple[str, ...]:
    """Every registered variant, in registration order."""
    return tuple(ALGORITHMS)
