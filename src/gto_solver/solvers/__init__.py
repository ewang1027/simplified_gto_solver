"""CFR solver machinery: regret storage, update rules, traversals, and the registry
of named (rule x traversal) variants.
"""

from gto_solver.solvers.base import (
    CFRSolver,
    InfoSetRecord,
    InfoSetStore,
    RegretUpdateRule,
    Traversal,
    regret_matching,
)
from gto_solver.solvers.registry import (
    ALGORITHMS,
    EXACT_ALGORITHMS,
    SAMPLED_ALGORITHMS,
    AlgorithmSpec,
    algorithm_names,
    get_algorithm,
)

__all__ = [
    "ALGORITHMS",
    "EXACT_ALGORITHMS",
    "SAMPLED_ALGORITHMS",
    "AlgorithmSpec",
    "CFRSolver",
    "InfoSetRecord",
    "InfoSetStore",
    "RegretUpdateRule",
    "Traversal",
    "algorithm_names",
    "get_algorithm",
    "regret_matching",
]
