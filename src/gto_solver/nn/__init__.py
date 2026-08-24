"""Small neural-network machinery, written from scratch in numpy.

Deep CFR needs a function approximator; this is one, at the scale the games here call
for. Keeping it in the package rather than importing a framework means the whole
project stays dependency-light and the gradients stay checkable.
"""

from gto_solver.nn.mlp import MLP

__all__ = ["MLP"]
