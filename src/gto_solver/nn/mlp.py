"""A small multilayer perceptron in numpy, with Adam and per-sample weights.

Written rather than imported, for the same reason the CFR engine was: this project's
whole claim is that the machinery is built from scratch, and a two-hidden-layer network
over a twelve-element input is roughly eighty lines. Pulling in a deep-learning
framework to run it would add a heavy dependency to CI for no understanding.

The obvious risk of hand-written backpropagation is that it is quietly wrong, so it is
checked against finite differences in `tests/test_mlp.py` -- every weight and bias in a
small network, to a relative tolerance of 1e-6. A gradient that is wrong by a constant
factor still trains, just worse, which is exactly the kind of bug that would be
invisible behind a loss curve that goes down.

Per-sample weights exist because Deep CFR weights each sample by the CFR iteration that
produced it: later iterations reflect a better strategy and should count more.
"""

from collections.abc import Sequence
from itertools import pairwise

import numpy as np


def _he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """He initialization: variance 2/fan_in, which is what ReLU wants.

    Xavier (1/fan_in) halves the signal at every ReLU, and a deep-enough stack of them
    trains visibly worse. At two hidden layers it barely matters, but getting it wrong
    is free to avoid.
    """
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


class MLP:
    """Fully-connected ReLU network trained with Adam on a weighted squared loss.

    `sizes` is (input, hidden..., output). The output layer is linear: Deep CFR's value
    network predicts counterfactual regret, which is signed and unbounded.
    """

    def __init__(self, sizes: Sequence[int], seed: int = 0, learning_rate: float = 1e-3):
        if len(sizes) < 2:
            raise ValueError(f"need at least an input and an output layer, got {sizes}")
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.learning_rate = learning_rate
        self.weights = [_he_init(rng, a, b) for a, b in pairwise(sizes)]
        self.biases = [np.zeros(b, dtype=np.float64) for b in sizes[1:]]
        # Adam state, one pair per parameter tensor.
        self._m = [np.zeros_like(p) for p in self.weights + self.biases]
        self._v = [np.zeros_like(p) for p in self.weights + self.biases]
        self._step = 0

    # --- forward / backward ------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Outputs, plus the per-layer activations backpropagation needs."""
        activations = [np.atleast_2d(x)]
        current = activations[0]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            if index < len(self.weights) - 1:  # linear output layer
                current = np.maximum(current, 0.0)
            activations.append(current)
        return current, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def gradients(
        self, x: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        """Gradients of the weighted mean squared error, and the loss itself.

        loss = sum_i w_i * ||f(x_i) - y_i||^2 / sum_i w_i
        """
        x = np.atleast_2d(x)
        targets = np.atleast_2d(targets)
        sample_weights = (
            np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights, float)
        )
        total_weight = sample_weights.sum()
        if total_weight <= 0.0:
            raise ValueError("sample weights must sum to something positive")

        outputs, activations = self.forward(x)
        residual = outputs - targets
        loss = float((sample_weights[:, None] * residual**2).sum() / total_weight)

        # d(loss)/d(outputs)
        delta = 2.0 * sample_weights[:, None] * residual / total_weight
        weight_grads: list[np.ndarray] = [None] * len(self.weights)
        bias_grads: list[np.ndarray] = [None] * len(self.biases)
        for index in reversed(range(len(self.weights))):
            weight_grads[index] = activations[index].T @ delta
            bias_grads[index] = delta.sum(axis=0)
            if index > 0:
                delta = (delta @ self.weights[index].T) * (activations[index] > 0.0)
        return weight_grads, bias_grads, loss

    # --- optimization ------------------------------------------------------

    def apply_gradients(
        self,
        weight_grads: Sequence[np.ndarray],
        bias_grads: Sequence[np.ndarray],
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """One Adam step over every parameter tensor."""
        self._step += 1
        parameters = self.weights + self.biases
        grads = list(weight_grads) + list(bias_grads)
        bias_correction1 = 1.0 - beta1**self._step
        bias_correction2 = 1.0 - beta2**self._step
        for i, (parameter, grad) in enumerate(zip(parameters, grads)):
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * grad
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * grad**2
            step = (self._m[i] / bias_correction1) / (
                np.sqrt(self._v[i] / bias_correction2) + epsilon
            )
            parameter -= self.learning_rate * step

    def fit(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray | None = None,
        epochs: int = 40,
        batch_size: int = 128,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Train on the whole set for `epochs`, returning the final epoch's mean loss.

        Batches are shuffled from `rng` so a caller holding a seeded generator keeps a
        reproducible run; passing None makes the shuffle deterministic rather than
        drawing from global state, which would make a "seeded" solver a lie.
        """
        x = np.atleast_2d(x)
        targets = np.atleast_2d(targets)
        if len(x) != len(targets):
            raise ValueError(f"{len(x)} inputs against {len(targets)} targets")
        sample_weights = (
            np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights, float)
        )
        rng = rng if rng is not None else np.random.default_rng(0)

        loss = float("nan")
        for _ in range(epochs):
            order = rng.permutation(len(x))
            losses, counts = [], []
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                if sample_weights[batch].sum() <= 0.0:
                    continue
                weight_grads, bias_grads, batch_loss = self.gradients(
                    x[batch], targets[batch], sample_weights[batch]
                )
                self.apply_gradients(weight_grads, bias_grads)
                losses.append(batch_loss)
                counts.append(len(batch))
            if losses:
                loss = float(np.average(losses, weights=counts))
        return loss

    def reset(self, seed: int) -> None:
        """Re-initialize every parameter and the optimizer state.

        Deep CFR retrains its value network *from scratch* each iteration rather than
        fine-tuning: the regret targets change as the strategy does, and a network
        carried over fits the previous iteration's targets first.
        """
        rng = np.random.default_rng(seed)
        self.weights = [_he_init(rng, a, b) for a, b in pairwise(self.sizes)]
        self.biases = [np.zeros(b, dtype=np.float64) for b in self.sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.weights + self.biases]
        self._v = [np.zeros_like(p) for p in self.weights + self.biases]
        self._step = 0
