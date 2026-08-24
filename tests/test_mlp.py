"""The from-scratch MLP, checked where hand-written backpropagation actually fails.

A gradient that is wrong by a constant factor still trains — just worse — and a sign
error in one layer still lets the loss fall, because Adam normalizes by the gradient's
own second moment. Neither shows up in a loss curve. So every weight and every bias is
compared against a central finite difference, which is the only check that catches them.
"""

import numpy as np
import pytest

from gto_solver.nn import MLP


def finite_difference(net: MLP, x, y, w, parameter: np.ndarray, index, epsilon=1e-6) -> float:
    """d(loss)/d(parameter[index]) by central difference."""
    original = parameter[index]
    parameter[index] = original + epsilon
    plus = net.gradients(x, y, w)[2]
    parameter[index] = original - epsilon
    minus = net.gradients(x, y, w)[2]
    parameter[index] = original
    return (plus - minus) / (2.0 * epsilon)


@pytest.fixture
def problem():
    rng = np.random.default_rng(3)
    net = MLP([4, 6, 5, 3], seed=1)
    x = rng.normal(size=(7, 4))
    y = rng.normal(size=(7, 3))
    weights = rng.uniform(0.5, 2.0, size=7)
    return net, x, y, weights


def test_every_weight_gradient_matches_finite_differences(problem):
    net, x, y, w = problem
    analytic, _, _ = net.gradients(x, y, w)
    for layer, grad in enumerate(analytic):
        for index in np.ndindex(grad.shape):
            numeric = finite_difference(net, x, y, w, net.weights[layer], index)
            assert grad[index] == pytest.approx(numeric, rel=1e-6, abs=1e-9), (layer, index)


def test_every_bias_gradient_matches_finite_differences(problem):
    net, x, y, w = problem
    _, analytic, _ = net.gradients(x, y, w)
    for layer, grad in enumerate(analytic):
        for index in np.ndindex(grad.shape):
            numeric = finite_difference(net, x, y, w, net.biases[layer], index)
            assert grad[index] == pytest.approx(numeric, rel=1e-6, abs=1e-9), (layer, index)


def test_gradients_are_still_right_with_unweighted_samples(problem):
    net, x, y, _ = problem
    analytic, _, _ = net.gradients(x, y, None)
    for index in np.ndindex(analytic[0].shape):
        numeric = finite_difference(net, x, y, None, net.weights[0], index)
        assert analytic[0][index] == pytest.approx(numeric, rel=1e-6, abs=1e-9), index


def test_sample_weights_actually_change_the_fit():
    """Deep CFR weights samples by the iteration that produced them, so a weight of
    zero must genuinely remove a sample rather than merely scale it.
    """
    x = np.array([[1.0], [1.0]])
    y = np.array([[1.0], [-1.0]])
    ignored_second = MLP([1, 8, 1], seed=0, learning_rate=0.05)
    ignored_second.fit(x, y, weights=np.array([1.0, 0.0]), epochs=400, batch_size=2)
    assert float(ignored_second.predict(np.array([[1.0]]))[0, 0]) == pytest.approx(1.0, abs=0.05)

    ignored_first = MLP([1, 8, 1], seed=0, learning_rate=0.05)
    ignored_first.fit(x, y, weights=np.array([0.0, 1.0]), epochs=400, batch_size=2)
    assert float(ignored_first.predict(np.array([[1.0]]))[0, 0]) == pytest.approx(-1.0, abs=0.05)


def test_it_learns_a_function_it_can_represent():
    """XOR: not linearly separable, so a network that fits it has working hidden units
    rather than a working bias term.
    """
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    net = MLP([2, 16, 16, 1], seed=0, learning_rate=0.02)
    loss = net.fit(x, y, epochs=800, batch_size=4)
    assert loss < 0.01
    assert np.abs(net.predict(x) - y).max() < 0.15


def test_the_output_layer_is_linear_and_can_predict_negatives():
    """Deep CFR's value network predicts counterfactual regret, which is signed. A
    ReLU on the output would silently clamp every negative target to zero.
    """
    x = np.array([[1.0], [2.0]])
    y = np.array([[-3.0], [-7.0]])
    net = MLP([1, 12, 1], seed=0, learning_rate=0.05)
    net.fit(x, y, epochs=600, batch_size=2)
    assert net.predict(x).max() < 0.0


def test_reset_restores_a_fresh_network():
    """Deep CFR retrains from scratch each iteration; a reset that left optimizer
    state behind would carry the previous iteration's targets into the next fit.
    """
    net = MLP([3, 5, 2], seed=4)
    before = [w.copy() for w in net.weights]
    net.fit(np.ones((4, 3)), np.ones((4, 2)), epochs=5, batch_size=2)
    assert not np.array_equal(net.weights[0], before[0])

    net.reset(seed=4)
    assert np.array_equal(net.weights[0], before[0])
    assert net._step == 0
    assert not np.any(net._m[0])


def test_the_same_seed_gives_the_same_network():
    assert np.array_equal(MLP([3, 4, 2], seed=9).weights[0], MLP([3, 4, 2], seed=9).weights[0])
    assert not np.array_equal(MLP([3, 4, 2], seed=9).weights[0], MLP([3, 4, 2], seed=10).weights[0])


@pytest.mark.parametrize("sizes", [[5], []])
def test_a_network_needs_an_input_and_an_output(sizes):
    with pytest.raises(ValueError):
        MLP(sizes)


def test_mismatched_inputs_and_targets_are_refused():
    with pytest.raises(ValueError, match="against"):
        MLP([2, 3, 1]).fit(np.ones((4, 2)), np.ones((3, 1)))


def test_all_zero_sample_weights_are_refused():
    """Silently dividing by a zero total would produce nan parameters and a network
    that predicts nan forever after.
    """
    with pytest.raises(ValueError, match="positive"):
        MLP([2, 3, 1]).gradients(np.ones((2, 2)), np.ones((2, 1)), np.zeros(2))
