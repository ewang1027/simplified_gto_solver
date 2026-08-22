"""Benchmarks the solved market-making game is checked against.

These guard the properties the Phase 4 gate depends on: the strategic optimum is a
functioning market (not the shutdown corner), and the spread responds to adverse
selection.
"""

import numpy as np
import pytest

from gto_solver.analysis.microstructure import (
    GMParams,
    competitive_half_spread,
    maker_profit,
    market_functions,
    strategic_half_spread,
    uninformed_trade_prob,
)

MUS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_value_distribution_is_a_symmetric_distribution():
    v, p = GMParams().values()
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()
    assert float(np.sum(p * v)) == pytest.approx(0.0)  # mid is 0
    assert v == pytest.approx(-v[::-1])


def test_uninformed_demand_falls_with_spread():
    params = GMParams()
    probs = [uninformed_trade_prob(params, s) for s in [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]]
    assert probs[0] == pytest.approx(1.0)
    assert probs[-1] == pytest.approx(0.0)  # nobody trades beyond the reservation cap
    assert probs == sorted(probs, reverse=True)


def test_strategic_spread_increases_with_adverse_selection():
    params = GMParams()
    spreads = [strategic_half_spread(params, mu) for mu in MUS]
    assert spreads == sorted(spreads), spreads
    assert spreads[-1] > spreads[0]


def test_competitive_spread_increases_with_adverse_selection():
    params = GMParams()
    spreads = [competitive_half_spread(params, mu) for mu in MUS]
    assert spreads == sorted(spreads), spreads


def test_strategic_maker_quotes_wider_than_competitive():
    """A monopolist charges more than a zero-profit market maker."""
    params = GMParams()
    for mu in MUS:
        assert strategic_half_spread(params, mu) >= competitive_half_spread(params, mu)


def test_market_functions_at_every_mu():
    """The optimum must be a working market, not the shutdown corner.

    A maker quoting so wide that nobody trades earns exactly zero, and that corner
    can sit strictly inside the quote grid -- so grid interiority alone would not
    catch it.
    """
    params = GMParams()
    for mu in MUS:
        assert market_functions(params, mu), mu


def test_strategic_spread_maximizes_maker_profit():
    params = GMParams()
    for mu in [0.05, 0.3, 0.7]:
        best = strategic_half_spread(params, mu)
        best_profit = maker_profit(params, mu, best)
        for s in params.quotes():
            assert maker_profit(params, mu, s) <= best_profit + 1e-12


def test_binary_value_gives_a_constant_spread_when_informed_are_cheap_to_exclude():
    """The control for why the model uses a multi-level value.

    With a binary value, |V| is a single number c. Once the half-spread reaches c
    the informed never trade, and profit is just the noise-trader term, maximized
    at T/2 for every mu. So when c <= T/2 the maker parks there and the spread
    carries no adverse-selection signal at all.
    """
    params = GMParams(num_values=2, sigma=0.5 / 2.5)  # |V| = 0.5, below T/2 = 0.8
    spreads = {strategic_half_spread(params, mu) for mu in MUS}
    assert len(spreads) == 1, spreads


def test_binary_value_saturates_even_when_it_does_respond():
    """The weaker but more general binary failure.

    With c > T/2 the spread does move with mu, but only until it reaches c --
    beyond that the informed are fully excluded and it stops responding. A
    multi-level value has no such ceiling because P(|V| > s) decays gradually.
    """
    c = 1.25
    params = GMParams(num_values=2, sigma=c / 2.5)
    spreads = [strategic_half_spread(params, mu) for mu in MUS]
    assert spreads == sorted(spreads)
    assert max(spreads) == pytest.approx(c, abs=0.05)
    assert spreads[-1] == pytest.approx(spreads[-3])  # saturated at the top end

    multi = GMParams()
    assert strategic_half_spread(multi, 0.7) > strategic_half_spread(multi, 0.5)


def test_maker_profit_falls_as_informed_share_rises():
    params = GMParams()
    profits = [maker_profit(params, mu, strategic_half_spread(params, mu)) for mu in MUS]
    assert profits == sorted(profits, reverse=True), profits
    assert profits[-1] > 0
