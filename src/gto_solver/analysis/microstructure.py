"""Glosten-Milgrom benchmarks: what the solver is checked against.

Two different market makers live here, and the difference matters:

  competitive_half_spread  -- the Glosten-Milgrom market maker, whose quotes are
      pinned by a ZERO-PROFIT condition (ask = E[V | buy]). It does not optimize.

  strategic_half_spread    -- a profit-MAXIMIZING market maker, which is what CFR
      solves for in a zero-sum game. Computed here by exhaustive search over the
      quote grid, so the CFR solution has something independent to reproduce.

These give different answers, and neither is presented as the other. See
docs/phase4-microstructure-design.md.

Every definition here must match games/glosten_milgrom.py exactly, or comparing
the solver against these benchmarks is meaningless.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GMParams:
    """Discretization of the Glosten-Milgrom market.

    Defaults are the verified non-degenerate configuration from the Phase 4
    design: the market functions (positive maker profit, both trader types
    trading) at every mu tested, and the strategic spread rises with mu.
    """

    num_values: int = 9  # a BINARY value is degenerate -- see the design doc
    sigma: float = 1.0
    reservation_max: float = 1.6  # uninformed reservation upper bound (T)
    num_quotes: int = 33
    max_half_spread: float = 2.0

    def values(self) -> tuple[np.ndarray, np.ndarray]:
        """Value grid and its probabilities, symmetric about a mid of 0."""
        v = np.linspace(-2.5 * self.sigma, 2.5 * self.sigma, self.num_values)
        p = np.exp(-0.5 * (v / self.sigma) ** 2)
        return v, p / p.sum()

    def quotes(self) -> np.ndarray:
        """Half-spreads the market maker may post."""
        return np.linspace(0.0, self.max_half_spread, self.num_quotes)


def uninformed_trade_prob(params: GMParams, half_spread: float) -> float:
    """Probability an uninformed trader accepts this half-spread.

    Reservation is uniform on [0, reservation_max], so this is exact rather than
    discretized. The uninformed trader is mechanical, so in the game tree this
    becomes a two-outcome chance node placed AFTER the maker quotes -- which keeps
    the reservation out of the root chance node and the tree small.
    """
    t = params.reservation_max
    return float(np.clip((t - half_spread) / t, 0.0, 1.0))


def maker_profit(params: GMParams, mu: float, half_spread: float) -> float:
    """Expected market-maker profit for one round at a given half-spread.

    Informed traders (probability mu) trade only when |V| exceeds the half-spread,
    and the maker loses |V| - half_spread when they do. Uninformed traders
    (probability 1 - mu) trade only within their reservation, and the maker earns
    the half-spread from them.
    """
    v, pv = params.values()
    adverse = float(np.sum(pv * np.maximum(np.abs(v) - half_spread, 0.0)))
    capture = half_spread * uninformed_trade_prob(params, half_spread)
    return -mu * adverse + (1 - mu) * capture


def strategic_half_spread(params: GMParams, mu: float) -> float:
    """Profit-maximizing half-spread, by exhaustive search over the quote grid.

    This is the benchmark the CFR solution must reproduce.
    """
    quotes = params.quotes()
    profits = [maker_profit(params, mu, s) for s in quotes]
    return float(quotes[int(np.argmax(profits))])


def competitive_half_spread(params: GMParams, mu: float) -> float:
    """Glosten-Milgrom zero-profit half-spread: ask = E[V | buy].

    Solved as a fixed point, since who buys depends on the ask. This is the
    classical GM quote -- reported for comparison, NOT as the CFR target.
    """
    v, pv = params.values()
    ask = 0.0
    for _ in range(500):
        informed_buy = pv * (v > ask)
        uninformed_buy = pv * 0.5
        weight = mu * informed_buy + (1 - mu) * uninformed_buy
        total = weight.sum()
        if total <= 0:
            break
        updated = float(np.sum(weight * v) / total)
        if abs(updated - ask) < 1e-12:
            break
        ask = updated
    return ask


def market_functions(params: GMParams, mu: float) -> bool:
    """Whether the market actually operates at the strategic optimum.

    Grid interiority is NOT sufficient: a maker that quotes so wide nobody trades
    sits at a degenerate corner that can fall strictly inside the grid, with zero
    profit. Require positive profit and both trader types actually trading.
    """
    v, pv = params.values()
    s = strategic_half_spread(params, mu)
    informed_trades = float(np.sum(pv * (np.abs(v) > s)))
    return (
        maker_profit(params, mu, s) > 1e-9
        and uninformed_trade_prob(params, s) > 0.0
        and informed_trades > 0.0
    )
