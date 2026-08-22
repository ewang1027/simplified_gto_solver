"""Kyle (1985) single-period equilibrium.

Kyle's market maker is COMPETITIVE: it prices at the Bayesian conditional
expectation E[v | y] and earns zero expected profit by construction -- it
does not optimize anything. That makes this a different object from a
two-player zero-sum game, so CFR and exploitability do not apply here (see
docs/phase4-microstructure-design.md). A strategic maker's PnL turns out to
rise monotonically in lambda with no interior optimum -- tested in
tests/test_kyle.py -- which is the concrete evidence for that call. Instead,
the equilibrium is found by iterating the trader's and maker's best-response
maps to their mutual fixed point.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KyleParams:
    """Single-period Kyle market: v ~ N(p0, sigma_v^2), u ~ N(0, sigma_u^2)."""

    sigma_v: float = 1.0
    sigma_u: float = 1.0
    p0: float = 0.0

    @property
    def prior_variance(self) -> float:
        """S0, the prior variance of v."""
        return self.sigma_v**2

    @property
    def equilibrium_lambda(self) -> float:
        """Closed-form price impact: sqrt(S0) / (2*sigma_u)."""
        return self.sigma_v / (2.0 * self.sigma_u)

    @property
    def equilibrium_beta(self) -> float:
        """Closed-form trading intensity: sigma_u / sqrt(S0)."""
        return self.sigma_u / self.sigma_v

    @property
    def expected_informed_profit(self) -> float:
        """Closed-form equilibrium informed-trader profit: sigma_u * sqrt(S0) / 2."""
        return self.sigma_u * self.sigma_v / 2.0

    @property
    def residual_variance(self) -> float:
        """Var(v | y) at equilibrium: S0 / 2 -- half the private information stays hidden."""
        return self.prior_variance / 2.0


def trader_best_response(lam: float) -> float:
    """beta maximizing (v - p0)*x - lambda*x^2 given a fixed price impact lambda."""
    return 1.0 / (2.0 * lam)


def maker_best_response(params: KyleParams, beta: float) -> float:
    """Bayesian zero-profit lambda given a fixed trading intensity beta."""
    s0 = params.prior_variance
    return beta * s0 / (beta**2 * s0 + params.sigma_u**2)


@dataclass(frozen=True)
class FixedPointResult:
    lam: float
    beta: float
    iterations: int
    converged: bool


def solve_fixed_point(
    params: KyleParams,
    initial_lambda: float = 1.0,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
) -> FixedPointResult:
    """Iterate trader/maker best responses to their mutual fixed point.

    Not an optimization: the maker's step is a Bayesian update (zero-profit
    pricing), not a best response to profit. See module docstring. Empirically
    this is an attracting fixed point that converges in well under 50 steps
    from any positive starting lambda.
    """
    lam = initial_lambda
    beta = trader_best_response(lam)
    for iteration in range(1, max_iterations + 1):
        new_lambda = maker_best_response(params, beta)
        converged = abs(new_lambda - lam) < tolerance
        lam = new_lambda
        beta = trader_best_response(lam)
        if converged:
            return FixedPointResult(lam=lam, beta=beta, iterations=iteration, converged=True)
    return FixedPointResult(lam=lam, beta=beta, iterations=max_iterations, converged=False)


def price_impact_regression(
    params: KyleParams,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """Recover lambda by simulation, independent of the fixed-point algebra above.

    Draws (v, u), applies the equilibrium trading rule x = beta*(v - p0), forms
    order flow y = x + u, and prices at p = E[v|y] using covariance/variance
    ESTIMATED from the sample rather than substituted from the closed-form
    S0/sigma_u^2 formula. lambda is then the OLS slope of p on y, which must
    match equilibrium_lambda if the best-response algebra above is right.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(params.p0, params.sigma_v, n_samples)
    u = rng.normal(0.0, params.sigma_u, n_samples)
    beta = params.equilibrium_beta
    y = beta * (v - params.p0) + u

    cov_vy = np.cov(v, y, ddof=1)[0, 1]
    var_y = np.var(y, ddof=1)
    slope = cov_vy / var_y
    p = (params.p0 - slope * np.mean(y)) + slope * y

    lam_hat, _ = np.polyfit(y, p, 1)
    return float(lam_hat)


def information_revealed(params: KyleParams) -> float:
    """Fraction of v's prior variance impounded into price -- exactly 1/2 at equilibrium."""
    return 1.0 - params.residual_variance / params.prior_variance


def informed_profit(params: KyleParams, lam: float) -> float:
    """Expected informed-trader profit at an arbitrary lambda (trader still best-responds)."""
    beta = trader_best_response(lam)
    return beta * params.prior_variance * (1.0 - lam * beta)


def strategic_maker_pnl(params: KyleParams, lam: float) -> float:
    """Counterfactual: maker PnL if it committed to `lam` against the best-responding trader.

    NOT the equilibrium object -- Kyle's maker is competitive (see module
    docstring). Included only to demonstrate PnL rises without bound in lam
    and is zero exactly at equilibrium_lambda: the reason CFR is not used here.
    """
    return lam * params.sigma_u**2 - informed_profit(params, lam)
