"""Tests for the Kyle (1985) fixed-point solver.

Kyle's market maker is competitive, not strategic, so there is no CFR game and
no exploitability metric here -- these tests check the fixed-point solver
against the closed form, plus an independent simulation-based regression
check, and document why a strategic maker has no interior optimum. See
kyle.py's module docstring and docs/phase4-microstructure-design.md.
"""

import pytest

from gto_solver.analysis.kyle import (
    KyleParams,
    information_revealed,
    maker_best_response,
    price_impact_regression,
    solve_fixed_point,
    strategic_maker_pnl,
    trader_best_response,
)

PARAM_COMBOS = [
    (1.0, 1.0),
    (2.0, 0.5),
    (0.5, 2.0),
    (3.0, 3.0),
    (0.1, 5.0),
    (5.0, 0.1),
]


# --- Solver matches closed form, across parameter combinations ---


@pytest.mark.parametrize("sigma_v, sigma_u", PARAM_COMBOS)
def test_solver_matches_closed_form(sigma_v, sigma_u):
    params = KyleParams(sigma_v=sigma_v, sigma_u=sigma_u)
    result = solve_fixed_point(params)
    assert result.converged
    assert result.iterations < 100
    assert result.lam == pytest.approx(params.equilibrium_lambda, rel=1e-8)
    assert result.beta == pytest.approx(params.equilibrium_beta, rel=1e-8)


def test_best_response_maps_agree_with_closed_form():
    params = KyleParams(sigma_v=2.0, sigma_u=0.5)
    br_beta = trader_best_response(params.equilibrium_lambda)
    br_lambda = maker_best_response(params, params.equilibrium_beta)
    assert br_beta == pytest.approx(params.equilibrium_beta)
    assert br_lambda == pytest.approx(params.equilibrium_lambda)


# --- The fixed point is an attractor: many starting lambdas land in the same place ---


@pytest.mark.parametrize("sigma_v, sigma_u", [(1.0, 1.0), (2.0, 0.5), (0.1, 5.0)])
@pytest.mark.parametrize("initial_lambda", [3.7, 0.001, 100.0, 1e-6, 50.0])
def test_converges_from_far_off_starting_points(sigma_v, sigma_u, initial_lambda):
    params = KyleParams(sigma_v=sigma_v, sigma_u=sigma_u)
    result = solve_fixed_point(params, initial_lambda=initial_lambda)
    assert result.converged
    assert result.lam == pytest.approx(params.equilibrium_lambda, rel=1e-6)


def test_far_off_start_converges_to_documented_fixed_point():
    """The design doc's example: lambda=3.7 converges to 0.5 when S0=su^2=1."""
    params = KyleParams(sigma_v=1.0, sigma_u=1.0)
    result = solve_fixed_point(params, initial_lambda=3.7)
    assert result.converged
    assert result.lam == pytest.approx(0.5, abs=1e-9)
    assert result.beta == pytest.approx(1.0, abs=1e-9)


# --- lambda*beta == 1/2 and information_revealed == 1/2, always ---


@pytest.mark.parametrize("sigma_v, sigma_u", PARAM_COMBOS)
def test_lambda_beta_product_is_one_half(sigma_v, sigma_u):
    params = KyleParams(sigma_v=sigma_v, sigma_u=sigma_u)
    assert params.equilibrium_lambda * params.equilibrium_beta == pytest.approx(0.5)


@pytest.mark.parametrize("sigma_v, sigma_u", PARAM_COMBOS)
def test_information_revealed_is_one_half(sigma_v, sigma_u):
    params = KyleParams(sigma_v=sigma_v, sigma_u=sigma_u)
    assert information_revealed(params) == pytest.approx(0.5)


# --- Expected informed profit ---


@pytest.mark.parametrize("sigma_v, sigma_u", PARAM_COMBOS)
def test_expected_informed_profit_matches_closed_form(sigma_v, sigma_u):
    params = KyleParams(sigma_v=sigma_v, sigma_u=sigma_u)
    assert params.expected_informed_profit == pytest.approx(sigma_u * sigma_v / 2.0)


# --- Independent regression check recovers the closed-form lambda ---


@pytest.mark.parametrize(
    "sigma_v, sigma_u, p0",
    [
        (1.0, 1.0, 0.0),
        (2.0, 0.5, 0.0),
        (0.5, 2.0, 3.0),
        (0.1, 5.0, -1.0),
        (5.0, 0.1, 0.0),
    ],
)
def test_price_impact_regression_recovers_closed_form_lambda(sigma_v, sigma_u, p0):
    params = KyleParams(sigma_v=sigma_v, sigma_u=sigma_u, p0=p0)
    lam_hat = price_impact_regression(params)
    assert lam_hat == pytest.approx(params.equilibrium_lambda, rel=0.01)


# --- Scaling: more private information -> more impact; more noise -> less impact ---


def test_lambda_rises_with_sigma_v():
    sigmas = [0.5, 1.0, 2.0, 4.0]
    lambdas = [solve_fixed_point(KyleParams(sigma_v=s, sigma_u=1.0)).lam for s in sigmas]
    assert lambdas == sorted(lambdas)


def test_lambda_falls_with_sigma_u():
    sigmas = [0.5, 1.0, 2.0, 4.0]
    lambdas = [solve_fixed_point(KyleParams(sigma_v=1.0, sigma_u=s)).lam for s in sigmas]
    assert lambdas == sorted(lambdas, reverse=True)


# --- Non-zero-sum finding: a strategic maker's PnL has no interior optimum ---


def test_strategic_maker_pnl_rises_without_bound_and_is_zero_at_equilibrium():
    """Evidence for why Kyle is solved by fixed-point iteration, not CFR: a
    profit-maximizing maker would widen lambda forever, and its PnL is exactly
    zero at the competitive equilibrium lambda.
    """
    params = KyleParams(sigma_v=1.0, sigma_u=1.0)
    lambdas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    pnls = [strategic_maker_pnl(params, lam) for lam in lambdas]

    assert pnls == sorted(pnls)  # monotonically increasing: no interior optimum
    assert pnls[lambdas.index(0.5)] == pytest.approx(0.0, abs=1e-9)
    assert strategic_maker_pnl(params, params.equilibrium_lambda) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("sigma_v, sigma_u", PARAM_COMBOS)
def test_strategic_maker_pnl_zero_only_at_equilibrium(sigma_v, sigma_u):
    params = KyleParams(sigma_v=sigma_v, sigma_u=sigma_u)
    below = strategic_maker_pnl(params, params.equilibrium_lambda * 0.5)
    above = strategic_maker_pnl(params, params.equilibrium_lambda * 2.0)
    assert below < 0.0
    assert above > 0.0
