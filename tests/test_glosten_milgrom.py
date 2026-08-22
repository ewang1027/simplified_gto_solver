"""Tests for GlostenMilgromState / GlostenMilgromGame -- the Phase 4 gate.

The most important test here is benchmark consistency: the expected maker payoff
computed by walking this game tree must equal maker_profit() from
analysis/microstructure.py exactly (up to floating-point noise). That equality is
what makes CFR's exploitability metric meaningful on this game -- see
docs/phase4-microstructure-design.md.
"""

import numpy as np
import pytest

from gto_solver.analysis.microstructure import GMParams, maker_profit, uninformed_trade_prob
from gto_solver.games.base import CHANCE
from gto_solver.games.glosten_milgrom import (
    BUY,
    INFORMED,
    PASS,
    SELL,
    UNINFORMED_BUY,
    UNINFORMED_SELL,
    GlostenMilgromGame,
    GlostenMilgromState,
)
from gto_solver.metrics.exploitability import best_response_value, exploitability

PARAMS = GMParams()


# --- Root chance node: joint draw of V and trader type ---


def test_root_chance_outcomes_sum_to_one():
    state = GlostenMilgromGame(mu=0.2, params=PARAMS).new_initial_state()
    assert state.is_chance()
    assert not state.is_terminal()
    assert state.current_player() == CHANCE

    outcomes = state.chance_outcomes()
    assert len(outcomes) == 9 * 3  # 9 value levels x {informed, uninformed buy, uninformed sell}
    assert sum(prob for _, prob in outcomes) == pytest.approx(1.0)


@pytest.mark.parametrize("mu", [0.02, 0.2, 0.5, 0.8])
def test_root_chance_split_matches_mu(mu):
    state = GlostenMilgromGame(mu=mu, params=PARAMS).new_initial_state()
    outcomes = state.chance_outcomes()
    informed_prob = sum(prob for (_, ttype), prob in outcomes if ttype == INFORMED)
    buy_prob = sum(prob for (_, ttype), prob in outcomes if ttype == UNINFORMED_BUY)
    sell_prob = sum(prob for (_, ttype), prob in outcomes if ttype == UNINFORMED_SELL)
    assert informed_prob == pytest.approx(mu)
    assert buy_prob == pytest.approx((1 - mu) / 2)
    assert sell_prob == pytest.approx((1 - mu) / 2)


def test_chance_outcomes_raises_at_decision_node():
    state = GlostenMilgromState(PARAMS, mu=0.2, value=1.0, trader_type=INFORMED)
    with pytest.raises(ValueError):
        state.chance_outcomes()


def test_legal_actions_raises_at_chance_node():
    state = GlostenMilgromGame(mu=0.2, params=PARAMS).new_initial_state()
    with pytest.raises(ValueError):
        state.legal_actions()


# --- Market maker's decision node ---


def test_maker_quotes_from_the_full_quote_grid():
    state = GlostenMilgromGame(mu=0.2, params=PARAMS).new_initial_state().apply((0.0, INFORMED))
    assert not state.is_chance()
    assert state.current_player() == 0
    assert state.legal_actions() == [float(s) for s in PARAMS.quotes()]


# --- Uninformed trader is mechanical: a two-outcome chance node after the quote ---


@pytest.mark.parametrize("quote_idx", [0, 8, 16, 24, 32])
def test_uninformed_trade_probability_matches_benchmark(quote_idx):
    quote = float(PARAMS.quotes()[quote_idx])
    state = GlostenMilgromState(PARAMS, mu=0.2, value=0.625, trader_type=UNINFORMED_BUY, quote=quote)
    assert state.is_chance()

    outcomes = dict(state.chance_outcomes())
    expected = uninformed_trade_prob(PARAMS, quote)
    assert outcomes[BUY] == pytest.approx(expected)
    assert outcomes[PASS] == pytest.approx(1 - expected)
    assert sum(outcomes.values()) == pytest.approx(1.0)


def test_uninformed_trade_probability_for_the_quote_actually_chosen():
    """Same check, but reached by actually walking chance -> maker's legal_actions()
    -> apply(), rather than hand-constructing the state.
    """
    state = GlostenMilgromGame(mu=0.2, params=PARAMS).new_initial_state()
    state = state.apply((1.25, UNINFORMED_SELL))
    quote = state.legal_actions()[20]
    state = state.apply(quote)

    assert state.is_chance()
    outcomes = dict(state.chance_outcomes())
    expected = uninformed_trade_prob(PARAMS, quote)
    assert outcomes[SELL] == pytest.approx(expected)
    assert outcomes[PASS] == pytest.approx(1 - expected)


def test_uninformed_direction_fixed_at_root_constrains_chance_outcomes():
    quote = float(PARAMS.quotes()[10])
    state = GlostenMilgromState(PARAMS, mu=0.2, value=-1.25, trader_type=UNINFORMED_SELL, quote=quote)
    outcomes = dict(state.chance_outcomes())
    assert SELL in outcomes
    assert PASS in outcomes
    assert BUY not in outcomes  # direction was fixed to sell at the root


# --- Informed trader's decision node ---


def test_informed_trader_chooses_buy_sell_or_pass():
    quote = float(PARAMS.quotes()[5])
    state = GlostenMilgromState(PARAMS, mu=0.2, value=1.25, trader_type=INFORMED, quote=quote)
    assert not state.is_chance()
    assert state.current_player() == 1
    assert state.legal_actions() == [BUY, SELL, PASS]


# --- Immutability ---


def test_apply_does_not_mutate_root_chance_state():
    original = GlostenMilgromGame(mu=0.2, params=PARAMS).new_initial_state()
    new_state = original.apply((0.625, INFORMED))
    assert original.value is None
    assert original.is_chance()
    assert new_state.value == 0.625
    assert not new_state.is_chance()


def test_apply_does_not_mutate_maker_decision_state():
    original = GlostenMilgromState(PARAMS, mu=0.2, value=0.625, trader_type=INFORMED)
    quote = float(PARAMS.quotes()[4])
    new_state = original.apply(quote)
    assert original.quote is None
    assert new_state.quote == quote


def test_apply_does_not_mutate_trader_decision_state():
    quote = float(PARAMS.quotes()[4])
    original = GlostenMilgromState(PARAMS, mu=0.2, value=0.625, trader_type=INFORMED, quote=quote)
    new_state = original.apply(BUY)
    assert original.flow_history == ()
    assert new_state.flow_history == (BUY,)
    assert new_state.quote is None  # reset for the next round


# --- Zero-sum payouts ---


@pytest.mark.parametrize(
    "value, quote, outcome",
    [
        (1.25, 0.5, BUY),  # informed buys profitably
        (-1.25, 0.5, SELL),  # informed sells profitably
        (0.625, 1.0, PASS),  # no trade
        (2.5, 0.0, BUY),  # zero spread, maximal adverse selection
        (-2.5, 2.0, SELL),  # max spread, maker still loses to an extreme value
        (0.3, 0.5, BUY),  # unprofitable buy (V < s) -- still must be zero-sum
    ],
)
def test_terminal_payouts_are_zero_sum(value, quote, outcome):
    state = GlostenMilgromState(
        PARAMS, mu=0.2, value=value, trader_type=INFORMED, flow_history=(outcome,), quote_history=(quote,)
    )
    assert state.is_terminal()
    assert state.payout(0) + state.payout(1) == pytest.approx(0.0)


def test_buy_payout_formula():
    state = GlostenMilgromState(
        PARAMS, mu=0.2, value=1.0, trader_type=INFORMED, flow_history=(BUY,), quote_history=(0.4,)
    )
    assert state.payout(0) == pytest.approx(0.4 - 1.0)  # maker sells at +s, delivers V
    assert state.payout(1) == pytest.approx(1.0 - 0.4)


def test_sell_payout_formula():
    state = GlostenMilgromState(
        PARAMS, mu=0.2, value=-1.0, trader_type=INFORMED, flow_history=(SELL,), quote_history=(0.4,)
    )
    assert state.payout(0) == pytest.approx(-1.0 + 0.4)  # maker buys at -s, receives V
    assert state.payout(1) == pytest.approx(1.0 - 0.4)


def test_pass_payout_is_zero():
    state = GlostenMilgromState(
        PARAMS, mu=0.2, value=1.0, trader_type=INFORMED, flow_history=(PASS,), quote_history=(0.4,)
    )
    assert state.payout(0) == 0.0
    assert state.payout(1) == 0.0


# --- Info sets: the property the whole model depends on ---


def test_maker_info_set_hides_value_and_trader_type():
    keys = set()
    for value in (-2.5, 0.0, 1.875):
        for trader_type in (INFORMED, UNINFORMED_BUY, UNINFORMED_SELL):
            state = GlostenMilgromState(PARAMS, mu=0.2, value=value, trader_type=trader_type)
            assert state.current_player() == 0
            keys.add(state.info_set_key())
    assert keys == {""}  # single round: exactly one maker info set, regardless of hidden state


def test_maker_info_set_is_flow_history_only_across_rounds():
    """Two round-2 maker nodes reached via different quotes, different V, and
    different trader types, but the same round-0 flow outcome, must share a key: the
    spec restricts the maker's info set to observed order flow, excluding even its
    own past quote choices.
    """
    s_low_quote = GlostenMilgromState(
        PARAMS,
        mu=0.2,
        num_rounds=2,
        value=1.25,
        trader_type=INFORMED,
        flow_history=(PASS,),
        quote_history=(0.0,),
    )
    s_high_quote = GlostenMilgromState(
        PARAMS,
        mu=0.2,
        num_rounds=2,
        value=-1.875,
        trader_type=UNINFORMED_SELL,
        flow_history=(PASS,),
        quote_history=(2.0,),
    )
    assert s_low_quote.current_player() == 0
    assert s_high_quote.current_player() == 0
    assert s_low_quote.info_set_key() == s_high_quote.info_set_key() == "p"


def test_informed_info_set_depends_on_value():
    quote = float(PARAMS.quotes()[5])
    low = GlostenMilgromState(PARAMS, mu=0.2, value=-1.25, trader_type=INFORMED, quote=quote)
    high = GlostenMilgromState(PARAMS, mu=0.2, value=1.25, trader_type=INFORMED, quote=quote)
    assert low.current_player() == 1
    assert high.current_player() == 1
    assert low.info_set_key() != high.info_set_key()


def test_informed_info_set_depends_on_quote():
    low_quote = GlostenMilgromState(PARAMS, mu=0.2, value=1.25, trader_type=INFORMED, quote=0.0)
    high_quote = GlostenMilgromState(PARAMS, mu=0.2, value=1.25, trader_type=INFORMED, quote=1.0)
    assert low_quote.info_set_key() != high_quote.info_set_key()


# --- Economic correctness ---


@pytest.mark.parametrize(
    "value, quote",
    [(1.5, 0.5), (0.7, 0.5), (2.0, 1.0), (0.5, 1.5), (-1.5, 0.5)],
)
def test_buying_is_profitable_for_the_trader_iff_value_exceeds_quote(value, quote):
    state = GlostenMilgromState(
        PARAMS, mu=0.2, value=value, trader_type=INFORMED, flow_history=(BUY,), quote_history=(quote,)
    )
    trader_profit = state.payout(1)
    assert (trader_profit > 0) == (value > quote)


@pytest.mark.parametrize(
    "value, quote",
    [(-1.5, 0.5), (-0.7, 0.5), (-2.0, 1.0), (-0.5, 1.5), (1.5, 0.5)],
)
def test_selling_is_profitable_for_the_trader_iff_value_below_negative_quote(value, quote):
    state = GlostenMilgromState(
        PARAMS, mu=0.2, value=value, trader_type=INFORMED, flow_history=(SELL,), quote_history=(quote,)
    )
    trader_profit = state.payout(1)
    assert (trader_profit > 0) == (value < -quote)


# --- Benchmark consistency: the phase gate ---


def _force_maker_quote(quotes: np.ndarray, idx: int) -> dict[str, np.ndarray]:
    one_hot = np.zeros(len(quotes))
    one_hot[idx] = 1.0
    return {"": one_hot}  # the maker's single round-0 info set


@pytest.mark.parametrize("mu", [0.02, 0.1, 0.2, 0.5, 0.7])
@pytest.mark.parametrize("quote_idx", [0, 1, 8, 16, 24, 31, 32])
def test_expected_maker_payoff_matches_benchmark(mu, quote_idx):
    """Force the maker to one quote -- expressible as a single-entry strategy dict
    because it has exactly one info set -- and let the informed trader best respond.
    The resulting maker payoff must equal maker_profit() from analysis/microstructure
    exactly (to floating-point noise) for every (mu, quote) pair: this is what proves
    the game and the benchmark encode the same model.
    """
    quotes = PARAMS.quotes()
    s = float(quotes[quote_idx])
    game = GlostenMilgromGame(mu=mu, params=PARAMS, num_rounds=1)
    strategy = _force_maker_quote(quotes, quote_idx)

    trader_best_response = best_response_value(game, strategy, br_player=1)
    maker_payoff = -trader_best_response  # exact zero-sum: payout(0) == -payout(1) always

    expected = maker_profit(PARAMS, mu, s)
    assert maker_payoff == pytest.approx(expected, abs=1e-9)


# --- Multi-round smoke test ---


def test_multi_round_tree_is_well_formed_and_zero_sum():
    game = GlostenMilgromGame(mu=0.3, params=PARAMS, num_rounds=2)
    action_counts: dict[tuple[int, str], int] = {}

    def walk(state):
        if state.is_terminal():
            assert len(state.flow_history) == 2
            assert len(state.quote_history) == 2
            assert state.payout(0) + state.payout(1) == pytest.approx(0.0)
            return
        if state.is_chance():
            outcomes = state.chance_outcomes()
            assert sum(prob for _, prob in outcomes) == pytest.approx(1.0)
            for outcome, _ in outcomes:
                walk(state.apply(outcome))
            return
        key = (state.current_player(), state.info_set_key())
        actions = state.legal_actions()
        if key in action_counts:
            assert action_counts[key] == len(actions), key  # one action set per info set
        else:
            action_counts[key] = len(actions)
        for action in actions:
            walk(state.apply(action))

    walk(game.new_initial_state())
    assert action_counts  # sanity: the walk actually visited decision nodes


def test_multi_round_exploitability_runs_without_error():
    """Exercises the full best-response machinery for both players on the multi-round
    tree. A structural bug (e.g. an info set whose member nodes disagree on the
    number of legal actions) raises inside best_response_value rather than just
    producing a wrong number, so this is a real correctness check, not only a timing
    smoke test.
    """
    game = GlostenMilgromGame(mu=0.3, params=PARAMS, num_rounds=2)
    value = exploitability(game, {})
    assert value >= 0.0


# --- Game metadata ---


def test_game_metadata():
    game = GlostenMilgromGame(mu=0.2, params=PARAMS, num_rounds=1)
    assert game.name == "glosten_milgrom"
    assert game.num_players == 2
    assert game.mu == 0.2
    assert game.num_rounds == 1


def test_default_params_used_when_not_provided():
    game = GlostenMilgromGame(mu=0.2)
    assert game.params.num_values == 9
    assert game.params.num_quotes == 33
