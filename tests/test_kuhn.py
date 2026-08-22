"""Tests for KuhnState / KuhnGame, ported from the original root-level test.py."""

import pytest

from gto_solver.games.base import CHANCE
from gto_solver.games.kuhn import BET, CARDS, PASS, KuhnGame, KuhnState

# --- Ported scenarios (card indices: J=0, Q=1, K=2) ---


def test_check_check_showdown():
    """K vs J, check-check -> K wins the 1-chip pot."""
    state = KuhnState(cards=(2, 0))  # P0=K, P1=J
    assert not state.is_terminal()
    assert state.current_player() == 0
    assert state.info_set_key() == "K"

    state = state.apply(PASS)  # P0 checks
    assert not state.is_terminal()
    assert state.current_player() == 1
    assert state.info_set_key() == "J|c"

    state = state.apply(PASS)  # P1 checks
    assert state.is_terminal()
    assert state.payout(0) == 1.0
    assert state.payout(1) == -1.0


def test_bet_fold_bluff_wins():
    """J vs Q, bet-fold -> P0's bluff wins the pot."""
    state = KuhnState(cards=(0, 1))  # P0=J, P1=Q
    state = state.apply(BET)
    assert state.info_set_key() == "Q|b"
    state = state.apply(PASS)  # P1 folds
    assert state.is_terminal()
    assert state.payout(0) == 1.0
    assert state.payout(1) == -1.0


def test_bet_call_showdown_loss():
    """J vs K, bet-call -> the called bettor loses at showdown."""
    state = KuhnState(cards=(0, 2))  # P0=J, P1=K
    state = state.apply(BET)
    state = state.apply(BET)  # P1 calls
    assert state.is_terminal()
    assert state.payout(0) == -2.0
    assert state.payout(1) == 2.0


def test_check_bet_call_showdown_win():
    """Q vs J, check-bet-call -> P0 calls P1's bluff and wins at showdown."""
    state = KuhnState(cards=(1, 0))  # P0=Q, P1=J
    state = state.apply(PASS)  # P0 checks
    state = state.apply(BET)  # P1 bets (bluff)
    assert not state.is_terminal()
    assert state.current_player() == 0
    assert state.info_set_key() == "Q|cb"
    state = state.apply(BET)  # P0 calls
    assert state.is_terminal()
    assert state.payout(0) == 2.0
    assert state.payout(1) == -2.0


def test_check_bet_fold():
    """Q vs K, check-bet-fold -> P0 folds and loses the ante."""
    state = KuhnState(cards=(1, 2))  # P0=Q, P1=K
    state = state.apply(PASS)
    state = state.apply(BET)  # P1 bets
    state = state.apply(PASS)  # P0 folds
    assert state.is_terminal()
    assert state.payout(0) == -1.0
    assert state.payout(1) == 1.0


def test_apply_does_not_mutate_original_decision_state():
    original = KuhnState(cards=(0, 1))
    new_state = original.apply(BET)
    assert original.history == ()
    assert new_state.history == (BET,)


# --- Chance-node deal ---


def test_initial_state_is_chance_node_with_six_outcomes():
    state = KuhnGame().new_initial_state()
    assert state.is_chance()
    assert not state.is_terminal()
    assert state.current_player() == CHANCE

    outcomes = state.chance_outcomes()
    assert len(outcomes) == 6

    deals = [deal for deal, _ in outcomes]
    assert len(set(deals)) == 6  # all 6 deals distinct
    for p0_card, p1_card in deals:
        assert p0_card != p1_card
        assert p0_card in range(3)
        assert p1_card in range(3)

    probs = [prob for _, prob in outcomes]
    assert all(prob == pytest.approx(1 / 6) for prob in probs)
    assert sum(probs) == pytest.approx(1.0)


def test_apply_does_not_mutate_original_chance_state():
    original = KuhnGame().new_initial_state()
    new_state = original.apply((2, 0))
    assert original.cards is None
    assert original.is_chance()
    assert new_state.cards == (2, 0)
    assert not new_state.is_chance()


def test_legal_actions_after_deal():
    state = KuhnGame().new_initial_state().apply((2, 0))
    assert state.legal_actions() == [PASS, BET]


def test_chance_outcomes_raises_at_decision_node():
    state = KuhnState(cards=(2, 0))
    with pytest.raises(ValueError):
        state.chance_outcomes()


def test_legal_actions_raises_at_chance_node():
    state = KuhnGame().new_initial_state()
    with pytest.raises(ValueError):
        state.legal_actions()


# --- Zero-sum payouts ---


@pytest.mark.parametrize(
    "cards, history",
    [
        ((2, 0), (PASS, PASS)),
        ((0, 1), (BET, PASS)),
        ((0, 2), (BET, BET)),
        ((1, 0), (PASS, BET, BET)),
        ((1, 2), (PASS, BET, PASS)),
    ],
)
def test_payouts_are_zero_sum(cards, history):
    state = KuhnState(cards=cards, history=history)
    assert state.is_terminal()
    assert state.payout(0) + state.payout(1) == 0.0


# --- Info sets must hide what the acting player cannot see ---


def test_info_set_key_hides_opponent_card():
    state_vs_j = KuhnState(cards=(2, 0), history=(PASS, BET))  # P0=K, P1=J
    state_vs_q = KuhnState(cards=(2, 1), history=(PASS, BET))  # P0=K, P1=Q
    assert state_vs_j.current_player() == 0
    assert state_vs_q.current_player() == 0
    assert state_vs_j.info_set_key() == "K|cb"
    assert state_vs_q.info_set_key() == "K|cb"


def test_card_rank_order():
    assert CARDS == ("J", "Q", "K")


def test_game_metadata():
    game = KuhnGame()
    assert game.name == "kuhn_poker"
    assert game.num_players == 2
