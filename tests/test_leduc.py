"""Tests for LeducState / LeducGame -- the Phase 3 gate: this must all pass driving
vanilla CFR on the *existing*, unmodified solver.
"""

import pytest

from gto_solver.games.base import CHANCE
from gto_solver.games.leduc import CALL, FOLD, NUM_CARDS, RAISE, LeducGame, LeducState
from gto_solver.metrics.exploitability import exploitability
from gto_solver.solvers.base import CFRSolver
from gto_solver.solvers.regret_rules import VanillaRegretMatching
from gto_solver.solvers.traversal import FullTraversal

# Card identities: 0,1 = J; 2,3 = Q; 4,5 = K (two suits per rank, suits interchangeable).
J0, J1, Q0, Q1, K0, K1 = range(6)

LEDUC_INFO_SETS = 288


# --- Chance nodes: private deal, then community card ---


def test_initial_deal_has_30_equally_likely_outcomes():
    state = LeducGame().new_initial_state()
    assert state.is_chance()
    assert not state.is_terminal()
    assert state.current_player() == CHANCE

    outcomes = state.chance_outcomes()
    assert len(outcomes) == 30  # 6 cards, ordered pairs without replacement

    deals = [deal for deal, _ in outcomes]
    assert len(set(deals)) == 30
    for p0_card, p1_card in deals:
        assert p0_card != p1_card
        assert p0_card in range(NUM_CARDS)
        assert p1_card in range(NUM_CARDS)

    probs = [prob for _, prob in outcomes]
    assert all(prob == pytest.approx(1 / 30) for prob in probs)
    assert sum(probs) == pytest.approx(1.0)


def test_community_card_outcomes_exclude_dealt_cards():
    state = LeducGame().new_initial_state().apply((K0, J0))
    state = state.apply(CALL).apply(CALL)  # check-check closes round 1
    assert state.is_chance()
    assert not state.is_terminal()

    outcomes = state.chance_outcomes()
    assert len(outcomes) == 4  # 6 dealt - 2 already in players' hands

    cards = [card for card, _ in outcomes]
    assert len(set(cards)) == 4
    assert K0 not in cards
    assert J0 not in cards

    probs = [prob for _, prob in outcomes]
    assert all(prob == pytest.approx(0.25) for prob in probs)
    assert sum(probs) == pytest.approx(1.0)


def test_chance_outcomes_raises_at_decision_node():
    state = LeducState(private=(K0, J0))
    with pytest.raises(ValueError):
        state.chance_outcomes()


def test_legal_actions_raises_at_chance_node():
    state = LeducGame().new_initial_state()
    with pytest.raises(ValueError):
        state.legal_actions()


def test_legal_actions_after_deal_is_check_or_raise():
    state = LeducGame().new_initial_state().apply((K0, J0))
    assert state.legal_actions() == [CALL, RAISE]  # no fold: nothing to fold to yet


# --- Immutability ---


def test_apply_does_not_mutate_original_chance_state():
    original = LeducGame().new_initial_state()
    new_state = original.apply((K0, J0))
    assert original.private is None
    assert original.is_chance()
    assert new_state.private == (K0, J0)
    assert not new_state.is_chance()


def test_apply_does_not_mutate_original_decision_state():
    original = LeducState(private=(K0, J0))
    new_state = original.apply(RAISE)
    assert original.round1 == ()
    assert new_state.round1 == (RAISE,)


# --- Betting rules: raise cap, bet sizes, fold ---


def test_facing_no_bet_cannot_fold():
    state = LeducState(private=(K0, J0))
    assert FOLD not in state.legal_actions()


def test_facing_a_bet_can_fold_below_cap():
    state = LeducState(private=(K0, J0), round1=(RAISE,))
    assert state.legal_actions() == [FOLD, CALL, RAISE]


def test_at_most_2_raises_per_round():
    state = LeducState(private=(K0, J0), round1=(RAISE, RAISE))
    assert state.legal_actions() == [FOLD, CALL]  # cap reached: no third raise


def test_raise_cap_resets_between_rounds():
    """Exercises both chance nodes plus the per-round cap: 2 raises in round 1 caps
    out, closes, board is dealt, and round 2 independently allows 2 more raises.
    """
    state = LeducGame().new_initial_state().apply((K0, J0))
    assert state.legal_actions() == [CALL, RAISE]

    state = state.apply(RAISE).apply(RAISE)  # cap out round 1's raises
    assert state.legal_actions() == [FOLD, CALL]

    state = state.apply(CALL)  # round 1 closes
    assert state.is_chance()
    state = state.apply(Q0)  # board card, distinct from both private cards
    assert not state.is_chance()
    assert state.board == Q0
    assert state.legal_actions() == [CALL, RAISE]  # cap reset for round 2

    state = state.apply(RAISE).apply(RAISE)  # cap out round 2's raises too
    assert state.legal_actions() == [FOLD, CALL]

    state = state.apply(CALL)
    assert state.is_terminal()
    assert state.payout(0) + state.payout(1) == 0.0


def test_round1_raise_size_is_2():
    state = LeducState(private=(K0, Q0), board=J0, round1=(RAISE, CALL), round2=(CALL, CALL))
    assert state.is_terminal()
    assert state.payout(0) == 3.0  # ante 1 + round1 raise 2; K beats Q, no round2 betting
    assert state.payout(1) == -3.0


def test_round2_raise_size_is_4():
    state = LeducState(private=(K0, Q0), board=J0, round1=(CALL, CALL), round2=(RAISE, CALL))
    assert state.is_terminal()
    assert state.payout(0) == 5.0  # ante 1 + round2 raise 4; K beats Q, no round1 betting
    assert state.payout(1) == -5.0


def test_fold_in_round1_ends_game_before_board_is_dealt():
    state = LeducState(private=(K0, J0), round1=(RAISE, FOLD))
    assert state.is_terminal()
    assert state.board is None  # community card never reached
    assert state.payout(0) == 1.0
    assert state.payout(1) == -1.0


def test_fold_in_round1_by_first_actor():
    state = LeducState(private=(J0, K0), round1=(CALL, RAISE, FOLD))
    assert state.is_terminal()
    assert state.payout(0) == -1.0
    assert state.payout(1) == 1.0


def test_fold_in_round2_ends_game_immediately():
    state = LeducState(private=(K0, Q0), board=J0, round1=(CALL, CALL), round2=(RAISE, FOLD))
    assert state.is_terminal()
    assert state.payout(0) == 1.0
    assert state.payout(1) == -1.0


# --- Showdown logic ---


def test_pair_beats_higher_unpaired_card():
    """P0 holds the low card but pairs the board; P1's unpaired high card loses."""
    state = LeducState(private=(J0, K0), board=J1, round1=(CALL, CALL), round2=(CALL, CALL))
    assert state.is_terminal()
    assert state.payout(0) == 1.0
    assert state.payout(1) == -1.0


def test_equal_rank_no_pair_is_a_tie():
    state = LeducState(private=(J0, J1), board=Q0, round1=(CALL, CALL), round2=(CALL, CALL))
    assert state.is_terminal()
    assert state.payout(0) == 0.0
    assert state.payout(1) == 0.0


def test_higher_unpaired_card_wins_at_showdown():
    state = LeducState(private=(K0, Q0), board=J0, round1=(CALL, CALL), round2=(CALL, CALL))
    assert state.is_terminal()
    assert state.payout(0) == 1.0
    assert state.payout(1) == -1.0


# --- Zero-sum payouts across a range of terminal scenarios ---


@pytest.mark.parametrize(
    "private, board, round1, round2",
    [
        ((J0, J1), Q0, (CALL, CALL), (CALL, CALL)),  # tie
        ((K0, J0), None, (RAISE, FOLD), ()),  # fold in round 1 (second actor)
        ((J0, K0), None, (CALL, RAISE, FOLD), ()),  # fold in round 1 (first actor)
        ((K0, Q0), J0, (CALL, CALL), (RAISE, FOLD)),  # fold in round 2
        ((J0, K0), J1, (CALL, CALL), (CALL, CALL)),  # pair beats high card
        ((K0, Q0), J0, (RAISE, CALL), (RAISE, CALL)),  # showdown with betting both rounds
    ],
)
def test_payouts_are_zero_sum(private, board, round1, round2):
    state = LeducState(private=private, board=board, round1=round1, round2=round2)
    assert state.is_terminal()
    assert state.payout(0) + state.payout(1) == 0.0


# --- Info sets must hide what the acting player cannot see ---


def test_info_set_key_hides_opponent_card():
    vs_j = LeducState(private=(K0, J0), round1=(CALL, RAISE))
    vs_q = LeducState(private=(K0, Q0), round1=(CALL, RAISE))
    assert vs_j.current_player() == 0
    assert vs_q.current_player() == 0
    assert vs_j.info_set_key() == vs_q.info_set_key() == "K|cr"


def test_info_set_key_hides_suit():
    with_k0 = LeducState(private=(K0, J0))
    with_k1 = LeducState(private=(K1, J0))
    assert with_k0.info_set_key() == with_k1.info_set_key() == "K|"


def test_board_appears_in_key_only_after_reveal():
    preflop = LeducState(private=(K0, J0), round1=(CALL,))
    assert preflop.info_set_key() == "J|c"  # P1 to act, no board mentioned

    postflop = LeducState(private=(K0, J0), board=Q0, round1=(CALL, CALL))
    assert postflop.info_set_key() == "KQ|cc|"  # P0 to act, board rank now included


# --- Game metadata ---


def test_game_metadata():
    game = LeducGame()
    assert game.name == "leduc_poker"
    assert game.num_players == 2


# --- The phase gate: vanilla CFR on the unmodified solver ---


def make_solver(seed: int = 0) -> CFRSolver:
    return CFRSolver(LeducGame(), VanillaRegretMatching(), FullTraversal(), seed=seed)


def test_discovers_every_info_set():
    solver = make_solver()
    solver.train(1)  # a full traversal visits every info set in one pass
    assert len(solver.store) == LEDUC_INFO_SETS


def test_exploitability_converges_substantially():
    """Numbers measured during development (seed=0; FullTraversal + VanillaRegretMatching
    do no sampling, so this is deterministic): exploitability ~0.34 at iteration 10,
    ~0.022 at iteration 500. No solver/metrics code was touched to make this happen --
    Leduc runs on the exact same CFRSolver as Kuhn.
    """
    game = LeducGame()
    solver = make_solver()

    solver.train(10)
    early = exploitability(game, solver.average_strategy())

    solver.train(490)  # cumulative 500 iterations
    late = exploitability(game, solver.average_strategy())

    assert late < early / 5
    assert late < 0.05

    for key, probs in solver.average_strategy().items():
        assert probs.sum() == pytest.approx(1.0), key
        assert (probs >= 0).all(), key
