from kuhn import KuhnState, Pass, Bet, cards

def test_game():
    """
    Walk through every possible game manually.

    Card indices: J=0, Q=1, K=2
    """

    # --- Test 1: K vs J, check-check → K wins +1 ---
    state = KuhnState(cards=[2, 0])  # P0=K, P1=J
    assert not state.is_terminal()
    assert state.current_player() == 0
    assert state.info_set() == "K"

    state = state.use_action(Pass)  # P0 checks
    assert not state.is_terminal()
    assert state.current_player() == 1
    assert state.info_set() == "J|c"

    state = state.use_action(Pass)  # P1 checks
    assert state.is_terminal()
    assert state.payout() == 1.0  # K beats J, P0 wins 1
    print("Test 1 Passed: K vs J, check-check → P0 wins 1")

    # --- Test 2: J vs Q, Bet-fold → P0(J) bluffs, P1(Q) folds ---
    state = KuhnState(cards=[0, 1])  # P0=J, P1=Q
    state = state.use_action(Bet)   # P0 Bets
    assert state.info_set() == "Q|b"
    state = state.use_action(Pass)  # P1 folds
    assert state.is_terminal()
    assert state.payout() == 1.0  # P0 wins (bluff worked)
    print("Test 2 Passed: J vs Q, Bet-fold → P0 wins 1 (bluff)")

    # --- Test 3: J vs K, Bet-call → K wins +2 ---
    state = KuhnState(cards=[0, 2])  # P0=J, P1=K
    state = state.use_action(Bet)
    state = state.use_action(Bet)   # P1 calls
    assert state.is_terminal()
    assert state.payout() == -2.0  # J loses showdown
    print("Test 3 Passed: J vs K, Bet-call → P0 loses 2")

    # --- Test 4: Q vs J, check-Bet-call → Q wins +2 ---
    state = KuhnState(cards=[1, 0])  # P0=Q, P1=J
    state = state.use_action(Pass)  # P0 checks
    state = state.use_action(Bet)   # P1 Bets (bluff)
    assert not state.is_terminal()
    assert state.current_player() == 0
    assert state.info_set() == "Q|cb"
    state = state.use_action(Bet)   # P0 calls
    assert state.is_terminal()
    assert state.payout() == 2.0  # Q beats J
    print("Test 4 Passed: Q vs J, check-Bet-call → P0 wins 2")

    # --- Test 5: Q vs K, check-Bet-fold → P1 wins +1 ---
    state = KuhnState(cards=[1, 2])  # P0=Q, P1=K
    state = state.use_action(Pass)
    state = state.use_action(Bet)   # P1 Bets
    state = state.use_action(Pass)  # P0 folds
    assert state.is_terminal()
    assert state.payout() == -1.0  # P0 folded, loses ante
    print("Test 5 Passed: Q vs K, check-Bet-fold → P0 loses 1")

    # --- Test 6: Verify immutability ---
    original = KuhnState(cards=[0, 1])
    new_state = original.use_action(Bet)
    assert len(original.history) == 0  # Original unchanged
    assert len(new_state.history) == 1
    print("Test 6 Passed: use_action doesn't mutate original")


if __name__ == "__main__":
    test_game()