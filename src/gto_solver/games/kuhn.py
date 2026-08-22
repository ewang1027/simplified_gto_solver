"""Kuhn poker: 3-card deck {J, Q, K}, one bet size, two players.

The deal is a chance node inside the tree: new_initial_state() returns a
state with no cards yet, and apply() on it resolves the deal and returns
the first decision state.
"""

from itertools import permutations
from typing import Any

from gto_solver.games.base import CHANCE, Game, GameState

CARDS = ("J", "Q", "K")  # index order is also rank order, J < Q < K
PASS = 0
BET = 1
ACTIONS = (PASS, BET)

# Ordered (player0_card, player1_card) index pairs: the 6 equally likely deals.
DEALS: tuple[tuple[int, int], ...] = tuple(permutations(range(len(CARDS)), 2))


class KuhnState(GameState):
    """cards is None exactly at the single chance node preceding the deal."""

    def __init__(self, cards: tuple[int, int] | None = None, history: tuple[int, ...] = ()):
        self.cards = cards
        self.history = history

    def is_chance(self) -> bool:
        return self.cards is None

    def chance_outcomes(self) -> list[tuple[Any, float]]:
        if not self.is_chance():
            raise ValueError("not a chance node: use legal_actions() instead")
        return [(deal, 1 / len(DEALS)) for deal in DEALS]

    def is_terminal(self) -> bool:
        if self.is_chance():
            return False
        h = self.history
        if len(h) < 2:
            return False
        if len(h) == 2:
            return h != (PASS, BET)  # check-bet leaves P0 to act again
        return True

    def current_player(self) -> int:
        if self.is_chance():
            return CHANCE
        return 0 if len(self.history) % 2 == 0 else 1

    def legal_actions(self) -> list[Any]:
        if self.is_chance():
            raise ValueError("chance node: use chance_outcomes() instead")
        return list(ACTIONS)

    def apply(self, action: Any) -> "KuhnState":
        if self.is_chance():
            return KuhnState(cards=action, history=())
        return KuhnState(cards=self.cards, history=self.history + (action,))

    def info_set_key(self) -> str:
        card = CARDS[self.cards[self.current_player()]]
        if not self.history:
            return card
        action_str = "".join("b" if a == BET else "c" for a in self.history)
        return f"{card}|{action_str}"

    def payout(self, player: int) -> float:
        h = self.history
        p0_card, p1_card = self.cards
        p0_wins = p0_card > p1_card

        if h == (PASS, PASS):  # both check, showdown
            p0_payout = 1.0 if p0_wins else -1.0
        elif h == (BET, PASS):  # P0 bets, P1 folds
            p0_payout = 1.0
        elif h == (PASS, BET, PASS):  # P0 checks, P1 bets, P0 folds
            p0_payout = -1.0
        elif h in ((PASS, BET, BET), (BET, BET)):  # called bet, showdown
            p0_payout = 2.0 if p0_wins else -2.0
        else:
            raise ValueError(f"Invalid history: {h}")

        return p0_payout if player == 0 else -p0_payout


class KuhnGame(Game):
    @property
    def name(self) -> str:
        return "kuhn_poker"

    @property
    def num_players(self) -> int:
        return 2

    def new_initial_state(self) -> KuhnState:
        return KuhnState()
