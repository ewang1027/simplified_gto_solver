"""Leduc Hold'em: 6-card deck (J, Q, K x 2 suits), two players, two betting rounds
with a public community card revealed between them.

Two chance nodes precede the two betting rounds: the private deal, then the
community card conditioned on what was dealt -- both are nodes inside the tree,
same as Kuhn's single deal. Suits are strategically irrelevant, so info_set_key()
keys on rank only; keying on the physical card would double the info-set count
with duplicate-strategy entries for suit-swapped hands.
"""

from itertools import permutations
from typing import Any

from gto_solver.games.base import CHANCE, Game, GameState

RANKS = ("J", "Q", "K")  # index order is also rank order, J < Q < K
RANK_OF: tuple[int, ...] = (0, 0, 1, 1, 2, 2)  # 6 physical cards, 2 per rank
NUM_CARDS = len(RANK_OF)

FOLD, CALL, RAISE = 0, 1, 2
MAX_RAISES = 2
ANTE = 1
ROUND1_BET = 2
ROUND2_BET = 4

# Ordered (player0_card, player1_card) identity pairs: the 30 equally likely deals.
DEALS: tuple[tuple[int, int], ...] = tuple(permutations(range(NUM_CARDS), 2))

_SYMBOL = {CALL: "c", RAISE: "r"}  # FOLD never appears mid-history: it ends the hand


def _facing_bet(history: tuple[int, ...]) -> bool:
    return bool(history) and history[-1] == RAISE


def _closed_by_call(history: tuple[int, ...]) -> bool:
    return len(history) >= 2 and history[-1] == CALL


def _folded(history: tuple[int, ...]) -> bool:
    return bool(history) and history[-1] == FOLD


def _last_actor(history: tuple[int, ...]) -> int:
    return (len(history) - 1) % 2


def _history_str(history: tuple[int, ...]) -> str:
    return "".join(_SYMBOL[a] for a in history)


def _round_contributions(history: tuple[int, ...], bet_size: int) -> tuple[int, int]:
    """Chips each player has put in this round via calls/raises (antes excluded)."""
    level = 0
    contributed = [0, 0]
    for i, action in enumerate(history):
        player = i % 2
        if action == RAISE:
            level += bet_size
        if action != FOLD:
            contributed[player] = level
    return contributed[0], contributed[1]


def _winner(p0_rank: int, p1_rank: int, board_rank: int) -> int | None:
    p0_pairs = p0_rank == board_rank
    p1_pairs = p1_rank == board_rank
    if p0_pairs != p1_pairs:  # only one card of board's rank remains, so not both pair
        return 0 if p0_pairs else 1
    if p0_rank == p1_rank:
        return None  # tie: equal ranks, neither pairs the board
    return 0 if p0_rank > p1_rank else 1


class LeducState(GameState):
    """private is None until the deal chance node resolves. board is None until the
    community-card chance node resolves, which itself only exists once round1 closes.
    """

    def __init__(
        self,
        private: tuple[int, int] | None = None,
        board: int | None = None,
        round1: tuple[int, ...] = (),
        round2: tuple[int, ...] = (),
    ):
        self.private = private
        self.board = board
        self.round1 = round1
        self.round2 = round2

    def is_chance(self) -> bool:
        if self.private is None:
            return True
        if self.board is None:
            return _closed_by_call(self.round1)
        return False

    def chance_outcomes(self) -> list[tuple[Any, float]]:
        if not self.is_chance():
            raise ValueError("not a chance node: use legal_actions() instead")
        if self.private is None:
            return [(deal, 1 / len(DEALS)) for deal in DEALS]
        remaining = [c for c in range(NUM_CARDS) if c not in self.private]
        return [(card, 1 / len(remaining)) for card in remaining]

    def is_terminal(self) -> bool:
        if self.private is None:
            return False
        if _folded(self.round1):
            return True
        if self.board is None:
            return False
        return _folded(self.round2) or _closed_by_call(self.round2)

    def current_player(self) -> int:
        if self.is_chance():
            return CHANCE
        history = self.round1 if self.board is None else self.round2
        return len(history) % 2

    def legal_actions(self) -> list[Any]:
        if self.is_chance():
            raise ValueError("chance node: use chance_outcomes() instead")
        history = self.round1 if self.board is None else self.round2
        actions = []
        if _facing_bet(history):
            actions.append(FOLD)
        actions.append(CALL)
        if sum(1 for a in history if a == RAISE) < MAX_RAISES:
            actions.append(RAISE)
        return actions

    def apply(self, action: Any) -> "LeducState":
        if self.private is None:
            return LeducState(private=action)
        if self.board is None and self.is_chance():
            return LeducState(private=self.private, board=action, round1=self.round1)
        if self.board is None:
            return LeducState(private=self.private, round1=self.round1 + (action,))
        return LeducState(
            private=self.private,
            board=self.board,
            round1=self.round1,
            round2=self.round2 + (action,),
        )

    def info_set_key(self) -> str:
        player = self.current_player()
        rank = RANKS[RANK_OF[self.private[player]]]
        r1 = _history_str(self.round1)
        if self.board is None:
            return f"{rank}|{r1}"
        board_rank = RANKS[RANK_OF[self.board]]
        r2 = _history_str(self.round2)
        return f"{rank}{board_rank}|{r1}|{r2}"

    def _p0_payout(self) -> float:
        r1_p0, r1_p1 = _round_contributions(self.round1, ROUND1_BET)
        r2_p0, r2_p1 = _round_contributions(self.round2, ROUND2_BET) if self.board is not None else (0, 0)
        contrib = (ANTE + r1_p0 + r2_p0, ANTE + r1_p1 + r2_p1)

        if _folded(self.round1):
            winner = 1 - _last_actor(self.round1)
        elif self.board is not None and _folded(self.round2):
            winner = 1 - _last_actor(self.round2)
        else:
            winner = _winner(RANK_OF[self.private[0]], RANK_OF[self.private[1]], RANK_OF[self.board])

        if winner is None:
            return 0.0
        loser_contrib = contrib[1 - winner]
        return float(loser_contrib if winner == 0 else -loser_contrib)

    def payout(self, player: int) -> float:
        p0_payout = self._p0_payout()
        return p0_payout if player == 0 else -p0_payout

    def payouts(self, num_players: int) -> list[float]:
        p0_payout = self._p0_payout()
        return [p0_payout, -p0_payout]


class LeducGame(Game):
    @property
    def name(self) -> str:
        return "leduc_poker"

    @property
    def num_players(self) -> int:
        return 2

    def new_initial_state(self) -> LeducState:
        return LeducState()
