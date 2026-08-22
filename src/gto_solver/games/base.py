"""Abstract extensive-form game interface used by every solver in this package."""

from abc import ABC, abstractmethod
from typing import Any

# Sentinel returned by current_player() at a chance node.
CHANCE = -1


class GameState(ABC):
    """One node in an extensive-form game tree.

    Implementations must be immutable: apply() returns a new state.
    """

    @abstractmethod
    def is_terminal(self) -> bool:
        """True when the game has ended and payout() is defined."""

    @abstractmethod
    def is_chance(self) -> bool:
        """True when the next move belongs to chance, not a player."""

    @abstractmethod
    def chance_outcomes(self) -> list[tuple[Any, float]]:
        """(outcome, probability) pairs at a chance node. Probabilities sum to 1."""

    @abstractmethod
    def current_player(self) -> int:
        """Index of the player to act, or CHANCE at a chance node."""

    @abstractmethod
    def legal_actions(self) -> list[Any]:
        """Actions available to the current player. Solvers index regret by position."""

    @abstractmethod
    def apply(self, action: Any) -> "GameState":
        """Return the successor state. Accepts a chance outcome at a chance node."""

    @abstractmethod
    def info_set_key(self) -> str:
        """Key identifying what the acting player knows. States a player cannot
        distinguish must return equal keys."""

    @abstractmethod
    def payout(self, player: int) -> float:
        """Terminal payoff for `player`."""


class Game(ABC):
    """Factory and static metadata for a game."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def num_players(self) -> int: ...

    @abstractmethod
    def new_initial_state(self) -> GameState: ...
