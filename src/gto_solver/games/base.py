"""Abstract extensive-form game interface used by every solver in this package."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # only for the annotation on features(); no runtime import
    import numpy as np

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

    def features(self) -> "np.ndarray | None":
        """A fixed-length numeric encoding of this info set, or None.

        Optional, and None by default. Tabular CFR needs only `info_set_key`, which
        says *whether* two states are the same; a function approximator needs to know
        how they are *related*, which a key cannot express -- one-hot encoding the key
        would give a network nothing to generalize over and reduce Deep CFR to a
        slower tabular solver.

        Returning None means this game does not support function approximation, and
        `DeepCFRSolver` says so rather than inventing an encoding on the game's behalf.
        Every state sharing an info-set key must return equal features; the tests
        check that directly, because a featurization that leaks hidden information
        would train a network on knowledge the player does not have.
        """
        return None

    def payouts(self, num_players: int) -> list[float]:
        """Terminal payoff for every player, as one call.

        Optional: the default below is correct for any game and needs no override.
        It exists because `payout(player)` is called once per player at every
        terminal node, and a two-player zero-sum game computes the same thing both
        times — Leduc rebuilds both players' chip contributions and re-decides the
        winner on each call, so the default costs exactly twice what it needs to.
        A game that can share that work overrides this and returns the vector.

        Whatever a game returns here must agree with `payout` exactly; the tests
        check both against each other at every terminal node of every game.
        """
        return [self.payout(player) for player in range(num_players)]


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
