"""Named games, the way `solvers/registry.py` names algorithms.

The CLI, the profiler and anything else that takes a game as a string needs one place
that maps "leduc" to a constructed game. Without it, every caller grows its own dict
and they drift: `scripts/profile_hotloop.py` had exactly that, and it is now the only
copy rather than one of several.

`parameters` is what makes the CLI honest. Kuhn and Leduc take no arguments, so a
`--mu` passed alongside `--game kuhn` is a mistake the caller should hear about rather
than have silently dropped -- a spread reported for the wrong mu is worse than an error.
"""

from collections.abc import Callable
from dataclasses import dataclass

from gto_solver.analysis.microstructure import GMParams
from gto_solver.games.base import Game
from gto_solver.games.glosten_milgrom import GlostenMilgromGame
from gto_solver.games.kuhn import KuhnGame
from gto_solver.games.leduc import LeducGame


@dataclass(frozen=True)
class GameSpec:
    """One named game, and which parameters it accepts."""

    name: str
    label: str
    description: str
    build: Callable[..., Game]
    parameters: tuple[str, ...] = ()

    def create(self, **kwargs) -> Game:
        """Build the game, refusing parameters it does not take.

        Raises rather than ignoring: `--mu 0.7 --game kuhn` means the caller believes
        something about the run that is not true.
        """
        unknown = sorted(set(kwargs) - set(self.parameters))
        if unknown:
            accepted = ", ".join(self.parameters) if self.parameters else "none"
            raise ValueError(
                f"game {self.name!r} does not take {', '.join(unknown)}; "
                f"it accepts: {accepted}"
            )
        return self.build(**kwargs)


def _glosten_milgrom(mu: float = 0.30, rounds: int = 1) -> Game:
    return GlostenMilgromGame(mu=mu, params=GMParams(), num_rounds=rounds)


GAMES: dict[str, GameSpec] = {
    spec.name: spec
    for spec in (
        GameSpec(
            name="kuhn",
            label="Kuhn poker",
            description="3-card poker, 12 info sets. The smallest game with a real bluff.",
            build=KuhnGame,
        ),
        GameSpec(
            name="leduc",
            label="Leduc Hold'em",
            description="6-card poker, two betting rounds, 288 info sets.",
            build=LeducGame,
        ),
        GameSpec(
            name="gm",
            label="Glosten-Milgrom",
            description="Market making against possibly-informed order flow, 298 info sets.",
            build=_glosten_milgrom,
            parameters=("mu", "rounds"),
        ),
    )
}


def get_game(name: str) -> GameSpec:
    try:
        return GAMES[name]
    except KeyError:
        known = ", ".join(sorted(GAMES))
        raise KeyError(f"unknown game {name!r}; known games are: {known}") from None


def game_names() -> tuple[str, ...]:
    return tuple(GAMES)
