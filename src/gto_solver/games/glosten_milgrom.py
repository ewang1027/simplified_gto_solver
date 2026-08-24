"""Glosten-Milgrom market making: each round a market maker posts a symmetric
half-spread `s` (ask = +s, bid = -s, mid = 0), then a trader responds.

Mirrors analysis/microstructure.py exactly -- GMParams, values(), quotes(), and
uninformed_trade_prob() are the single source of truth and are imported here, not
reimplemented. See docs/phase4-microstructure-design.md for why the model is shaped
this way: a binary V is degenerate (no margin for the maker to trade off adverse
selection against spread), and a single round has a dominant-strategy trader, which
only becomes genuinely strategic (profit now vs. revealing V) over multiple rounds.

Root chance draws V and the trader type jointly: informed with probability mu, else
uninformed with a direction (buy/sell, equally likely) fixed for the whole episode.
The market maker's info set is the observed order-flow history ONLY -- never V, never
the trader type -- so a single round gives it exactly one info set. An uninformed
trader is mechanical: after the quote, a two-outcome chance node decides whether it
trades (in its pre-drawn direction), never a real decision node for player 1.
"""

from typing import Any

from gto_solver.analysis.microstructure import GMParams, uninformed_trade_prob
from gto_solver.games.base import CHANCE, Game, GameState

# Trade-flow actions/outcomes: chosen by the informed trader, or by the mechanical
# chance node standing in for the uninformed trader. Same vocabulary either way, since
# both land in flow_history identically.
BUY, SELL, PASS = 0, 1, 2
_FLOW_SYMBOL = {BUY: "b", SELL: "s", PASS: "p"}

# Trader type, drawn once at the root and fixed for every round of the episode.
INFORMED, UNINFORMED_BUY, UNINFORMED_SELL = 0, 1, 2


def _flow_str(flow_history: tuple[int, ...]) -> str:
    return "".join(_FLOW_SYMBOL[a] for a in flow_history)


class GlostenMilgromState(GameState):
    """value/trader_type are None only at the root, before the joint chance draw.
    quote is None between rounds (market maker to act) and set once it has quoted
    (trader/chance to act). flow_history and quote_history hold completed rounds
    only, in parallel, and grow by one entry each time a round resolves.
    """

    def __init__(
        self,
        params: GMParams,
        mu: float,
        num_rounds: int = 1,
        value: float | None = None,
        trader_type: int | None = None,
        quote: float | None = None,
        flow_history: tuple[int, ...] = (),
        quote_history: tuple[float, ...] = (),
    ):
        self.params = params
        self.mu = mu
        self.num_rounds = num_rounds
        self.value = value
        self.trader_type = trader_type
        self.quote = quote
        self.flow_history = flow_history
        self.quote_history = quote_history

    def is_chance(self) -> bool:
        if self.value is None:
            return True  # root: joint draw of V and trader type
        return self.quote is not None and self.trader_type != INFORMED

    def chance_outcomes(self) -> list[tuple[Any, float]]:
        if not self.is_chance():
            raise ValueError("not a chance node: use legal_actions() instead")
        if self.value is None:
            v_grid, p_grid = self.params.values()
            outcomes: list[tuple[Any, float]] = []
            for v, pv in zip(v_grid, p_grid):
                v, pv = float(v), float(pv)
                outcomes.append(((v, INFORMED), pv * self.mu))
                outcomes.append(((v, UNINFORMED_BUY), pv * (1.0 - self.mu) / 2))
                outcomes.append(((v, UNINFORMED_SELL), pv * (1.0 - self.mu) / 2))
            return outcomes
        # Mechanical uninformed trader: direction was fixed at the root, whether it
        # trades this round is redrawn each round via the reservation-price formula.
        p = uninformed_trade_prob(self.params, self.quote)
        direction = BUY if self.trader_type == UNINFORMED_BUY else SELL
        return [(direction, p), (PASS, 1.0 - p)]

    def is_terminal(self) -> bool:
        return self.value is not None and len(self.flow_history) >= self.num_rounds

    def current_player(self) -> int:
        if self.is_chance():
            return CHANCE
        return 0 if self.quote is None else 1

    def legal_actions(self) -> list[Any]:
        if self.is_chance():
            raise ValueError("chance node: use chance_outcomes() instead")
        if self.quote is None:
            return [float(s) for s in self.params.quotes()]
        return [BUY, SELL, PASS]

    def apply(self, action: Any) -> "GlostenMilgromState":
        if self.value is None:
            v, ttype = action
            return GlostenMilgromState(self.params, self.mu, self.num_rounds, value=v, trader_type=ttype)
        if self.quote is None:  # market maker just quoted this round's half-spread
            return GlostenMilgromState(
                self.params,
                self.mu,
                self.num_rounds,
                value=self.value,
                trader_type=self.trader_type,
                quote=action,
                flow_history=self.flow_history,
                quote_history=self.quote_history,
            )
        # This round's outcome just resolved -- via the informed trader's own choice,
        # or via the chance node standing in for the uninformed trader. Same handling
        # either way: `action` is already a BUY/SELL/PASS flow code.
        return GlostenMilgromState(
            self.params,
            self.mu,
            self.num_rounds,
            value=self.value,
            trader_type=self.trader_type,
            flow_history=self.flow_history + (action,),
            quote_history=self.quote_history + (self.quote,),
        )

    def info_set_key(self) -> str:
        flow_str = _flow_str(self.flow_history)
        if self.quote is None:  # market maker: order-flow history only, never V/type
            return flow_str
        return f"{self.value:.6f}|{flow_str}|{self.quote:.6f}"  # informed trader

    def _p0_payout(self) -> float:
        total = 0.0
        for outcome, s in zip(self.flow_history, self.quote_history):
            if outcome == BUY:  # maker sells at the ask, delivers an asset worth V
                total += s - self.value
            elif outcome == SELL:  # maker buys at the bid, receives an asset worth V
                total += self.value + s
        return total

    def payout(self, player: int) -> float:
        total = self._p0_payout()
        return total if player == 0 else -total

    def payouts(self, num_players: int) -> list[float]:
        total = self._p0_payout()
        return [total, -total]


class GlostenMilgromGame(Game):
    """Market making under adverse selection, as a two-player zero-sum game.

    Player 0 is the market maker, which posts a half-spread each round and sees
    only the order flow it has observed -- never the asset value or the trader
    type. Player 1 is the informed trader. Uninformed traders are mechanical, so
    they are chance nodes rather than a player.

    `mu` is the probability the trader is informed, the adverse-selection knob:
    the solved spread widens as it rises.

    `num_rounds=1` is the case the analytical benchmark in
    analysis/microstructure.py validates, and the trader's best response there is
    a dominant strategy. With more rounds the trader becomes genuinely strategic,
    since trading reveals information that moves later quotes. Tree size is
    9*99^R + 18*66^R terminal nodes, so rounds get expensive quickly.
    """

    def __init__(self, mu: float, params: GMParams | None = None, num_rounds: int = 1):
        self.mu = mu
        self.params = params if params is not None else GMParams()
        self.num_rounds = num_rounds

    @property
    def name(self) -> str:
        return "glosten_milgrom"

    @property
    def num_players(self) -> int:
        return 2

    def new_initial_state(self) -> GlostenMilgromState:
        return GlostenMilgromState(self.params, self.mu, self.num_rounds)
