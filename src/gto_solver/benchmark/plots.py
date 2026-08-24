"""Convergence charts for benchmark results.

matplotlib is imported lazily and only here, so it stays an optional dependency:
the solver, the metrics and the whole test suite run without it, and only plotting
asks for `pip install -e '.[viz]'`.

Figures are built on `matplotlib.figure.Figure` directly rather than through pyplot.
pyplot carries global state and picks a GUI backend on import, which in a headless
CI run is either a warning or a hang; a bare Figure has neither problem and saves
through the writer implied by the file extension.

Two encodings are deliberate:

* **Line style carries the traversal.** Exact traversals are solid, sampled ones
  dashed, so the exact-vs-sampled split survives being printed in grayscale.
* **A shaded band is always the seed envelope**, never the bootstrap CI. They mean
  different things (see `stats.py`), the envelope is the wider and more honest one to
  draw, and every figure that has one says so in its footnote.
"""

import textwrap
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gto_solver.benchmark.runner import ConvergenceRun, WallclockRun
from gto_solver.benchmark.stats import DEFAULT_ENVELOPE

# Okabe-Ito, minus the yellow, which is unreadable as a thin line on white.
_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#333333")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X")
_ENVELOPE_ALPHA = 0.18

# Fixed geometry rather than tight_layout: the legend sits below the axes and the
# footnote below that, and a layout engine that does not know about either kept
# clipping both off the right edge. Rect is (left, bottom, width, height).
_FIG_SIZE = (9.0, 6.4)
_AXES_RECT = (0.085, 0.290, 0.885, 0.560)
_TITLE_SIZE = 13.0
_SUBTITLE_SIZE = 8.5
_FOOTNOTE_SIZE = 7.0
# Wrap widths in characters, sized to the axes width at each font size.
_SUBTITLE_WRAP = 118
_FOOTNOTE_WRAP = 145
_LEGEND_COLUMNS = 2


def _require_matplotlib():
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "plotting needs matplotlib, which is an optional dependency of this package. "
            "Install it with: pip install -e '.[viz]'"
        ) from exc
    return Figure


def _style(index: int, deterministic: bool) -> dict:
    return {
        "color": _PALETTE[index % len(_PALETTE)],
        "marker": _MARKERS[index % len(_MARKERS)],
        "linestyle": "-" if deterministic else "--",
        "linewidth": 1.6,
        "markersize": 4.0,
    }


def _legend_label(run: ConvergenceRun | WallclockRun) -> str:
    """Legend text that says how many seeds are behind the line.

    A deterministic variant says so rather than reporting "1 seed", which would
    read as a thin measurement instead of an exact one.
    """
    if run.deterministic:
        return f"{run.label} (deterministic)"
    return f"{run.label} (median of {len(run.seeds)} seeds)"


def _new_figure(Figure):
    fig = Figure(figsize=_FIG_SIZE)
    return fig, fig.add_axes(_AXES_RECT)


def _finish(fig, ax, title: str, subtitle: str | None, footnote: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.30, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    left = _AXES_RECT[0]
    fig.text(left, 0.965, title, fontsize=_TITLE_SIZE, va="top", ha="left")
    if subtitle:
        fig.text(
            left,
            0.922,
            textwrap.fill(subtitle, _SUBTITLE_WRAP),
            fontsize=_SUBTITLE_SIZE,
            color="#555555",
            va="top",
            ha="left",
        )
    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.135),
        ncols=_LEGEND_COLUMNS,
        frameon=False,
    )
    fig.text(
        left,
        0.018,
        textwrap.fill(footnote, _FOOTNOTE_WRAP),
        fontsize=_FOOTNOTE_SIZE,
        color="#555555",
        va="bottom",
        ha="left",
    )


def _save(fig, path: str | Path, dpi: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return path


def figure_convergence(
    runs: Sequence[ConvergenceRun],
    title: str = "CFR variants on Kuhn poker",
    subtitle: str | None = None,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
):
    """Exploitability against iteration count, log-log, one line per variant.

    Returns the figure rather than saving it, so the dashboard renders exactly the
    chart the README publishes instead of drawing its own lookalike.
    """
    Figure = _require_matplotlib()
    if not runs:
        raise ValueError("plot_convergence needs at least one run")

    fig, ax = _new_figure(Figure)
    for index, run in enumerate(runs):
        style = _style(index, run.deterministic)
        x = np.asarray(run.checkpoints, dtype=float)
        aggregates = run.aggregates(envelope=envelope)
        median = np.array([a.median for a in aggregates])
        ax.plot(x, median, label=_legend_label(run), **style)
        if len(run.seeds) > 1:
            ax.fill_between(
                x,
                [a.p_low for a in aggregates],
                [a.p_high for a in aggregates],
                color=style["color"],
                alpha=_ENVELOPE_ALPHA,
                linewidth=0,
            )

    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Exploitability")
    low, high = envelope
    _finish(
        fig,
        ax,
        title,
        subtitle,
        f"Solid = exact traversal, dashed = sampled.  Shaded = {low:g}-{high:g}% seed "
        f"envelope (spread across runs, not a confidence interval).  "
        f"Exploitability evaluation is excluded from every timing.",
    )
    return fig


def plot_convergence(
    runs: Sequence[ConvergenceRun],
    path: str | Path,
    title: str = "CFR variants on Kuhn poker",
    subtitle: str | None = None,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
    dpi: int = 160,
) -> Path:
    """`figure_convergence`, saved."""
    return _save(figure_convergence(runs, title, subtitle, envelope), path, dpi)


def figure_wallclock(
    runs: Sequence[WallclockRun],
    title: str = "Exact traversal vs sampling at equal wall-clock budget",
    subtitle: str | None = None,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
    annotate_iterations: bool = True,
):
    """Exploitability against training seconds -- the only currency exact traversal
    and sampling share, since one MCCFR iteration is a sampled path and one exact
    iteration is a whole tree.
    """
    Figure = _require_matplotlib()
    if not runs:
        raise ValueError("plot_wallclock needs at least one run")

    fig, ax = _new_figure(Figure)
    for index, run in enumerate(runs):
        style = _style(index, run.deterministic)
        x = np.asarray(run.median_seconds(), dtype=float)
        aggregates = run.aggregates(envelope=envelope)
        median = np.array([a.median for a in aggregates])
        ax.plot(x, median, label=_legend_label(run), **style)
        if len(run.seeds) > 1:
            ax.fill_between(
                x,
                [a.p_low for a in aggregates],
                [a.p_high for a in aggregates],
                color=style["color"],
                alpha=_ENVELOPE_ALPHA,
                linewidth=0,
            )
        if annotate_iterations:
            ax.annotate(
                f"{round(run.median_iterations()[-1]):,} it",
                xy=(x[-1], median[-1]),
                xytext=(-6, -12),
                textcoords="offset points",
                fontsize=7.5,
                color=style["color"],
                ha="right",
            )

    ax.set_xlabel("Training seconds (exploitability evaluation excluded)")
    ax.set_ylabel("Exploitability")
    low, high = envelope
    _finish(
        fig,
        ax,
        title,
        subtitle,
        f"Solid = exact traversal, dashed = sampled.  Shaded = {low:g}-{high:g}% seed "
        f"envelope.  Labels give iterations completed in the final budget.  "
        f"x is measured time, not requested budget.",
    )
    return fig


def plot_wallclock(
    runs: Sequence[WallclockRun],
    path: str | Path,
    title: str = "Exact traversal vs sampling at equal wall-clock budget",
    subtitle: str | None = None,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
    annotate_iterations: bool = True,
    dpi: int = 160,
) -> Path:
    """`figure_wallclock`, saved."""
    figure = figure_wallclock(runs, title, subtitle, envelope, annotate_iterations)
    return _save(figure, path, dpi)


def figure_seed_spread(
    run: ConvergenceRun,
    title: str | None = None,
    subtitle: str | None = None,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
):
    """Every individual seed of one stochastic run, drawn behind its median.

    This is the figure that argues for the rest of Phase 5. A single MCCFR curve
    looks like a result; twenty of them on the same axes show how much of that
    "result" was the seed, and how far a reader would be misled by whichever run
    happened to be published.
    """
    Figure = _require_matplotlib()
    if len(run.seeds) < 2:
        raise ValueError(
            f"plot_seed_spread needs a run with several seeds; {run.algorithm} has "
            f"{len(run.seeds)} (it is {'deterministic' if run.deterministic else 'stochastic'})"
        )

    fig, ax = _new_figure(Figure)
    x = np.asarray(run.checkpoints, dtype=float)
    for curve in run.exploitability_by_seed:
        ax.plot(x, curve, color="#0072B2", alpha=0.30, linewidth=0.9)
    aggregates = run.aggregates(envelope=envelope)
    ax.plot(
        x,
        [a.median for a in aggregates],
        color="#D55E00",
        linewidth=2.2,
        marker="o",
        markersize=4.0,
        label=f"median of {len(run.seeds)} seeds",
    )
    ax.fill_between(
        x,
        [a.ci_low for a in aggregates],
        [a.ci_high for a in aggregates],
        color="#D55E00",
        alpha=0.25,
        linewidth=0,
        label="95% bootstrap CI of the median",
    )
    ax.plot([], [], color="#0072B2", alpha=0.45, linewidth=0.9, label="individual seeds")

    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Exploitability")
    final = aggregates[-1]
    spread = "infinite" if not np.isfinite(final.spread_ratio) else f"{final.spread_ratio:.1f}x"
    _finish(
        fig,
        ax,
        title or f"Seed variance: {run.label} on {run.game}",
        subtitle,
        f"At {run.checkpoints[-1]:,} iterations the luckiest seed is {spread} better than the "
        f"unluckiest.  The shaded band here is the bootstrap CI of the median -- how well "
        f"{len(run.seeds)} seeds pin the centre down -- not the spread between runs.",
    )
    return fig


def plot_seed_spread(
    run: ConvergenceRun,
    path: str | Path,
    title: str | None = None,
    subtitle: str | None = None,
    envelope: tuple[float, float] = DEFAULT_ENVELOPE,
    dpi: int = 160,
) -> Path:
    """`figure_seed_spread`, saved."""
    return _save(figure_seed_spread(run, title, subtitle, envelope), path, dpi)
