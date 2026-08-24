"""The Streamlit dashboard, driven by Streamlit's own AppTest.

These run the real app rather than importing its functions: the interesting failures
in a dashboard are wiring failures -- a widget key that does not exist, a figure built
from the wrong object, an expensive solve that fires on load -- and none of them show
up when you call the helpers directly.

Skips cleanly without the optional dashboard extra.
"""

import pytest

pytest.importorskip("streamlit", reason="the dashboard needs the optional dashboard extra")

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "gto_solver" / "dashboard.py")
DEFAULT_TIMEOUT = 90


def run_app(**session_state) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=DEFAULT_TIMEOUT)
    for key, value in session_state.items():
        app.session_state[key] = value
    return app.run()


@pytest.fixture(scope="module")
def app() -> AppTest:
    return run_app()


def test_the_app_loads_without_exceptions(app):
    assert not app.exception


def test_it_solves_kuhn_on_load_and_reports_a_real_number(app):
    """Kuhn is milliseconds, so it is allowed to solve without being asked."""
    values = [m.value for m in app.metric]
    assert values, "no metrics rendered"
    exploitability = float(values[0])
    assert 0.0 <= exploitability < 0.05, values


def solve_metrics(app) -> dict[str, str]:
    """Only the Solve tab's metrics. The Microstructure tab renders its own on every
    load, so an unscoped `app.metric` never distinguishes 'solved' from 'did not'.
    """
    wanted = ("Exploitability", "Value to player 0", "Info sets")
    return {m.label: m.value for m in app.metric if m.label in wanted}


def test_leduc_does_not_solve_until_asked():
    """A 288-info-set tree is seconds per run. Solving it on a widget change would
    make every unrelated interaction slow, so it sits behind a button.
    """
    app = run_app()
    app.selectbox(key="solve_game").set_value("leduc").run()
    assert not app.exception
    assert any("behind the button" in info.value for info in app.info)
    assert solve_metrics(app) == {}, "Leduc solved without being asked"


def test_pressing_the_button_solves_leduc():
    app = run_app()
    app.selectbox(key="solve_game").set_value("leduc").run()
    app.select_slider(key="solve_iterations").set_value(100).run()
    app.button(key="solve_run").click().run()
    assert not app.exception
    metrics = solve_metrics(app)
    assert metrics["Info sets"] == "288"


def test_a_sampled_variant_offers_a_seed_and_an_exact_one_does_not():
    """A seed box beside a deterministic run would read as if it mattered."""
    sampled = run_app(solve_algorithm="mccfr")
    assert any(widget.key == "solve_seed" for widget in sampled.number_input)

    exact = run_app(solve_algorithm="vanilla")
    assert not any(widget.key == "solve_seed" for widget in exact.number_input)
    assert any("never reads the rng" in caption.value for caption in exact.caption)


def test_the_market_making_game_exposes_its_mu_and_kuhn_does_not():
    """Scoped by widget key: the Microstructure tab has a mu slider of its own that is
    always present, so matching on the label alone would pass no matter what.
    """
    app = run_app()
    app.selectbox(key="solve_game").set_value("gm").run()
    assert any(slider.key == "solve_mu" for slider in app.slider)

    app.selectbox(key="solve_game").set_value("kuhn").run()
    assert not any(slider.key == "solve_mu" for slider in app.slider)


def test_the_benchmarks_tab_reads_published_results_and_shows_the_notes(app):
    """The notes exist to stop a reduced or capped run reading like a complete one,
    so a dashboard that renders the numbers and drops them would undo the point.
    """
    results = Path(__file__).resolve().parents[1] / "results"
    if not any(results.glob("*.json")):
        pytest.skip("no published results to display")
    labels = [expander.label for expander in app.expander]
    assert any("Notes" in label for label in labels), labels
    assert any("Seed variance" in label for label in labels), labels


def test_every_band_is_described_as_a_seed_envelope_not_a_confidence_interval(app):
    """The distinction the whole stats module exists to hold."""
    text = " ".join(c.value for c in app.caption) + " ".join(m.value for m in app.markdown)
    assert "seed envelope" in text
    assert "never a confidence interval" in text


def test_the_microstructure_tab_reports_cfr_against_both_benchmarks(app):
    labels = [m.label for m in app.metric]
    assert "CFR spread" in labels
    assert "Brute-force optimum" in labels
    assert "Competitive (zero-profit)" in labels


def test_the_solved_spread_matches_the_independent_benchmark(app):
    """The Phase 4 gate, surfaced in the UI: the delta shown next to the brute-force
    optimum must be zero, since that benchmark shares no code with the solver.
    """
    solved = next(m for m in app.metric if m.label == "CFR spread")
    brute = next(m for m in app.metric if m.label == "Brute-force optimum")
    assert float(solved.value) == pytest.approx(float(brute.value), abs=1e-9)
    assert float(brute.delta) == pytest.approx(0.0, abs=1e-9)


def test_charts_are_rendered_for_every_tab(app):
    """Matplotlib figures reach AppTest as image elements: the live Kuhn solve, the
    selected suite, and the seed-spread chart inside its expander.
    """
    assert len(app.get("image")) >= 2
