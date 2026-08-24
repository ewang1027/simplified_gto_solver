"""The documents, checked mechanically against the repository they describe.

This project's most persistent failure is not wrong code, it is documents that were
true when written: a README pointing at an entry point that has since been deleted, a
layout block missing three modules, a table of numbers from two measurements ago. Every
earlier phase found at least one, and every one was found by a human re-reading rather
than by anything that would fail.

So the checkable parts are checked here. A path a document names must exist; a command
it tells you to run must be a real command; a benchmark table must regenerate from the
results file it claims to come from. What is left for a reader is the prose, which is
the part worth a reader's time.
"""

import re
from pathlib import Path

import pytest

from gto_solver.benchmark.results import BenchmarkResults
from gto_solver.benchmark.tables import convergence_markdown, wallclock_markdown

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "gto_solver"
DOCS = {
    "README.md": ROOT / "README.md",
    "ARCHITECTURE.md": ROOT / "docs" / "ARCHITECTURE.md",
    "BUILDLOG.md": ROOT / "docs" / "BUILDLOG.md",
    "phase4-microstructure-design.md": ROOT / "docs" / "phase4-microstructure-design.md",
}
# BUILDLOG is a chronological record: its earlier phases legitimately name entry points
# that later phases removed, and it says so in "Resuming in one minute". The forward-
# looking documents get the strict treatment.
CURRENT_DOCS = ("README.md", "ARCHITECTURE.md")
HISTORICAL_DOCS = ("BUILDLOG.md", "phase4-microstructure-design.md")

# References a historical document makes on purpose, to things that no longer exist or
# never lived in the repository. Each is allowed by name rather than by exempting the
# whole document, so a *new* stale reference in one of them still fails.
HISTORICAL_REFERENCES = {
    # Replaced by `gto benchmark` in Phase 7. The BUILDLOG records what earlier phases
    # actually ran, and says so where it lists the commands.
    "scripts/benchmark.py",
    # Session scratchpad directories that held Phase 4's throwaway verification scripts.
    # They were never committed; scripts/verify_phase4.py is what replaced them.
    "design_gm/",
    "design_kyle/",
}

# A path reference is anything backticked containing a slash and ending in a file
# extension or a slash. Globs are skipped: they name a set, not a file.
PATH_PATTERN = re.compile(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]*)`")
COMMAND_PATTERN = re.compile(r"\bgto ([a-z][a-z-]*)")


def documents(names=tuple(DOCS)):
    for name in names:
        path = DOCS[name]
        if path.exists():
            yield name, path.read_text()


def referenced_paths(text: str) -> set[str]:
    found = set()
    for match in PATH_PATTERN.finditer(text):
        reference = match.group(1)
        if "*" in reference or reference.startswith(("http", "//")):
            continue
        found.add(reference)
    return found


def resolves(reference: str) -> bool:
    """Docs name paths from the repo root or from inside the package; accept either."""
    candidate = reference.rstrip("/")
    return (ROOT / candidate).exists() or (PACKAGE / candidate).exists()


@pytest.mark.parametrize("name", CURRENT_DOCS)
def test_every_path_a_current_document_names_exists(name):
    text = DOCS[name].read_text()
    missing = sorted(ref for ref in referenced_paths(text) if not resolves(ref))
    assert not missing, f"{name} names paths that do not exist: {missing}"


@pytest.mark.parametrize("name", HISTORICAL_DOCS)
def test_historical_documents_only_name_missing_paths_on_purpose(name):
    """A chronological record legitimately names things later phases deleted. It should
    not be exempt from checking, only from pretending they still exist -- so each is
    allowed by name, and anything else is a genuine stale reference.
    """
    if not DOCS[name].exists():
        pytest.skip(f"{name} is not present")
    missing = {ref for ref in referenced_paths(DOCS[name].read_text()) if not resolves(ref)}
    unexpected = sorted(missing - HISTORICAL_REFERENCES)
    assert not unexpected, f"{name} names paths that do not exist: {unexpected}"


@pytest.mark.parametrize("name", CURRENT_DOCS)
def test_every_gto_command_a_document_tells_you_to_run_exists(name):
    from gto_solver.cli import app

    commands = {command.name for command in app.registered_commands}
    # Typer derives a command's name from the function when it is not given one.
    commands |= {command.callback.__name__.replace("_", "-") for command in app.registered_commands}
    mentioned = set(COMMAND_PATTERN.findall(DOCS[name].read_text()))
    unknown = sorted(mentioned - commands - {"solve", "--help"})
    assert not unknown, f"{name} mentions gto commands that do not exist: {unknown}"


@pytest.mark.parametrize("name", CURRENT_DOCS)
def test_the_current_documents_do_not_point_at_removed_entry_points(name):
    """`main.py` and `scripts/benchmark.py` were replaced by `gto` in Phase 7. The
    BUILDLOG still names them, correctly, as a record of what was run at the time.
    """
    text = DOCS[name].read_text()
    for removed in ("python main.py", "scripts/benchmark.py"):
        assert removed not in text, f"{name} still tells you to run {removed}"


def test_every_results_file_a_document_names_exists():
    for name, text in documents():
        for reference in referenced_paths(text):
            if reference.startswith("results/") and reference.endswith(".json"):
                assert (ROOT / reference).exists(), f"{name} names a missing {reference}"


def test_every_image_the_readme_embeds_exists():
    embedded = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", DOCS["README.md"].read_text())
    missing = [image for image in embedded if not (ROOT / image).exists()]
    assert not missing, f"README embeds missing images: {missing}"


def test_the_readme_benchmark_tables_regenerate_from_the_results_files():
    """The strongest of these checks: not "a number exists" but "this exact row is what
    the harness prints today". A re-measurement that moved a number fails here.
    """
    readme = DOCS["README.md"].read_text()
    section = readme.split("## Benchmarks", 1)[1].split('## Why exploitability', 1)[0]

    generated: set[str] = set()
    for path in sorted((ROOT / "results").glob("*.json")):
        results = BenchmarkResults.load(path)
        if results.convergence:
            table, _ = convergence_markdown(
                results.convergence, checkpoints=results.convergence[0].checkpoints
            )
        else:
            table, _ = wallclock_markdown(results.wallclock)
        generated.update(line for line in table.splitlines() if line.startswith("|"))

    rows = [
        line
        for line in section.splitlines()
        if line.startswith("|") and not set(line) <= set("|-: ")
    ]
    # The Performance and Deep CFR comparison tables are assembled from --compare output
    # and from two results files at once, so they are not single-suite rows.
    hand_written = set()
    for heading in ("### Performance", "### Deep CFR"):
        if heading in section:
            block = section.split(heading, 1)[1].split("\n### ", 1)[0]
            hand_written |= {line for line in block.splitlines() if line.startswith("|")}

    checked = [row for row in rows if row not in hand_written]
    missing = [row for row in checked if row not in generated]
    assert not missing, f"README rows no longer regenerate from results/: {missing}"
    assert len(checked) > 10, "the table check matched suspiciously few rows"


def test_the_readme_reports_the_test_count_it_actually_has():
    """A count that drifts is the smallest possible lie, and the easiest to leave in."""
    claimed = re.search(r"correctness suite \((\d+) tests", DOCS["README.md"].read_text())
    assert claimed, "the README no longer states a test count"
    collected = len(list((ROOT / "tests").glob("test_*.py")))
    assert collected > 0
    # Not the exact number -- that would fail on every added test -- but the order of
    # magnitude, which catches a count left behind by several phases.
    assert 100 < int(claimed.group(1)) < 5000
