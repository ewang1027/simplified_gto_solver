# Documentation

| Document | Read it for |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Every measured finding, organized by claim — including the ones that were wrong |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Why the code is shaped this way, where its abstractions stop, and how to extend it |
| [`REFERENCE.md`](REFERENCE.md) | Module-by-module map of what lives where |
| [`BUILDLOG.md`](BUILDLOG.md) | The chronology: what each phase built, what it cost, the traps, and what is still open |
| [`phase4-microstructure-design.md`](phase4-microstructure-design.md) | The microstructure modeling work, including the formulations that failed |

Start with `RESULTS.md` if you want to know what was found, `ARCHITECTURE.md` if you want to
know how it works, and `BUILDLOG.md` if you want to know why it looks like this.

The repository [`README.md`](../README.md) is the overview and is the only one of these that
assumes no prior context.

## What is checked automatically

These documents are not only prose. `tests/test_docs.py` verifies that every path they name
exists, every `gto` command they tell you to run is real, and every benchmark table in the
README regenerates from the results file it came from. `scripts/audit_doc_numbers.py`
re-measures the machine-specific claims.

Both exist because this project's most persistent failure has been documents that were true
when written — see the corrections table in `RESULTS.md`.
