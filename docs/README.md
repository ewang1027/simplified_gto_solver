# Documentation

| Document | Read it for |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Every measured finding, organized by claim — including the ones that were wrong |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Why the code is shaped this way, where its abstractions stop, and how to extend it |
| [`REFERENCE.md`](REFERENCE.md) | Module-by-module map of what lives where |

Start with `RESULTS.md` if you want to know what was found, and `ARCHITECTURE.md` if you
want to know how it works.

The repository [`README.md`](../README.md) is the overview and is the only one of these that
assumes no prior context.

## What is checked automatically

These documents are not only prose. `tests/test_docs.py` verifies that every path they name
exists and every `gto` command they tell you to run is real. `scripts/audit_doc_numbers.py`
re-measures the machine-specific claims.

Both exist because this project's most persistent failure has been documents that were true
when written — see the corrections table in `RESULTS.md`.
