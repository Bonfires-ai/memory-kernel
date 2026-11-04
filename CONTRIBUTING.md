# Contributing

## Set up

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[build,parser,dev]"
```

or `make venv`. The spaCy pipeline is a wheel URL, so `make venv` installs it for you; with plain pip see the README.
`mk setup` fetches the runtime data (engine, frames, models) if you want to run a build.

## Checks

```bash
make lint format-check     # ruff check and ruff format --check on src and tests
make test                  # pytest tests -q (fetches the WordNet corpus first, through `mk setup`)
make wheel                 # build the wheel, install it in a clean venv, run mk --help and mk setup --check --json
```

The tests run under a blocked environment (no real API key, a dead HTTPS proxy, the Hugging Face hub offline): `tests/conftest.py`
sets it for every test and for the child processes they start, so a test that tries the network fails instead of billing.
There is no public end-to-end test: the suite is the import smoke (every module imports in a fresh process), contract tests
for the CLI, the graph contract, serving and the provider replay, and the hazard ratchet. The render tests need the NLTK WordNet
corpus: `make test` runs `make wordnet` (`mk setup --component wordnet` into `$(DATA_DIR)`) and points the tests at it with
`NLTK_DATA`; with plain pytest, export `NLTK_DATA` to a directory that holds `corpora/wordnet` yourself. Lint and format are pinned
to one ruff version (`[dev]`), so a different release can disagree with CI.

`tests/unit/graph/test_kg_contract.py` has two cross-checks against `graphiti_core`. They are optional: they skip with a reason when
that package is not installed, and the package does not import it.

## The hazard ratchet

A test scans every tracked file for private identifiers and development-layout leaks and fails on a match. It carries no plaintext
pattern itself. If it flags your change, remove the text; do not add an exemption.

## What does not go in the tree

- No dataset text: no benchmark conversations, questions or answers, and nothing derived from them. Examples are invented.
- No secrets, personal identifiers, machine paths or private hostnames, in code, comments, docs or tests.
- No hand-written vocabulary lists: word lists are package data (see `memory_kernel/data/`).
- Do not edit the frozen modules whose bytes are pinned (the ledger's `MK_CODE` files);
  a change there changes the code identity of every build.

## Commits

Small commits, one concern each, with explicit paths (`git add path ...`, not `git add -A`). Run `ruff format` and `ruff check` on the
files you touched. Message: a short imperative subject, then a body that says why. Do not rewrite published history.
