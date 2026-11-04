# Developer entry points. `.github/workflows/ci.yml` runs the same targets.
#
#   make venv          create $(VENV) with the package, its extras and the spaCy pipeline
#   make lint          ruff check on src and tests
#   make format-check  ruff format --check on src and tests
#   make wordnet       fetch the WordNet corpus through `mk setup` into $(DATA_DIR) (the tests read it through NLTK_DATA)
#   make test          pytest tests -q under the blocked environment; needs the WordNet corpus (make wordnet runs first)
#   make wheel         build the wheel, install it in a clean venv, run mk --help, mk setup --check and an import smoke
#   make setup         fetch the pinned runtime data (mk setup)

VENV      ?= .venv
PY        := $(VENV)/bin/python
PYTHON    ?= python3
WHEEL_DIR ?= .wheel-check
# Where `make wordnet` places the corpus. Outside the checkout: the hazard ratchet walks the whole tree when there is no .git.
DATA_DIR  ?= $(HOME)/.cache/memory-kernel-test-data
EXTRAS    ?= build,parser
SPACY_WHL := https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# A test must never reach the network or bill an account.
BLOCKED := OPENROUTER_API_KEY=replay-invalid HTTPS_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.PHONY: help venv lint format-check wordnet test wheel setup

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## create/refresh the dev venv (package + extras + the spaCy pipeline)
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -e ".[$(EXTRAS),dev]"
	$(PY) -m pip install "en_core_web_sm @ $(SPACY_WHL)"

lint: ## ruff check
	$(PY) -m ruff check src tests

format-check: ## ruff format --check
	$(PY) -m ruff format --check src tests

wordnet: ## fetch the WordNet corpus the tests need (mk setup; a no-op once it is there)
	$(VENV)/bin/mk setup --component wordnet --data-dir $(DATA_DIR)

test: wordnet ## pytest tests -q, blocked environment; the render tests need the WordNet corpus (NLTK_DATA)
	$(BLOCKED) NLTK_DATA=$(DATA_DIR)/nltk_data $(PY) -m pytest tests -q

wheel: ## build the wheel; clean-venv install; mk --help; mk setup --check --json (rc 0 or 1, parses as JSON, no Traceback); import smoke
	rm -rf dist $(WHEEL_DIR)
	mkdir -p $(WHEEL_DIR)/cwd $(WHEEL_DIR)/data
	$(PYTHON) -m venv $(WHEEL_DIR)/build-venv
	$(WHEEL_DIR)/build-venv/bin/pip install build
	$(WHEEL_DIR)/build-venv/bin/python -m build --wheel --outdir dist .
	$(PYTHON) -m venv $(WHEEL_DIR)/venv
	$(WHEEL_DIR)/venv/bin/pip install "$$(ls $(CURDIR)/dist/*.whl)[$(EXTRAS)]"
	cd $(WHEEL_DIR)/cwd && env -u PYTHONPATH $(BLOCKED) MK_DATA_DIR=$(CURDIR)/$(WHEEL_DIR)/data $(CURDIR)/$(WHEEL_DIR)/venv/bin/mk --help
	cd $(WHEEL_DIR)/cwd && env -u PYTHONPATH $(BLOCKED) MK_DATA_DIR=$(CURDIR)/$(WHEEL_DIR)/data \
	  $(CURDIR)/$(WHEEL_DIR)/venv/bin/mk setup --check --json >setup-check.json 2>setup-check.err; \
	  rc=$$?; echo "mk setup --check rc=$$rc"; test $$rc -le 1 \
	  && $(CURDIR)/$(WHEEL_DIR)/venv/bin/python -I -c "import json, sys; \
	  doc = json.load(open('setup-check.json')); \
	  bad = [c for c in doc['components'] if c['status'] not in ('ok', 'missing')]; \
	  sys.exit('bad component status: %s' % bad if bad else 0)" \
	  && ! grep -q Traceback setup-check.err
	cd $(WHEEL_DIR)/cwd && env -u PYTHONPATH $(BLOCKED) MK_DATA_DIR=$(CURDIR)/$(WHEEL_DIR)/data \
	  $(CURDIR)/$(WHEEL_DIR)/venv/bin/python -I -c "import importlib.util, pathlib; \
	  root = pathlib.Path(importlib.util.find_spec('memory_kernel').origin).parent; \
	  print('\n'.join('.'.join(f.relative_to(root.parent).with_suffix('').parts).removesuffix('.__init__') \
	  for f in sorted(root.rglob('*.py'))))" \
	  | xargs -P 8 -n 1 sh -c 'env -u PYTHONPATH $(CURDIR)/$(WHEEL_DIR)/venv/bin/python -I -c "import $$0" || { echo "import failed: $$0"; exit 255; }'
	@echo "wheel check ok"

setup: ## fetch the pinned runtime data
	$(VENV)/bin/mk setup
