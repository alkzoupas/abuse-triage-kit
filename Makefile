PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

.PHONY: test lint typecheck demo confirm check

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

typecheck:
	$(PY) -m mypy

check: lint typecheck test

demo:
	$(PY) -m triage_kit.demo

confirm:
	$(PY) -m triage_kit.confirm
