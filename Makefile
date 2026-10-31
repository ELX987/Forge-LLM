PY ?= python

.PHONY: test test-gpu lint format run install dev

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests -q

test-gpu:
	$(PY) -m pytest tests_gpu -q -m gpu

lint:
	$(PY) -m ruff check forge tests tests_gpu

format:
	$(PY) -m ruff format forge tests tests_gpu
	$(PY) -m ruff check --fix forge tests tests_gpu

run:
	$(PY) -m forge.cli serve-ui
