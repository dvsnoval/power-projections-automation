.PHONY: all format lint format-check lint-check test

all: test format-check lint-check

format:
	uv run ruff format utils/ tests/

lint:
	uv run ruff check --fix utils/ tests/

format-check:
	uv run ruff format --check utils/ tests/

lint-check:
	uv run ruff check utils/ tests/

test:
	uv run pytest
