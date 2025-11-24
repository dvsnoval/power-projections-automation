.PHONY: format lint format-check lint-check

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

lint-check:
	uv run ruff check .
