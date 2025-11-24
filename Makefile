.PHONY: format lint format-check lint-check

format:
	ruff format .

lint:
	ruff check --fix .

format-check:
	ruff format --check .

lint-check:
	ruff check .
