.PHONY: setup lint fmt type test demo api check

setup:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .

type:
	uv run pyright

test:
	uv run pytest --cov=nba --cov-report=term-missing

demo:
	uv run python scripts/run_demo.py

api:
	uv run uvicorn nba.api.app:app --reload

check: lint type test
