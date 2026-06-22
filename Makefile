.PHONY: setup lint fmt type test demo api check monitoring-up monitoring-down drift-demo metrics-exporter

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

monitoring-up:
	./scripts/monitoring_stack.sh up

monitoring-down:
	./scripts/monitoring_stack.sh down

drift-demo:
	uv run python scripts/simulate_drift_demo.py --n-pre 5000 --n-post 3000 --shifts 4 --seed 7

metrics-exporter:
	NBA_METRICS_EXPORTER_ENABLED=1 uv run python scripts/run_metrics_exporter.py

check: lint type test
