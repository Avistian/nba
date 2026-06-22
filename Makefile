.PHONY: setup lint fmt type test demo api check monitoring-up monitoring-down drift-demo metrics-exporter online-drift-demo metrics-exporter-demo

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

online-drift-demo:
	NBA_USE_DRIFT_MONITORING=1 NBA_ALERT_EMAIL_ENABLED=1 \
	uv run python scripts/run_online_drift_demo.py \
		--warmup 3000 --events-per-tick 200 --ticks 12 --tick-seconds 15 \
		--drift-mode ramp --drift-onset 4 --seed 7

metrics-exporter-demo:
	NBA_METRICS_EXPORTER_ENABLED=1 \
	NBA_DB_PATH=artifacts/drift_demo/events.db \
	NBA_DEPLOYED_MODEL_MANIFEST=artifacts/drift_demo/models/deployed.json \
	NBA_MONITORING_REPORT_PATH=artifacts/drift_demo/monitoring/drift_reports.jsonl \
	NBA_RETRAIN_AUDIT_PATH=artifacts/drift_demo/monitoring/retrain_audit.jsonl \
	uv run python scripts/run_metrics_exporter.py

check: lint type test
