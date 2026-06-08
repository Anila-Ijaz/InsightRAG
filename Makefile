.PHONY: help install dev lint format test eval up down logs ingest

help:
	@echo "Available targets:"
	@echo "  install   Install package + dev dependencies"
	@echo "  dev       Run the API locally (requires services up)"
	@echo "  lint      Run ruff + mypy"
	@echo "  format    Auto-format with ruff"
	@echo "  test      Run pytest with coverage"
	@echo "  eval      Run RAGAS evaluation against running API"
	@echo "  up        Start full stack via docker-compose"
	@echo "  down      Stop docker-compose stack"
	@echo "  logs      Tail API logs"
	@echo "  ingest    Ingest a sample 10-K (TICKER=AAPL by default)"

install:
	pip install -e ".[dev,eval]"
	pre-commit install || true

dev:
	uvicorn insightrag.api.main:app --reload --host 0.0.0.0 --port 8000

lint:
	ruff check src/ tests/
	mypy src/insightrag --ignore-missing-imports

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest

eval:
	python evals/run_ragas.py --testset evals/testset.jsonl --output evals/results.json

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

TICKER ?= AAPL
ingest:
	curl -X POST http://localhost:8000/v1/ingest \
		-H "Content-Type: application/json" \
		-d '{"ticker":"$(TICKER)","limit":1}'
