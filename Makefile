.PHONY: install dev-backend dev-frontend test lint build eval

install:
	cd backend && python -m pip install -r requirements-dev.txt
	cd frontend && npm ci

dev-backend:
	cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

test:
	cd backend && python -m pytest -q

lint:
	cd backend && python -m ruff check app tests
	cd backend && python -m mypy app

build:
	cd frontend && npm run build

eval:
	@echo "Eval harness arrives in Stage 7."