PY := .venv/bin/python
COMPOSE := docker compose -f deploy/docker-compose.yml

.PHONY: install lock lint test check run report audit eval fetch-model docker-up docker-up-llm docker-down

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --require-hashes -r requirements-dev.txt

lock:
	uv pip compile requirements.in --generate-hashes --python-version 3.12 -o requirements.txt
	uv pip compile requirements-dev.in --generate-hashes --python-version 3.12 -o requirements-dev.txt
	uv pip compile requirements-audit.in --generate-hashes --python-version 3.12 -o requirements-audit.txt

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest --cov=core --cov=app --cov-report=term

check: lint test

run:
	$(PY) -m streamlit run app/main.py

report:
	$(PY) -m core.report --compact

audit:
	uv venv .audit
	uv pip install --python .audit/bin/python --require-hashes -r requirements-audit.txt
	.audit/bin/pip-audit --requirement requirements.txt --require-hashes --disable-pip --strict

eval:
	PYTHONPATH=. $(PY) -m app.evaluation --output evals/results.json

fetch-model:
	deploy/fetch-model.sh

docker-up:
	$(COMPOSE) up -d --build

docker-up-llm: fetch-model
	$(COMPOSE) --profile llm up -d --build

docker-down:
	$(COMPOSE) --profile llm down
