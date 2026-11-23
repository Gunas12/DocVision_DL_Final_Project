.PHONY: setup test lint data reproduce

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt pytest ruff

test:
	pytest -q

lint:
	ruff check src tests

# The targets below are available once the corresponding scripts are implemented.
data:
	bash scripts/download_data.sh

reproduce:
	bash scripts/reproduce.sh
