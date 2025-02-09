SHELL := /bin/bash

bootstrap: venv req-sync

conda:
	./conda_bootstrap.sh --location=current --name=gnn-recsys --version=3.10

venv:
	test -d venv || uv venv

avenv: venv
	source .venv/bin/activate

uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

clean:
	py3clean .
	rm -rf .ruff_cache
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf **/.pytest_cache
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf catboost_info

typecheck:
	uvx mypy -p src

lint:
	uvx ruff check --fix src/

format:
	uvx ruff format src/

check: typecheck lint format