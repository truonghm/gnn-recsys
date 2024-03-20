SHELL := /bin/bash

bootstrap: env req-sync

conda:
	./conda_bootstrap.sh --location=current --name=gnn-recsys --version=3.10

venv:
	test -d venv || uv venv

avenv: venv
	source .venv/bin/activate

uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

req-compile:
	uv pip compile requirements/requirements-base.in requirements/requirements-cpu.in -o requirements.txt
	uv pip compile requirements/requirements-base.in requirements/requirements-cuda.in -o requirements-cuda.txt

req-install:
	uv pip install -r requirements.txt

req-install-cuda:
	uv pip install -r requirements-cuda.txt

req-sync:
	uv pip sync requirements.txt

req-sync-cuda:
	uv pip sync requirements-cuda.txt

req-cs: req-compile req-sync

req-csc: req-compile req-sync-cuda

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