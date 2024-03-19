bootstrap: env req-sync

dev-container: uv venv req-install

env:
	./conda_bootstrap.sh --location=current --name=gnn-recsys --version=3.10

venv:
	uv venv
	source .venv/bin/activate

uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

req-compile:
	uv pip compile pyproject.toml -o requirements.txt
	if grep -q "torch" requirements.txt; then sed -i '/torch/i -f https://download.pytorch.org/whl/torch_stable.html' requirements.txt; fi

req-install:
	uv pip install -r requirements.txt

req-sync:
	uv pip sync requirements.txt

req-cs: req-compile req-sync

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