include .env

bootstrap: env req-sync

env:
	./conda_bootstrap.sh --location=current --name=gnn-recsys --version=3.10

uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

req-compile:
	uv pip compile pyproject.toml -o requirements.txt
	if grep -q "torch" requirements.txt; then sed -i '/torch/i -f https://download.pytorch.org/whl/torch_stable.html' requirements.txt; fi

req-install:
	uv pip install -r requirements.txt

req-sync: req-compile
	uv pip sync requirements.txt