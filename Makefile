env:
	./conda_bootstrap.sh

add-torch:
	poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu117
	poetry add --source pytorch torch

add-torch-cpu:
	poetry source add -p explicit pytorch-cpu https://download.pytorch.org/whl/cpu
	poetry add --source pytorch torch

poetry:
	pip install poetry
	poetry install
	poetry shell