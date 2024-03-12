env:
	./conda_bootstrap.sh

add-torch:
	poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu117
	poetry add --source pytorch torch