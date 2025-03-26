# For local development setup
PYTHON_VERSION=3.11
PYENV_NAME=mlpype

init: _init_env _init_deps _init_precommit

# Initialize UV project and virtual environment
_init_env:
	uv venv --python=$(PYTHON_VERSION)

# Editable install for easy development.
_init_deps:
	uv sync --all-packages --extra dev --extra test --extra strict

# Install pre-commit
_init_precommit:
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg

# To test if packages can be build
build-packages:
	uv build --all-packages

clean:
	rm -rf dist/

host-pypi-local:
	mkdir -p packages
	pypi-server run -p 8080 packages -a . -P . --overwrite &
	twine upload --repository-url http://0.0.0.0:8080 dist/* -u '' -p ''

build-and-host-local: clean build-packages host-pypi-local

# Test and coverage commands
test-unit:
	uv run python -m pytest -m "not spark and not wheel"

test-all:
	uv run python -m pytest

coverage-unit:
	uv run python -m pytest -m "not spark and not wheel" --cov-report term-missing --cov mlpype -ra

coverage-all:
	uv run python -m pytest --cov-report term-missing --cov mlpype -ra

# Document code
build-docs:
	uv run python -m scripts.build_mkdown
	uv run python -m mkdocs build

serve-docs:
	uv run python -m mkdocs serve

pre-commit-run:
	uv run pre-commit run --all-files
