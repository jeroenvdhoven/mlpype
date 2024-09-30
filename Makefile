# For local development setup
PYTHON_VERSION=3.11
PYENV_NAME=mlpype

pyenv-delete:
	pyenv virtualenv-delete -f ${PYENV_NAME}

pyenv:
	pyenv install -s ${PYTHON_VERSION}
	pyenv virtualenv ${PYTHON_VERSION} ${PYENV_NAME} -f
	echo ${PYENV_NAME} > .python-version

pyenv-dev-install: pyenv dev-install

pyenv-dev-setup: pyenv dev-setup

# Development setup
dev-setup: dev-install pre-commit-install

dev-install:
	./scripts/dev_install.sh -e 1

local-install:
	./scripts/dev_install.sh

# To test if packages can be build
build-packages:
	./scripts/build.sh

clean:
	rm -rf dist/ packages/

host-pypi-local:
	mkdir -p packages
	pypi-server run -p 8080 packages -a . -P . --overwrite &
	twine upload --repository-url http://0.0.0.0:8080 dist/* -u '' -p ''

build-and-host-local: clean build-packages host-pypi-local

# Test and coverage commands
test-unit:
	python -m pytest -m "not spark and not wheel"

test-all:
	python -m pytest

coverage-unit:
	python -m pytest -m "not spark and not wheel" --cov-report term-missing --cov mlpype -ra

coverage-all:
	python -m pytest --cov-report term-missing --cov mlpype -ra

# Document code
create-docs:
	cd docs && sphinx-build -b html . _build

# Pre-commit defaults
pre-commit-install:
	pip install pre-commit
	pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg

pre-commit-run:
	pre-commit run --all-files
