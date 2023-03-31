# For local development setup
dev-install:
	./scripts/dev_install.sh -e 1

# To test if packages can be build
build:
	./scripts/build.sh

clean:
	rm -rf dist/ packages/

host-pypi-local:
	mkdir -p packages
	pypi-server run -p 8080 packages -a . -P . &
	twine upload --repository-url http://0.0.0.0:8080 dist/* -u '' -p ''

build-and-host-local: clean build host-pypi-local

# Test and coverage commands
test-without-spark:
	python -m pytest -m "not spark"

test-all:
	python -m pytest

coverage-without-spark:
	python -m pytest -m "not spark" --cov-report term-missing --cov pype -ra

coverage-all:
	python -m pytest --cov-report term-missing --cov pype -ra

# Document code
create-docs:
	pdoc ./pype -o ./docs -d google

# Pre-commit defaults
pre-commit-install:
	pip install pre-commit
	pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg

pre-commit-run:
	pre-commit run --all-files

# Development setup
dev-setup: dev-install pre-commit-install