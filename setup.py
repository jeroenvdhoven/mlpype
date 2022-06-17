from setuptools import setup

dev_deps = ["pre-commit"]

test_deps = ["pytest"]

deps: list[str] = []

setup(
    name="pype",
    requires=deps,
    extras_require={"dev": deps + dev_deps + test_deps, "test": deps + test_deps},
)
