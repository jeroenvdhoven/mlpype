from setuptools import find_namespace_packages, setup

dev_deps = ["pre-commit"]

test_deps = ["pytest"]

deps: list[str] = []

setup(
    name="pype",
    packages=find_namespace_packages(include=["pype.*"]),
    namespace_packages=["pype"],
    install_requires=deps,
    python_requires=">=3.10",
    extras_require={"dev": deps + dev_deps + test_deps, "test": deps + test_deps},
)
