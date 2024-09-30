"""Main setup script for MLpype."""
from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.6.0"

    dev_deps = [
        "pre-commit",
        "build==0.8.0",
        "pypiserver==1.5.1",
        "twine==4.0.1",
        "importlib-metadata<8.0.0",
        "sphinx-autoapi==3.1.2",
        "sphinx-rtd-theme==2.0.0",
        "myst-parser==3.0.1",
    ]
    test_deps = ["pytest", "pytest-cov"]
    deps = [
        "docstring_parser>=0.14.1",
        "pydantic>=1.10.7",
        "joblib>=1.1.1",
        "PyYAML==6.0.1",
        "jinja2==3.1.4",
    ]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-base",
        install_requires=deps,
        extras_require={
            "dev": strict_deps + dev_deps + test_deps,
            "test": strict_deps + test_deps,
            "strict": strict_deps,
        },
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
