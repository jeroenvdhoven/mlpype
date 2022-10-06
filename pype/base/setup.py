from setuptools import find_namespace_packages, setup

dev_deps = ["pre-commit", "uvicorn==0.18.2"]
test_deps = ["pytest", "pytest-cov"]
deps: list[str] = ["docstring_parser==0.14.1", "pydantic==1.9.1", "joblib==1.1.0", "fastapi==0.79.0"]

setup(
    name="pype-base",
    install_requires=deps,
    extras_require={"dev": deps + dev_deps + test_deps, "test": deps + test_deps},
    packages=find_namespace_packages(include=["pype.*"]),
)
