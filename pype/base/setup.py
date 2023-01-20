from setuptools import find_namespace_packages, setup

dev_deps = ["pre-commit", "build==0.8.0", "pypiserver==1.5.1", "twine==4.0.1"]
test_deps = ["pytest", "pytest-cov"]
deps = ["docstring_parser>=0.14.1", "pydantic>=1.9.1", "joblib>=1.1.0"]
strict_deps = [s.replace(">=", "==") for s in deps]

setup(
    name="pype-base",
    install_requires=deps,
    extras_require={"dev": strict_deps + dev_deps + test_deps, "test": strict_deps + test_deps, "strict": strict_deps},
    packages=find_namespace_packages(include=["pype.*"]),
    version="0.2.0",
)
