"""Main setup script for MLpype - Matplotlib."""
from typing import List

from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.6.1"
    deps: List[str] = [f"mlpype-base=={version}", "matplotlib>=3.7.5", "numpy>=1.26.4", "pandas>=1.5.3", "shap>=0.46.0"]
    dev_deps = []
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-matplotlib",
        install_requires=deps,
        extras_require={"dev": strict_deps + dev_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
