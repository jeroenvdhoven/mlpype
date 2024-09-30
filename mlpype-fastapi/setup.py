"""Main setup script for MLpype - FastAPI."""
from typing import List

from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.6.0"
    deps: List[str] = [f"mlpype-base=={version}", "fastapi>=0.86.0", "anyio<4", "typing-extensions>=4.8.0"]
    dev_deps = ["uvicorn==0.18.2"]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-fastapi",
        install_requires=deps,
        extras_require={"dev": strict_deps + dev_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
