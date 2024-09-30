"""Main setup script for MLpype - scikit-learn."""
from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.6.0"

    deps = [
        f"mlpype-base=={version}",
        "numpy>=1.26.4",
        "scikit-learn>=1.2.2",
        "pandas>=1.5.3",
        "dill>=0.3.8",
    ]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-sklearn",
        install_requires=deps,
        extras_require={"dev": strict_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
