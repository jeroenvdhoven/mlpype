"""An extension of MLpype for XGBoost models."""
from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.6.0"

    deps = [
        f"mlpype-base=={version}",
        f"mlpype-sklearn=={version}",
        "xgboost>=1.6.2",
    ]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-xgboost",
        install_requires=deps,
        extras_require={"dev": strict_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
