from typing import List

from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.4.9"
    deps: List[str] = [f"mlpype-base=={version}", "hyperopt>=0.2.7"]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-hyperopt",
        install_requires=deps,
        extras_require={"dev": strict_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
