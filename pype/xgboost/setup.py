from setuptools import find_namespace_packages, setup

version = "0.2.0"

deps = [
    f"pype-base=={version}",
    "pype-sklearn",
    "xgboost>=1.6.2",
]
strict_deps = [s.replace(">=", "==") for s in deps]

setup(
    name="pype-xgboost",
    install_requires=deps,
    extras_require={"dev": strict_deps, "strict": strict_deps},
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
