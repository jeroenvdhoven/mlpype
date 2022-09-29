from setuptools import find_namespace_packages, setup

deps: list[str] = ["pype-base", "hyperopt==0.2.7"]

setup(
    name="pype-hyperopt",
    install_requires=deps,
    packages=find_namespace_packages(include=["pype.*"]),
)
