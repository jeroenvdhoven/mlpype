from setuptools import find_namespace_packages, setup

version = "0.1.0"

setup(
    name="pype-xgboost",
    install_requires=[
        f"pype-base=={version}",
        "pype-sklearn",
        "xgboost==1.6.2",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
