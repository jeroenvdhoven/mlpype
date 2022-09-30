from setuptools import find_namespace_packages, setup

setup(
    name="pype-xgboost",
    install_requires=[
        "pype-base",
        "pype-sklearn",
        "xgboost==1.6.2",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
)
