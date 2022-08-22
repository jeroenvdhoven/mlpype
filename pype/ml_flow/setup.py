from setuptools import find_namespace_packages, setup

setup(
    name="pype-ml-flow",
    install_requires=[
        "pype-base",
        "mlflow==1.28.0",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
)
