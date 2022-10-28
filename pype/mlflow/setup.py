from setuptools import find_namespace_packages, setup

setup(
    name="pype-mlflow",
    install_requires=[
        "pype-base",
        "mlflow==1.28.0",
        "GitPython==3.1.27",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
)
