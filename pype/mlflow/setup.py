from setuptools import find_namespace_packages, setup

version = "0.1.0"

setup(
    name="pype-mlflow",
    install_requires=[
        f"pype-base=={version}",
        "mlflow==1.28.0",
        "GitPython==3.1.27",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
