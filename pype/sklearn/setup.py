from setuptools import find_namespace_packages, setup

version = "0.1.0"

setup(
    name="pype-sklearn",
    install_requires=[
        f"pype-base=={version}",
        "numpy==1.23.0",
        "scikit-learn==1.1.1",
        "pandas==1.4.3",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
