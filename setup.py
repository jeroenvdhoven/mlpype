from setuptools import find_namespace_packages, setup

setup(
    name="pype",
    packages=find_namespace_packages(include=["pype.*"]),
    namespace_packages=["pype"],
    python_requires=">=3.10",
)
