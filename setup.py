from setuptools import find_namespace_packages, setup

setup(
    name="mlpype",
    packages=find_namespace_packages(include=["mlpype.*"]),
    python_requires=">=3.8",
    version="0.4.9",
    license="MIT",
    author="Jeroen van den Hoven",
    url="https://github.com/jeroenvdhoven/mlpype",
)
