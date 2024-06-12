from setuptools import find_namespace_packages, setup

setup(
    name="mlpype",
    packages=find_namespace_packages(include=["mlpype.*"]),
    python_requires=">=3.9,<3.12",
    version="0.5.2",
    license="MIT",
    author="Jeroen van den Hoven",
    url="https://github.com/jeroenvdhoven/mlpype",
)
