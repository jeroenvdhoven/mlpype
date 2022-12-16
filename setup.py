from setuptools import find_namespace_packages, setup

setup(
    name="pype",
    packages=find_namespace_packages(include=["pype.*"]),
    namespace_packages=["pype"],
    python_requires=">=3.10",
    version="0.1.0",
    license="MIT",
    author="Jeroen van den Hoven",
    url="https://github.com/jeroenvdhoven/pype",
)
