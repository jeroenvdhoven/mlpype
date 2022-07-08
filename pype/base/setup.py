from setuptools import find_namespace_packages, setup

setup(
    name="pype-base",
    install_requires=[
        "joblib==1.1.0",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
)
