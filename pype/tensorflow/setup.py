from setuptools import find_namespace_packages, setup

setup(
    name="pype-tensorflow",
    install_requires=[
        "pype-base",
        # on mac, it is recommended to use conda/mamba or source to install tensorflow
        "tensorflow==2.9.1",
        "numpy==1.23.0",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
)
