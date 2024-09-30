"""Main setup script for MLpype - TensorFlow."""
from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.6.0"

    deps = [
        f"mlpype-base=={version}",
        # on mac, it is recommended to use conda/mamba or source to install tensorflow
        "tensorflow>=2.14.1",
        "numpy>=1.26.4",
        "protobuf>=3.20.3",
    ]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-tensorflow",
        install_requires=deps,
        extras_require={"dev": strict_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
