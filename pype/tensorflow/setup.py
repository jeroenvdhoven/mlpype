from setuptools import find_namespace_packages, setup

version = "0.2.0"

deps = [
    f"pype-base=={version}",
    # on mac, it is recommended to use conda/mamba or source to install tensorflow
    "tensorflow>=2.9.1",
    "numpy>=1.23.0",
    "protobuf>=3.19",
]
strict_deps = [s.replace(">=", "==") for s in deps]

setup(
    name="pype-tensorflow",
    install_requires=deps,
    extras_require={"dev": strict_deps, "strict": strict_deps},
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
