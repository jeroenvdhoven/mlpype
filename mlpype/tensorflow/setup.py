from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.4.9"

    deps = [
        f"mlpype-base=={version}",
        # on mac, it is recommended to use conda/mamba or source to install tensorflow
        "tensorflow>=2.12",
        "numpy>=1.23.0",
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
