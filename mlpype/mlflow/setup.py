from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.4.1"

    deps = [
        f"mlpype-base=={version}",
        "mlflow>=1.28.0, <2.0",
        "GitPython>=3.1.27",
    ]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-mlflow",
        install_requires=deps,
        extras_require={"dev": strict_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
