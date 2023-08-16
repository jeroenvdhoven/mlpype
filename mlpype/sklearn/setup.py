from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.4.9"

    deps = [
        f"mlpype-base=={version}",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.1",
        "pandas>=1.4.3",
    ]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-sklearn",
        install_requires=deps,
        extras_require={"dev": strict_deps, "strict": strict_deps},
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
