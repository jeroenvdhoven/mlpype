from setuptools import find_namespace_packages, setup

version = "0.2.0"

deps = [
    f"pype-base=={version}",
    "numpy>=1.23.0",
    "scikit-learn>=1.1.1",
    "pandas>=1.4.3",
]
strict_deps = [s.replace(">=", "==") for s in deps]

setup(
    name="pype-sklearn",
    install_requires=deps,
    extras_require={"dev": strict_deps, "strict": strict_deps},
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
