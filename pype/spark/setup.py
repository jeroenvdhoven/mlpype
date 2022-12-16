from setuptools import find_namespace_packages, setup

version = "0.1.0"

deps = [
    f"pype-base=={version}",
    # We will provide absolute no guarantees that our integration will work with
    # EVERY version of pyspark. This has been developed under pyspark==3.2.1.
    "pyspark>=3.2.1",
]
test_deps = ["pandas"]

setup(
    name="pype-spark",
    install_requires=deps,
    extras_require={"dev": deps + test_deps, "test": deps + test_deps},
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
