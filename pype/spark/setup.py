from setuptools import find_namespace_packages, setup

setup(
    name="pype-spark",
    install_requires=[
        "pype-base",
        # We will provide absolute no guarantees that our integration will work with
        # EVERY version of pyspark. This has been developed under pyspark==3.3.0.
        "pyspark",
    ],
    packages=find_namespace_packages(include=["pype.*"]),
)
