from setuptools import find_namespace_packages, setup

deps: list[str] = ["pype-base", "fastapi==0.79.0"]
dev_deps = ["uvicorn==0.18.2"]

setup(
    name="pype-fastapi",
    install_requires=deps,
    extras_require={"dev": deps + dev_deps},
    packages=find_namespace_packages(include=["pype.*"]),
)
