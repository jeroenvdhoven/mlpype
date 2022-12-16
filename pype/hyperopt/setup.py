from setuptools import find_namespace_packages, setup

version = "0.1.0"
deps: list[str] = [f"pype-base=={version}", "hyperopt>=0.2.7"]
strict_deps = [s.replace(">=", "==") for s in deps]

setup(
    name="pype-hyperopt",
    install_requires=deps,
    extras_require={"dev": strict_deps, "strict": strict_deps},
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
