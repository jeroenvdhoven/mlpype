from setuptools import find_namespace_packages, setup

version = "0.1.0"
deps: list[str] = [f"pype-base=={version}", "fastapi>=0.79.0"]
dev_deps = ["uvicorn==0.18.2"]
strict_deps = [s.replace(">=", "==") for s in deps]

setup(
    name="pype-fastapi",
    install_requires=deps,
    extras_require={"dev": strict_deps + dev_deps, "strict": strict_deps},
    packages=find_namespace_packages(include=["pype.*"]),
    version=version,
)
