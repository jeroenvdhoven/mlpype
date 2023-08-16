from setuptools import find_namespace_packages, setup

if __name__ == "__main__":
    version = "0.4.9"

    dev_deps = ["pre-commit", "build==0.8.0", "pypiserver==1.5.1", "twine==4.0.1", "pdoc==13.1.0"]
    test_deps = ["pytest", "pytest-cov"]
    deps = [
        f"mlpype=={version}",
        "docstring_parser>=0.14.1",
        "pydantic>=1.9.1",
        "joblib>=1.1.0",
        "PyYAML==6.0",
        "jinja2==3.1.2",
    ]
    strict_deps = [s.replace(">=", "==") for s in deps]

    setup(
        name="mlpype-base",
        install_requires=deps,
        extras_require={
            "dev": strict_deps + dev_deps + test_deps,
            "test": strict_deps + test_deps,
            "strict": strict_deps,
        },
        packages=find_namespace_packages(include=["mlpype.*"]),
        version=version,
    )
