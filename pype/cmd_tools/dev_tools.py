"""Only intended for development command line tools."""

import re
import subprocess
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from setuptools import find_namespace_packages


def run_dev_install(parser: ArgumentParser) -> None:
    """Runs the development installation command.

    This command will install all sub-namespace packages in
    editable mode, with dependencies.

    The base package will be installed first. Other packages
    requiring prioritisation can be configured in the code
    of this function.

    Args:
        parser (ArgumentParser): Argument parser from the top-level entrypoint.
    """
    # please ensure this is properly sorted!
    prioritised_packages = ["pype.base", "pype.sklearn"]

    parser.add_argument("--editable", default=True, action=BooleanOptionalAction)
    parser.add_argument("--host", default=None, type=str)
    parsed = parser.parse_args()
    use_editable = parsed.editable

    base_path = Path(__file__).parent.parent.parent.absolute()
    packages = _find_packages(base_path)

    editable_message = " IN DEV MODE" if use_editable else ""
    print(f"PREPARING TO INSTALL PYPE{editable_message}\nInstalling the following packages:\n\t", packages)

    # Installing prioritised packages first.
    for package in prioritised_packages:
        assert package in packages, f"Prioritised package was not found: {package}"
        _install_package(package, use_editable, base_path, parsed.host)

    for package in packages:
        if package not in prioritised_packages:
            _install_package(package, use_editable, base_path, parsed.host)


def _install_package(package: str, use_editable: bool, base_path: Path, host: str | None = None) -> None:
    """Runs an installation command for the given sub-namespace package.

    Args:
        package (str): The sub-namespace package to be installed.
        use_editable (bool): If an editable version of the package should be installed.
        base_path (Path): Top-level directory of this reposity.
        host (str | None): The host to install from. Defaults to None, meaning from local directory.
            Useful for installing from a local pypi server. Ignores the editable command if set.
    """
    assert re.match(r"^pype\.[a-z_0-9]+$", package) is not None, f"`{package}` is not a valid package name. Exiting"

    if host is not None:
        command = ["pip", "install", f"{package}[dev]", "-i", host]
    else:
        editable_cmd = "-e" if use_editable else ""
        path = base_path.joinpath(*package.split("."))
        command = ["pip", "install", editable_cmd, f"{path}[dev]"]

    print(f"\nInstalling:\n\t{package} from {path}\n")
    subprocess.run(command)


def _find_packages(path: Path) -> list[str]:
    return find_namespace_packages(str(path), include=["pype.*"], exclude=["pype.cmd_tools", "pype.*.*"])


def run_dev_build(parser: ArgumentParser) -> None:
    """Builds all packages in this repo and stores them in a command directory.

    Supported args:
        - `--output-dir`: the output directory. Defaults to project top-level `dist` folder
        - `--packages`: specific packages to build. Defaults to all packages if not provided.

    Args:
        parser (ArgumentParser): Argument parser from the top-level entrypoint.
    """
    parser.add_argument("--output-dir", default=None, type=str, required=False)
    parser.add_argument("--packages", default=None, type=str, required=False, nargs="+")
    parsed = parser.parse_args()

    base_path = Path(__file__).parent.parent.parent.absolute()
    if parsed.packages is None:
        packages = _find_packages(base_path)
    else:
        packages = parsed.packages

    if parsed.output_dir is None:
        output_dir = base_path / "dist"
    else:
        output_dir = Path(parsed.output_dir)

    for package in packages:
        _build_package(package, base_path, output_dir)


def _build_package(package: str, base_path: Path, output_dir: Path) -> None:
    """Builds a package for the given sub-namespace package.

    Args:
        package (str): The sub-namespace package to be build.
        base_path (Path): Top-level directory of this reposity.
        output_dir (Path): Main output directory for build packages.
    """
    assert re.match(r"^pype\.[a-z_0-9]+$", package) is not None, f"`{package}` is not a valid package name. Exiting"

    path = base_path.joinpath(*package.split("."))
    command = ["python", "-m", "build", str(path), "--outdir", str(output_dir)]

    print(f"\nBuilding:\n\t{package} from {path}\n")
    subprocess.run(command)
