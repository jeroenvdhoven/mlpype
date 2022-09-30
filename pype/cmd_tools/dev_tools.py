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
    parsed = parser.parse_args()
    use_editable = parsed.editable

    base_path = Path(__file__).parent.parent.parent.absolute()
    packages = find_namespace_packages(str(base_path), include=["pype.*"], exclude=["pype.cmd_tools", "pype.*.*"])

    editable_message = " IN DEV MODE" if use_editable else ""
    print(f"PREPARING TO INSTALL PYPE{editable_message}\nInstalling the following packages:\n\t", packages)

    # Installing prioritised packages first.
    for package in prioritised_packages:
        assert package in packages, f"Prioritised package was not found: {package}"
        _install_package(package, use_editable, base_path)

    for package in packages:
        if package not in prioritised_packages:
            _install_package(package, use_editable, base_path)


def _install_package(package: str, use_editable: bool, base_path: Path) -> None:
    """Runs an installation command for the given sub-namespace package.

    Args:
        package (str): The sub-namespace package to be installed.
        use_editable (bool): If an editable version of the package should be installed.
        base_path (Path): Top-level directory of this reposity.
    """
    assert re.match(r"^pype\.[a-z_0-9]+$", package) is not None, f"`{package}` is not a valid package name. Exiting"

    editable_cmd = "-e" if use_editable else ""
    path = base_path.joinpath(*package.split("."))
    command = ["pip", "install", editable_cmd, f"{path}[dev]"]

    print(f"\nInstalling:\n\t{package} from {path}\n")
    subprocess.run(command)
