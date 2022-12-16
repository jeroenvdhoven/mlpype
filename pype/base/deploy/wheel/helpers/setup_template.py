import os
from pathlib import Path
from typing import List, Union

from setuptools import setup


def package_files(directory: Union[Path, str]) -> List[str]:
    """Find all files from the output directory of an Experiment run to include.

    Args:
        directory (Union[Path, str]): The main output directory to search.

    Returns:
        List[str]: The files to include.
    """
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files(Path().parent.absolute())

setup(
    name="{package_name}",
    install_requires=["{install_requires}"],
    version="{version}",
    include_package_data=True,
    package_data={"": extra_files},
)
