import importlib
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Generator, Set


@contextmanager
def switch_workspace(target_workspace: str | Path, extra_files: list[str | Path] | None = None) -> Generator:
    """Allows you to temporarily switch Python over to using a different working directory.

    This changes the following:
        - The current working directory will change to `target_workspace`
        - `target_workspace` is added to the system path
        - Any files provided in `extra_files` will be assumed to follow a relative path
            from `target_workspace` to python files that will be imported. Variables
            (incl. classes and functions) defined in those files will be imported into
            the __main__ module, and the files will be added to the path. We do not overwrite variables
            already existing in __main__.

    These changes will be undone once the context manager ends.

    Args:
        target_workspace (str | Path): The directory to switch to.
        extra_files (list[str | Path] | None, optional): Any additional files whose contents should
            be added to the current python program. Defaults to no extra files.
    """
    if extra_files is None:
        extra_files = []

    # create backups of the workspace and path.
    target_workspace = Path(target_workspace)
    old_workspace = os.getcwd()
    old_sys_path = sys.path.copy()

    # Add each file to  the path, and import variables specific to that file into globals.
    main_lib = importlib.import_module("__main__")
    main_to_remove = _create_temporary_workspace(target_workspace, extra_files, main_lib)
    # Yield so the normal code in the managed context can run.
    yield

    _reset_workspace(old_workspace, old_sys_path, main_lib, main_to_remove)


def _reset_workspace(
    old_workspace: str, old_sys_path: list[str], main_lib: ModuleType, main_to_remove: Set[str]
) -> None:
    os.chdir(old_workspace)
    sys.path = old_sys_path
    for var in main_to_remove:
        del main_lib.__dict__[var]


def _create_temporary_workspace(
    target_workspace: Path, extra_files: list[str | Path], main_lib: ModuleType
) -> Set[str]:
    """Perform the actual workspace switching.

    Args:
        target_workspace (Path): The directory to switch to.
        extra_files (list[str  |  Path]): Any additional files whose contents should
            be added to the current python program. Empty lists are accepted.
        main_lib (ModuleType): The __main__ module.

    Returns:
        Set[str]: The set of variable names that should be removed from __main__ again
            after the context manager quits.
    """
    logger = logging.getLogger(__name__)
    main_to_remove = set()

    # change directory and path
    os.chdir(target_workspace)
    sys.path.append(str(target_workspace))
    directory_name = target_workspace.name

    for file in extra_files:
        file = str(file)
        if not file.endswith(".py"):
            logger.warning(f"Not adding {file} to path since it is not a python file!")
            continue

        # add the python file to the path. This makes sure we can import variables from other files.
        relative_file_name = str(os.path.join(".", file))
        sys.path.append(relative_file_name)

        # import the file as a module.
        module_name = f'{directory_name}.{file.replace("/", ".").replace(".py", "")}'
        lib = importlib.import_module(module_name)

        # find all variables defined in this file, avoiding things like numpy/pandas imports.
        sub_vars = {
            name: value
            for name, value in lib.__dict__.items()
            if hasattr(value, "__module__") and value.__module__ == module_name
        }

        # add the variable to the main module. This makes sure we can import variables from the
        # same 'main' file that was used to run the experiment.
        # out of safety reasons we only add a variable if it doesn't exist in 'main' yet.
        for var_name, value in sub_vars.items():
            if var_name not in main_lib.__dict__:
                main_to_remove.add(var_name)
                main_lib.__dict__[var_name] = value
    return main_to_remove
