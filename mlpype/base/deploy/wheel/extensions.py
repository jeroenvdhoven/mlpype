import re
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List, Tuple


@dataclass
class WheelExtension:
    """An extension to mlpypes deployment process in wheel files.

    This class provides standardised configuration to add plugins to mlpype wheel files.

    Extensions require 3 parameters:
        - name: A unique name for the extension. This will be used to split imports and files.
        - functionality: A list of (path, functions) pairs. These are files to be copied over to any
            wheel package being created. Please make sure any imports in these files are NOT relative
            and do not otherwise rely on file structure/ordering. Ideally you should make sure each
            file is self-contained, only relying on external dependencies.
        - libraries that should be installed as extra dependencies to make the plugin work.

    All extensions will be placed in a similar fashion inside the wheel file:
        - <name of the model/package>
        |
        --  - <plugin name 1>
                - files for plugin 1
            - <plugin name 2>
            - <plugin name 3>
    Files from a plugin will be place in a flat manner inside the folder. Duplicate file names will cause issues.
    """

    name: str
    functionality: List[Tuple[Path, List[str]]]
    libraries: List[str]

    def __post_init__(self) -> None:
        """Perform checks on inputs."""
        self._check_file_name(self.name)
        for path, _ in self.functionality:
            self._check_file_name(path.name.replace(".py", ""))

        files = set()
        function_names = set()

        for path, functions in self.functionality:
            assert path not in files, f"{path} is a duplicated file name, this is not allowed."
            files.add(path)

            for func in functions:
                assert (
                    func not in function_names
                ), f"{func} is a duplicated function name from {path}, this is not allowed."
                function_names.add(func)

    @staticmethod
    def _check_file_name(name: str) -> None:
        """Check that the given name is an okay filename for imports.

        Args:
            name (str): The name to check

        Raises:
            AssertionError: if the name starts with a non-letter or contains a space or dot.
        """
        assert (
            re.search(r"^[a-zA-Z]", name) is not None
        ), f"Wheel extension file names need to start with a letter: {name}"

        for character in [" ", "."]:
            assert character not in name, f"{character} is not supported in wheel extension names: {name}"

    def extend(
        self,
        package_name: str,
        libraries: List[str],
        wheel_package_dir: Path,
    ) -> None:
        """Extends the given wheel file with this extension.

        Args:
            package_name (str): The name of the main package.
            libraries (List[str]): The current set of libraries to be installed.
            wheel_package_dir (Path): The Path to the main package.
        """
        libraries.extend(self.libraries)

        init_file_path = wheel_package_dir / "__init__.py"
        subfolder = wheel_package_dir / self.name
        subfolder.mkdir(exist_ok=False)

        with open(subfolder / "__init__.py", "w") as f:
            f.write("")

        with open(init_file_path, "a") as init_file:
            for file, imports in self.functionality:
                copyfile(file, subfolder / file.name)
                import_file_name = file.name.replace(".py", "")
                imports_str = ", ".join(imports)

                init_line = f"from {package_name}.{self.name}.{import_file_name} import {imports_str}\n"
                init_file.write(init_line)
