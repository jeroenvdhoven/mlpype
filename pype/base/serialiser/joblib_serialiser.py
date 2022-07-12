from pathlib import Path
from typing import Any

from joblib import dump, load

from pype.base.serialiser.serialiser import Serialiser


class JoblibSerialiser(Serialiser):
    def serialise(self, object: Any, file: str | Path) -> None:
        """Serialise the given object to the given file.

        Args:
            object (Any): The object to serialise.
            file (str | Path): The file to serialise to.
        """
        dump(object, file)

    def deserialise(self, file: str | Path) -> Any:
        """Deserialise the object in the given file.

        Args:
            file (str | Path): The file to deserialise.

        Returns:
            Any: The python object stored in the file.
        """
        return load(file)
