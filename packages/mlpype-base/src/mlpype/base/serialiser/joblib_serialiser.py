"""Provides a Serialiser to integrate Joblib with mlpype."""
from pathlib import Path
from typing import Any, Union

from joblib import dump, load

from mlpype.base.serialiser.serialiser import Serialiser


class JoblibSerialiser(Serialiser):
    """A Serialiser to integrate Joblib with mlpype."""

    def serialise(self, object: Any, file: Union[str, Path]) -> Union[str, Path]:
        """Serialise the given object to the given file.

        Args:
            object (Any): The object to serialise.
            file (Union[str, Path]): The file to serialise to.

        Returns:
            Union[str, Path]: The path to the serialised object (file).
        """
        dump(object, file)
        return file

    def deserialise(self, file: Union[str, Path]) -> Any:
        """Deserialise the object in the given file.

        Args:
            file (Union[str, Path]): The file to deserialise.

        Returns:
            Any: The python object stored in the file.
        """
        return load(file)
