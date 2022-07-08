from pathlib import Path
from typing import Any

from joblib import dump, load

from pype.base.serialiser import Serialiser


class JoblibSerialiser(Serialiser):
    def serialise(self, object: Any, file: str | Path) -> None:
        dump(object, file)

    def deserialise(self, file: str | Path) -> Any:
        return load(file)
