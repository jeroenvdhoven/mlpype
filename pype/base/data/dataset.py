from typing import Dict, Generic, Iterable, List

from pype.base.data.data import Data


class DataSet(Generic[Data]):
    def __init__(self, data: Dict[str, Data]):
        """A DataSet consisting of multiple name-Data pairs.

        The Data can be multiple types of objects:
        - pandas DataFrames (or of other languages)
        - spark DataFrames
        - strings, integers, etc.

        Args:
            data (Dict[str, Data]): A dictionary of name-Data pairs that
                form this DataSet.
        """
        self.data = data

    def __getitem__(self, key: str) -> Data:
        """Return the data associated with the given key.

        Args:
            key (str): The key of the data to return.

        Returns:
            Data: The data associated with the given key.
        """
        return self.data[key]

    def get_all(self, keys: List[str]) -> List[Data]:
        """Returns all data associated with the given keys, in order.

        Args:
            keys (List[str]): The keys of all data to return.

        Returns:
            List[Data]: All data associated with the given keys.
        """
        return [self[key] for key in keys]

    def copy(self) -> "DataSet":
        """Returns a shallow copy of this DataSet.

        Only the dictionary is copied: all data is only copied by reference.

        Returns:
            DataSet: A shallow copy of this DataSet.
        """
        return DataSet(self.data.copy())

    def __setitem__(self, key: str, data: Data) -> None:
        """Store the given data on the given key.

        This will overwrite any existing data on the given key.

        Args:
            key (str): The key/name of the data.
            data (Data): The data to store.
        """
        self.data[key] = data

    def set_all(self, keys: List[str], data: Iterable[Data]) -> None:
        """Set all data to the given keys, in order.

        This assumes the keys and data set are of the same length.

        Args:
            keys (List[str]): The keys of the given data.
            data (Iterable[Data]): The data to store.
        """
        for key, d in zip(keys, data):
            self[key] = d
