from typing import Dict, TypeVar

from pype.base.data.data_source import DataSource
from pype.base.data.dataset import DataSet

Data = TypeVar("Data")


class DataSetSource(Dict[str, DataSource[Data]]):
    """A collection of DataSources that together form a DataSet when loaded."""

    def read(self) -> DataSet[Data]:
        """Read all DataSources and generate a DataSet.

        Names of DataSources are preserved when loading the data.

        Returns:
            DataSet[Data]: The DataSet constructed from the DataSources.
        """
        return DataSet.from_dict({name: data.read() for name, data in self.items()})

    def __str__(self, indents: int = 0) -> str:
        """Create string representation of this DataSetSource.

        Args:
            indents (int, optional): The number of preceding tabs. Defaults to 0.

        Returns:
            str: A string representation of this DataSetSource.
        """
        tab = "\t" * indents
        return "\n".join([f"{tab}{name}: {source}" for name, source in self.items()])
