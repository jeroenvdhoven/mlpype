from typing import Dict, Generic

from pype.base.data.data import Data
from pype.base.data.data_source import DataSource
from pype.base.data.dataset import DataSet


class DataSetSource(Generic[Data]):
    def __init__(self, data_sources: Dict[str, DataSource[Data]]):
        """A collection of DataSources that together form a DataSet when loaded.

        Args:
            data_sources (Dict[str, DataSource[Data]]): The data sourcers that form
                a DataSet together.
        """
        self.data_sources = data_sources

    def read(self) -> DataSet[Data]:
        """Read all DataSources and generate a DataSet.

        Names of DataSources are preserved when loading the data.

        Returns:
            DataSet[Data]: The DataSet constructed from the DataSources.
        """
        return DataSet({name: data.read() for name, data in self.data_sources.items()})
