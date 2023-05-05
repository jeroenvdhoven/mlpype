import importlib
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar, Union

import yaml

from mlpype.base.data.data_source import DataSource
from mlpype.base.data.dataset import DataSet

Data = TypeVar("Data")


class DataCatalog(Dict[str, DataSource[Data]]):
    """A collection of DataSources that together form a DataSet when loaded."""

    def read(self) -> DataSet[Data]:
        """Read all DataSources and generate a DataSet.

        Names of DataSources are preserved when loading the data.

        Returns:
            DataSet[Data]: The DataSet constructed from the DataSources.
        """
        return DataSet.from_dict({name: data.read() for name, data in self.items()})

    def __str__(self, indents: int = 0) -> str:
        """Create string representation of this DataCatalog.

        Args:
            indents (int, optional): The number of preceding tabs. Defaults to 0.

        Returns:
            str: A string representation of this DataCatalog.
        """
        tab = "\t" * indents
        return "\n".join([f"{tab}{name}: {source}" for name, source in self.items()])

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DataCatalog":
        """Read a catalog in from a configuraiton YAML file.

        We expect the following format:
        ```
        <dataset name>:
            callable: <python path to class, method on a class, or function.
                Should produce a DataCatalog.
            args:
                <name of argument>: <value>
        ```

        Values can be a plain value (like string, bool, etc) or a complex object.
        These will follow the same format as the callable-args structure of the
        DataSource's, but don't need to be DataSource's. e.g.:

        ```
        dataframe:
            callable: mlpype.sklearn.data.DataFrameSource
            args:
                df:
                    callable: pandas.DataFrame
                    args:
                        data:
                            x:
                                - 1.0
                                - 2.0
                            y:
                                - "a"
                                - "b"
        ```

        To use a method on a class, use `path.to.class:function`. This will only work for
        static or class methods.

        For security reasons, please make sure you trust any YAML file you read using this
        method! It will lead to code execution based on the imports, which can be abused.

        Args:
            path (Union[str, Path]): The path to the YAML file.

        Returns:
            DataCatalog: A DataCatalog with DataSources based on the YAML file.
        """
        data_sources = {}

        path = Path(path)
        with open(path, "r") as f:
            configuration = yaml.safe_load(f)
        assert isinstance(configuration, dict), "Cannot process YAML as DataCatalog: should be a dictionary."

        for dataset_name, parameters in configuration.items():
            dataset = cls._parse_object(parameters)
            assert isinstance(
                dataset, DataSource
            ), "Please make sure objects in your DataCatalog only translate to DataSources."
            data_sources[dataset_name] = dataset
        return DataCatalog(**data_sources)

    @classmethod
    def _parse_object(cls, dct: Dict[str, Any]) -> Any:
        assert cls._is_valid_parseable_object(
            dct
        ), "DataCatalog: any dictionary parsed should have a `class` and `args` entry."

        callable_ = cls._load_class(dct["callable"])
        args = dct["args"]
        assert isinstance(args, dict), "Arguments to a parseable object should be a dict."

        # Parse arguments recursively
        parsed_args = {}
        for arg_name, arg_value in args.items():
            # print(f"{arg_name}: {arg_value}")
            if cls._is_valid_parseable_object(arg_value):
                parsed_args[arg_name] = cls._parse_object(arg_value)
            else:
                parsed_args[arg_name] = arg_value

        return callable_(**parsed_args)

    @staticmethod
    def _load_class(full_path: str) -> Callable:
        method_split = full_path.split(":")
        assert len(method_split) <= 2, "We do not accept paths with more than 1 `:`"
        callable_path = method_split[0]

        split_path = callable_path.split(".")
        module_path = ".".join(split_path[:-1])

        module = importlib.import_module(module_path)
        callable = getattr(module, split_path[-1])

        if len(method_split) > 1:
            return getattr(callable, method_split[1])
        else:
            return callable

    @staticmethod
    def _is_valid_parseable_object(dct: Dict[str, Any]) -> bool:
        return isinstance(dct, dict) and len(dct) == 2 and "callable" in dct and "args" in dct
