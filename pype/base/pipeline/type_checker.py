import warnings
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, create_model

from pype.base.data.data_source import Data
from pype.base.data.dataset import DataSet
from pype.base.pipeline.operator import Operator
from pype.base.pipeline.pipe import Pipe


class DataModel(BaseModel, ABC):
    @abstractmethod
    def convert(self) -> Any:
        """Convert the Model to actual data (e.g. numpy or pandas).

        Returns:
            Any: The converted data.
        """

    @classmethod
    @abstractmethod
    def to_model(cls, data: Any) -> "DataModel":
        """Convert the Model to actual data (e.g. numpy or pandas).

        Returns:
            Any: The converted data.
        """


class DataSetModel(BaseModel):
    def convert(self) -> DataSet:
        """Converts this DataSetModel to a DataSet, converting all data at the same time.

        Returns:
            DataSet: A DataSet based on this DataSetModel.
        """
        return DataSet({name: value.convert() for name, value in self.__dict__.items()})

    @classmethod
    def to_model(cls, ds: DataSet) -> "DataSetModel":
        """Converts a DataSet to a DataSetModel, which can be serialised.

        We convert data in the DataSet using known DataModels.

        Args:
            ds (DataSet): A DataSet to serialise.

        Returns:
            DataSetModel: A serialisable version of the DataSet.
        """
        dct = {}
        fields = cls.__fields__
        for name, dataset in ds.items():
            dct[name] = fields[name].type_.to_model(dataset)
        return cls(**dct)


class TypeChecker(Operator[Data], ABC):
    def __init__(self) -> None:
        """Defines a type checker class that can be used to check the format of new data."""
        super().__init__()

    @abstractmethod
    def fit(self, data: Data) -> "Operator":
        """Fits this Type Checker to the given Data.

        Args:
            data (Data): The data to fit.

        Returns:
            Operator: self.
        """

    @abstractmethod
    def transform(self, data: Data) -> Data:
        """Checks if the new data conforms to the specifications of the fitted data.

        Args:
            data (Data): The data to check.

        Returns:
            Data: data, if the data conforms. Otherwise an AssertionError will be thrown.
        """

    @abstractmethod
    def get_pydantic_type(self) -> type[DataModel]:
        """Returns a Pydantic type for data serialisation/deserialistion based on this pipe.

        Returns:
            type[DataModel]: The data model for the data this was fitted on.
        """


class TypeCheckerPipe(Pipe):
    def __init__(
        self, name: str, input_names: list[str], type_checker_classes: list[tuple[type, type[TypeChecker]]]
    ) -> None:
        """A pipe that fully checks an incoming DataSet for type consistency.

        Args:
            name (str): The name of the pipe.
            input_names (list[str]): The names of datasets to be checked by this type checker.
                We highly recommend to make sure this encompasses the datasets used by
                the model at inference time (or the predicted datasets for the output type checker).
            type_checker_classes (list[tuple[type, type[TypeChecker]]]): A list of pairs
                of data types and the type checker that should be used for that class.
                E.g. (pandas.DataFrame, PandasTypeChecker) for pandas data.
        """
        super().__init__(
            name,
            DataSetTypeChecker,
            input_names,
            [],
            {
                "input_names": input_names,
                "type_checker_classes": type_checker_classes,
            },
        )
        self.input_names = input_names

    def get_pydantic_types(self, names: list[str] | None = None) -> type[DataSetModel]:
        """Generate a Pydantic model for the given dataset names.

        Args:
            names (list[str] | None, optional): The type checker for which we want data models
                to be returned. Defaults to all input datasets.

        Returns:
            type[DataSetModel]: A pydantic model to serialise/deserialise data for the given
                names.
        """
        if names is None:
            names = self.input_names

        return self.operator.get_pydantic_types(names)


class DataSetTypeChecker(Operator[Data]):
    def __init__(self, input_names: list[str], type_checker_classes: list[tuple[type, type[TypeChecker]]]) -> None:
        """A TypeChecker that conforms to the Operator class.

        Mainly used by TypeCheckerPipe for easy type checking of incoming data.

        Args:
            input_names (list[str]): The names of datasets to be checked by this type checker.
                We highly recommend to make sure this encompasses the datasets used by
                the model at inference time (or the predicted datasets for the output type checker).
            type_checker_classes (list[tuple[type, type[TypeChecker]]]): A list of pairs
                of data types and the type checker that should be used for that class.
                E.g. (pandas.DataFrame, PandasTypeChecker) for pandas data.
        """
        super().__init__()
        self.input_names = input_names

        self.type_checker_classes = type_checker_classes

        self.type_checkers: dict[str, TypeChecker] = {}

    def fit(self, *data: Data) -> "Operator":
        """Fits all type checkers to the given data.

        Returns:
            Operator: self.
        """
        self.type_checkers = {}
        for ds_name, dataset in zip(self.input_names, data):
            type_checker_class = self._get_type_checker(dataset)

            if type_checker_class is None:
                warnings.warn(f"{ds_name} has no supported type checker!")
            else:
                checker = type_checker_class()
                checker.fit(dataset)
                self.type_checkers[ds_name] = checker
        return self

    def transform(self, *data: Data) -> tuple[Data, ...]:
        """Checks if the given data fits the fitted Type Checkers.

        Returns:
            tuple[Data, ...]: data, if all data fits the Type Checkers.
        """
        for ds_name, dataset in zip(self.input_names, data):
            assert ds_name in self.type_checkers, f"{ds_name} does not have a type checker"

            checker = self.type_checkers[ds_name]
            checker.transform(dataset)

        return data

    def _get_type_checker(self, data: Data) -> type[TypeChecker] | None:
        for ref_type, type_checker in self.type_checker_classes:
            if isinstance(data, ref_type):
                return type_checker
        return None

    def get_pydantic_types(self, names: list[str] | None = None) -> type[DataSetModel]:
        """Generates a DataModel for the given input datasets after fitting.

        Args:
            names (list[str] | None, optional): The type checker for which we want data models
                to be returned. Defaults to all input datasets.

        Returns:
            type[DataSetModel]: A pydantic model to serialise/deserialise data for the given
                input_names.
        """
        if names is None:
            names = self.input_names

        pydantic_types = {
            name: (checker.get_pydantic_type(), ...) for name, checker in self.type_checkers.items() if name in names
        }
        return create_model("DataSetModel", **pydantic_types, __base__=DataSetModel)


# class TypeCheckerOptions:
#     """Please register any Type Checkers you want to use!
#
#     For the ones defined in your script, just us the @register wrapper.
#     For the ones defined in standalone code (e.g. libs), put this import high
#     in the priority list, e.g. top level __init__
#
#     Returns:
#         _type_: _description_
#     """
#     registry: dict[str, tuple[type, type[TypeChecker]]] = {}
#
#     @classmethod
#     def get_type_checker(cls, type_: type) -> type[TypeChecker] | None:
#         for ref_type, type_checker in cls.registry.values():
#             if ref_type == type_:
#                 return type_checker
#         return None
#
#     @classmethod
#     def register(cls, name: str, type_: type):
#         def inner_wrapper(wrapped_class):
#             if name in cls.registry:
#                 print(f'Class {name} already exists. Will replace it')
#             cls.registry[name] = (type_, wrapped_class)
#             return wrapped_class
#         return inner_wrapper
