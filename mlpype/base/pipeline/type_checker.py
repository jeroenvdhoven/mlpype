import logging
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, create_model

from mlpype.base.data.data_source import Data
from mlpype.base.data.dataset import DataSet
from mlpype.base.pipeline.operator import Operator
from mlpype.base.pipeline.pipe import Pipe


class DataModel(BaseModel, ABC):
    @abstractmethod
    def convert(self) -> Any:
        """Convert the Model to actual data (e.g. numpy or pandas).

        Returns:
            Any: The converted data.
        """

    @abstractclassmethod
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
    def get_pydantic_type(self) -> Type[DataModel]:
        """Returns a Pydantic type for data serialisation/deserialistion based on this pipe.

        Returns:
            Type[DataModel]: The data model for the data this was fitted on.
        """

    @classmethod
    @abstractmethod
    def supports_object(cls, obj: Any) -> bool:
        """Class method to determine if this TypeChecker supports the given object.

        Args:
            obj (Any): The object to check.

        Returns:
            bool: True if this TypeChecker supports the given object, False otherwise.
        """


class TypeCheckerPipe(Pipe):
    def __init__(self, name: str, input_names: List[str], type_checker_classes: List[Type[TypeChecker]]) -> None:
        """A pipe that fully checks an incoming DataSet for type consistency.

        Args:
            name (str): The name of the pipe.
            input_names (List[str]): The names of datasets to be checked by this type checker.
                We highly recommend to make sure this encompasses the datasets used by
                the model at inference time (or the predicted datasets for the output type checker).
            type_checker_classes (List[Type[TypeChecker]]): A list of TypeChecker classes.
                E.g. PandasTypeChecker for pandas data (requires mlpype.sklearn).
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

    def get_pydantic_types(self, names: Optional[List[str]] = None) -> Type[DataSetModel]:
        """Generate a Pydantic model for the given dataset names.

        Args:
            names (Optional[List[str]]): The type checker for which we want data models
                to be returned. Defaults to all input datasets.

        Returns:
            Type[DataSetModel]: A pydantic model to serialise/deserialise data for the given
                names.
        """
        if names is None:
            names = self.input_names
        assert isinstance(
            self.operator, DataSetTypeChecker
        ), f"Operators of TypeCheckerPipes should be DataSetTypeChecker, got: `{type(self.operator)}`"
        return self.operator.get_pydantic_types(names)


class DataSetTypeChecker(Operator[Data]):
    def __init__(self, input_names: List[str], type_checker_classes: List[Type[TypeChecker]]) -> None:
        """A TypeChecker that conforms to the Operator class.

        Mainly used by TypeCheckerPipe for easy type checking of incoming data.

        Args:
            input_names (List[str]): The names of datasets to be checked by this type checker.
                We highly recommend to make sure this encompasses the datasets used by
                the model at inference time (or the predicted datasets for the output type checker).
            type_checker_classes (List[Type[TypeChecker]]): A list of TypeChecker classes.
                E.g. PandasTypeChecker for pandas data (requires mlpype.sklearn).
        """
        super().__init__()
        self.input_names = input_names
        self.type_checker_classes = type_checker_classes
        self.type_checkers: Dict[str, TypeChecker] = {}

    def fit(self, *data: Data) -> "Operator":
        """Fits all type checkers to the given data.

        Returns:
            Operator: self.
        """
        self.type_checkers = {}
        for ds_name, dataset in zip(self.input_names, data):
            type_checker_class = self._get_type_checker(dataset)

            if type_checker_class is None:
                logger = logging.getLogger(__name__)
                logger.warning(f"{ds_name} has no supported type checker!")
            else:
                checker = type_checker_class()
                checker.fit(dataset)
                self.type_checkers[ds_name] = checker
        return self

    def transform(self, *data: Data) -> Tuple[Data, ...]:
        """Checks if the given data fits the fitted Type Checkers.

        Returns:
            Tuple[Data, ...]: returned result from each type_checker, if
                all data fits the Type Checkers.
        """
        data_result = []
        for ds_name, dataset in zip(self.input_names, data):
            assert ds_name in self.type_checkers, f"{ds_name} does not have a type checker"

            checker = self.type_checkers[ds_name]
            data_result.append(checker.transform(dataset))

        return tuple(data_result)

    def _get_type_checker(self, data: Data) -> Union[Type[TypeChecker], None]:
        for type_checker in self.type_checker_classes:
            if type_checker.supports_object(data):
                return type_checker
        return None

    def get_pydantic_types(self, names: Optional[List[str]] = None) -> Type[DataSetModel]:
        """Generates a DataModel for the given input datasets after fitting.

        Args:
            names (Optional[List[str]]): The type checker for which we want data models
                to be returned. Defaults to all input datasets.

        Returns:
            Type[DataSetModel]: A pydantic model to serialise/deserialise data for the given
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
#     registry: Dict[str, Tuple[type, Type[TypeChecker]]] = {}
#
#     @classmethod
#     def get_type_checker(cls, type_: type) -> Type[TypeChecker] or None:
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
