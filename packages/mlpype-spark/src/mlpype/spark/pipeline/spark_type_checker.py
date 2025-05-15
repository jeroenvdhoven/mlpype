"""Provides tools for type checking Spark DataFrames and serialising/deserialising them."""
from datetime import date
from typing import Any, Dict, List, Tuple, Type, Union

from loguru import logger
from pydantic import create_model
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    AtomicType,
    BooleanType,
    DateType,
    IntegerType,
    NumericType,
    StringType,
    StructField,
    StructType,
    UserDefinedType,
)

from mlpype.base.pipeline.type_checker import DataModel, TypeChecker


class SparkData(DataModel):
    """A serialisable version of a Spark DataFrame."""

    def convert(self) -> SparkDataFrame:
        """Converts this object to a spark DataFrame.

        Returns:
            SparkDataFrame: The spark DataFrame contained by this object.
        """
        list_format: List[Dict[str, Any]] = []
        length = None
        for name, value in self.__dict__.items():
            if length is None:
                length = len(value)
                list_format = [{} for _ in range(length)]
            else:
                assert length == len(value), f"Warning: {name} did not have the same length as others"
            for i, v in enumerate(value):
                list_format[i][name] = v

        spark_session = SparkSession.builder.getOrCreate()

        return spark_session.createDataFrame(list_format)  # type: ignore

    @classmethod
    def to_model(cls, data: SparkDataFrame) -> "SparkData":
        """Converts a spark DataFrame to a SparkData model, which can be serialised.

        Args:
            data (SparkDataFrame): A spark DataFrame to serialise.

        Returns:
            SparkData: A serialisable version of the DataFrame.
        """
        rows = data.collect()
        columns: Dict[str, List[Any]] = {}
        for row in rows:
            for name, value in row.asDict().items():
                if name in columns:
                    columns[name].append(value)
                else:
                    columns[name] = [value]

        return cls(**columns)


class SparkTypeChecker(TypeChecker[SparkDataFrame]):
    """A TypeChecker for Spark DataFrames."""

    def __init__(self, name: str = "no name", error_on_missing_column: bool = False) -> None:
        """A TypeChecker for Spark DataFrames.

        Args:
            name (str, optional): Name of the dataset. Used to create named DataModels. Defaults to "no name".
            error_on_missing_column (bool, optional): Whether to throw an error if a column is missing.
                Since Pyspark's ML models tend to require all data being present in 1 DataFrame, it's easy
                to get errors if the target column is missing. This tries to avoid that, and will instead print
                a warning. Defaults to False.
        """
        super().__init__(name=name)
        # self.raw_types: Dict[str, Tuple[Type[Union[UserDefinedType, AtomicType]], type]] = {}
        self.error_on_missing_column = error_on_missing_column

    def fit(self, data: SparkDataFrame) -> "SparkTypeChecker":
        """Fit this SparkTypeChecker to the given data.

        Args:
            data (SparkDataFrame): The data to fit.

        Returns:
            SparkTypeChecker: self
        """
        self.raw_types = self._convert_dtypes(data.schema)
        return self

    def _convert_dtypes(self, schema: StructType) -> Dict[str, Tuple[Type[Union[UserDefinedType, AtomicType]], type]]:
        return {field.name: self._convert_dtype(field) for field in schema.fields}

    def _convert_dtype(self, field: StructField) -> Tuple[Type[Union[UserDefinedType, AtomicType]], type]:
        if isinstance(field.dataType, StringType):
            return (StringType, str)
        elif isinstance(field.dataType, IntegerType):
            return (IntegerType, int)
        elif isinstance(field.dataType, NumericType):
            return (NumericType, float)
        elif isinstance(field.dataType, BooleanType):
            return (BooleanType, bool)
        elif isinstance(field.dataType, DateType):
            return (DateType, date)
        elif isinstance(field.dataType, VectorUDT):
            return (VectorUDT, list[float])
        else:
            raise ValueError(f"{field.dataType} not supported")

    def transform(self, data: SparkDataFrame) -> SparkDataFrame:
        """Checks if the given data fits the specifications this TypeChecker was fitted for.

        Args:
            data (SparkDataFrame): The data to check.

        Returns:
            SparkDataFrame: data, if the data fits the specifications. Otherwise, an assertion error is thrown.
        """
        assert isinstance(data, SparkDataFrame), "Please provide a spark DataFrame!"
        dtype_dict = {field.name: field.dataType for field in data.schema.fields}

        available_cols = []
        for name, (spark_type, _) in self.raw_types.items():
            if name in dtype_dict:
                available_cols.append(name)
                actual_dtype = dtype_dict[name]
                assert isinstance(actual_dtype, spark_type), (
                    f"Dtypes did not match up for col {name}: Expected {spark_type.__name__}, "
                    f"got {actual_dtype.__class__.__name__}"
                )
            elif self.error_on_missing_column:
                raise AssertionError(f"`{name}` is missing from the dataset")
            else:
                logger.warning(f"`{name}` is missing from the dataset")

        return data.select(available_cols)

    def get_pydantic_type(self) -> Type[SparkData]:
        """Creates a Pydantic model for this data to handle serialisation/deserialisation.

        Returns:
            Type[SparkData]: A SparkData model that fits the data this wat fitted on.
        """
        data_type = {
            name: (Union[List[dtype], Dict[str or int, dtype]], ...)  # type: ignore
            for name, (_, dtype) in self.raw_types.items()
        }

        model = create_model(f"SparkData[{self.name}]", **data_type, __base__=SparkData)

        return model

    @classmethod
    def supports_object(cls, obj: Any) -> bool:
        """Returns True if the object is a Spark DataFrame.

        Args:
            obj (Any): The object to check.

        Returns:
            bool: True if the given object is a Spark DataFrame, False otherwise.
        """
        return isinstance(obj, SparkDataFrame)
