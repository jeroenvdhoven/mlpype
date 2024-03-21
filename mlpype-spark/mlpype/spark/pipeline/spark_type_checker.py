from typing import Any, Dict, List, Tuple, Type, Union

from pydantic import create_model
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from mlpype.base.pipeline.type_checker import DataModel, TypeChecker


class SparkData(DataModel):
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

        return spark_session.createDataFrame(list_format)

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
    def fit(self, data: SparkDataFrame) -> "SparkTypeChecker":
        """Fit this SparkTypeChecker to the given data.

        Args:
            data (SparkDataFrame): The data to fit.

        Returns:
            SparkTypeChecker: self
        """
        self.raw_types = self._convert_dtypes(dict(data.dtypes))
        return self

    def _convert_dtypes(self, dct: Dict[str, str]) -> Dict[str, Tuple[str, type]]:
        return {name: self._convert_dtype(string_type) for name, string_type in dct.items()}

    def _convert_dtype(self, string_type: str) -> Tuple[str, type]:
        conv_type: Union[Type, None] = None
        if string_type in ["str", "string"]:
            conv_type = str
        elif string_type in ["int", "bigint"]:
            conv_type = int
        elif string_type in ["float", "double"]:
            conv_type = float
        elif string_type == "bool":
            conv_type = bool
        else:
            raise ValueError(f"{string_type} not supported")

        return (string_type, conv_type)

    def transform(self, data: SparkDataFrame) -> SparkDataFrame:
        """Checks if the given data fits the specifications this TypeChecker was fitted for.

        Args:
            data (SparkDataFrame): The data to check.

        Returns:
            SparkDataFrame: data, if the data fits the specifications. Otherwise, an assertion error is thrown.
        """
        assert isinstance(data, SparkDataFrame), "Please provide a spark DataFrame!"
        colnames = list(self.raw_types.keys())
        for col in colnames:
            assert col in data.columns, f"`{col}` is missing from the dataset"
        dtype_dict = dict(data.dtypes)

        data = data.select(colnames)

        for name, (spark_type, _) in self.raw_types.items():
            actual_dtype = dtype_dict[name]
            assert (
                actual_dtype == spark_type
            ), f"Dtypes did not match up for col {name}: Expected {spark_type}, got {actual_dtype}"
        return data

    def get_pydantic_type(self) -> Type[SparkData]:
        """Creates a Pydantic model for this data to handle serialisation/deserialisation.

        Returns:
            Type[SparkData]: A SparkData model that fits the data this wat fitted on.
        """
        data_type = {
            name: (Union[List[dtype], Dict[str or int, dtype]], ...)  # type: ignore
            for name, (_, dtype) in self.raw_types.items()
        }

        model = create_model("SparkData", **data_type, __base__=SparkData)

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
