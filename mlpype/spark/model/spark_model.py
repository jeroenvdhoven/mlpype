import json
import typing
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union

from pyspark.ml import Model as BaseSparkModel
from pyspark.ml import Predictor
from pyspark.sql import DataFrame as SparkDataFrame

from mlpype.base.data import DataSet
from mlpype.base.experiment.argument_parsing import add_args_to_parser_for_class
from mlpype.base.model import Model
from mlpype.base.serialiser import JoblibSerialiser

T = TypeVar("T", bound=Predictor)


class SparkModel(Model[SparkDataFrame], ABC, Generic[T]):
    mlpype_MODEL_CONFIG = "config.json"
    SPARK_PREDICTOR_PATH = "predictor"
    SPARK_MODEL_PATH = "spark_model"
    SPARK_MODEL_CLASS_PATH = "spark_model_class"

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        output_col: Optional[str] = None,
        model: Optional[BaseSparkModel] = None,
        predictor: Optional[T] = None,
        seed: int = 1,
        **model_args: Any,
    ) -> None:
        """A mlpype-compliant framework for using Spark Models.

        Args:
            inputs (List[str]): The name of the input DataFrame. Should be a list of 1 string.
            outputs (List[str]): The name of the output DataFrame. Should be a list of 1 string, the same as `inputs`
            output_col (Optional[str]): The name of the column where the model will put the output.
                Defaults to None, which means we won't select any columns and instead return the full output
                of the model.
            predictor (Optional[Predictor]): The Spark Predictor. If not set, we try to instantiate it
                using `model_args`
            model (Optional[BaseSparkModel]): The Spark Model. Defaults to None. If set to None,
                this model can't be serialised or used for inference.
            seed (int, optional): Spark Seed. Currently ignored, unfortunately. Defaults to 1.
        """
        assert len(inputs) == 1, "SparkML only requires a single DataFrame as input and output, the same one."
        assert len(outputs) == 1, "SparkML only requires a single DataFrame as input and output, the same one."
        assert inputs == outputs, "SparkML only requires a single DataFrame as input and output, the same one."
        super().__init__(inputs, outputs, seed)
        if predictor is None:
            predictor = self._init_model(model_args)

        self.model = model
        self.predictor = predictor
        self.output_col = output_col

    @abstractmethod
    def _init_model(self, args: Dict[str, Any]) -> T:
        raise NotImplementedError

    @classmethod
    def _get_annotated_class(cls) -> Type[T]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    def set_seed(self) -> None:
        """Sets the RNG seed."""
        # TODO: set seed properly in pyspark
        print("Currently, setting a seed is not yet supported.")

    def fit(self, data: DataSet) -> "Model":
        """Fits the Model to the given DataSet.

        The DataSet should contain all inputs. For Spark models, this means 1 DataFrame
        that contains a feature vector and the label.

        Args:
            data (DataSet): The DataSet to fit this Model on.
        """
        self._fit(*data.get_all(self.inputs))
        return self

    def _fit(self, *data: SparkDataFrame) -> None:
        assert len(data) == 1, f"SparkML needs a single DataFrame as input, got {len(data)}"
        self.model = self.predictor.fit(data[0])

    def _transform(
        self, *data: SparkDataFrame, reduce_columns_if_possible: bool = True
    ) -> Union[Iterable[SparkDataFrame], SparkDataFrame]:
        assert len(data) == 1, f"SparkML needs a single DataFrame as input, got {len(data)}"
        assert self.model is not None, "Please fit this model before transforming data."

        result = self.model.transform(data[0])
        if self.output_col is not None and reduce_columns_if_possible:
            result = result.select(self.output_col)
        return result

    def transform_for_evaluation(self, data: DataSet) -> DataSet:
        """Applies the SparkModel to the given DataSet in an evaluation way.

        The DataSet should contain all inputs. This will not reduce the output dataset
        to the predicted columns such that we can use it in evaluation steps.

        Args:
            data (DataSet): The DataSet to transform using this Model.

        Returns:
            DataSet: The outputs as a Dataset.
        """
        result = self._transform(*data.get_all(self.inputs), reduce_columns_if_possible=False)
        if len(self.outputs) == 1:
            result = [result]  # type: ignore

        # type check handled by above check.
        return DataSet.from_dict({name: data for name, data in zip(self.outputs, result)})  # type: ignore

    def _save(self, folder: Path) -> None:
        assert self.model is not None, "Please fit this model before transforming data."

        serialiser = JoblibSerialiser()
        config = {"output_col": self.output_col}
        with open(folder / self.mlpype_MODEL_CONFIG, "w") as f:
            json.dump(config, f)

        self.predictor.save(str(folder / self.SPARK_PREDICTOR_PATH))
        self.model.save(str(folder / self.SPARK_MODEL_PATH))
        serialiser.serialise(type(self.model), str(folder / self.SPARK_MODEL_CLASS_PATH))

    @classmethod
    def _load(cls: Type["SparkModel"], folder: Path, inputs: List[str], outputs: List[str]) -> "SparkModel":
        serialiser = JoblibSerialiser()

        with open(folder / cls.mlpype_MODEL_CONFIG, "r") as f:
            config = json.load(f)
        output_col = config["output_col"]

        predictor_class: Type[Predictor] = cls._get_annotated_class()
        model_class: Type[BaseSparkModel] = serialiser.deserialise(str(folder / cls.SPARK_MODEL_CLASS_PATH))

        predictor: Predictor = predictor_class.load(str(folder / cls.SPARK_PREDICTOR_PATH))
        model: BaseSparkModel = model_class.load(str(folder / cls.SPARK_MODEL_PATH))

        return cls(inputs=inputs, outputs=outputs, predictor=predictor, model=model, output_col=output_col, seed=1)

    @classmethod
    def get_parameters(cls: Type["SparkModel"], parser: ArgumentParser) -> None:
        """Get and add parameters to initialise this class.

        SparkModel's will work by requiring 2 ways to instantiate a Model:
            - through `model`, which is a spark model.
            - through parameters, which will instantiate the model internally.

        Args:
            parser (ArgumentParser): The ArgumentParser to add arguments to.
        """
        super().get_parameters(parser)
        BaseModel = cls._get_annotated_class()

        add_args_to_parser_for_class(
            parser, BaseModel, "model", [], excluded_args=["seed", "inputs", "outputs", "model"]
        )
