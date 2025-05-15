"""A mlpype-compliant framework for using Spark Models."""
import shutil
import tarfile
import typing
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union

from loguru import logger
from pyspark.sql import DataFrame as SparkDataFrame

from mlpype.base.data import DataSet
from mlpype.base.experiment.argument_parsing import add_args_to_parser_for_class
from mlpype.base.model import Model
from mlpype.base.serialiser import JoblibSerialiser
from mlpype.spark.model.types import SerialisablePredictor, SerialisableSparkModel

T = TypeVar("T", bound=SerialisablePredictor)


class SparkModel(Model[SparkDataFrame], ABC, Generic[T]):
    """A mlpype-compliant framework for using Spark Models."""

    SPARK_PREDICTOR_PATH = "predictor"
    SPARK_MODEL_PATH = "spark_model"
    SPARK_MODEL_CLASS_PATH = "spark_model_class"

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        model: Optional[T] = None,
        predictor: Optional[T] = None,
        seed: int = 1,
        **model_args: Any,
    ) -> None:
        """A mlpype-compliant framework for using Spark Models.

        Args:
            inputs (List[str]): The name of the input DataFrame. Should be a list of 1 string.
            outputs (List[str]): The name of the output DataFrame. Should be a list of 1 string, the same as `inputs`
            model (Optional[T]): The Spark Model. Defaults to None. If set to None,
                this model can't be serialised or used for inference.
            predictor (Optional[T]): The Spark Predictor. If not set, we try to instantiate it
                using `model_args`
            seed (int, optional): Spark Seed. Currently ignored, unfortunately. Defaults to 1.
            **model_args (Any): any keywords arguments to be passed to _init_model.
        """
        assert len(inputs) == 1, "SparkML only requires a single DataFrame as input and output, the same one."
        assert len(outputs) == 1, "SparkML only requires a single DataFrame as input and output, the same one."
        assert inputs == outputs, "SparkML only requires a single DataFrame as input and output, the same one."
        super().__init__(inputs, outputs, seed)
        if predictor is None:
            predictor = self._init_model(model_args)

        self.model: Optional[SerialisableSparkModel] = model
        self.predictor = predictor

    @abstractmethod
    def _init_model(self, args: Dict[str, Any]) -> T:
        raise NotImplementedError

    @classmethod
    def _get_annotated_class(cls) -> Type[T]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    def set_seed(self) -> None:
        """Sets the RNG seed."""
        # TODO: set seed properly in pyspark
        logger.warning("Currently, setting a seed is not yet supported.")

    def fit(self, data: DataSet) -> "Model":
        """Fits the Model to the given DataSet.

        The DataSet should contain all inputs. For Spark models, this means 1 DataFrame
        that contains a feature vector and the label.

        Args:
            data (DataSet): The DataSet to fit this Model on.

        Returns:
            Model: self
        """
        self._fit(*data.get_all(self.inputs))
        return self

    def _fit(self, *data: SparkDataFrame) -> None:
        assert len(data) == 1, f"SparkML needs a single DataFrame as input, got {len(data)}"
        self.model = self.predictor.fit(data[0])

    def _transform(
        self,
        *data: SparkDataFrame,
    ) -> Union[Iterable[SparkDataFrame], SparkDataFrame]:
        assert len(data) == 1, f"SparkML needs a single DataFrame as input, got {len(data)}"
        assert self.model is not None, "Please fit this model before transforming data."

        result = self.model.transform(data[0])
        return result.drop(self.model.getOrDefault("featuresCol"))

    @classmethod
    def _get_spark_prefix(cls, folder: Path) -> str:
        if folder.is_absolute():
            spark_prefix = "file://"
        else:
            logger.warning(
                "Loading model from a relative path. This may not work as expected, and will fail in Databricks!"
            )
            spark_prefix = ""
        return spark_prefix

    def _save(self, folder: Path) -> None:
        assert self.model is not None, "Please fit this model before transforming data."
        spark_prefix = self._get_spark_prefix(folder)

        serialiser = JoblibSerialiser()

        self.predictor.write().overwrite().save(spark_prefix + str(folder / self.SPARK_PREDICTOR_PATH))
        self.model.write().overwrite().save(spark_prefix + str(folder / self.SPARK_MODEL_PATH))
        serialiser.serialise(type(self.model), str(folder / self.SPARK_MODEL_CLASS_PATH))

        self._create_zip(folder / self.SPARK_PREDICTOR_PATH)
        self._create_zip(folder / self.SPARK_MODEL_PATH)

    def _create_zip(self, folder: Path) -> None:
        """Creates a zip of the serialised model or predictor."""
        folder_name = folder.name
        logger.info(f"Turning {folder_name} into a tar file")
        # assert folder.is_dir(), f"Folder {folder} does not exist."

        with tarfile.open(folder.parent / f"{folder_name}.tar", "w") as tar_ref:
            tar_ref.add(folder, arcname=folder_name)

        # Cleanup: remove input folder.
        shutil.rmtree(str(folder), ignore_errors=True)

    @classmethod
    def _load(cls: Type["SparkModel"], folder: Path, inputs: List[str], outputs: List[str]) -> "SparkModel":
        serialiser = JoblibSerialiser()
        spark_prefix = cls._get_spark_prefix(folder)

        predictor_class: Type[SerialisablePredictor] = cls._get_annotated_class()
        model_class: Type[SerialisableSparkModel] = serialiser.deserialise(str(folder / cls.SPARK_MODEL_CLASS_PATH))

        cls._unzip(folder / f"{cls.SPARK_PREDICTOR_PATH}.tar")
        cls._unzip(folder / f"{cls.SPARK_MODEL_PATH}.tar")

        predictor: SerialisablePredictor = predictor_class.load(spark_prefix + str(folder / cls.SPARK_PREDICTOR_PATH))
        model: SerialisableSparkModel = model_class().load(spark_prefix + str(folder / cls.SPARK_MODEL_PATH))

        return cls(inputs=inputs, outputs=outputs, predictor=predictor, model=model, seed=1)

    @classmethod
    def _unzip(cls, tar_file: Path) -> None:
        """Unzips the model or predictor."""
        logger.info(f"Untarring: {tar_file}")
        with tarfile.open(tar_file, "r") as tar_ref:
            tar_ref.extractall(tar_file.parent)

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
            parser,
            BaseModel,
            "model",
            [],
            excluded_args=["seed", "inputs", "outputs", "model"],
        )
