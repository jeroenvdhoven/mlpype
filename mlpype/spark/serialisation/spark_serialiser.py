import json
from pathlib import Path
from typing import Any, List, Type, Union

import joblib
from pyspark.ml import Transformer

from mlpype.base.pipeline import Pipe, Pipeline
from mlpype.base.serialiser import Serialiser
from mlpype.spark.pipeline.spark_pipe import SparkPipe


class SparkSerialiser(Serialiser):
    """A Serialiser to integrate Spark with mlpype.

    It is highly recommended to initialise SparkSession manually before calling this,
    with all proper configuration done beforehand. A SparkSession is needed to
    serialise / deserialise objects stored using this class.
    """

    SUB_PIPE_PREFIX = "__sub_pipe_"
    BASE_PIPE_FILE = "pipe.pkl"
    SPARK_TRANSFORMER_FILE = "spark_transformer"
    SPARK_TRANSFORMER_CLASS_FILE = "spark_transformer_class"
    STEPS_FILE = "steps.json"

    def serialise(self, object: Any, file: Union[str, Path]) -> None:
        """Serialse a given object.

        If the object is:
            - A mlpype Pipeline: we serialise each step separately.
            - Otherwise, we use joblib.

        Args:
            object (Any): The object to serialise.
            file (Union[str, Path]): The path to serialise to.
        """
        if isinstance(object, Pipeline):
            self._serialise_pipeline(object, file)
        else:
            return self._serialise_joblib(object, file)

    def deserialise(self, file: Union[str, Path]) -> Any:
        """Deserialise the object in the given file.

        This function can handle mlpype Pipelines with PySpark elements.

        Args:
            file (Union[str, Path]): The file containing a python object to deserialise.

        Returns:
            Any: The deserialised object.
        """
        file = Path(file)
        if file.is_dir() and (file / self.STEPS_FILE).is_file():
            return self._deserialise_pipeline(file)
        else:
            return self._deserialise_joblib(file)

    def _serialise_pipeline(self, pipeline: Pipeline, file: Union[str, Path]) -> None:
        file = Path(file)

        file.mkdir(exist_ok=True)
        steps = []
        i = 0
        for step in pipeline:
            if isinstance(step, SparkPipe):
                name = step.name
                pipe_path = file / name
                pipe_path.mkdir(exist_ok=True)
                self._serialise_joblib(step, pipe_path / self.BASE_PIPE_FILE)

                # store the class for re-importing
                transformer: Transformer = step.fitted
                self._serialise_joblib(type(transformer), pipe_path / self.SPARK_TRANSFORMER_CLASS_FILE)
                transformer.save(str(pipe_path / self.SPARK_TRANSFORMER_FILE))
            elif isinstance(step, Pipe):
                name = step.name
                self._serialise_joblib(step, file / name)
            else:
                name = f"{self.SUB_PIPE_PREFIX}{i}"
                i += 1
                self._serialise_pipeline(step, file / name)

            steps.append(name)

        with open(file / self.STEPS_FILE, "w") as f:
            json.dump({"steps": steps}, f)

    def _serialise_joblib(self, object: Any, file: Union[str, Path]) -> None:
        """Serialise the given object to the given file.

        Args:
            object (Any): The object to serialise.
            file (Union[str, Path]): The file to serialise to.
        """
        joblib.dump(object, file)

    def _deserialise_pipeline(self, file: Union[str, Path]) -> Pipeline:
        file = Path(file)
        assert file.is_dir()

        with open(file / self.STEPS_FILE, "r") as f:
            steps: List[str] = json.load(f)["steps"]

        pipes = []
        for step in steps:
            step_path = file / step
            if step_path.is_file():
                pipes.append(self._deserialise_joblib(step_path))
            elif step_path.is_dir():
                if step.startswith(self.SUB_PIPE_PREFIX):
                    pipes.append(self._deserialise_pipeline(step_path))
                else:
                    pipe: SparkPipe = self._deserialise_joblib(step_path / self.BASE_PIPE_FILE)
                    fitted_class: Type[Transformer] = self._deserialise_joblib(
                        step_path / self.SPARK_TRANSFORMER_CLASS_FILE
                    )
                    pipe.fitted = fitted_class.load(str(step_path / self.SPARK_TRANSFORMER_FILE))
                    pipes.append(pipe)
            else:
                raise ValueError(f"{step_path} was not an existing path")

        return Pipeline(pipes)

    def _deserialise_joblib(self, file: Union[str, Path]) -> Any:
        """Deserialise the object in the given file.

        Args:
            file (Union[str, Path]): The file to deserialise.

        Returns:
            Any: The python object stored in the file.
        """
        return joblib.load(file)
