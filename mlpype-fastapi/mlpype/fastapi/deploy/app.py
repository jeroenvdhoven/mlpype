from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from fastapi.applications import FastAPI
from fastapi.background import BackgroundTasks

from mlpype.base.data import DataSink
from mlpype.base.data.dataset import DataSet
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.pipeline.type_checker import DataModel, DataSetModel


@dataclass
class mlpypeApp:
    name: str
    folder: Union[str, Path]
    tracking_servers: Optional[Dict[str, DataSink]] = None

    def __post_init__(self):  # type: ignore
        """Makes sure the folder is an actual Path."""
        if isinstance(self.folder, str):
            self.folder = Path(self.folder)

    def create_app(self) -> FastAPI:
        """Creates a FastAPI app for a saved mlpype experiment.

        Returns:
            FastAPI: The FastAPI server that can be served.
        """
        app = FastAPI()
        inferencer = self._load_model()
        logger = getLogger(self.name)

        InputType: Type[DataSetModel] = inferencer.input_type_checker.get_pydantic_types()
        OutputType: Type[DataSetModel] = inferencer.output_type_checker.get_pydantic_types()
        self._verify_tracking_servers(InputType, OutputType, logger)

        @app.get("/")
        async def home_page() -> str:
            logger.info("Homepage request")
            return f"Welcome to the mlpype FastAPI app for {self.name}"

        @app.post("/predict")
        async def predict(inputs: InputType, background: BackgroundTasks) -> OutputType:  # type: ignore
            logger.info("Prediction request")
            converted: DataSet = inputs.convert()  # type: ignore
            prediction = inferencer.predict(converted)
            result = OutputType.to_model(prediction)

            self._handle_tracking(converted, prediction, logger, background)
            logger.info("Finishing prediction request")
            return result

        return app

    def _handle_tracking(self, inputs: DataSet, outputs: DataSet, logger: Logger, background: BackgroundTasks) -> None:
        if self.tracking_servers is not None:
            logger.info("Logging records request")
            for name, sink in self.tracking_servers.items():
                try:
                    # Try to find the name of the dataset in the tracking servers.
                    ds: Union[DataModel, None] = None
                    if name in outputs:
                        ds = outputs[name]
                    elif name in inputs:
                        ds = inputs[name]

                    if ds is None:
                        logger.warning(f"Not storing data to sink `{name}` since no data by that name was found.")
                    else:
                        logger.info(f"Storing data to sink `{name}`.")
                        background.add_task(write_in_background, sink=sink, data=ds)

                except Exception as e:
                    logger.error(f"Encountered error while sending data to {name}: {str(e)}")

    def _verify_tracking_servers(
        self,
        inputs: Type[DataSetModel],
        outputs: Type[DataSetModel],
        logger: Logger,
    ) -> None:
        # Since it could be possible someone wants to log an intermediate dataset,
        # we can't put a hard restriction on logging only input or output data.
        # We will only log the latest version of a dataset though.
        if self.tracking_servers is not None:
            for name in self.tracking_servers:
                if not (name in inputs.__fields__ or name in outputs.__fields__):
                    logger.warning(f"No dataset named `{name}` found in the fields of input or output DataSetModels")

    def _load_model(self) -> Inferencer:
        assert isinstance(self.folder, Path)
        return Inferencer.from_folder(self.folder)


def write_in_background(sink: DataSink, data: Any) -> None:
    """Background function to write data to a given DataSink.

    Args:
        sink (DataSink): The Sink to write to.
        data (Any): The Data to write.
    """
    sink.write(data)
