from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from pype.base.data.dataset import DataSet
from pype.base.deploy.inference import Inferencer
from pype.base.pipeline.type_checker import DataSetModel


@dataclass
class PypeApp:
    name: str
    folder: str | Path

    def __post_init__(self):  # type: ignore
        """Makes sure the folder is an actual Path."""
        if isinstance(self.folder, str):
            self.folder = Path(self.folder)

    def create_app(self) -> FastAPI:
        """Creates a FastAPI app for a saved Pype experiment.

        Returns:
            FastAPI: The FastAPI server that can be served.
        """
        app = FastAPI()
        inferencer = self._load_model()

        InputType: type[DataSetModel] = inferencer.input_type_checker.get_pydantic_types()
        OutputType: type[DataSetModel] = inferencer.output_type_checker.get_pydantic_types()

        @app.get("/")
        async def home_page() -> str:
            return f"Welcome to the Pype FastAPI app for {self.name}"

        @app.post("/predict")
        async def predict(inputs: InputType) -> OutputType:  # type: ignore
            converted: DataSet = inputs.convert()  # type: ignore
            prediction = inferencer.predict(converted)
            return OutputType.to_model(prediction)

        return app

    def _load_model(self) -> Inferencer:
        assert isinstance(self.folder, Path)
        return Inferencer.from_folder(self.folder)
