import json
from pathlib import Path
from unittest.mock import patch

from experiment.experiment import Experiment
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import fixture

from pype.base.data.dataset import DataSet
from pype.base.deploy.app import PypeApp
from pype.base.deploy.inference import Inferencer
from tests.test_utils import DummyDataModel, dummy_experiment

dummy_experiment


@fixture(scope="module")
def experiment_path(dummy_experiment: Experiment):
    dummy_experiment.run()

    yield dummy_experiment.output_folder


@fixture
def app(experiment_path: Path):
    return PypeApp("dummy-app", experiment_path)


@fixture()
def test_client(app: PypeApp) -> TestClient:
    return TestClient(app.create_app())


class Test_create_app:
    def test_loading(self, app: PypeApp):
        with patch.object(app, "_load_model") as mock_load:
            mock_inferencer = mock_load.return_value
            mock_inferencer.input_type_checker.get_pydantic_types.return_value = DummyDataModel
            mock_inferencer.output_type_checker.get_pydantic_types.return_value = DummyDataModel

            created_app = app.create_app()
            assert isinstance(created_app, FastAPI)

            mock_load.assert_called_once_with()
            mock_inferencer.input_type_checker.get_pydantic_types.assert_called_once_with()
            mock_inferencer.output_type_checker.get_pydantic_types.assert_called_once_with()


class Test_app:
    def test_home_page(self, test_client: TestClient):
        response = test_client.get("/")
        assert response.status_code == 200

        content = json.loads(response.content.decode())
        assert content == "Welcome to the Pype FastAPI app for dummy-app"

    def test_predict(self, test_client: TestClient, experiment_path: Path):
        x = [1, 2, 3, 4]
        response = test_client.post("/predict", json={"x": {"data": x}})
        assert response.status_code == 200

        content = json.loads(response.content.decode())
        assert "y" in content
        assert "data" in content["y"]
        prediction = content["y"]["data"]

        assert isinstance(prediction, list)
        for p in prediction:
            assert isinstance(p, (float, int))

        # compare to local reading
        inferencer = Inferencer.from_folder(experiment_path)
        y_true = inferencer.predict(DataSet(x=x))["y"]

        assert y_true == prediction
