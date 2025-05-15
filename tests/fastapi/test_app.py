import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, call, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import fixture

from mlpype.base.data.data_sink import DataSink
from mlpype.base.data.dataset import DataSet
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.experiment import Experiment
from mlpype.base.pipeline.type_checker import DataSetModel
from mlpype.fastapi.deploy.app import mlpypeApp, write_in_background
from tests.shared_fixtures import dummy_experiment
from tests.utils import AnyArg, DummyDataModel, DummyDataSet, DummyDataSink, get_dummy_data

dummy_experiment


@fixture(scope="module")
def experiment_path(dummy_experiment: Experiment):
    dummy_experiment.run()

    yield dummy_experiment.output_folder


@fixture
def app(experiment_path: Path):
    return mlpypeApp("dummy-app", experiment_path)


@fixture
def tracking():
    return {
        "x": DummyDataSink(),
        "y": DummyDataSink(),
    }


@fixture
def app_with_tracking(experiment_path: Path, tracking: Dict[str, DataSink]):
    return mlpypeApp("dummy-app", experiment_path, tracking_servers=tracking)


@fixture()
def test_client(app: mlpypeApp) -> TestClient:
    return TestClient(app.create_app())


@fixture()
def test_client_with_tracking(app_with_tracking: mlpypeApp) -> TestClient:
    return TestClient(app_with_tracking.create_app())


class Test_create_app:
    def test_loading(self, app: mlpypeApp):
        with patch.object(app, "_load_model") as mock_load, patch.object(
            app, "_verify_tracking_servers"
        ) as mock_verify:
            mock_inferencer = mock_load.return_value
            mock_inferencer.input_type_checker.get_pydantic_types.return_value = DummyDataSet
            mock_inferencer.output_type_checker.get_pydantic_types.return_value = DummyDataSet

            created_app = app.create_app()
            assert isinstance(created_app, FastAPI)

            mock_load.assert_called_once_with()
            mock_inferencer.input_type_checker.get_pydantic_types.assert_called_once_with()
            mock_inferencer.output_type_checker.get_pydantic_types.assert_called_once_with()

            # input and output
            mock_verify.assert_called_once_with(DummyDataSet, DummyDataSet)


class Test_verify_tracking_servers:
    def test_verify_tracking_servers_without_tracking(self, app: mlpypeApp):
        # This should just run, since nothing is checked.
        app._verify_tracking_servers(MagicMock(), MagicMock())

    def test_verify_tracking_servers_with_tracking(self, app_with_tracking: mlpypeApp):
        # This should just run, since nothing is checked.
        with patch("mlpype.fastapi.deploy.app.logger") as mock_logger:
            app_with_tracking._verify_tracking_servers(DummyDataSet, DummyDataSet)

        mock_logger.warning.assert_not_called()

    def test_verify_tracking_servers_with_tracking_raise_warning(self, app_with_tracking: mlpypeApp):
        # This should just run, since nothing is checked.
        class DummyDataSet(DataSetModel):
            x: DummyDataModel
            z: DummyDataModel

        with patch("mlpype.fastapi.deploy.app.logger") as mock_logger:
            app_with_tracking._verify_tracking_servers(DummyDataSet, DummyDataSet)

        mock_logger.warning.assert_called_once_with(
            f"No dataset named `y` found in the fields of input or output DataSetModels"
        )


class Test_handle_tracking:
    def test_handle_tracking_without_tracking(self, app: mlpypeApp):
        data = get_dummy_data(10, 2, 1)

        # should just run.
        app._handle_tracking(DataSet(x=data["x"]), DataSet(y=data["y"]), MagicMock())

    def test_handle_tracking_with_tracking(self, app_with_tracking: mlpypeApp, tracking: Dict[str, MagicMock]):
        data = get_dummy_data(10, 2, 1).read()

        background_tasks = MagicMock()
        # should just run.
        app_with_tracking._handle_tracking(DataSet(x=data["x"]), DataSet(y=data["y"]), background_tasks)

        background_tasks.add_task.assert_has_calls(
            [
                call(write_in_background, sink=tracking["x"], data=data["x"]),
                call(write_in_background, sink=tracking["y"], data=data["y"]),
            ],
            any_order=True,
        )

    def test_handle_tracking_handles_crashes(self, app_with_tracking: mlpypeApp, tracking: Dict[str, MagicMock]):
        data = get_dummy_data(10, 2, 1).read()

        background_tasks = MagicMock()
        # This will call add_task to throw an error, causing only 1 background task to be executed.
        background_tasks.add_task.side_effect = [ConnectionError("Dummy connection error!"), None]

        with patch("mlpype.fastapi.deploy.app.logger") as mock_logger:
            app_with_tracking._handle_tracking(DataSet(x=data["x"]), DataSet(y=data["y"]), background_tasks)

        background_tasks.add_task.assert_has_calls(
            [
                call(write_in_background, sink=tracking["x"], data=data["x"]),
                call(write_in_background, sink=tracking["y"], data=data["y"]),
            ],
            any_order=True,
        )

        mock_logger.error.assert_called_once_with(f"Encountered error while sending data to x: Dummy connection error!")


def test_write_in_background():
    sink = MagicMock()
    data = MagicMock()
    write_in_background(sink, data)

    sink.write.assert_called_once_with(data)


class Test_app:
    def test_home_page(self, test_client: TestClient):
        response = test_client.get("/")
        assert response.status_code == 200

        content = json.loads(response.content.decode())
        assert content == "Welcome to the mlpype FastAPI app for dummy-app"

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

    def test_predict_with_tracking(self, app_with_tracking: mlpypeApp):
        x = [1.0, 2.0, 3.0, 4.0]

        with patch.object(app_with_tracking, "_handle_tracking") as mock_handle:
            app = app_with_tracking.create_app()
            test_client = TestClient(app)

            response = test_client.post("/predict", json={"x": {"data": x}})
        assert response.status_code == 200

        content = json.loads(response.content.decode())
        prediction = content["y"]["data"]

        # Cannot check `background` argument
        mock_handle.assert_called_once_with(DataSet(x=x), DataSet(y=prediction), AnyArg())

    def test_predict_with_tracking_integration(
        self, test_client_with_tracking: TestClient, tracking: Dict[str, DummyDataSink]
    ):
        x = [1.0, 2.0, 3.0, 4.0]

        response = test_client_with_tracking.post("/predict", json={"x": {"data": x}})
        assert response.status_code == 200

        content = json.loads(response.content.decode())
        prediction = content["y"]["data"]

        # Wait for max 2 seconds
        current_time = datetime.now()
        succeeded = False
        while not succeeded and ((datetime.now() - current_time).seconds < 2):
            try:
                assert tracking["x"].data == x
                assert tracking["y"].data == prediction
                succeeded = True
            except AssertionError as e:
                # Potentially expected, just continue
                pass
        assert succeeded, "Did not manage to receive background task result within 2 seconds!"
