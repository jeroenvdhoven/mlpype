from pathlib import Path
from unittest.mock import MagicMock, patch

from pytest import mark

from pype.ml_flow.deploy.load_experiment import (
    _download_and_load,
    load_experiment_from_mlflow,
)
from tests.utils import pytest_assert


def dummy_run_info(run_id: str) -> MagicMock:
    mock = MagicMock()
    mock.run_id = run_id
    return mock


def dummy_experiment(run_id: str) -> MagicMock:
    mock = MagicMock()
    mock.experiment_id = run_id
    return mock


class Test_load_experiment_from_mlflow:
    def test_assertion_valid_experiment(self):
        url = "some_url"
        name = "some name"
        run_id = "jkldsf9032sd"
        experiment_id = "238jlkdfala932la@@"
        with patch("pype.ml_flow.deploy.load_experiment.mlflow") as mock_mlflow, patch(
            "pype.ml_flow.deploy.load_experiment._download_and_load"
        ):
            # experiment not found
            mock_mlflow.get_experiment_by_name.return_value = None

            with pytest_assert(AssertionError, f"Experiment {name} does not exist in {url}."):
                load_experiment_from_mlflow(url, name, run_id)

    def test_assertion_valid_run_id(self):
        url = "some_url"
        name = "some name"
        run_id = "jkldsf9032sd"
        experiment_id = "238jlkdfala932la@@"
        with patch("pype.ml_flow.deploy.load_experiment.mlflow") as mock_mlflow, patch(
            "pype.ml_flow.deploy.load_experiment._download_and_load"
        ):
            # experiment not found
            mock_mlflow.get_experiment_by_name.return_value = dummy_experiment(experiment_id)
            mock_mlflow.list_run_infos.return_value = [dummy_run_info("others")]

            with pytest_assert(AssertionError, f"Run ID {run_id} is not present in the given experiment."):
                load_experiment_from_mlflow(url, name, run_id)

    @mark.parametrize(["directory"], [["str-directory"], [Path("path-directory")]])
    def test_with_folder(self, directory: str | Path):
        url = "some_url"
        name = "some name"
        run_id = "jkldsf9032sd"
        experiment_id = "238jlkdfala932la@@"
        with patch("pype.ml_flow.deploy.load_experiment.mlflow") as mock_mlflow, patch(
            "pype.ml_flow.deploy.load_experiment._download_and_load"
        ) as mock_download:
            mock_mlflow.get_experiment_by_name.return_value = dummy_experiment(experiment_id)
            mock_mlflow.list_run_infos.return_value = [dummy_run_info(run_id), dummy_run_info("others")]

            result = load_experiment_from_mlflow(url, name, run_id, directory)

            mock_mlflow.set_tracking_uri.assert_called_once_with(url)
            mock_mlflow.get_experiment_by_name.assert_called_once_with(name)
            mock_mlflow.list_run_infos.assert_called_once_with(experiment_id)

            mock_download.assert_called_once_with(run_id, Path(directory))
            assert result == mock_download.return_value

    def test_with_tmp_folder(self):
        url = "some_url"
        name = "some name"
        run_id = "jkldsf9032sd"
        experiment_id = "238jlkdfala932la@@"
        with patch("pype.ml_flow.deploy.load_experiment.mlflow") as mock_mlflow, patch(
            "pype.ml_flow.deploy.load_experiment._download_and_load"
        ) as mock_download, patch("pype.ml_flow.deploy.load_experiment.TemporaryDirectory") as mock_tmp_dir:
            tmp_path = "tmp_path"
            mock_tmp_dir.return_value.__enter__.return_value = tmp_path

            mock_mlflow.get_experiment_by_name.return_value = dummy_experiment(experiment_id)
            mock_mlflow.list_run_infos.return_value = [dummy_run_info(run_id), dummy_run_info("others")]

            result = load_experiment_from_mlflow(url, name, run_id)

            mock_mlflow.set_tracking_uri.assert_called_once_with(url)
            mock_mlflow.get_experiment_by_name.assert_called_once_with(name)
            mock_mlflow.list_run_infos.assert_called_once_with(experiment_id)

            mock_tmp_dir.assert_called_once_with(prefix="mlflow_model")

            mock_download.assert_called_once_with(run_id, Path(tmp_path))
            assert result == mock_download.return_value


def test_download_and_load():
    run_id = "run_id"
    directory = Path("folder_with_model")
    with patch("pype.ml_flow.deploy.load_experiment.mlflow.artifacts.download_artifacts") as mock_download, patch(
        "pype.ml_flow.deploy.load_experiment.Inferencer.from_folder"
    ) as mock_load:
        result = _download_and_load(run_id=run_id, directory=directory)

        mock_download.assert_called_once_with(f"runs:/{run_id}/", dst_path="folder_with_model")
        mock_load.assert_called_once_with(directory)
        assert result == mock_load.return_value
