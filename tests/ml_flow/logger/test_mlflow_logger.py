from unittest.mock import MagicMock, patch

from pytest import fixture

from pype.ml_flow.logger.mlflow_logger import MlflowLogger
from tests.test_utils import pytest_assert


class Test_mlflow_logger:
    @fixture
    def logger(self) -> MlflowLogger:
        return MlflowLogger("example", "http://localhost:5000")

    def test_enter(self, logger: MlflowLogger):
        assert logger.run is None
        with patch("pype.ml_flow.logger.mlflow_logger.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = []  # just not None
            logger.__enter__()

            mock_mlflow.set_tracking_uri.assert_called_once_with(logger.uri)
            mock_mlflow.get_experiment_by_name.assert_called_once_with(logger.name)
            mock_mlflow.create_experiment.assert_not_called()
            mock_mlflow.set_experiment.assert_called_once_with(experiment_name=logger.name)

            mock_mlflow.start_run.assert_called_once_with()
            run = mock_mlflow.start_run.return_value
            run.__enter__.assert_called_once_with()
            active_run = run.__enter__.return_value

            assert logger.run == active_run

    def test_enter_new_experiment(self, logger: MlflowLogger):
        assert logger.run is None
        with patch("pype.ml_flow.logger.mlflow_logger.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            logger.__enter__()

            mock_mlflow.set_tracking_uri.assert_called_once_with(logger.uri)
            mock_mlflow.get_experiment_by_name.assert_called_once_with(logger.name)
            mock_mlflow.create_experiment.assert_called_once_with(logger.name, None)
            mock_mlflow.set_experiment.assert_called_once_with(experiment_name=logger.name)

            mock_mlflow.start_run.assert_called_once_with()
            run = mock_mlflow.start_run.return_value
            run.__enter__.assert_called_once_with()
            active_run = run.__enter__.return_value

            assert logger.run == active_run

    def test_exit(self, logger: MlflowLogger):
        mock_run = MagicMock()
        logger = mock_run
        with patch("pype.ml_flow.logger.mlflow_logger.mlflow"):
            a = MagicMock()
            b = MagicMock()
            c = MagicMock()
            logger.__exit__(a, b, c)
            mock_run.__exit__.assert_called_once_with(a, b, c)

    def test_exit_assert_run_started(self, logger: MlflowLogger):
        with patch("pype.ml_flow.logger.mlflow_logger.mlflow"):
            a = MagicMock()
            b = MagicMock()
            c = MagicMock()
            with pytest_assert(AssertionError, "Run not started, cannot exit!"):
                logger.__exit__(a, b, c)

    def test_log_metrics(self, logger: MlflowLogger):
        with patch("pype.ml_flow.logger.mlflow_logger.mlflow") as mock_mlflow:
            metrics = {"a": 1, "b": 3}
            logger._log_metrics(metrics)

            mock_mlflow.log_metrics.assert_called_once_with(metrics)

    def test_log_parameters(self, logger: MlflowLogger):
        with patch("pype.ml_flow.logger.mlflow_logger.mlflow") as mock_mlflow:
            parameters = {"a": 1, "b": 3}
            logger.log_parameters(parameters)

            mock_mlflow.log_params.assert_called_once_with(parameters)

    def test_log_file(self, logger: MlflowLogger):
        with patch("pype.ml_flow.logger.mlflow_logger.mlflow") as mock_mlflow:
            file = "some file"
            logger.log_file(file)

            mock_mlflow.log_artifact.assert_called_once_with(file)
