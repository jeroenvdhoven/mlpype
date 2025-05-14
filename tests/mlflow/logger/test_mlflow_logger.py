import pickle
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from git import InvalidGitRepositoryError
from pytest import fixture, mark

from mlpype.base.logger.experiment_logger import ExperimentLogger
from mlpype.mlflow.logger.mlflow_logger import MlflowLogger
from mlpype.mlflow.model.model import PypeMLFlowModel
from tests.utils import pytest_assert


class Test_mlflow_logger:
    @fixture
    def logger(self) -> MlflowLogger:
        return MlflowLogger("example", "http://localhost:5000")

    def test_enter(self, logger: MlflowLogger):
        assert logger.run is None
        with patch("mlpype.mlflow.logger.mlflow_logger.set_tracking_uri") as mock_set_tracking_uri, patch(
            "mlpype.mlflow.logger.mlflow_logger.get_experiment_by_name"
        ) as mock_get_experiment_by_name, patch(
            "mlpype.mlflow.logger.mlflow_logger.create_experiment"
        ) as mock_create_experiment, patch(
            "mlpype.mlflow.logger.mlflow_logger.set_experiment"
        ) as mock_set_experiment, patch(
            "mlpype.mlflow.logger.mlflow_logger.start_run"
        ) as mock_start_run, patch.object(
            logger, "log_branch"
        ) as mock_log_branch:
            mock_get_experiment_by_name.return_value = []  # just not None
            logger.__enter__()

            mock_set_tracking_uri.assert_called_once_with(logger.uri)
            mock_get_experiment_by_name.assert_called_once_with(logger.name)
            mock_create_experiment.assert_not_called()
            mock_set_experiment.assert_called_once_with(experiment_name=logger.name)

            mock_start_run.assert_called_once_with()
            mock_log_branch.assert_called_once_with()
            run = mock_start_run.return_value
            run.__enter__.assert_called_once_with()
            active_run = run.__enter__.return_value

            assert logger.run == active_run

    def test_enter_new_experiment(self, logger: MlflowLogger):
        assert logger.run is None
        with patch("mlpype.mlflow.logger.mlflow_logger.set_tracking_uri") as mock_set_tracking_uri, patch(
            "mlpype.mlflow.logger.mlflow_logger.get_experiment_by_name"
        ) as mock_get_experiment_by_name, patch(
            "mlpype.mlflow.logger.mlflow_logger.create_experiment"
        ) as mock_create_experiment, patch(
            "mlpype.mlflow.logger.mlflow_logger.set_experiment"
        ) as mock_set_experiment, patch(
            "mlpype.mlflow.logger.mlflow_logger.start_run"
        ) as mock_start_run, patch.object(
            logger, "log_branch"
        ) as mock_log_branch:
            mock_get_experiment_by_name.return_value = None
            logger.__enter__()

            mock_set_tracking_uri.assert_called_once_with(logger.uri)
            mock_get_experiment_by_name.assert_called_once_with(logger.name)
            mock_create_experiment.assert_called_once_with(logger.name, None)
            mock_set_experiment.assert_called_once_with(experiment_name=logger.name)

            mock_start_run.assert_called_once_with()
            mock_log_branch.assert_called_once_with()
            run = mock_start_run.return_value
            run.__enter__.assert_called_once_with()
            active_run = run.__enter__.return_value

            assert logger.run == active_run

    def test_log_branch(self, logger: MlflowLogger):
        branch_name = "branch/name"
        repo = MagicMock()
        repo.active_branch.name = branch_name
        with patch.object(logger, "_find_encapsulating_repo", return_value=repo) as mock_find, patch(
            "mlpype.mlflow.logger.mlflow_logger.set_tag"
        ) as mock_set_tag:
            logger.run = MagicMock()
            logger.log_branch()

            mock_find.assert_called_once_with(Path("."))
            mock_set_tag.assert_called_once_with("git_branch", branch_name)

    def test_log_branch_check_run(self, logger: MlflowLogger):
        with patch.object(logger, "_find_encapsulating_repo"), patch(
            "mlpype.mlflow.logger.mlflow_logger.set_tag"
        ), pytest_assert(AssertionError, "Please start the experiment first"):
            logger.log_branch()

    def test_log_branch_no_repo_found(self, logger: MlflowLogger):
        repo = None
        with patch.object(logger, "_find_encapsulating_repo", return_value=repo) as mock_find, patch(
            "mlpype.mlflow.logger.mlflow_logger.set_tag"
        ) as mock_set_tag:
            logger.run = MagicMock()
            logger.log_branch()

            mock_find.assert_called_once_with(Path("."))
            mock_set_tag.assert_called_once_with("git_branch", "unknown_branch")

    def test_find_encapsulating_repo(self, logger: MlflowLogger):
        repo = MagicMock()
        directory = Path("a_path") / "sub_dir"
        with patch("mlpype.mlflow.logger.mlflow_logger.Repo", return_value=repo) as mock_repo_class:
            result = logger._find_encapsulating_repo(directory)
            mock_repo_class.assert_called_once_with(directory.absolute())

            assert result == repo

    def test_find_encapsulating_repo_find_repo_up(self, logger: MlflowLogger):
        repo = MagicMock()
        directory = Path("a_path") / "sub_dir"
        with patch(
            "mlpype.mlflow.logger.mlflow_logger.Repo", side_effect=[InvalidGitRepositoryError, repo]
        ) as mock_repo_class:
            result = logger._find_encapsulating_repo(directory)
            mock_repo_class.assert_has_calls([call(directory.absolute()), call(directory.absolute().parent)])

            assert result == repo

    def test_find_encapsulating_repo_find_no_repo(self, logger: MlflowLogger):
        directory = Path("/") / "1_dir" / "2_dir"
        with patch(
            "mlpype.mlflow.logger.mlflow_logger.Repo", side_effect=[InvalidGitRepositoryError] * 2
        ) as mock_repo_class:
            result = logger._find_encapsulating_repo(directory)
            mock_repo_class.assert_has_calls(
                [
                    call(Path("/") / "1_dir" / "2_dir"),
                    call(Path("/") / "1_dir"),
                ]
            )

            assert result == None

    def test_exit(self, logger: MlflowLogger):
        mock_run = MagicMock()
        logger = mock_run
        a = MagicMock()
        b = MagicMock()
        c = MagicMock()
        logger.__exit__(a, b, c)
        mock_run.__exit__.assert_called_once_with(a, b, c)

    def test_exit_assert_run_started(self, logger: MlflowLogger):
        a = MagicMock()
        b = MagicMock()
        c = MagicMock()
        with pytest_assert(AssertionError, "Run not started, cannot exit!"):
            logger.__exit__(a, b, c)

    def test_log_metrics(self, logger: MlflowLogger):
        with patch("mlpype.mlflow.logger.mlflow_logger.log_metrics") as mock_log_metrics:
            metrics = {"a": 1, "b": 3}
            logger._log_metrics(metrics)

            mock_log_metrics.assert_called_once_with(metrics)

    def test_log_parameters(self, logger: MlflowLogger):
        with patch("mlpype.mlflow.logger.mlflow_logger.log_params") as mock_log_params:
            parameters = {"a": 1, "b": 3}
            logger.log_parameters(parameters)

            mock_log_params.assert_called_once_with(parameters)

    def test_log_file(self, logger: MlflowLogger):
        with patch("mlpype.mlflow.logger.mlflow_logger.log_artifact") as mock_log_artifact:
            file = "outputs/file.txt"
            logger.log_file(file)

            mock_log_artifact.assert_called_once_with(file)

    def test_nested_log_file(self, logger: MlflowLogger):
        with patch("mlpype.mlflow.logger.mlflow_logger.log_artifact") as mock_log_artifact:
            file = Path("outputs", "dummy", "file.txt")
            logger.log_file(file)

            mock_log_artifact.assert_called_once_with(str(file))

    @mark.parametrize(["name", "model_name"], [["no registration", None], ["registration", "very_fancy_model_name"]])
    def test_model_no_register(self, name: str, model_name: str):
        folder = Path("folder")
        model = MagicMock()
        run = MagicMock()
        run_id = "run_id_32"
        run.info.run_id = run_id
        logger = MlflowLogger("example", "http://localhost:5000", model_name=model_name)
        logger.run = run

        with patch.object(ExperimentLogger, "log_model") as mock_log_model, patch(
            "mlpype.mlflow.logger.mlflow_logger.mlflow_log_model"
        ) as mock_mlflow_log_model, patch(
            "mlpype.mlflow.logger.mlflow_logger.PypeMLFlowModel"
        ) as mock_integration_model, patch.object(
            logger, "register_mlpype_model"
        ) as mock_register_mlpype_model:
            logger.log_model(model, folder)

        model.save.assert_called_once_with(folder)
        mock_log_model.assert_called_once_with(model, folder)
        mock_integration_model.assert_called_once_with()
        mock_mlflow_log_model.assert_called_once_with(
            artifact_path=logger.ARTIFACT_FOLDER,
            python_model=mock_integration_model.return_value,
            artifacts={"folder": str(folder.parent)},
        )

        if model_name is None:
            mock_register_mlpype_model.assert_not_called()
        else:
            mock_register_mlpype_model.assert_called_once_with(run_id, model_name)

    @mark.parametrize(
        ["run_id", "model_name"],
        [
            [None, None],
            ["run_id_32", None],
            [None, "very_fancy_model_name"],
            ["run_id_32", "very_fancy_model_name"],
        ],
    )
    def test_register_mlpype_model(self, logger: MlflowLogger, run_id: str, model_name: str):
        logger.run = MagicMock()
        logger.run.info.run_id = "default_run_id"

        logger.model_name = "default_model_name"

        with patch("mlpype.mlflow.logger.mlflow_logger.register_model") as mock_register_model:
            logger.register_mlpype_model(run_id, model_name)

        expected_run = run_id if run_id is not None else logger.run.info.run_id
        expected_model_name = model_name if model_name is not None else logger.model_name
        mock_register_model.assert_called_once_with(
            f"runs:/{expected_run}/{logger.ARTIFACT_FOLDER}", expected_model_name
        )

    @mark.parametrize(
        ["model_name"],
        [
            [None],
            ["very_fancy_model_name"],
        ],
    )
    def test_log_model(self, logger: MlflowLogger, model_name: str):
        model = MagicMock()
        folder = Path("folder")

        with patch.object(ExperimentLogger, "log_model") as mock_load_model, patch(
            "mlpype.mlflow.logger.mlflow_logger.mlflow_log_model"
        ) as mock_log_model, patch.object(logger, "register_mlpype_model") as mock_register_mlpype_model, patch(
            "mlpype.mlflow.logger.mlflow_logger.PypeMLFlowModel"
        ) as mock_pype:
            logger.model_name = model_name
            logger.run = MagicMock()
            logger.log_model(model, folder)

        model.save.assert_called_once_with(folder)
        mock_load_model.assert_called_once_with(model, folder)
        mock_pype.assert_called_once_with()
        mock_log_model.assert_called_once_with(
            artifact_path=logger.ARTIFACT_FOLDER,
            python_model=mock_pype.return_value,
            artifacts={"folder": str(folder.parent)},
        )

        if model_name is not None:
            mock_register_mlpype_model.assert_called_once_with(logger.run.info.run_id, model_name)
        else:
            mock_register_mlpype_model.assert_not_called()

    def test_log_model_with_exception(self, logger: MlflowLogger):
        model = MagicMock()
        folder = Path("folder")

        with patch.object(ExperimentLogger, "log_model") as mock_load_model, patch(
            "mlpype.mlflow.logger.mlflow_logger.mlflow_log_model", side_effect=Exception
        ) as mock_log_model, patch.object(logger, "register_mlpype_model") as mock_register_mlpype_model, patch(
            "mlpype.mlflow.logger.mlflow_logger.PypeMLFlowModel"
        ) as mock_pype:
            logger.model_name = "very_fancy_model_name"
            logger.run = MagicMock()
            logger.log_model(model, folder)

        model.save.assert_called_once_with(folder)
        mock_load_model.assert_called_once_with(model, folder)
        mock_pype.assert_called_once_with()
        mock_log_model.assert_called_once_with(
            artifact_path=logger.ARTIFACT_FOLDER,
            python_model=mock_pype.return_value,
            artifacts={"folder": str(folder.parent)},
        )

        mock_register_mlpype_model.assert_not_called()


def test_PypeMLFlowModel_pickle_state():
    model = PypeMLFlowModel()
    model.inferencer = MagicMock()

    res = pickle.dumps(model)
    loaded = pickle.loads(res)

    assert loaded.inferencer is None
