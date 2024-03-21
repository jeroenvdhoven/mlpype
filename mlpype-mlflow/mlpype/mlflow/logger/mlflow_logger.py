import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

from git import InvalidGitRepositoryError
from git.repo import Repo
from mlflow import create_experiment  # type: ignore
from mlflow import get_experiment_by_name  # type: ignore
from mlflow import log_artifact  # type: ignore
from mlflow import log_metrics  # type: ignore
from mlflow import log_params  # type: ignore
from mlflow import set_experiment  # type: ignore
from mlflow import set_tag  # type: ignore
from mlflow import set_tracking_uri  # type: ignore
from mlflow import start_run  # type: ignore

from mlpype.base.logger import ExperimentLogger


class MlflowLogger(ExperimentLogger):
    def __init__(self, name: str, uri: str, artifact_location: Optional[str] = None) -> None:
        """A logger using mlflow for mlpype.

        Args:
            name (str): The name of the experiment. For runs done on databricks, make sure you preface
                this with a workspace, like /Users/<your databricks user name>/<experiment name>.
            uri (str): The tracking uri. Should most likely start with `http://` or `databricks`.
                It can also equal `databricks`.
            artifact_location (Optional[str]): The artificat uri location. This is needed if you set up
                your own artifact location, to make sure files are copied over to any remote tracking server.
                This is not required when using hosted databricks. Defaults to None, like in
        """
        super().__init__()
        logger = logging.getLogger(__name__)
        self.name = name

        if not uri.startswith("http://") and not uri.startswith("databricks"):
            logger.warning(f"Most often `uri` has to start with `http://` or `databricks`. Got: {uri}")
        self.uri = uri
        self.artifact_location = artifact_location
        self.run = None

    def __enter__(self) -> None:
        """Setup the mlflow experiment and call start_run."""
        set_tracking_uri(self.uri)
        experiment = get_experiment_by_name(self.name)

        if experiment is None:
            # this is a new experiment: create it explicitly with the artifact uri
            # TODO: make sure this works on S3
            create_experiment(self.name, self.artifact_location)

        set_experiment(experiment_name=self.name)

        self.run = start_run().__enter__()
        self.log_branch()

    def log_branch(self) -> None:
        """Set the current branch as a tag on the experiment."""
        assert self.run is not None, "Please start the experiment first"
        repo = self._find_encapsulating_repo(Path("."))

        if repo is None:
            branch_name = "unknown_branch"
        else:
            branch_name = repo.active_branch.name

        set_tag("git_branch", branch_name)

    def _find_encapsulating_repo(self, directory: Path) -> Union[Repo, None]:
        """Finds the current repo we're in, if any.

        We'll recursively move up from `directory` until we find a valid
        git repo.

        Args:
            directory (Path): The directory to start looking for. Usually set
                to the current directory.

        Returns:
            Repo or None: None if no repo is found, otherwise a GitPython Repo.
        """
        repo = None
        directory = directory.absolute()
        while str(directory) != "/":
            try:
                repo = Repo(directory)
                break
            except InvalidGitRepositoryError:
                directory = directory.parent
        return repo

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """Exits a run started by __enter__.

        Args:
            exc_type (Optional[Type[BaseException]]): Passed to ActiveRun.__exit__()
            exc_val (Optional[BaseException]): Passed to ActiveRun.__exit__()
            exc_tb (Optional[TracebackType]): Passed to ActiveRun.__exit__()
        """
        assert self.run is not None, "Run not started, cannot exit!"
        self.run.__exit__(exc_type, exc_val, exc_tb)

    def _log_metrics(self, metrics: Dict[str, Union[str, float, int, str, bool]]) -> None:
        """Perform the actual logging of metrics in mlflow.

        Args:
            metrics (Dict[str, Union[str, float, int, str, bool]]): A dictionary of metric names and values.
        """
        log_metrics(metrics)

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Logs the parameters for a given run.

        These will by default be passed on with prefixes such as `model__`.

        Args:
            parameters (Dict[str, Any]): The parameters of a given run. Ideally these
                parameters should be no more complicated than string, float, int, bool, or a
                list of these.
        """
        log_params(parameters)

    def log_file(self, file: Union[str, Path]) -> None:
        """Logs a given file as part of an experiment.

        This only works well if:
            - you work with a databricks-hosted mlflow experiment.
            - you have set a artifact uri and configured your server properly.
            - you have run mlflow ui/server at the same folder where mlruns is located.

        Args:
            file (Union[str, Path]): The file to log.
        """
        log_artifact(str(file))
