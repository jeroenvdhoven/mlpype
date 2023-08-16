from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from mlflow.artifacts import download_artifacts
from mlflow.tracking import set_tracking_uri
from mlflow.tracking.fluent import get_experiment_by_name, search_runs

from mlpype.base.deploy import Inferencer


def load_experiment_from_mlflow(
    url: str,
    experiment_name: str,
    run_id: str,
    directory: Optional[Union[Path, str]] = None,
) -> Inferencer:
    """Download and import a trained mlpype Model from mlflow's artifact store.

    Args:
        url (str): The tracking url.
        experiment_name (str): The name of the experiment. Used to verify that the
            run_id is the correct one.
        run_id (str): The run id. Used to pull the actual experiment.
        directory (Optional[Union[Path, str]]): Optional directory to store the results
            in. By default, a temporary directory is used.

    Returns:
        Inferencer: The Inferencer loaded from the trained model.
    """
    set_tracking_uri(url)

    # verify the run ID provided is part of the given experiment.
    exp_id = get_experiment_by_name(experiment_name)
    assert exp_id is not None, f"Experiment {experiment_name} does not exist in {url}."
    runs = search_runs(exp_id.experiment_id, output_format="list")
    assert isinstance(runs, list)
    assert run_id in [run.info.run_id for run in runs], f"Run ID {run_id} is not present in the given experiment."

    if directory is None:
        with TemporaryDirectory(prefix="mlflow_model") as tmp_dir:
            return _download_and_load(run_id, Path(tmp_dir))
    else:
        if isinstance(directory, str):
            directory = Path(directory)
        return _download_and_load(run_id, directory)


def _download_and_load(
    run_id: str,
    directory: Path,
) -> Inferencer:
    download_artifacts(f"runs:/{run_id}/", dst_path=str(directory))
    return Inferencer.from_folder(directory)
