import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pip
from fastapi import FastAPI
from pytest import fixture, mark

from pype.base.constants import Constants
from pype.base.deploy.wheel.builder import WheelBuilder
from pype.base.experiment import Experiment
from pype.fastapi.deploy import FastApiExtension
from tests.shared_fixtures import dummy_experiment

dummy_experiment


@fixture(scope="module")
def run_experiment(dummy_experiment: Experiment):
    dummy_experiment.run()
    yield dummy_experiment


@mark.wheel
def test_fastapi_wheel_extension(run_experiment: Experiment):
    # this will install the trained model into your current environment.
    # an upgrade would be to use a new environment.
    # however, this way you will use the same pype libraries as used for training.
    # Make sure we print requirements as well for easy debugging
    req_file = run_experiment.output_folder / Constants.REQUIREMENTS_FILE
    with open(req_file, "r") as f:
        print("Requirements:")
        for req_line in f.readlines():
            print(req_line, end="")

    with TemporaryDirectory() as f:
        output_folder = Path(f) / "wheel"
        model_name = "example_fastapi_model_for_testing_purposes"

        builder = WheelBuilder(
            model_folder=run_experiment.output_folder,
            model_name=model_name,
            version="0.0.1",
            output_wheel_file=output_folder,
            extensions=[FastApiExtension],
        )

        builder.build()
        result = output_folder / os.listdir(output_folder)[0]

        try:
            install_result = pip.main(["install", str(result), "--force-reinstall"])
            assert install_result == 0, f"Installation failed! {install_result}"

            imported_model = importlib.import_module(model_name)
            app = imported_model.load_app()

            assert isinstance(app, FastAPI)
        finally:
            pip.main(["uninstall", str(result), "-y"])
