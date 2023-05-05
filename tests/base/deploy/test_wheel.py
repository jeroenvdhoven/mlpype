import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pip
from pytest import fixture, mark

from mlpype.base.constants import Constants
from mlpype.base.deploy.wheel import WheelBuilder, WheelExtension
from mlpype.base.deploy.wheel.builder import BaseExtension
from mlpype.base.experiment.experiment import Experiment
from tests.shared_fixtures import dummy_experiment
from tests.utils import pytest_assert

dummy_experiment


@fixture(scope="module")
def run_experiment(dummy_experiment: Experiment):
    dummy_experiment.run()
    yield dummy_experiment


class Test_WheelBuilder:
    @mark.wheel
    def test_integration(self, run_experiment: Experiment):
        # this will install the trained model into your current environment.
        # an upgrade would be to use a new environment.
        # however, this way you will use the same mlpype libraries as used for training.
        # Make sure we print requirements as well for easy debugging
        req_file = run_experiment.output_folder / Constants.REQUIREMENTS_FILE
        with open(req_file, "r") as f:
            print("Requirements:")
            for req_line in f.readlines():
                print(req_line, end="")

        with TemporaryDirectory() as f:
            output_folder = Path(f) / "wheel"
            model_name = "example_model_for_testing_purposes"

            builder = WheelBuilder(
                model_folder=run_experiment.output_folder,
                model_name=model_name,
                version="0.0.1",
                output_wheel_file=output_folder,
            )

            builder.build()

            result = output_folder / os.listdir(output_folder)[0]

            try:
                # dependencies are already present, so this will help speed things up.
                install_result = pip.main(["install", str(result), "--no-deps", "--force-reinstall"])
                assert install_result == 0, f"Installation failed! {install_result}"

                imported_model = importlib.import_module(model_name)
                model = imported_model.load_model()

                input_dataset = run_experiment.data_sources["train"].read()
                predictions = model.predict(input_dataset)

                assert "y" in predictions
                assert len(predictions["y"]) == len(input_dataset["x"])
            finally:
                pip.main(["uninstall", str(result), "-y"])

    def test_post_init(self, run_experiment: Experiment):
        with patch.object(WheelBuilder, "_validate_extensions") as mock_validate:
            builder = WheelBuilder(
                model_folder=run_experiment.output_folder, model_name="example_model", version="0.0.1"
            )

        assert builder.extensions == [BaseExtension]
        assert builder.output_wheel_file == Path(os.getcwd()) / "wheel_output"
        mock_validate.assert_called_once_with()

    def test_validate_extensions_success(self, run_experiment: Experiment):
        # the BaseExtension added to the builder should be validated.
        with patch.object(WheelBuilder, "_validate_extensions") as mock_validate:
            builder = WheelBuilder(
                model_folder=run_experiment.output_folder, model_name="example_model", version="0.0.1"
            )
        builder._validate_extensions()

    def test_validate_extensions_duplicate_extensions(self, run_experiment: Experiment):
        # the BaseExtension added to the builder should be validated.
        with patch.object(WheelBuilder, "_validate_extensions") as mock_validate:
            extension1 = WheelExtension("ext1", [], [])
            extension2 = WheelExtension("ext1", [], [])

            builder = WheelBuilder(
                model_folder=run_experiment.output_folder,
                model_name="example_model",
                version="0.0.1",
                extensions=[extension1, extension2],
            )

        with pytest_assert(AssertionError, "ext1 is a duplicated extension name, this is not allowed."):
            builder._validate_extensions()

    def test_validate_extensions_duplicate_imports(self, run_experiment: Experiment):
        # the BaseExtension added to the builder should be validated.
        with patch.object(WheelBuilder, "_validate_extensions") as mock_validate:
            extension1 = WheelExtension("ext1", [(Path(".").absolute(), ["func1", "func2"])], [])
            extension2 = WheelExtension("ext2", [(Path(".").absolute(), ["func3", "func2"])], [])

            builder = WheelBuilder(
                model_folder=run_experiment.output_folder,
                model_name="example_model",
                version="0.0.1",
                extensions=[extension1, extension2],
            )

        with pytest_assert(AssertionError, "func2 is a duplicated function name from ext2, this is not allowed."):
            builder._validate_extensions()


class Test_WheelExtension:
    pass
