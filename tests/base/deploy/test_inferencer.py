import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, mock_open, patch

from pytest import mark

from mlpype.base.constants import Constants
from mlpype.base.data import DataCatalog, DataSet, DataSource
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.experiment import Experiment
from mlpype.base.logger import LocalLogger
from mlpype.base.serialiser import JoblibSerialiser
from tests.utils import DummyModel, get_dummy_data, get_dummy_evaluator, get_dummy_pipeline, get_dummy_type_checkers


class DummySource(DataSource):
    def __init__(self, x: Any) -> None:
        super().__init__()
        self.x = x

    def read(self) -> Any:
        return self.x


def create_experiment_in_folder(folder: Path) -> Experiment:
    train = get_dummy_data(20, 1, 0)
    test = get_dummy_data(5, 2, -1)

    input_checker, output_checker = get_dummy_type_checkers()

    return Experiment(
        data_sources={"train": train, "test": test},
        model=DummyModel(inputs=["x"], outputs=["y"]),
        pipeline=get_dummy_pipeline(),
        logger=LocalLogger(),
        evaluator=get_dummy_evaluator(),
        serialiser=JoblibSerialiser(),
        output_folder=folder,
        input_type_checker=input_checker,
        output_type_checker=output_checker,
        parameters={"param_1": 2},
    )


class Test_Inferencer:
    @mark.parametrize(["name", "use_inferencer_directly"], [["without inferencer", False], ["with inferencer", True]])
    def test_from_folder(self, name: str, use_inferencer_directly: bool):
        path = Path("something")
        abs_path = path.absolute()

        pipeline = MagicMock()
        inputs = MagicMock()
        outputs = MagicMock()
        mocked_serialiser = MagicMock(spec=JoblibSerialiser)
        mocked_serialiser.deserialise.side_effect = [pipeline, inputs, outputs]

        read_data = '{"paths": ["1", "2"]}'
        m_open = mock_open(read_data=read_data)
        with patch("mlpype.base.deploy.inference.Model.load") as mock_model_load, patch(
            "mlpype.base.deploy.inference.JoblibSerialiser.deserialise", return_value=mocked_serialiser
        ) as mock_deserialise, patch("mlpype.base.deploy.inference.switch_workspace") as mock_switch, patch(
            "mlpype.base.deploy.inference.open", m_open
        ):
            serialiser = mocked_serialiser if use_inferencer_directly else None
            result = Inferencer.from_folder(path, serialiser=serialiser)

            m_open.assert_called_once_with(abs_path / Constants.EXTRA_FILES, "r")

            mock_switch.assert_called_once_with(abs_path, ["1", "2"])
            mock_model_load.assert_called_once_with(abs_path / Constants.MODEL_FOLDER)

            if use_inferencer_directly:
                mock_deserialise.assert_not_called()
            else:
                mock_deserialise.assert_called_once_with(abs_path / Constants.SERIALISER_FILE)
            mocked_serialiser.deserialise.assert_has_calls(
                [
                    call(abs_path / Constants.PIPELINE_FILE),
                    call(abs_path / Constants.INPUT_TYPE_CHECKER_FILE),
                    call(abs_path / Constants.OUTPUT_TYPE_CHECKER_FILE),
                ]
            )

            assert result.model == mock_model_load.return_value
            assert result.pipeline == pipeline
            assert result.input_type_checker == inputs
            assert result.output_type_checker == outputs

    def test_from_experiment(self):
        pipeline = MagicMock()
        model = MagicMock()
        itc = MagicMock()
        otc = MagicMock()

        experiment = Experiment(
            data_sources={"train": MagicMock()},
            pipeline=pipeline,
            model=model,
            evaluator=MagicMock(),
            input_type_checker=itc,
            output_type_checker=otc,
            logger=MagicMock(),
        )

        result = Inferencer.from_experiment(experiment)

        assert result.model == model
        assert result.pipeline == pipeline
        assert result.input_type_checker == itc
        assert result.output_type_checker == otc

    def test_predict(self):
        model = MagicMock()
        pipeline = MagicMock()
        inputs = MagicMock()
        outputs = MagicMock()
        dataset = MagicMock()

        inferencer = Inferencer(model, pipeline, inputs, outputs)
        result = inferencer.predict(dataset)

        inputs.transform.assert_called_once_with(dataset)
        pipeline.transform.assert_called_once_with(inputs.transform.return_value, is_inference=True)
        model.transform.assert_called_once_with(pipeline.transform.return_value)
        outputs.transform.assert_called_once_with(model.transform.return_value)
        transformed_data = pipeline.transform.return_value

        transformed_data.set_all.assert_not_called()
        assert result == outputs.transform.return_value

    def test_predict_with_transformed_data(self):
        model = MagicMock()
        pipeline = MagicMock()
        inputs = MagicMock()
        outputs = MagicMock()
        dataset = MagicMock()

        inferencer = Inferencer(model, pipeline, inputs, outputs)
        result = inferencer.predict(dataset, return_transformed_data=True)

        inputs.transform.assert_called_once_with(dataset)
        pipeline.transform.assert_called_once_with(inputs.transform.return_value, is_inference=True)
        transformed_data = pipeline.transform.return_value

        model.transform.assert_called_once_with(transformed_data)
        outputs.transform.assert_called_once_with(model.transform.return_value)
        output_data = outputs.transform.return_value

        transformed_data.set_all.assert_called_once_with(output_data.keys.return_value, output_data.values.return_value)

        assert result == transformed_data

    def test_predict_read_from_source(self):
        model = MagicMock()
        pipeline = MagicMock()
        inputs = MagicMock()
        outputs = MagicMock()
        dataset = MagicMock(spec=DataCatalog)

        inferencer = Inferencer(model, pipeline, inputs, outputs)
        result = inferencer.predict(dataset)

        dataset.read.assert_called_once_with()
        inputs.transform.assert_called_once_with(dataset.read.return_value)
        pipeline.transform.assert_called_once_with(inputs.transform.return_value, is_inference=True)
        model.transform.assert_called_once_with(pipeline.transform.return_value)
        outputs.transform.assert_called_once_with(model.transform.return_value)

        assert result == outputs.transform.return_value

    @mark.parametrize(
        ["folder"],
        [
            [Path("/tmp/outputs")],
            [Path("outputs__1")],
            [Path("./outputs__2")],
        ],
    )
    def test_integration(self, folder: Path):
        assert not folder.is_dir(), "Folder should be empty to start"
        try:
            exp = create_experiment_in_folder(folder)
            exp.run()
            assert folder.is_dir(), "Folder should now exist"
            data = get_dummy_data(10, 1, 0)
            y = data["y"].read()
            inferencer = Inferencer.from_folder(folder)

            prediction = inferencer.predict(data)
            assert isinstance(prediction, DataSet)

            assert len(y) == len(prediction["y"])
            assert type(y) == type(prediction["y"])

            manual_pred = inferencer.model.transform(inferencer.pipeline.transform(data.read()))
            assert manual_pred == prediction
        finally:
            shutil.rmtree(str(folder))
            assert not folder.is_dir(), "Folder should be empty at end"

    def test_integration_from_external_run(self):
        try:
            # TODO: reuse this format for example tests
            # Make sure we use the python path to the experiment file.
            # This mainly tests that switch_workspace does what it is supposed to do:
            # load the proper files from the folder into memory.
            path = str((Path(__file__).parent / "integration_experiment").absolute().relative_to(os.getcwd())).replace(
                "/", "."
            )
            subprocess.run([sys.executable, "-m", path])

            from .integration_experiment import _make_data, output_folder

            inferencer = Inferencer.from_folder(output_folder)
            _, test_x, _, test_y = _make_data()
            ds = {
                "test": DataCatalog(
                    x=DummySource(test_x),
                ),
            }
            result = inferencer.predict(ds["test"])
            assert "y" in result
            assert result["y"].shape[0] == test_y.shape[0]
        finally:
            from .integration_experiment import output_folder

            shutil.rmtree(str(output_folder), ignore_errors=True)
