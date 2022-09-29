from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

from pytest import fixture

from pype.base.constants import Constants
from pype.base.data.dataset import DataSet
from pype.base.data.dataset_source import DataSetSource
from pype.base.deploy.inference import Inferencer
from pype.base.experiment import Experiment
from tests.test_utils import get_dummy_data


@fixture(scope="module")
def experiment_path(dummy_experiment: Experiment):
    dummy_experiment.run()
    yield dummy_experiment.output_folder


class Test_Inferencer:
    def test_from_folder(self):
        path = Path("something")

        pipeline = MagicMock()
        inputs = MagicMock()
        outputs = MagicMock()
        deserialised_values = [pipeline, inputs, outputs]

        read_data = '{"paths": ["1", "2"]}'
        m_open = mock_open(read_data=read_data)
        with patch("pype.base.deploy.inference.Model.load") as mock_model_load, patch(
            "pype.base.deploy.inference.JoblibSerialiser.deserialise", side_effect=deserialised_values
        ) as mock_deserialise, patch("pype.base.deploy.inference.switch_workspace") as mock_switch, patch(
            "pype.base.deploy.inference.open", m_open
        ):
            result = Inferencer.from_folder(path)

            m_open.assert_called_once_with(path / Constants.EXTRA_FILES, "r")

            mock_switch.assert_called_once_with(path, ["1", "2"])
            mock_model_load.assert_called_once_with(Constants.MODEL_FOLDER)

            mock_deserialise.assert_has_calls(
                [
                    call(Constants.PIPELINE_FILE),
                    call(Constants.INPUT_TYPE_CHECKER_FILE),
                    call(Constants.OUTPUT_TYPE_CHECKER_FILE),
                ]
            )

            assert result.model == mock_model_load.return_value
            assert result.pipeline == pipeline
            assert result.input_type_checker == inputs
            assert result.output_type_checker == outputs

    def test_predict(self):
        model = MagicMock()
        pipeline = MagicMock()
        inputs = MagicMock()
        outputs = MagicMock()
        dataset = MagicMock()

        inferencer = Inferencer(model, pipeline, inputs, outputs)
        result = inferencer.predict(dataset)

        inputs.transform.assert_called_once_with(dataset)
        pipeline.transform.assert_called_once_with(dataset)
        model.transform.assert_called_once_with(pipeline.transform.return_value)
        outputs.transform.assert_called_once_with(model.transform.return_value)

        assert result == model.transform.return_value

    def test_predict_read_from_source(self):
        model = MagicMock()
        pipeline = MagicMock()
        inputs = MagicMock()
        outputs = MagicMock()
        dataset = MagicMock(spec=DataSetSource)

        inferencer = Inferencer(model, pipeline, inputs, outputs)
        result = inferencer.predict(dataset)

        dataset.read.assert_called_once_with()
        inputs.transform.assert_called_once_with(dataset.read.return_value)
        pipeline.transform.assert_called_once_with(dataset.read.return_value)
        model.transform.assert_called_once_with(pipeline.transform.return_value)
        outputs.transform.assert_called_once_with(model.transform.return_value)

        assert result == model.transform.return_value

    def test_integration(self, experiment_path: Path):
        data = get_dummy_data(10, 1, 0)
        y = data["y"].read()
        inferencer = Inferencer.from_folder(experiment_path)

        prediction = inferencer.predict(data)
        assert isinstance(prediction, DataSet)

        assert len(y) == len(prediction["y"])
        assert type(y) == type(prediction["y"])

        manual_pred = inferencer.model.transform(inferencer.pipeline.transform(data.read()))
        assert manual_pred == prediction
