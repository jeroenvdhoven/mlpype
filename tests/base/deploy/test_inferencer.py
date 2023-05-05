from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

from pytest import fixture, mark

from mlpype.base.constants import Constants
from mlpype.base.data import DataCatalog, DataSet
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.experiment import Experiment
from mlpype.base.serialiser import JoblibSerialiser
from tests.shared_fixtures import dummy_experiment
from tests.utils import get_dummy_data

dummy_experiment


@fixture(scope="module")
def experiment_path(dummy_experiment: Experiment):
    dummy_experiment.run()
    yield dummy_experiment.output_folder


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
        pipeline.transform.assert_called_once_with(dataset, is_inference=True)
        model.transform.assert_called_once_with(pipeline.transform.return_value)
        outputs.transform.assert_called_once_with(model.transform.return_value)

        assert result == model.transform.return_value

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
        pipeline.transform.assert_called_once_with(dataset.read.return_value, is_inference=True)
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
