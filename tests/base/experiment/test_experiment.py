import shutil
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from pytest import fixture

from pype.base.constants import Constants
from pype.base.experiment.experiment import Experiment
from tests.test_utils import get_dummy_data, get_dummy_experiment, pytest_assert


class Test_get_cmd_args:
    @fixture(scope="class")
    def run_experiment(self) -> tuple[Experiment, dict]:
        experiment = get_dummy_experiment()
        metrics = experiment.run()

        yield experiment, metrics

        shutil.rmtree(experiment.output_folder)

    def test_unit(self):
        data_sources = {"train": MagicMock(), "test": MagicMock()}
        model = MagicMock()
        pipeline = MagicMock()
        evaluator = MagicMock()
        logger = MagicMock()
        serialiser = MagicMock()
        input_type_checker = MagicMock()
        output_type_checker = MagicMock()
        output_folder = Path("tmp")
        additional_files_to_store = ["a", "b"]

        experiment = Experiment(
            data_sources=data_sources,
            model=model,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            output_folder=output_folder,
            additional_files_to_store=additional_files_to_store,
        )

        train_transform = MagicMock()
        test_transform = MagicMock()
        pipeline.transform.side_effect = [train_transform, test_transform]

        train_performance = MagicMock()
        test_performance = MagicMock()
        evaluator.evaluate.side_effect = [train_performance, test_performance]

        with patch.object(experiment, "_create_output_folders") as mock_create_folders:
            result = experiment.run()

        logger.__enter__.assert_called_once()
        logger.__exit__.assert_called_once()

        # getting data
        data_sources["train"].read.assert_called_once_with()
        data_sources["test"].read.assert_called_once_with()

        # input checker
        dataset_train = data_sources["train"].read.return_value
        dataset_test = data_sources["test"].read.return_value
        input_type_checker.fit.assert_called_once_with(dataset_train)
        input_type_checker.transform.assert_has_calls([call(dataset_train), call(dataset_test)])

        # pipeline fitting
        pipeline.fit.assert_called_once_with(dataset_train)
        pipeline.transform.assert_has_calls([call(dataset_train), call(dataset_test)])

        # model fitting
        model.fit.assert_called_once_with(train_transform)
        model.transform.assert_called_once_with(train_transform)
        output_type_checker.fit.assert_called_once_with(model.transform.return_value)

        # evaluation
        evaluator.evaluate.assert_has_calls([call(model, train_transform), call(model, test_transform)])
        assert result == {
            "train": train_performance,
            "test": test_performance,
        }

        # logging
        logger.log_metrics.assert_has_calls(
            [
                call("train", train_performance),
                call("test", test_performance),
            ]
        )

        mock_create_folders.assert_called_once_with()
        logger.log_model.assert_called_once_with(model, output_folder / Constants.MODEL_FOLDER)
        logger.log_artifact.assert_has_calls(
            [
                call(output_folder / Constants.PIPELINE_FILE, serialiser, object=pipeline),
                call(output_folder / Constants.INPUT_TYPE_CHECKER_FILE, serialiser, object=input_type_checker),
                call(output_folder / Constants.OUTPUT_TYPE_CHECKER_FILE, serialiser, object=output_type_checker),
            ]
        )
        logger.log_parameters.assert_called_once_with({})
        logger.log_file.assert_has_calls([call("a"), call("b")])
        # of = self.output_folder
        # self.experiment_logger.log_model(self.model, of / Constants.MODEL_FOLDER)
        # self.experiment_logger.log_artifact(of / Constants.PIPELINE_FILE, self.serialiser, object=self.pipeline)
        # self.experiment_logger.log_artifact(
        #     of / Constants.INPUT_TYPE_CHECKER_FILE, self.serialiser, object=self.input_type_checker
        # )
        # self.experiment_logger.log_artifact(
        #     of / Constants.OUTPUT_TYPE_CHECKER_FILE, self.serialiser, object=self.output_type_checker
        # )
        # self.experiment_logger.log_parameters(self.parameters)

        # for extra_file in self.additional_files_to_store:
        #     self.experiment_logger.log_file(extra_file)

        # self.logger.info("Done")

    def test_integration(self, run_experiment: tuple[Experiment, dict]):
        experiment, metrics = run_experiment
        for ds_name in ["train", "test"]:
            assert ds_name in metrics
            assert "diff" in metrics[ds_name]

        test_data = get_dummy_data(10, 2, 3).read()
        y = test_data["y"]
        predictions = experiment.model.transform(experiment.pipeline.transform(test_data))["y"]

        assert len(y) == len(predictions)


class Test_init:
    def test_assert_train_name(self):
        data_sources = {"test": MagicMock()}
        model = MagicMock()
        pipeline = MagicMock()
        evaluator = MagicMock()
        logger = MagicMock()
        serialiser = MagicMock()
        input_type_checker = MagicMock()
        output_type_checker = MagicMock()
        output_folder = MagicMock()
        additional_files_to_store = MagicMock()

        args = {}

        with pytest_assert(AssertionError, "Must provide a 'train' entry in the data_sources dictionary."):
            Experiment(
                data_sources=data_sources,
                model=model,
                pipeline=pipeline,
                evaluator=evaluator,
                logger=logger,
                serialiser=serialiser,
                input_type_checker=input_type_checker,
                output_type_checker=output_type_checker,
                output_folder=output_folder,
                additional_files_to_store=additional_files_to_store,
                parameters=args,
            )

    def test_warning_on_no_args(self):
        data_sources = {"train": MagicMock()}
        model = MagicMock()
        pipeline = MagicMock()
        evaluator = MagicMock()
        logger = MagicMock()
        serialiser = MagicMock()
        input_type_checker = MagicMock()
        output_type_checker = MagicMock()
        output_folder = MagicMock()
        additional_files_to_store = MagicMock()

        with patch("pype.base.experiment.experiment.getLogger") as mock_get_logger:
            Experiment(
                data_sources=data_sources,
                model=model,
                pipeline=pipeline,
                evaluator=evaluator,
                logger=logger,
                serialiser=serialiser,
                input_type_checker=input_type_checker,
                output_type_checker=output_type_checker,
                output_folder=output_folder,
                additional_files_to_store=additional_files_to_store,
            )

            mock_get_logger.assert_called_once()
            logger = mock_get_logger.return_value

            logger.warning.assert_called_once_with(
                """
It is highly recommended to provide the parameters used to initialise your
run here for logging purposes. Consider using the `from_command_line` or
`from_dictionary` initialisation methods
                """
            )


def test_from_dictionary():
    data_sources = MagicMock()
    model_class = MagicMock()
    pipeline = MagicMock()
    evaluator = MagicMock()
    logger = MagicMock()
    serialiser = MagicMock()
    input_type_checker = MagicMock()
    output_type_checker = MagicMock()
    model_inputs = MagicMock()
    model_outputs = MagicMock()
    output_folder = MagicMock()
    additional_files_to_store = MagicMock()
    seed = 3

    args = {"model__a": 1, "model__b": 3, "pipeline__step1__c": 3, "pipeline__step2__d": 5}

    with patch.object(Experiment, "__init__", return_value=None) as mock_init:
        result = Experiment.from_dictionary(
            data_sources=data_sources,
            model_class=model_class,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
            output_folder=output_folder,
            additional_files_to_store=additional_files_to_store,
            parameters=args,
            seed=seed,
        )

        assert isinstance(result, Experiment)
        model_class.assert_called_once_with(a=1, b=3, seed=seed, inputs=model_inputs, outputs=model_outputs)
        pipeline.reinitialise.assert_called_once_with({"step1__c": 3, "step2__d": 5})

        mock_init.assert_called_once_with(
            data_sources=data_sources,
            model=model_class.return_value,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            output_folder=output_folder,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            additional_files_to_store=additional_files_to_store,
            parameters=args,
        )


def test_from_command_line():
    class ParserOutput:
        def __init__(self, a, b) -> None:
            self.a = a
            self.b = b

    data_sources = MagicMock()
    model_class = MagicMock()
    pipeline = MagicMock()
    evaluator = MagicMock()
    logger = MagicMock()
    serialiser = MagicMock()
    input_type_checker = MagicMock()
    output_type_checker = MagicMock()
    model_inputs = MagicMock()
    model_outputs = MagicMock()
    output_folder = MagicMock()
    additional_files_to_store = MagicMock()
    seed = 3

    args = ParserOutput(2, 4)

    with patch.object(Experiment, "_get_cmd_args") as mock_get_cmd, patch.object(
        Experiment, "from_dictionary"
    ) as mock_from_dictionary:
        mock_parser = mock_get_cmd.return_value
        mock_parser.parse_known_args.return_value = (args, None)

        result = Experiment.from_command_line(
            data_sources=data_sources,
            model_class=model_class,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
            output_folder=output_folder,
            additional_files_to_store=additional_files_to_store,
            seed=seed,
        )

        assert result == mock_from_dictionary.return_value
        mock_get_cmd.assert_called_once_with(model_class, pipeline)
        mock_parser.parse_known_args.assert_called_once_with()

        mock_from_dictionary.assert_called_once_with(
            data_sources=data_sources,
            model_class=model_class,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
            output_folder=output_folder,
            additional_files_to_store=additional_files_to_store,
            parameters=args.__dict__,
            seed=seed,
        )


def test_get_cmd_args():
    with patch("pype.base.experiment.experiment.ArgumentParser") as mock_parser_class, patch(
        "pype.base.experiment.experiment.add_args_to_parser_for_pipeline"
    ) as mock_add_args:
        mock_parser = mock_parser_class.return_value
        mock_model_class = MagicMock()
        pipeline = MagicMock()

        result = Experiment._get_cmd_args(mock_model_class, pipeline)

        assert result == mock_parser
        mock_parser_class.assert_called_once_with()

        mock_model_class.get_parameters.assert_called_once_with(mock_parser)
        mock_add_args.assert_called_once_with(mock_parser, pipeline)
