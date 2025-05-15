from pathlib import Path
from typing import Iterable, Tuple
from unittest.mock import MagicMock, call, patch

from pytest import fixture, mark

from mlpype.base.data import DataCatalog
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.pipeline.type_checker import TypeCheckerPipe
from tests.shared_fixtures import dummy_experiment, dummy_experiment_with_tcc
from tests.utils import get_dummy_data, pytest_assert

dummy_experiment
dummy_experiment_with_tcc


class Test_run:
    @fixture(scope="class")
    def run_experiment(self, dummy_experiment: Experiment) -> Iterable[Tuple[Experiment, dict]]:
        metrics = dummy_experiment.run()

        yield dummy_experiment, metrics

    @fixture(scope="class")
    def run_experiment_with_tcc(self, dummy_experiment_with_tcc: Experiment) -> Iterable[Tuple[Experiment, dict]]:
        metrics = dummy_experiment_with_tcc.run()

        yield dummy_experiment_with_tcc, metrics

    @mark.parametrize(["is_source", "train_source"], [[False, MagicMock()], [True, MagicMock(spec=DataCatalog)]])
    def test_unit(self, is_source: bool, train_source: MagicMock):
        data_sources = {"train": train_source, "test": MagicMock(spec=DataCatalog)}
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
        train_checked = MagicMock()
        test_checked = MagicMock()
        input_type_checker.transform.side_effect = [train_checked, test_checked]

        train_transform = MagicMock()
        test_transform = MagicMock()
        pipeline.transform.side_effect = [train_transform, test_transform]

        train_performance = MagicMock()
        test_performance = MagicMock()
        evaluator.evaluate.side_effect = [train_performance, test_performance]

        result = experiment.run()

        logger.__enter__.assert_called_once()
        logger.__exit__.assert_called_once()

        # getting data
        if is_source:
            data_sources["train"].read.assert_called_once_with()
        else:
            data_sources["train"].read.assert_not_called()
        data_sources["test"].read.assert_called_once_with()

        # input checker
        if is_source:
            dataset_train = data_sources["train"].read.return_value
        else:
            dataset_train = train_source

        dataset_test = data_sources["test"].read.return_value
        input_type_checker.fit.assert_called_once_with(dataset_train)
        input_type_checker.transform.assert_has_calls([call(dataset_train), call(dataset_test)])

        # pipeline fitting
        pipeline.fit.assert_called_once_with(train_checked)
        pipeline.transform.assert_has_calls(
            [call(train_checked, is_inference=False), call(test_checked, is_inference=False)]
        )

        # model fitting
        model.fit.assert_called_once_with(train_transform)
        model.transform.assert_called_once_with(train_transform)
        output_type_checker.fit.assert_called_once_with(model.transform.return_value)

        # evaluation
        evaluator.evaluate.assert_has_calls([call(model, train_transform), call(model, test_transform)])
        expected_metrics = {
            "train": train_performance,
            "test": test_performance,
        }

        # logging
        logger.log_run.assert_called_with(
            experiment, expected_metrics, {"train": train_transform, "test": test_transform}
        )
        assert result == expected_metrics

    def test_integration(self, run_experiment: Tuple[Experiment, dict]):
        experiment, metrics = run_experiment
        for ds_name in ["train", "test"]:
            assert ds_name in metrics
            assert "diff" in metrics[ds_name]

        test_data = get_dummy_data(10, 2, 3).read()
        y = test_data["y"]
        predictions = experiment.model.transform(experiment.pipeline.transform(test_data))["y"]

        assert len(y) == len(predictions)

    def test_integration_with_tcc_instantiation(self, run_experiment_with_tcc: Tuple[Experiment, dict]):
        experiment, metrics = run_experiment_with_tcc
        for ds_name in ["train", "test"]:
            assert ds_name in metrics
            assert "diff" in metrics[ds_name]

        test_data = get_dummy_data(10, 2, 3).read()
        y = test_data["y"]
        predictions = experiment.model.transform(experiment.pipeline.transform(test_data))["y"]

        assert len(y) == len(predictions)


class Test_init:
    def test_assert_tcc(self):
        data_sources = {"train": MagicMock()}
        model = MagicMock()
        pipeline = MagicMock()
        evaluator = MagicMock()
        logger = MagicMock()
        serialiser = MagicMock()
        output_folder = MagicMock()
        additional_files_to_store = MagicMock()

        args = {}

        with pytest_assert(
            ValueError,
            "Either type_checker_classes needs to be set or input_type_checker AND output_type_checker have to be set.",
        ):
            Experiment(
                data_sources=data_sources,
                model=model,
                pipeline=pipeline,
                evaluator=evaluator,
                logger=logger,
                serialiser=serialiser,
                output_folder=output_folder,
                additional_files_to_store=additional_files_to_store,
                parameters=args,
            )

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

    def test_instantiate_tcc(self):
        data_sources = {"train": MagicMock()}
        model = MagicMock()
        pipeline = MagicMock()
        evaluator = MagicMock()
        logger = MagicMock()
        serialiser = MagicMock()
        output_folder = MagicMock()
        tcc = [MagicMock(), MagicMock()]
        additional_files_to_store = MagicMock()

        exp = Experiment(
            data_sources=data_sources,
            model=model,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            type_checker_classes=tcc,
            output_folder=output_folder,
            additional_files_to_store=additional_files_to_store,
        )

        assert isinstance(exp.input_type_checker, TypeCheckerPipe)
        assert isinstance(exp.output_type_checker, TypeCheckerPipe)

        pipeline.get_input_datasets_names.assert_called_once_with()
        assert exp.input_type_checker.input_names == pipeline.get_input_datasets_names.return_value
        assert exp.output_type_checker.input_names == model.outputs

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

        with patch("mlpype.base.experiment.experiment.loguru_logger") as mock_logger:
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

            mock_logger.warning.assert_called_once_with(
                """It is highly recommended to provide the parameters used to initialise your
run here for logging purposes. Consider using the `from_command_line` or
`from_dictionary` initialisation methods"""
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
    type_checker_classes = MagicMock()
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
            type_checker_classes=type_checker_classes,
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
            type_checker_classes=type_checker_classes,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            additional_files_to_store=additional_files_to_store,
            parameters=args,
        )


def test_copy(dummy_experiment: Experiment):
    exp = dummy_experiment
    params = {"model__a": 4}

    result = exp.copy(params)

    assert result.model.a == 4
    assert result.model != exp.model
    assert result.model.__class__ == exp.model.__class__

    assert result.pipeline != exp.pipeline
    assert len(result.pipeline) == len(exp.pipeline)
    assert result.pipeline[0].name == exp.pipeline[0].name

    assert result.data_sources == exp.data_sources
    assert result.additional_files_to_store == exp.additional_files_to_store
    assert result.evaluator == exp.evaluator
    assert result.experiment_logger == exp.experiment_logger
    assert result.input_type_checker == exp.input_type_checker
    assert result.output_type_checker == exp.output_type_checker
    assert result.output_folder == exp.output_folder
    assert result.serialiser == exp.serialiser


def test_copy_with_predefined_parameters(dummy_experiment: Experiment):
    extra_params = {"model__b": 9, "model__a": 1, "pipeline__minus 1__c": 3}
    exp = dummy_experiment
    exp.parameters = extra_params

    params = {"model__a": 4}

    assert exp.model.b != extra_params["model__b"]
    assert exp.pipeline["minus 1"].operator.c != extra_params["pipeline__minus 1__c"]
    result = exp.copy(params)

    assert result.model.a != extra_params["model__a"]
    assert result.model.a == params["model__a"]
    assert result.model.b == extra_params["model__b"]
    assert result.pipeline["minus 1"].operator.c == extra_params["pipeline__minus 1__c"]
    assert result.model != exp.model
    assert result.model.__class__ == exp.model.__class__

    assert result.pipeline != exp.pipeline
    assert len(result.pipeline) == len(exp.pipeline)
    assert result.pipeline[0].name == exp.pipeline[0].name

    assert result.data_sources == exp.data_sources
    assert result.additional_files_to_store == exp.additional_files_to_store
    assert result.evaluator == exp.evaluator
    assert result.experiment_logger == exp.experiment_logger
    assert result.input_type_checker == exp.input_type_checker
    assert result.output_type_checker == exp.output_type_checker
    assert result.output_folder == exp.output_folder
    assert result.serialiser == exp.serialiser

    expected_params = extra_params.copy()
    expected_params.update(params)
    assert result.parameters == expected_params


@mark.parametrize(
    ["name", "fixed_args", "expected_dict_extras"],
    [
        ["no extras", None, {}],
        ["extras", {"model__8": 9}, {"model__8": 9}],
        ["overwrite", {"a": 120}, {"a": 120}],
    ],
)
def test_from_command_line(name, fixed_args, expected_dict_extras):
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
    type_checker_classes = MagicMock()
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

        expected_args = args.__dict__.copy()
        expected_args.update(expected_dict_extras)

        result = Experiment.from_command_line(
            data_sources=data_sources,
            model_class=model_class,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            type_checker_classes=type_checker_classes,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
            output_folder=output_folder,
            additional_files_to_store=additional_files_to_store,
            seed=seed,
            fixed_arguments=fixed_args,
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
            type_checker_classes=type_checker_classes,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
            output_folder=output_folder,
            additional_files_to_store=additional_files_to_store,
            parameters=expected_args,
            seed=seed,
        )


def test_get_cmd_args():
    with patch("mlpype.base.experiment.experiment.ArgumentParser") as mock_parser_class, patch(
        "mlpype.base.experiment.experiment.add_args_to_parser_for_pipeline"
    ) as mock_add_args:
        mock_parser = mock_parser_class.return_value
        mock_model_class = MagicMock()
        pipeline = MagicMock()

        result = Experiment._get_cmd_args(mock_model_class, pipeline)

        assert result == mock_parser
        mock_parser_class.assert_called_once_with()

        mock_model_class.get_parameters.assert_called_once_with(mock_parser)
        mock_add_args.assert_called_once_with(mock_parser, pipeline)
