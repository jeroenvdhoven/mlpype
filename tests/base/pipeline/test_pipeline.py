from typing import List
from unittest.mock import MagicMock, call, patch

from pytest import fixture, mark

from mlpype.base.data.dataset import DataSet
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from tests.utils import pytest_assert


class Test_Pipeline:
    @fixture
    def input_data(self):
        return DataSet(
            a=MagicMock(),
            b=MagicMock(),
        )

    @fixture
    def pipes(self):
        res = []
        for i in range(3):
            pipe = MagicMock(spec=Pipe)
            pipe.name = f"step{i}"
            res.append(pipe)
        return res

    @fixture
    def pipeline(self, pipes):
        return Pipeline(
            [
                pipes[0],
                pipes[1],
            ]
        )

    @fixture
    def pipeline_with_pipeline(self, pipeline, pipes):
        return Pipeline([pipeline, pipes[2]])

    def test_init_call_assert_all_names_different(self):
        with patch.object(Pipeline, "_assert_all_names_different") as mock_assert:
            Pipeline([])
            mock_assert.assert_called_once_with()

    def test_assert_all_names_different_success(self, pipeline: Pipeline):
        # assert no alert raised
        pipeline._assert_all_names_different()

    def test_assert_all_names_different_failure(self):
        name = "name"
        with pytest_assert(AssertionError, f"{name} is used multiple times."):
            pipeline = Pipeline(
                [
                    Pipe(name, MagicMock(), ["a", "b"], ["c"]),
                    Pipe(name, MagicMock(), ["c", "a"], ["d"]),
                ]
            )

    def test_assert_all_names_different_with_names(self, pipeline: Pipeline):
        name = "step1"
        with pytest_assert(AssertionError, f"{name} is used multiple times."):
            pipeline._assert_all_names_different(set([name]))

    def test_fit(self, pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]):
        result = pipeline.fit(input_data)

        assert result == pipeline

        pipes[0].fit.assert_called_once_with(input_data)
        pipes[0].transform.assert_called_once_with(input_data, is_inference=False)

        pipes[1].fit.assert_called_once_with(pipes[0].transform.return_value)
        pipes[1].transform.assert_called_once_with(pipes[0].transform.return_value, is_inference=False)

        pipes[2].fit.assert_not_called()
        pipes[2].transform.assert_not_called()

    def test_fit_with_pipeline(self, pipeline_with_pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]):
        result = pipeline_with_pipeline.fit(input_data)

        assert result == pipeline_with_pipeline

        # pipes 0 and 1 will get their transform called twice. Once during their pipeline's fit method,
        # once during the main pipeline's transform.
        pipes[0].fit.assert_called_once_with(input_data)
        pipes[0].transform.assert_has_calls(
            [call(input_data, is_inference=False), call(input_data, is_inference=False)]
        )

        pipes[1].fit.assert_called_once_with(pipes[0].transform.return_value)
        pipes[1].transform.assert_has_calls(
            [
                call(pipes[0].transform.return_value, is_inference=False),
                call(pipes[0].transform.return_value, is_inference=False),
            ]
        )

        # pipe 2's transform will only be called once.
        pipes[2].fit.assert_called_once_with(pipes[1].transform.return_value)
        pipes[2].transform.assert_called_once_with(pipes[1].transform.return_value, is_inference=False)

    def test_transform(self, pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]):
        pipeline.transform(input_data)

        pipes[0].transform.assert_called_once_with(input_data, is_inference=False)
        pipes[1].transform.assert_called_once_with(pipes[0].transform.return_value, is_inference=False)
        pipes[2].transform.assert_not_called()

    def test_transform_in_inference(self, pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]):
        pipeline.transform(input_data, is_inference=True)

        pipes[0].transform.assert_called_once_with(input_data, is_inference=True)
        pipes[1].transform.assert_called_once_with(pipes[0].transform.return_value, is_inference=True)
        pipes[2].transform.assert_not_called()

    def test_transform_with_pipeline(
        self, pipeline_with_pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]
    ):
        pipeline_with_pipeline.transform(input_data)

        pipes[0].transform.assert_called_once_with(input_data, is_inference=False)
        pipes[1].transform.assert_called_once_with(pipes[0].transform.return_value, is_inference=False)
        pipes[2].transform.assert_called_once_with(pipes[1].transform.return_value, is_inference=False)

    def test_inverse_transform(self, pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]):
        pipeline.inverse_transform(input_data)

        pipes[1].inverse_transform.assert_called_once_with(input_data, is_inference=False)
        pipes[0].inverse_transform.assert_called_once_with(pipes[1].inverse_transform.return_value, is_inference=False)

        pipes[2].inverse_transform.assert_not_called()

    def test_inverse_transform_in_inference(self, pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]):
        pipeline.inverse_transform(input_data, is_inference=True)

        pipes[1].inverse_transform.assert_called_once_with(input_data, is_inference=True)
        pipes[0].inverse_transform.assert_called_once_with(pipes[1].inverse_transform.return_value, is_inference=True)

        pipes[2].inverse_transform.assert_not_called()

    def test_inverse_transform_with_pipeline(
        self, pipeline_with_pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]
    ):
        pipeline_with_pipeline.inverse_transform(input_data)

        pipes[2].inverse_transform.assert_called_once_with(input_data, is_inference=False)
        pipes[1].inverse_transform.assert_called_once_with(pipes[2].inverse_transform.return_value, is_inference=False)
        pipes[0].inverse_transform.assert_called_once_with(pipes[1].inverse_transform.return_value, is_inference=False)

    def test_reinitialise(self, pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]):
        args = {
            "step0__a": 1,
            "step0__b": 2,
            "step1__c": 3,
            "step2__d": 4,  # ignored
        }
        pipeline.reinitialise(args)

        p0 = pipes[0]
        p0.reinitialise.assert_called_once_with(
            {
                "a": 1,
                "b": 2,
            }
        )

        p1 = pipes[1]
        p1.reinitialise.assert_called_once_with(
            {
                "c": 3,
            }
        )

    def test_reinitialise_with_pipeline(
        self, pipeline_with_pipeline: Pipeline, input_data: DataSet, pipes: List[MagicMock]
    ):
        args = {
            "step0__a": 1,
            "step0__b": 2,
            "step1__c": 3,
            "step2__d": 4,
        }
        pipeline_with_pipeline.reinitialise(args)

        p0 = pipes[0]
        p0.reinitialise.assert_called_once_with(
            {
                "a": 1,
                "b": 2,
            }
        )

        p1 = pipes[1]
        p1.reinitialise.assert_called_once_with(
            {
                "c": 3,
            }
        )
        p2 = pipes[2]
        p2.reinitialise.assert_called_once_with(
            {
                "d": 4,
            }
        )

    def test_copy(self):
        sub_pipe_1_op_class = MagicMock()
        sub_pipe_1 = Pipe("step0", sub_pipe_1_op_class, [], [])

        sub_pipe_2_op_class = MagicMock()
        sub_pipe_2 = Pipe("step1", sub_pipe_2_op_class, [], [])

        sub_pipe_pipeline_op_class = MagicMock()
        sub_pipe_pipeline = Pipe("step2", sub_pipe_pipeline_op_class, [], [])

        sub_pipeline = Pipeline([sub_pipe_pipeline])
        pipeline = Pipeline([sub_pipe_1, sub_pipe_2, sub_pipeline])

        args = {
            "step0__a": 1,
            "step0__b": 2,
            "step1__c": 3,
            "step2__d": 4,
        }
        result = pipeline.copy(args)

        expected_args = [{"a": 1, "b": 2}, {"c": 3}, {"d": 4}]
        expected_pipes = [sub_pipe_1, sub_pipe_2, sub_pipe_pipeline]
        for expected, actual, args in zip(expected_pipes, result.pipes, expected_args):
            assert isinstance(actual, Pipe)
            assert expected != actual

            assert actual.args == args
            assert actual.operator_class == expected.operator_class
            actual.operator_class.assert_called_with(**args)
            assert actual.name == expected.name
            assert actual.inputs == expected.inputs
            assert actual.outputs == expected.outputs
            assert actual.fit_inputs == expected.fit_inputs

    def test_copy_with_old_args(self):
        sub_pipe_1_op_class = MagicMock()
        sub_pipe_1 = Pipe("step0", sub_pipe_1_op_class, [], [], kw_args={"a": 10, "b": 9, "c": 4})

        sub_pipe_2_op_class = MagicMock()
        sub_pipe_2 = Pipe("step1", sub_pipe_2_op_class, [], [])

        sub_pipe_pipeline_op_class = MagicMock()
        sub_pipe_pipeline = Pipe("step2", sub_pipe_pipeline_op_class, [], [])

        sub_pipeline = Pipeline([sub_pipe_pipeline])
        pipeline = Pipeline([sub_pipe_1, sub_pipe_2, sub_pipeline])

        args = {
            "step0__a": 1,
            "step0__b": 2,
            "step1__c": 3,
            "step2__d": 4,
        }
        result = pipeline.copy(args)

        expected_args = [{"a": 1, "b": 2, "c": 4}, {"c": 3}, {"d": 4}]
        expected_pipes = [sub_pipe_1, sub_pipe_2, sub_pipe_pipeline]
        for expected, actual, args in zip(expected_pipes, result.pipes, expected_args):
            assert isinstance(actual, Pipe)
            assert expected != actual

            assert actual.args == args
            assert actual.operator_class == expected.operator_class
            actual.operator_class.assert_called_with(**args)
            assert actual.name == expected.name
            assert actual.inputs == expected.inputs
            assert actual.outputs == expected.outputs
            assert actual.fit_inputs == expected.fit_inputs

    def test_get_item_int(self, pipeline: Pipeline, pipes: List[MagicMock]):
        assert pipeline[0] == pipes[0]
        assert pipeline[1] == pipes[1]

    def test_get_item_str(self, pipeline: Pipeline, pipes: List[MagicMock]):
        assert pipeline["step0"] == pipes[0]
        assert pipeline["step1"] == pipes[1]

    def test_get_slice(self, pipeline: Pipeline, pipes: List[MagicMock]):
        result = pipeline[:1]
        assert isinstance(result, Pipeline)
        assert result.pipes == pipes[:1]

    def test_get_item_nested(self, pipeline_with_pipeline: Pipeline, pipeline: Pipeline, pipes: List[MagicMock]):
        assert pipeline_with_pipeline[0] == pipeline
        assert pipeline_with_pipeline["step0"] == pipes[0]
        assert pipeline_with_pipeline["step1"] == pipes[1]
        assert pipeline_with_pipeline["step2"] == pipes[2]

    def test_iter(self, pipeline: Pipeline, pipes: List[MagicMock]):
        for p_in_pipeline, pipe in zip(pipeline, pipes[:2]):
            assert pipe == p_in_pipeline

    def test_len(self, pipeline: Pipeline):
        assert len(pipeline) == 2

    def test_len_with_pipeline(self, pipeline_with_pipeline: Pipeline):
        assert len(pipeline_with_pipeline) == 3

    def test_add_pipe(self, pipeline: Pipeline, pipes: List[MagicMock]):
        result = pipeline + pipes[2]
        assert len(result) == 3
        assert result[2] == pipes[2]
        assert result != pipeline

    def test_add_pipeline(self, pipeline: Pipeline, pipes: List[MagicMock]):
        pipe_3 = MagicMock(spec=Pipe)
        pipe_3.name = "another step"
        pipeline_2 = Pipeline([pipes[2], pipe_3])

        result = pipeline + pipeline_2
        assert len(result) == 4
        assert result[2] == pipes[2]
        assert result[3] == pipe_3
        assert result != pipeline

    def test_add_error_on_same_name(self, pipeline: Pipeline):
        with pytest_assert(AssertionError, "step0 is used multiple times."):
            pipeline + pipeline

    def test_insert(self, pipeline: Pipeline):
        L = len(pipeline)
        pipe = MagicMock()

        pipeline.insert(1, pipe)

        assert len(pipeline) == L + 1
        assert pipeline[1] == pipe

    @mark.parametrize(
        ["name", "dummy_pipeline", "add_fit", "expected"],
        [
            ["simple", Pipeline([Pipe("1", MagicMock, inputs=["a"], outputs=["b"])]), False, ["a"]],
            [
                "nested",
                Pipeline(
                    [
                        Pipe("1", MagicMock, inputs=["a"], outputs=["b"]),
                        Pipeline(
                            [
                                Pipe("2", MagicMock, inputs=["b"], outputs=["c"]),
                                Pipe("3", MagicMock, inputs=["d"], outputs=["e"]),
                            ]
                        ),
                    ]
                ),
                False,
                ["a", "d"],
            ],
            [
                "self-referential",
                Pipeline(
                    [
                        Pipe("1", MagicMock, inputs=["a"], outputs=["a"]),
                        Pipe("2", MagicMock, inputs=["a"], outputs=["a"]),
                        Pipe("3", MagicMock, inputs=["a"], outputs=["a"]),
                        Pipe("4", MagicMock, inputs=["a"], outputs=["a"]),
                    ]
                ),
                False,
                ["a"],
            ],
            [
                "complex",
                Pipeline(
                    [
                        Pipe("1", MagicMock, inputs=["a"], outputs=["b"]),
                        Pipeline(
                            [
                                Pipe("2", MagicMock, inputs=["b"], outputs=["c"]),
                                Pipe("3", MagicMock, inputs=["d"], outputs=["e"]),
                                Pipe("4", MagicMock, inputs=["b", "d", "f"], outputs=["e", "g"]),
                            ]
                        ),
                        Pipe("5", MagicMock, inputs=["e"], outputs=["a"]),
                    ]
                ),
                False,
                ["a", "d", "f"],
            ],
            [
                "ignore-fit-nested",
                Pipeline(
                    [
                        Pipe("1", MagicMock, inputs=["a"], outputs=["b"], fit_inputs=["b"]),
                        Pipe("2", MagicMock, inputs=["b"], outputs=["c"], fit_inputs=["d"]),
                    ]
                ),
                False,
                ["a"],
            ],
            [
                "ignore-inference",
                Pipeline(
                    [
                        Pipe("1", MagicMock, inputs=["a"], outputs=["b"], fit_inputs=["b"]),
                        Pipe("2", MagicMock, inputs=["b"], outputs=["c"], fit_inputs=["d"], skip_on_inference=True),
                    ]
                ),
                False,
                ["a"],
            ],
            [
                "add-fit-nested",
                Pipeline(
                    [
                        Pipe("1", MagicMock, inputs=["a"], outputs=["b"], fit_inputs=["b"]),
                        Pipe("2", MagicMock, inputs=["b"], outputs=["c"], fit_inputs=["d"], skip_on_inference=True),
                    ]
                ),
                True,
                ["a", "b", "d"],
            ],
        ],
    )
    def test_get_input_datasets_names(self, name: str, dummy_pipeline: Pipeline, add_fit: bool, expected: List[str]):
        result = dummy_pipeline.get_input_datasets_names(add_fit)
        assert result == expected
