from unittest.mock import MagicMock

from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.base.pipeline.pipe import Pipe
from tests.utils import pytest_assert


class TestPipe:
    @fixture
    def data(self):
        return DataSet(
            a=9,
            b=10,
            c=43,
            out=2,
            out2=5,
        )

    @fixture
    def operator(self):
        return MagicMock()

    @fixture
    def pipe(self, operator):
        return Pipe(
            "name",
            operator=operator,
            inputs=["a", "b", "c"],
            outputs=["out", "out2"],
        )

    def test_init(self, operator: MagicMock):
        Pipe(
            "name",
            operator=operator,
            inputs=["a", "b"],
            outputs=["out", "out2"],
            fit_inputs=["c"],
            kw_args={"a": 1, "b": "satr"},
        )
        operator.assert_called_once_with(a=1, b="satr")

    def test_init_name_check(self, operator: MagicMock):
        with pytest_assert(AssertionError, "Pipe names cannot contain the string `__`"):
            Pipe(
                "name__something",
                operator=operator,
                inputs=["a", "b"],
                outputs=["out", "out2"],
                fit_inputs=["c"],
                kw_args={"a": 1, "b": "satr"},
            )

    def test_fit(self, data: DataSet[int], pipe: Pipe, operator: MagicMock):
        mock_operator = operator.return_value
        result = pipe.fit(data)

        assert result == pipe
        mock_operator.fit.assert_called_once_with(9, 10, 43)

    def test_fit_with_extra_args(self, data: DataSet[int], operator: MagicMock):
        pipe = Pipe("name", operator=operator, inputs=["a", "b"], outputs=["out", "out2"], fit_inputs=["c"])
        mock_operator = operator.return_value
        result = pipe.fit(data)

        assert result == pipe
        mock_operator.fit.assert_called_once_with(9, 10, 43)

    def test_transform(self, data: DataSet[int], operator: MagicMock):
        pipe = Pipe("name", operator=operator, inputs=["a", "b"], outputs=["out", "out2"], fit_inputs=["c"])
        mock_operator = operator.return_value
        mock_operator.transform.return_value = [1, 2]
        result = pipe.transform(data)

        input_backup = data.copy()
        expected = DataSet(
            a=9,
            b=10,
            c=43,
            out=1,
            out2=2,
        )
        assert result == expected
        assert input_backup == data
        mock_operator.transform.assert_called_once_with(9, 10)

    def test_transform_with_inference_skip(self, data: DataSet[int], operator: MagicMock):
        pipe = Pipe(
            "name",
            operator=operator,
            inputs=["a", "b"],
            outputs=["out", "out2"],
            fit_inputs=["c"],
            skip_on_inference=True,
        )
        mock_operator = operator.return_value

        result = pipe.transform(data, is_inference=True)

        assert result == data
        mock_operator.transform.assert_not_called()

    def test_transform_with_extra_args(self, data: DataSet[int], pipe: Pipe, operator: MagicMock):
        mock_operator = operator.return_value
        mock_operator.transform.return_value = [1, 2]
        result = pipe.transform(data)

        input_backup = data.copy()
        expected = DataSet(
            a=9,
            b=10,
            c=43,
            out=1,
            out2=2,
        )
        assert result == expected
        assert input_backup == data
        mock_operator.transform.assert_called_once_with(9, 10, 43)

    def test_inverse_transform(self, data: DataSet[int], pipe: Pipe, operator: MagicMock):
        mock_operator = operator.return_value
        mock_operator.inverse_transform.return_value = [1, 2, 0]
        result = pipe.inverse_transform(data)

        expected = DataSet(
            a=1,
            b=2,
            c=0,
            out=2,
            out2=5,
        )
        assert result == expected
        mock_operator.inverse_transform.assert_called_once_with(2, 5)

    def test_inverse_transform_with_inference_skip(self, data: DataSet[int], operator: MagicMock):
        pipe = Pipe(
            "name",
            operator=operator,
            inputs=["a", "b"],
            outputs=["out", "out2"],
            fit_inputs=["c"],
            skip_on_inference=True,
        )
        mock_operator = operator.return_value

        result = pipe.inverse_transform(data, is_inference=True)

        assert result == data
        mock_operator.inverse_transform.assert_not_called()

    def test_inverse_transform_passed(self, pipe: Pipe, operator: MagicMock):
        mock_operator = operator.return_value
        data = DataSet(
            a=9,
            b=10,
            c=43,
            out2=5,  # missing `out`
        )

        result = pipe.inverse_transform(data)
        assert result == data
        mock_operator.inverse_transform.assert_not_called()

    def test_reinitialise(self, pipe: Pipe, operator: MagicMock):
        args = {"a": 3, "b": 2}

        mock_class = MagicMock()
        pipe.operator_class = mock_class
        pipe.reinitialise(args)

        assert pipe.operator != operator
        assert pipe.operator == mock_class.return_value
        mock_class.assert_called_once_with(a=3, b=2)

    def test_copy(self):
        operator = MagicMock()
        pipe = Pipe(
            name="name",
            operator=operator,
            inputs=["a", "b"],
            outputs=["out", "out2"],
            fit_inputs=["c"],
            kw_args={"a": 1, "b": "satr"},
            skip_on_inference=True,
        )

        # instantiating the operator generates different objects
        operator.side_effect = [1, 2]
        result = pipe.copy()

        assert result.args == pipe.args
        assert result.operator_class == pipe.operator_class
        assert result.name == pipe.name
        assert result.inputs == pipe.inputs
        assert result.outputs == pipe.outputs
        assert result.fit_inputs == pipe.fit_inputs
        assert result.skip_on_inference == pipe.skip_on_inference

        # the object is not completely the same.
        assert result.operator != pipe.operator

    def test_copy_with_args(self, pipe: Pipe, operator: MagicMock):
        # instantiating the operator generates different objects
        operator.side_effect = [1, 2]
        args = {"a": 3}
        result = pipe.copy(args)

        assert result.args != pipe.args
        assert result.args == args
        assert result.operator_class == pipe.operator_class
        assert result.name == pipe.name
        assert result.inputs == pipe.inputs
        assert result.outputs == pipe.outputs
        assert result.fit_inputs == pipe.fit_inputs

        # the object is not completely the same.
        assert result.operator != pipe.operator
        result.operator_class.assert_called_with(a=3)
