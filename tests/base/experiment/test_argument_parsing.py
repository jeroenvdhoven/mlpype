import inspect
import sys
from argparse import ArgumentParser
from typing import Callable, Iterable, List, Tuple
from unittest.mock import MagicMock, call, patch

from pytest import mark

from mlpype.base.experiment.argument_parsing import (
    _convert_bool,
    _get_conversion_function,
    _parse_docs_to_type_args,
    _parse_type_name,
    add_args_to_parser_for_class,
    add_args_to_parser_for_function,
    add_args_to_parser_for_pipeline,
    add_argument,
)
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from tests.utils import pytest_assert


class Test_convert_bool:
    @mark.parametrize(
        ["inputs", "expected"],
        [
            ["1", True],
            ["true", True],
            ["0", False],
            ["321s", False],
        ],
    )
    def test(self, inputs, expected):
        result = _convert_bool(inputs)
        assert expected == result


class Test_get_conversion_function:
    def test_bool(self):
        result = _get_conversion_function(bool)
        assert result == _convert_bool

    @mark.parametrize(
        ["inputs"],
        [
            [str],
            [int],
            [MagicMock],
        ],
    )
    def test_other(self, inputs):
        result = _get_conversion_function(inputs)
        assert inputs == result


class Test_add_argument:
    @mark.parametrize(["class_"], [[int], [str], [float]])
    def test_normal_run(self, class_: type):
        parser = MagicMock()
        name = "arg_name"
        prefix = "prefix"
        add_argument(parser, name, prefix, class_, is_required=True, default=None)

        parser.add_argument.assert_called_once_with(f"--{prefix}__{name}", type=class_, required=True, default=None)

    def test_boolean(self):
        parser = MagicMock()
        name = "arg_name"
        prefix = "prefix"
        class_ = bool
        add_argument(parser, name, prefix, class_, is_required=True, default=None)

        parser.add_argument.assert_called_once_with(
            f"--{prefix}__{name}", type=_convert_bool, required=True, default=None
        )

    def test_default(self):
        parser = MagicMock()
        name = "arg_name"
        prefix = "prefix"
        class_ = int
        default = "some value"
        add_argument(parser, name, prefix, class_, is_required=False, default=default)

        parser.add_argument.assert_called_once_with(f"--{prefix}__{name}", type=class_, required=False, default=default)

    @mark.parametrize(["name"], [["cls"], ["self"]])
    def test_excluded_defaults(self, name: str):
        parser = MagicMock()
        prefix = "prefix"
        class_ = int
        default = "some value"

        add_argument(parser, name, prefix, class_, is_required=False, default=default)
        parser.add_argument.assert_not_called()

        # check that normally this would work
        alternative_name = "other name"
        add_argument(parser, alternative_name, prefix, class_, is_required=False, default=default)
        parser.add_argument.assert_called_once_with(
            f"--{prefix}__{alternative_name}", type=class_, required=False, default=default
        )

    def test_excluded_extra(self):
        parser = MagicMock()
        name = "excluded"
        prefix = "prefix"
        class_ = int
        default = "some value"

        add_argument(parser, name, prefix, class_, is_required=False, default=default, excluded=[name])
        parser.add_argument.assert_not_called()

        # check that normally this would work
        alternative_name = "other name"
        add_argument(parser, alternative_name, prefix, class_, is_required=False, default=default)
        parser.add_argument.assert_called_once_with(
            f"--{prefix}__{alternative_name}", type=class_, required=False, default=default
        )

    def test_not_required_check_default_value(self):
        parser = MagicMock()
        name = "arg_name"
        prefix = "prefix"
        class_ = int

        with pytest_assert(AssertionError, "A default value must be provided if the argument is not required"):
            add_argument(parser, name, prefix, class_, is_required=False, default=inspect._empty)

    def test_none_class(self):
        parser = MagicMock()
        name = "arg_name"
        prefix = "prefix"
        class_ = None
        default = "some value"

        add_argument(parser, name, prefix, class_, is_required=False, default=default)
        parser.add_argument.assert_not_called()

        # check that normally this would work
        class_alt = int
        add_argument(parser, name, prefix, class_alt, is_required=False, default=default)
        parser.add_argument.assert_called_once_with(
            f"--{prefix}__{name}", type=class_alt, required=False, default=default
        )

    @mark.parametrize(
        ["class_", "conversion"],
        [
            [List[int], int],
            [Tuple[float], float],
            [Tuple[float], float],
            [Iterable[str], str],
            [List[bool], _convert_bool],
        ],
    )
    def test_list_classes(self, class_: type, conversion: Callable):
        parser = MagicMock()
        name = "arg_name"
        prefix = "prefix"
        default = "some value"

        add_argument(parser, name, prefix, class_, is_required=False, default=default)
        parser.add_argument.assert_called_once_with(
            f"--{prefix}__{name}", type=conversion, required=False, default=default, nargs="+"
        )

    def test_warning(self):
        parser = MagicMock()
        name = "arg_name"
        prefix = "prefix"
        default = "some value"
        class_ = MagicMock

        with patch("mlpype.base.experiment.argument_parsing.warnings.warn") as mock_warn:
            add_argument(parser, name, prefix, class_, is_required=False, default=default)
        parser.add_argument.assert_not_called()
        mock_warn.assert_called_once_with(
            f"Currently the class {str(class_)} is not supported for automatic command line importing."
        )

    def test_integration(self):
        arguments = sys.argv.copy()

        name = "arg_0"
        prefix = "model"
        class_ = float

        try:
            sys.argv = ["0", "--model__arg_0", "9.2"]
            parser = ArgumentParser()
            add_argument(parser, name, prefix, class_, is_required=True, default=inspect._empty)

            parsed = parser.parse_args()
            assert parsed.model__arg_0 == 9.2
        finally:
            sys.argv = arguments


class Test_parse_type_name:
    @mark.parametrize(
        ["inputs", "expected"],
        [
            # simple parsing
            ["int: 8", int],
            ["float: 8", float],
            ["str: 8", str],
            ["bool: 8", bool],
            ["this is int: 8", int],
            ["this is float: 8", float],
            ["this is str: 8", str],
            ["this is bool: 8", bool],
            # capitalized
            ["this is Int: 8", int],
            ["this is Float: 8", float],
            ["this is Str: 8", str],
            ["this is Bool: 8", bool],
            # do not use parts of words!
            ["Complex: intelligence", None],
            ["we ignore strings", None],
            ["floating rocks", None],
            ["boolean values, also ignored", None],
            ["a random assortment of words", None],
        ],
    )
    def test(self, inputs: str, expected: type):
        result = _parse_type_name(inputs)
        assert expected == result

    def test_extra_mappings(self):
        s = "some random string that doesnt normally map"
        result = _parse_type_name(s)
        assert result is None

        mapper = {r"map": float}
        result_2 = _parse_type_name(s, mapper)
        assert result_2 == float


class Test_parse_docs_to_type_args:
    def example(self, I, F, S, C):
        """example!

        Args:
            I (int): int
            F (float): float
            S (str): string
            C (List[MagicMock, complex]): aaaa
        """

    def test(self):
        result = _parse_docs_to_type_args(self.example)

        exepected = {"I": int, "F": float, "S": str}
        assert exepected == result

    def test_extra_mappings(self):
        result = _parse_docs_to_type_args(self.example, {"MagicMock": MagicMock})

        exepected = {"I": int, "F": float, "S": str, "C": MagicMock}
        assert exepected == result

    def test_keep_unknown_type_args(self):
        result = _parse_docs_to_type_args(self.example, include_none_args=True)

        exepected = {"I": int, "F": float, "S": str, "C": None}
        assert exepected == result

    def test_no_doc(self):
        result = _parse_docs_to_type_args(self.test_no_doc)
        assert {} == result

    def test_ignore_attributes_through_parameters(self):
        def func(a, b):
            """Test.

            Short text.

            Parameters
            ----------
            a : bool, default=True
                Something text

            b : int
                More description

            Attributes
            ----------
            coef_: float
                Hey a coefficient
            """

        result = _parse_docs_to_type_args(func)
        assert {"a": bool, "b": int} == result

    def test_sklearn_example(self):
        def func(fit_intercept, copy_X, normalize, n_jobs, positive):
            """Ordinary least squares Linear Regression.

            LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
            to minimize the residual sum of squares between the observed targets in
            the dataset, and the targets predicted by the linear approximation.

            Parameters
            ----------
            fit_intercept : bool, default=True
                Whether to calculate the intercept for this model. If set
                to False, no intercept will be used in calculations
                (i.e. data is expected to be centered).

            normalize : bool, default=False
                This parameter is ignored when ``fit_intercept`` is set to False.
                If True, the regressors X will be normalized before regression by
                subtracting the mean and dividing by the l2-norm.
                If you wish to standardize, please use
                :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
                on an estimator with ``normalize=False``.

                .. deprecated:: 1.0
                   `normalize` was deprecated in version 1.0 and will be
                   removed in 1.2.

            copy_X : bool, default=True
                If True, X will be copied; else, it may be overwritten.

            n_jobs : int, default=None
                The number of jobs to use for the computation. This will only provide
                speedup in case of sufficiently large problems, that is if firstly
                `n_targets > 1` and secondly `X` is sparse or if `positive` is set
                to `True`. ``None`` means 1 unless in a
                :obj:`joblib.parallel_backend` context. ``-1`` means using all
                processors. See :term:`Glossary <n_jobs>` for more details.

            positive : bool, default=False
                When set to ``True``, forces the coefficients to be positive. This
                option is only supported for dense arrays.

                .. versionadded:: 0.24

            Attributes
            ----------
            coef_ : array of shape (n_features, ) or (n_targets, n_features)
                Estimated coefficients for the linear regression problem.
                If multiple targets are passed during the fit (y 2D), this
                is a 2D array of shape (n_targets, n_features), while if only
                one target is passed, this is a 1D array of length n_features.

            rank_ : int
                Rank of matrix `X`. Only available when `X` is dense.

            singular_ : array of shape (min(X, y),)
                Singular values of `X`. Only available when `X` is dense.

            intercept_ : float or array of shape (n_targets,)
                Independent term in the linear model. Set to 0.0 if
                `fit_intercept = False`.

            n_features_in_ : int
                Number of features seen during :term:`fit`.

                .. versionadded:: 0.24

            feature_names_in_ : ndarray of shape (`n_features_in_`,)
                Names of features seen during :term:`fit`. Defined only when `X`
                has feature names that are all strings.

                .. versionadded:: 1.0

            See Also
            --------
            Ridge : Ridge regression addresses some of the
                problems of Ordinary Least Squares by imposing a penalty on the
                size of the coefficients with l2 regularization.
            Lasso : The Lasso is a linear model that estimates
                sparse coefficients with l1 regularization.
            ElasticNet : Elastic-Net is a linear regression
                model trained with both l1 and l2 -norm regularization of the
                coefficients.

            Notes
            -----
            From the implementation point of view, this is just plain Ordinary
            Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
            (scipy.optimize.nnls) wrapped as a predictor object.

            Examples
            --------
            >>> import numpy as np
            >>> from sklearn.linear_model import LinearRegression
            >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
            >>> # y = 1 * x_0 + 2 * x_1 + 3
            >>> y = np.dot(X, np.array([1, 2])) + 3
            >>> reg = LinearRegression().fit(X, y)
            >>> reg.score(X, y)
            1.0
            >>> reg.coef_
            array([1., 2.])
            >>> reg.intercept_
            3.0...
            >>> reg.predict(np.array([[3, 5]]))
            array([16.])
            """

        # dummy function to use sklearn docsstring parsing
        result = _parse_docs_to_type_args(func)
        expected = {
            "fit_intercept": bool,
            "normalize": bool,
            "copy_X": bool,
            "n_jobs": int,
            "positive": bool,
        }
        assert expected == result


class Test_add_args_to_parser_for_function:
    def func_1(self, a: int, c, b: float = 1.0, d=8):
        pass

    def test(self):
        parser = MagicMock()
        prefix = "prefix"
        exc = ["a different one"]

        with patch("mlpype.base.experiment.argument_parsing.add_argument") as mock_add:
            add_args_to_parser_for_function(parser, self.func_1, prefix=prefix, excluded=exc)

            mock_add.assert_has_calls(
                [
                    call(parser, "a", prefix, int, True, inspect._empty, exc),
                    call(parser, "b", prefix, float, False, 1.0, exc),
                    call(parser, "c", prefix, inspect._empty, True, inspect._empty, exc),
                    call(parser, "d", prefix, inspect._empty, False, 8, exc),
                ],
                any_order=True,
            )

    def test_docstring(self):
        parser = MagicMock()
        prefix = "prefix"
        exc = ["a different one"]
        cda = {"c": str, "d": int}

        with patch("mlpype.base.experiment.argument_parsing.add_argument") as mock_add:
            add_args_to_parser_for_function(parser, self.func_1, prefix=prefix, excluded=exc, class_docstring_args=cda)

            mock_add.assert_has_calls(
                [
                    call(parser, "a", prefix, int, True, inspect._empty, exc),
                    call(parser, "b", prefix, float, False, 1.0, exc),
                    call(parser, "c", prefix, str, True, inspect._empty, exc),
                    call(parser, "d", prefix, int, False, 8, exc),
                ],
                any_order=True,
            )


class Test_add_args_to_parser_for_class:
    class A:
        def __init__(self, a: int, b: float, c, d=9) -> None:
            pass

    class B(A):
        def __init__(self, a: int) -> None:
            pass

    class C(A):
        def __init__(self, **kwargs) -> None:
            pass

    def test(self):
        parser = MagicMock()
        prefix = "prefix"

        with patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_function"
        ) as mock_add_function, patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_class"
        ) as mock_add_class, patch(
            "mlpype.base.experiment.argument_parsing._parse_docs_to_type_args"
        ) as mock_parse:
            add_args_to_parser_for_class(parser, self.A, prefix=prefix, excluded_superclasses=[])

            mock_add_function.assert_called_once_with(
                parser, self.A.__init__, prefix, None, class_docstring_args=mock_parse.return_value
            )
            mock_add_class.assert_not_called()

    def test_class_docstring(self):
        parser = MagicMock()
        prefix = "prefix"

        with patch("mlpype.base.experiment.argument_parsing.add_args_to_parser_for_function"), patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_class"
        ), patch(
            "mlpype.base.experiment.argument_parsing._parse_docs_to_type_args", return_value={"C": float}
        ) as mock_parse:
            add_args_to_parser_for_class(parser, self.A, prefix=prefix, excluded_superclasses=[])

            mock_parse.assert_called_once_with(self.A)

    def test_class_docstring_init_alternative(self):
        parser = MagicMock()
        prefix = "prefix"

        with patch("mlpype.base.experiment.argument_parsing.add_args_to_parser_for_function"), patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_class"
        ), patch("mlpype.base.experiment.argument_parsing._parse_docs_to_type_args", return_value={}) as mock_parse:
            add_args_to_parser_for_class(parser, self.A, prefix=prefix, excluded_superclasses=[])

            mock_parse.assert_has_calls([call(self.A), call(self.A.__init__)])

    def test_superclass_not_called(self):
        parser = MagicMock()
        prefix = "prefix"

        with patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_function"
        ) as mock_add_function, patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_class"
        ) as mock_add_class, patch(
            "mlpype.base.experiment.argument_parsing._parse_docs_to_type_args"
        ) as mock_parse:
            add_args_to_parser_for_class(parser, self.B, prefix=prefix, excluded_superclasses=[])
            mock_add_class.assert_not_called()

    def test_superclass_excluded(self):
        parser = MagicMock()
        prefix = "prefix"

        with patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_function"
        ) as mock_add_function, patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_class"
        ) as mock_add_class, patch(
            "mlpype.base.experiment.argument_parsing._parse_docs_to_type_args"
        ) as mock_parse:
            add_args_to_parser_for_class(parser, self.C, prefix=prefix, excluded_superclasses=[self.A])
            mock_add_class.assert_not_called()

    def test_superclass_called(self):
        parser = MagicMock()
        prefix = "prefix"

        with patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_function"
        ) as mock_add_function, patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_class"
        ) as mock_add_class, patch(
            "mlpype.base.experiment.argument_parsing._parse_docs_to_type_args"
        ) as mock_parse:
            add_args_to_parser_for_class(parser, self.C, prefix=prefix, excluded_superclasses=[])
            mock_add_class.assert_called_once_with(parser, self.A, prefix, [], None)


class Test_add_args_to_parser_for_pipeline:
    def test(self):
        parser = MagicMock()
        pipeline = Pipeline(
            [
                Pipe("1", MagicMock, [], []),
                Pipe("2", MagicMock, [], []),
                Pipeline(
                    [
                        Pipe("3", MagicMock, [], []),
                    ]
                ),
            ]
        )

        with patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_pipeline"
        ) as mock_add_pipeline, patch(
            "mlpype.base.experiment.argument_parsing.add_args_to_parser_for_class"
        ) as mock_add_class:
            add_args_to_parser_for_pipeline(parser, pipeline)

            mock_add_pipeline.assert_called_once_with(parser, pipeline[2])
            mock_add_class.assert_has_calls(
                [
                    call(parser, pipeline[0].operator_class, f"pipeline__1", []),
                    call(parser, pipeline[1].operator_class, f"pipeline__2", []),
                ]
            )
