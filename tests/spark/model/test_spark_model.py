from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pytest import mark

from mlpype.base.data import DataSet
from mlpype.spark.model import LinearSparkModel, SparkModel
from tests.spark.utils import spark_session
from tests.utils import pytest_assert

spark_session


class Test_SparkModel:
    # Tested using LinearSparkModel for initialisation purposes

    @mark.parametrize(
        ["inputs", "outputs"],
        [
            [[], ["x"]],
            [["x"], []],
            [["1", "2"], ["x"]],
            [["x"], ["1", "2"]],
            [["x"], ["y"]],
        ],
    )
    def test_init_assert(self, inputs: List[str], outputs: List[str]):
        with pytest_assert(
            AssertionError, "SparkML only requires a single DataFrame as input and output, the same one."
        ):
            LinearSparkModel(
                inputs=inputs,
                outputs=outputs,
            )

    def test_init(self):
        with patch.object(LinearSparkModel, "_init_model") as mock_init_model:
            LinearSparkModel(inputs=["x"], outputs=["x"], arg1=1, arg2=4)

            mock_init_model.assert_called_once_with({"arg1": 1, "arg2": 4})

    def test_init_with_predictor(self):
        with patch.object(LinearSparkModel, "_init_model") as mock_init_model:
            LinearSparkModel(inputs=["x"], outputs=["x"], predictor=MagicMock())

            mock_init_model.assert_not_called()

    def test_get_annotated_class(self):
        model = LinearSparkModel(inputs=["x"], outputs=["x"], predictor=MagicMock())

        result = model._get_annotated_class()

        assert result == LinearRegression

    def test_fit(self):
        model = LinearSparkModel(inputs=["x"], outputs=["x"], predictor=MagicMock())
        x = MagicMock()
        data = DataSet(x=x, y=MagicMock())

        with patch.object(model, "_fit") as mock_fit:
            result = model.fit(data)

        assert result == model
        mock_fit.assert_called_once_with(x)

    def test__fit_assertions(self):
        model = LinearSparkModel(inputs=["x"], outputs=["x"], predictor=MagicMock())
        x = MagicMock()
        y = MagicMock()

        with pytest_assert(AssertionError, "SparkML needs a single DataFrame as input, got 2"):
            model._fit(x, y)

    def test__fit(self):
        predictor = MagicMock()
        model = LinearSparkModel(inputs=["x"], outputs=["x"], predictor=predictor)
        x = MagicMock()

        model._fit(x)

        predictor.fit.assert_called_once_with(x)
        assert model.model == predictor.fit.return_value

    def test_transform_assertions_excess_data(self):
        model = LinearSparkModel(inputs=["x"], outputs=["x"], predictor=MagicMock())
        x = MagicMock()
        y = MagicMock()

        with pytest_assert(AssertionError, "SparkML needs a single DataFrame as input, got 2"):
            model._transform(x, y)

    def test_transform_assertions_unfitted(self):
        model = LinearSparkModel(inputs=["x"], outputs=["x"], predictor=MagicMock())
        x = MagicMock()

        with pytest_assert(AssertionError, "Please fit this model before transforming data."):
            model._transform(x)

    def test_transform(self):
        predictor = MagicMock()
        model = LinearSparkModel(
            inputs=["x"],
            outputs=["x"],
            predictor=predictor,
        )
        fcol = "features"
        x = MagicMock()
        data = DataSet(x=x, y=MagicMock())
        pred_model = predictor.fit.return_value
        pred_model.getOrDefault.return_value = fcol

        model.fit(data)
        result = model.transform(data)

        pred_model.transform.assert_called_once_with(x)

        predictions = pred_model.transform.return_value
        pred_model.getOrDefault.assert_called_once_with("featuresCol")
        predictions.drop.assert_called_once_with(fcol)
        assert result["x"] == predictions.drop.return_value

    def test_create_and_unzip(self):
        model = LinearSparkModel(inputs=["x"], outputs=["x"], predictor=MagicMock())
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            zipdir = tmp_dir / "inputs"
            zipdir.mkdir()
            files = ["f1", "f2"]
            for f in files:
                with open(zipdir / f, "w") as f_write:
                    f_write.write(f)

            model._create_zip(zipdir)
            assert (tmp_dir / "inputs.tar").is_file()
            assert not zipdir.is_dir()

            SparkModel._unzip(tmp_dir / "inputs.tar")
            assert zipdir.is_dir()

            for f in files:
                assert (zipdir / f).is_file()
                with open(zipdir / f, "r") as f_read:
                    assert f_read.read() == f

    @mark.parametrize(
        ["path", "expected"],
        [
            [Path("/absolute/path"), "file://"],
            [Path("relative/path"), ""],
        ],
    )
    def test_get_spark_prefix(self, path, expected):
        assert expected == SparkModel._get_spark_prefix(path)

    def test_save_assertions(self):
        model = LinearSparkModel(
            inputs=["x"],
            outputs=["x"],
            predictor=MagicMock(),
        )

        with pytest_assert(
            AssertionError, "Please fit this model before transforming data."
        ), TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)

    @mark.parametrize("use_absolute_path", [False, True])
    def test_save(self, use_absolute_path: bool):
        tmp_dir = "/tmp/" if use_absolute_path else "./tmp"
        predictor = MagicMock()
        model = MagicMock()
        mlpype_model = LinearSparkModel(
            inputs=["x"],
            outputs=["x"],
            predictor=predictor,
            model=model,
            output_col=None,
        )

        with patch("mlpype.spark.model.spark_model.JoblibSerialiser.serialise") as mock_serialise, patch.object(
            LinearSparkModel, "_create_zip"
        ) as mock_create_zip:
            tmp_dir = Path(tmp_dir)
            mlpype_model._save(tmp_dir)

            mock_create_zip.assert_has_calls(
                [
                    call(tmp_dir / LinearSparkModel.SPARK_PREDICTOR_PATH),
                    call(tmp_dir / LinearSparkModel.SPARK_MODEL_PATH),
                ]
            )
            spark_prefix = "file://" if use_absolute_path else ""

            model.write.assert_called_once_with()
            model.write.return_value.overwrite.assert_called_once_with()
            model.write.return_value.overwrite.return_value.save.assert_called_once_with(
                spark_prefix + str(tmp_dir / mlpype_model.SPARK_MODEL_PATH)
            )
            predictor.write.assert_called_once_with()
            predictor.write.return_value.overwrite.assert_called_once_with()
            predictor.write.return_value.overwrite.return_value.save.assert_called_once_with(
                spark_prefix + str(tmp_dir / mlpype_model.SPARK_PREDICTOR_PATH)
            )
            mock_serialise.assert_called_once_with(type(model), str(tmp_dir / mlpype_model.SPARK_MODEL_CLASS_PATH))

    @mark.parametrize("use_absolute_path", [False, True])
    def test_load(self, use_absolute_path: bool):
        tmp_dir = "/tmp/" if use_absolute_path else "./tmp"
        with patch.object(LinearSparkModel, "_get_annotated_class") as mock_annotate, patch(
            "mlpype.spark.model.spark_model.JoblibSerialiser.deserialise"
        ) as mock_deserialise, patch.object(LinearSparkModel, "_unzip") as mock_unzip:
            tmp_dir = Path(tmp_dir)

            result = LinearSparkModel._load(tmp_dir, ["x"], ["x"])

            mock_unzip.assert_has_calls(
                [
                    call(tmp_dir / f"{LinearSparkModel.SPARK_PREDICTOR_PATH}.tar"),
                    call(tmp_dir / f"{LinearSparkModel.SPARK_MODEL_PATH}.tar"),
                ]
            )
            spark_prefix = "file://" if use_absolute_path else ""

            mock_annotate.assert_called_once_with()
            mock_pred_class = mock_annotate.return_value
            mock_pred_class.load.assert_called_once_with(
                spark_prefix + str(tmp_dir / LinearSparkModel.SPARK_PREDICTOR_PATH)
            )

            mock_deserialise.assert_called_once_with(str(tmp_dir / LinearSparkModel.SPARK_MODEL_CLASS_PATH))
            mock_model_class = mock_deserialise.return_value
            mock_model_class.assert_called_once_with()
            mock_model_class.return_value.load.assert_called_once_with(
                spark_prefix + str(tmp_dir / LinearSparkModel.SPARK_MODEL_PATH)
            )

            assert result.inputs == ["x"]
            assert result.outputs == ["x"]
            assert result.predictor == mock_pred_class.load.return_value
            assert result.model == mock_model_class.return_value.load.return_value

    def test_get_parameters(self):
        parser = MagicMock()

        with patch("mlpype.spark.model.spark_model.add_args_to_parser_for_class") as mock_add:
            LinearSparkModel.get_parameters(parser)

        mock_add.assert_called_once_with(
            parser, LinearRegression, "model", [], excluded_args=["seed", "inputs", "outputs", "model"]
        )

    def test_integration(self, spark_session: SparkSession):
        df = pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                    3,
                    4,
                ],
                "y": [9, 4, 6, 1],
                "response": [3, 5, 6, 7],
            }
        )
        spark_df = spark_session.createDataFrame(df)
        spark_df = VectorAssembler(inputCols=["x", "y"], outputCol="vector").transform(spark_df)

        dataset = DataSet(df=spark_df)

        model = LinearSparkModel(
            inputs=["df"],
            outputs=["df"],
            predictionCol="prediction_col",
            featuresCol="vector",
            labelCol="response",
            fitIntercept=False,
        )

        model.fit(dataset)
        transformed = model.transform(dataset)

        assert "df" in transformed

        transformed_df = transformed["df"].toPandas()
        for col in ["prediction_col", "response", "x", "y"]:
            assert col in transformed_df.columns
        assert "vector" not in transformed_df.columns

        assert transformed_df.shape == (df.shape[0], 4)
        assert np.all(~transformed_df.isna())

        # saving/loading
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            model.save(tmp_dir)
            loaded_model = LinearSparkModel.load(tmp_dir)
            loaded_preds = loaded_model.transform(dataset)["df"].toPandas()

            assert_frame_equal(transformed_df, loaded_preds)
