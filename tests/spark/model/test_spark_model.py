import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pytest import mark

from pype.base.data import DataSet
from pype.spark.model import LinearSparkModel
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

    @mark.parametrize(
        ["name", "prediction_col"],
        [
            ["return full set", None],
            ["return only column", "pred_col"],
        ],
    )
    def test_transform(self, name: str, prediction_col: Optional[str]):
        predictor = MagicMock()
        model = LinearSparkModel(
            inputs=["x"],
            outputs=["x"],
            predictor=predictor,
            output_col=prediction_col,
        )
        x = MagicMock()
        data = DataSet(x=x, y=MagicMock())

        model.fit(data)
        result = model.transform(data)

        model = predictor.fit.return_value
        model.transform.assert_called_once_with(x)

        predictions = model.transform.return_value

        if prediction_col is None:
            assert result["x"] == predictions
        else:
            predictions.select.assert_called_once_with(prediction_col)
            assert result["x"] == predictions.select.return_value

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

    def test_save(self):
        predictor = MagicMock()
        model = MagicMock()
        pype_model = LinearSparkModel(
            inputs=["x"],
            outputs=["x"],
            predictor=predictor,
            model=model,
            output_col=None,
        )

        with TemporaryDirectory() as tmp_dir, patch(
            "pype.spark.model.spark_model.JoblibSerialiser.serialise"
        ) as mock_serialise:
            tmp_dir = Path(tmp_dir)
            pype_model._save(tmp_dir)

            with open(tmp_dir / pype_model.PYPE_MODEL_CONFIG, "r") as f:
                config = json.load(f)
                assert config == {"output_col": None}

            model.save.assert_called_once_with(str(tmp_dir / pype_model.SPARK_MODEL_PATH))
            predictor.save.assert_called_once_with(str(tmp_dir / pype_model.SPARK_PREDICTOR_PATH))
            mock_serialise.assert_called_once_with(type(model), str(tmp_dir / pype_model.SPARK_MODEL_CLASS_PATH))

    def test_load(self):
        with TemporaryDirectory() as tmp_dir, patch.object(
            LinearSparkModel, "_get_annotated_class"
        ) as mock_annotate, patch("pype.spark.model.spark_model.JoblibSerialiser.deserialise") as mock_deserialise:
            tmp_dir = Path(tmp_dir)

            with open(tmp_dir / LinearSparkModel.PYPE_MODEL_CONFIG, "w") as f:
                json.dump({"output_col": None}, f)

            result = LinearSparkModel._load(tmp_dir, ["x"], ["x"])

            mock_annotate.assert_called_once_with()
            mock_pred_class = mock_annotate.return_value
            mock_pred_class.load.assert_called_once_with(str(tmp_dir / LinearSparkModel.SPARK_PREDICTOR_PATH))

            mock_deserialise.assert_called_once_with(str(tmp_dir / LinearSparkModel.SPARK_MODEL_CLASS_PATH))
            mock_model_class = mock_deserialise.return_value
            mock_model_class.load.assert_called_once_with(str(tmp_dir / LinearSparkModel.SPARK_MODEL_PATH))

            assert result.inputs == ["x"]
            assert result.outputs == ["x"]
            assert result.output_col is None
            assert result.predictor == mock_pred_class.load.return_value
            assert result.model == mock_model_class.load.return_value

    def test_get_parameters(self):
        parser = MagicMock()

        with patch("pype.spark.model.spark_model.add_args_to_parser_for_class") as mock_add:
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
            output_col="prediction_col",
            predictionCol="prediction_col",
            featuresCol="vector",
            labelCol="response",
            fitIntercept=False,
        )

        model.fit(dataset)
        transformed = model.transform(dataset)

        assert "df" in transformed
        assert ["prediction_col"] == transformed["df"].columns

        transformed_df = transformed["df"].toPandas()
        assert transformed_df.shape == (df.shape[0], 1)
        assert np.all(~transformed_df.isna())

        # saving/loading
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            model.save(tmp_dir)
            loaded_model = LinearSparkModel.load(tmp_dir)
            loaded_preds = loaded_model.transform(dataset)["df"].toPandas()

            assert_frame_equal(transformed_df, loaded_preds)

    def test_transform_for_evaluation(self, spark_session: SparkSession):
        df = MagicMock()
        data = DataSet(df=df)
        model = LinearSparkModel(
            inputs=["df"],
            outputs=["df"],
            output_col="prediction_col",
            predictionCol="prediction_col",
            featuresCol="vector",
            labelCol="response",
            fitIntercept=False,
        )
        with patch.object(model, "_transform") as mock_transform:
            result = model.transform_for_evaluation(data)

            mock_transform.assert_called_once_with(df, reduce_columns_if_possible=False)
            assert isinstance(result, DataSet)
            assert len(result) == 1
            assert result["df"] == mock_transform.return_value
