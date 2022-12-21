import shutil
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import numpy as np
import tensorflow as tf
from keras import Model
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam
from pytest import mark

from pype.base.data import DataSet
from pype.base.model import Model as PypeModel
from pype.tensorflow.model import KerasPypeModel, MLPKeras


class DummyKerasModel(KerasPypeModel[MLPKeras]):
    def _init_model(self, args: Dict[str, Any]) -> MLPKeras:
        return MLPKeras(10, 2, 2)


class Test_keras_pype_model:
    # tested through MLPPypeModel since that can be instantiated.
    @mark.parametrize(
        ["model"],
        [
            [None],
            [MLPKeras(10, 2, 2)],
        ],
    )
    def test_init(self, model):
        arguments = {"a": None}
        with patch.object(DummyKerasModel, "_init_model") as mock_init:
            pype_model = DummyKerasModel(["x"], ["y"], MagicMock(), MagicMock(), model=model, **arguments)

            if model is None:
                mock_init.assert_called_once_with(arguments)
            else:
                mock_init.assert_not_called()

    def test_get_annotated_class(self):
        result = DummyKerasModel._get_annotated_class()

        assert result == MLPKeras

    def test_set_seed(self):
        seed = 3
        pype_model = DummyKerasModel(["x"], ["y"], MagicMock(), MagicMock(), seed=seed)

        with patch("pype.tensorflow.model.keras_pype_model.set_seed") as mock_seed:
            pype_model.set_seed()
            mock_seed.assert_called_once_with(seed)

    def test_fit(self):
        model = MagicMock()
        loss = MagicMock()
        optimizer_class = MagicMock()
        metrics = [MagicMock(), MagicMock()]
        pype_model = DummyKerasModel(
            ["x"], ["y"], loss=loss, optimizer_class=optimizer_class, model=model, metrics=metrics
        )

        x = MagicMock()
        y = MagicMock()
        pype_model._fit(x, y)

        model.compile.assert_called_once_with(optimizer=optimizer_class.return_value, loss=loss, metrics=metrics)

        model.fit.assert_called_once()

    def test_transform(self):
        model = MagicMock()
        loss = MagicMock()
        optimizer_class = MagicMock()
        pype_model = DummyKerasModel(["x"], ["y"], loss=loss, optimizer_class=optimizer_class, model=model)

        x = MagicMock()
        y = MagicMock()
        pype_model._transform(x, y)

        model.assert_called_once_with(x, y)

    def test_integration_with_save(self):
        folder = Path(__file__).parent / "tmp"
        folder.mkdir()
        ds = DataSet(
            x=tf.zeros((25, 5)),
            y=tf.zeros(25),
        )
        loss = MeanAbsoluteError()

        try:
            model = MLPKeras(layer_size=20, n_layers=3, output_size=1)
            pype_model = DummyKerasModel(["x"], ["y"], loss, Adam, model=model)

            pype_model.fit(ds)
            initial_result = pype_model.transform(ds)

            pype_model.save(folder)

            loaded = DummyKerasModel.load(folder)
            loaded_result = loaded.transform(ds)

            np.testing.assert_array_almost_equal(initial_result["y"].numpy(), loaded_result["y"].numpy())
        finally:
            shutil.rmtree(folder)

    def test_get_parameters(self):
        parser = MagicMock()

        with patch("pype.tensorflow.model.keras_pype_model.add_args_to_parser_for_class") as mock_args:
            DummyKerasModel.get_parameters(parser)

            mock_args.assert_has_calls(
                [
                    call(parser, MLPKeras, "model", [Model], excluded_args=[]),
                    call(
                        parser,
                        DummyKerasModel,
                        "model",
                        [PypeModel],
                        excluded_args=["seed", "inputs", "outputs", "model", "loss", "optimizer_class"],
                    ),
                ]
            )
