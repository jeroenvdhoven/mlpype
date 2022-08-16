# we test shared functions through LocalLogger

from pathlib import Path
from unittest.mock import MagicMock, patch

from pype.base.logger.local_logger import LocalLogger
from pype.base.serialiser.serialiser import Serialiser


class Test_experiment_logger:
    def test_log_metrics(self):
        logger = LocalLogger()
        metrics = {"a": 2, "b4": 9}
        ds_name = "some name"

        with patch.object(logger, "_log_metrics") as mock_log:
            logger.log_metrics(ds_name, metrics)

            mock_log.assert_called_once_with(
                {
                    f"{ds_name}_a": 2,
                    f"{ds_name}_b4": 9,
                }
            )

    def test_log_model(self):
        model = MagicMock()
        folder = Path("some")
        logger = LocalLogger()

        with patch.object(logger, "log_file") as mock_log:
            logger.log_model(model, folder)
        model.save.assert_called_once_with(folder)
        mock_log.assert_called_once_with(folder)

    def test_log_artifact(self):
        file = Path("dummy")
        obj = MagicMock()
        serialiser = MagicMock(spec=Serialiser)
        logger = LocalLogger()

        with patch.object(logger, "log_file") as mock_log_file:
            logger.log_artifact(file, serialiser, obj)

        serialiser.serialise.assert_called_once_with(obj, file)
        mock_log_file.assert_called_once_with(file)
