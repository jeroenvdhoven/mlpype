# we test shared functions through LocalLogger

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from pytest import mark

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

    def test_log_local_file_only_file(self):
        logger = LocalLogger()

        file = "a.py"
        output = "target"
        message = "hello!"

        with TemporaryDirectory() as source_dir, TemporaryDirectory() as target_dir, patch.object(
            logger, "log_file"
        ) as mock_log_file:
            source_dir = Path(source_dir)
            target_dir = Path(target_dir)

            source_file = source_dir / file
            target_file = target_dir / output
            with open(source_file, "w") as f:
                f.write(message)

            assert not target_file.exists()
            logger.log_local_file(source_file, target_file)
            assert target_file.exists()

            with open(target_file, "r") as f:
                assert message == f.read()

            mock_log_file.assert_called_once_with(target_file)

    def test_log_local_file_nested_file(self):
        logger = LocalLogger()

        file = "sub/folder/a.py"
        output = "different/target.py"
        message = "hello!"

        with TemporaryDirectory() as source_dir, TemporaryDirectory() as target_dir, patch.object(
            logger, "log_file"
        ) as mock_log_file:
            source_dir = Path(source_dir)
            target_dir = Path(target_dir)

            source_file = source_dir / file
            target_file = target_dir / output
            source_file.parent.mkdir(parents=True, exist_ok=True)

            with open(source_file, "w") as f:
                f.write(message)

            assert not target_file.exists()
            logger.log_local_file(source_file, target_file)
            assert target_file.exists()

            with open(target_file, "r") as f:
                assert message == f.read()

            mock_log_file.assert_called_once_with(target_file)

    @mark.parametrize(["output_dir_exists"], [[True], [False]])
    def test_log_local_file_folder(self, output_dir_exists: bool):
        logger = LocalLogger()

        folder = Path("sub/folder")
        file1 = "a.py"
        file2 = "b.py"
        output = "different/targets"
        message1 = "hello!"
        message2 = "goodbey!"

        with TemporaryDirectory() as source_dir, TemporaryDirectory() as target_dir, patch.object(
            logger, "log_file"
        ) as mock_log_file:
            source_dir = Path(source_dir)
            target_dir = Path(target_dir)

            source_folder = source_dir / folder
            source_folder.mkdir(parents=True, exist_ok=True)
            source_file1 = source_folder / file1
            source_file2 = source_folder / file2
            target_folder = target_dir / output

            if output_dir_exists:
                target_folder.mkdir(parents=True, exist_ok=True)
                with open(target_folder / file1, "w") as f:
                    f.write("a dummy message")

            with open(source_file1, "w") as f:
                f.write(message1)
            with open(source_file2, "w") as f:
                f.write(message2)

            if not output_dir_exists:
                assert not target_folder.exists()
            logger.log_local_file(source_folder, target_folder)
            assert target_folder.exists()
            assert target_folder.is_dir()

            with open(source_file1, "r") as f:
                assert message1 == f.read()
            with open(source_file2, "r") as f:
                assert message2 == f.read()

            mock_log_file.assert_called_once_with(target_folder)
