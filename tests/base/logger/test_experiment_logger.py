# we test shared functions through LocalLogger

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import MagicMock, call, mock_open, patch

from pytest import fixture, mark

from mlpype.base.constants import Constants
from mlpype.base.data.dataset import DataSet
from mlpype.base.evaluate.plot import BasePlotter
from mlpype.base.logger import LocalLogger
from mlpype.base.logger.local_logger import LocalLogger
from mlpype.base.serialiser.serialiser import Serialiser


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


def test_log_plots():
    model = MagicMock()

    data = {
        "train": DataSet(
            x=MagicMock(),
            y=MagicMock(),
        ),
        "test": DataSet(
            x=MagicMock(),
            y=MagicMock(),
        ),
    }
    predictions = [
        DataSet(
            y=MagicMock(),
        ),
        DataSet(
            y=MagicMock(),
        ),
    ]
    plotters = [MagicMock(spec=BasePlotter), MagicMock(spec=BasePlotter), MagicMock(spec=BasePlotter)]

    model.transform.side_effect = predictions
    for i, p in enumerate(plotters):
        p.plot.side_effect = [[f"train{i}"], [f"test{i}"]]

    exp = MagicMock()
    exp.model = model
    exp.plots = plotters
    logger = LocalLogger()
    with TemporaryDirectory() as tmp_dir, patch.object(logger, "log_file") as mock_log_file:
        tmp_dir = Path(tmp_dir)
        exp.output_folder = tmp_dir
        logger._log_plots(exp, data)

    model.transform.assert_has_calls([call(data["train"]), call(data["test"])])
    expected_datasets = {
        "train": DataSet(
            **{
                "x": data["train"]["x"],
                "y": data["train"]["y"],
                f"y{Constants.PREDICTION_SUFFIX}": predictions[0]["y"],
            }
        ),
        "test": DataSet(
            **{
                "x": data["test"]["x"],
                "y": data["test"]["y"],
                f"y{Constants.PREDICTION_SUFFIX}": predictions[1]["y"],
            }
        ),
    }
    for plotter in plotters:
        plotter.plot.assert_has_calls(
            [call(tmp_dir / Constants.PLOT_FOLDER / name, expected_datasets[name], exp) for name in ["train", "test"]],
            any_order=True,
        )

    mock_log_file.assert_has_calls(
        [call(f"{name}{i}") for name in ["train", "test"] for i, _ in enumerate(plotters)], any_order=True
    )


class TestLogExtraFiles:
    @fixture
    def upper_file(self) -> List[Path]:
        folder_path = Path(__file__).parent.parent.absolute() / "_tmp_dir_"
        try:
            assert not folder_path.is_dir()
            folder_path.mkdir(parents=True, exist_ok=False)
            file1_path = folder_path / "a.py"
            with open(file1_path, "w") as f:
                f.write("tmp_data")
            file2_path = folder_path / "b" / "c.py"
            file2_path.parent.mkdir(parents=False, exist_ok=False)
            with open(file2_path, "w") as f:
                f.write("tmp_data_2")
            yield [folder_path, file1_path, file2_path]
        finally:
            shutil.rmtree(str(folder_path), ignore_errors=True)

    def test_inside_cwd(self):
        cwd = Path(__file__).parent

        logger = MagicMock()

        additional_files_to_store = [cwd / "a.py"]
        logger = LocalLogger()

        m_open = mock_open()
        with patch("mlpype.base.logger.experiment_logger.os.getcwd", return_value=cwd) as mock_getcwd, patch(
            "mlpype.base.logger.experiment_logger.open", m_open
        ), patch("mlpype.base.logger.experiment_logger.json.dump") as mock_dump, patch.object(
            logger, "log_local_file"
        ) as mock_log_local_file, patch.object(
            logger, "log_file"
        ) as mock_log_file, TemporaryDirectory() as tmp_dir:
            output_folder = Path(tmp_dir)
            logger._log_extra_files(output_folder=output_folder, additional_files_to_store=additional_files_to_store)

            mock_getcwd.assert_called_once_with()
            mock_log_local_file.assert_called_once_with(Path("a.py"), output_folder / "a.py")
            mock_log_file.assert_called_once_with(output_folder / Constants.EXTRA_FILES)

            m_open.assert_called_once_with(output_folder / Constants.EXTRA_FILES, "w")
            opened_obj = m_open.return_value

            mock_dump.assert_called_once_with({"paths": ["a.py"]}, opened_obj)

    def test_outside_of_cwd(self, upper_file: List[Path]):
        extra_folder = upper_file[0]
        cwd = Path(__file__).parent

        logger = LocalLogger()

        m_open = mock_open()
        with patch("mlpype.base.logger.experiment_logger.os.getcwd", return_value=cwd) as mock_getcwd, patch(
            "mlpype.base.logger.experiment_logger.open", m_open
        ), patch("mlpype.base.logger.experiment_logger.json.dump") as mock_dump, patch.object(
            logger, "log_local_file"
        ) as mock_log_local_file, patch.object(
            logger, "log_file"
        ) as mock_log_file, TemporaryDirectory() as tmp_dir:
            output_folder = Path(tmp_dir)
            logger._log_extra_files(output_folder=output_folder, additional_files_to_store=[extra_folder])

            mock_getcwd.assert_called_once_with()

            mock_log_local_file.assert_called_once_with(extra_folder, output_folder / extra_folder.name)
            mock_log_file.assert_called_once_with(output_folder / Constants.EXTRA_FILES)

            m_open.assert_called_once_with(output_folder / Constants.EXTRA_FILES, "w")
            opened_obj = m_open.return_value

            mock_dump.assert_called_once_with({"paths": [str(extra_folder.name)]}, opened_obj)


@dataclass
class VersionInfo:
    major: int
    minor: int
    micro: int


@fixture
def version_info():
    old_version = sys.version_info

    sys.version_info = VersionInfo(major=4, minor=10, micro=129)

    yield sys.version_info
    sys.version_info = old_version


def test_log_requirements(version_info: VersionInfo):
    logger = LocalLogger()
    output_folder = Path("tmp")

    f1 = MagicMock()
    f2 = MagicMock()
    req_text = b"pandas==0.1.0\nnumpy=1.0.1"
    with patch("mlpype.base.logger.experiment_logger.open", side_effect=[f1, f2]) as mock_open, patch(
        "mlpype.base.logger.experiment_logger.subprocess.check_output", return_value=req_text
    ) as mock_check_output, patch("mlpype.base.logger.experiment_logger.json.dump") as mock_dump, patch.object(
        logger, "log_file"
    ) as mock_log_file:
        logger._log_requirements(output_folder)

        mock_open.assert_has_calls(
            [
                call(output_folder / Constants.PYTHON_VERSION_FILE, "w"),
                call(output_folder / Constants.REQUIREMENTS_FILE, "w"),
            ]
        )

        # logger.log_local_file.assert_called_once_with(Path("a.py"), output_folder / "a.py")
        # logger.log_file.assert_called_once_with(output_folder / Constants.EXTRA_FILES)
        # python version
        mock_dump.assert_called_once_with(
            {
                "python_version": "4.10.129",
                "major": version_info.major,
                "minor": version_info.minor,
                "micro": version_info.micro,
            },
            f1.__enter__.return_value,
        )

        # requirements
        mock_check_output.assert_called_once_with([sys.executable, "-m", "pip", "freeze"])
        f2.__enter__.return_value.write.assert_called_once_with(req_text.decode())

        # logging
        mock_log_file.assert_has_calls(
            [
                call(output_folder / Constants.PYTHON_VERSION_FILE),
                call(output_folder / Constants.REQUIREMENTS_FILE),
            ],
            any_order=True,
        )


def test_log_run():
    logger = LocalLogger()
    metrics = {
        "train": {"a": 1},
        "test": {"a": 2},
    }
    transformed = {
        "train": MagicMock(),
        "test": MagicMock(),
    }

    with patch.object(logger, "log_file") as mock_log_file, patch.object(
        logger, "log_artifact"
    ) as mock_log_artifact, patch.object(logger, "_create_output_folders") as mock_create_output_folders, patch.object(
        logger, "_log_plots"
    ) as mock_log_plots, patch.object(
        logger, "_log_requirements"
    ) as mock_log_requirements, patch.object(
        logger, "_log_extra_files"
    ) as mock_log_extra_files, patch.object(
        logger, "log_metrics"
    ) as mock_log_metrics, patch.object(
        logger, "log_model"
    ) as mock_log_model, patch(
        "mlpype.base.logger.experiment_logger.JoblibSerialiser"
    ) as mock_jl_serialiser_cls, TemporaryDirectory() as tmp_dir:
        output_folder = Path(tmp_dir)
        exp = MagicMock()
        exp.output_folder = output_folder
        logger.log_run(exp, metrics, transformed)
        mock_jl_serialiser_cls.assert_called_once_with()
        mock_jl_serialiser = mock_jl_serialiser_cls.return_value

        mock_log_metrics.assert_has_calls(
            [
                call("train", metrics["train"]),
                call("test", metrics["test"]),
            ]
        )
        mock_create_output_folders.assert_called_once_with(output_folder)

        mock_log_plots.assert_called_once_with(exp, transformed)
        mock_log_requirements.assert_called_once_with(output_folder)
        mock_log_extra_files.assert_called_once_with(output_folder, exp.additional_files_to_store)
        mock_log_artifact.assert_has_calls(
            [
                call(output_folder / Constants.PIPELINE_FILE, exp.serialiser, object=exp.pipeline),
                call(output_folder / Constants.INPUT_TYPE_CHECKER_FILE, exp.serialiser, object=exp.input_type_checker),
                call(
                    output_folder / Constants.OUTPUT_TYPE_CHECKER_FILE, exp.serialiser, object=exp.output_type_checker
                ),
                call(output_folder / Constants.SERIALISER_FILE, mock_jl_serialiser, object=exp.serialiser),
            ],
            any_order=True,
        )

        mock_log_model.assert_called_once_with(exp.model, output_folder / Constants.MODEL_FOLDER)


def test_create_output_folders():
    logger = LocalLogger()

    with TemporaryDirectory() as tmp_dir:
        output_folder = Path(tmp_dir)
        logger._create_output_folders(output_folder)

        assert output_folder.exists()
        assert (output_folder / Constants.MODEL_FOLDER).exists()

        # Make sure we can safely call it twice
        logger._create_output_folders(output_folder)
        assert output_folder.exists()
        assert (output_folder / Constants.MODEL_FOLDER).exists()
