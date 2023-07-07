import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

from mlpype.base.utils.workspace import (
    _create_temporary_workspace,
    _find_all_py_files,
    _reset_workspace,
    switch_workspace,
)
from tests.utils import pytest_assert


def test_reset_workspace():
    old_workspace = "old_workspace"
    old_sys_path = ["a", "b"]
    main_lib = MagicMock()

    v3 = MagicMock()
    main_lib.__dict__ = {"var1": MagicMock(), "var2": MagicMock(), "var3": v3}
    main_to_remove = {"var1", "var2"}

    with patch("mlpype.base.utils.workspace.os") as mock_os, patch("mlpype.base.utils.workspace.sys") as mock_sys:
        _reset_workspace(old_workspace, old_sys_path, main_lib, main_to_remove)

    mock_os.chdir.assert_called_once_with(old_workspace)

    assert mock_sys.path == old_sys_path
    assert main_lib.__dict__ == {"var3": v3}


def make_moduled_variable(module: str):
    var = MagicMock()
    var.__module__ = module
    return var


def test_create_temporary_workspace():
    target_workspace = Path("directory")
    extra_files = ["a.py", "example/b.py", "dummy_file.txt"]  # the last file will be ignored.
    extra_files_expected = ["a.py", "example/b.py"]
    main_lib = MagicMock()

    v1 = MagicMock()
    v2 = MagicMock()
    v3 = MagicMock()
    main_lib.__dict__ = {"var1": v1, "var2": v2, "var3": v3}

    a_lib = MagicMock()
    a_lib.__dict__ = {
        "var1": make_moduled_variable("a"),  # will be ignored because of name
        "var_new_1": make_moduled_variable("other_module"),  # will be ignored because of module
        "var_new_2": make_moduled_variable("a"),  # will be added
    }
    example_b_lib = MagicMock()
    example_b_lib.__dict__ = {
        "var_new_2": make_moduled_variable("a"),  # will be ignored since a_lib has it
        "var_new_3": make_moduled_variable("example.b"),  # will be added
    }
    libs = [a_lib, example_b_lib]

    with patch("mlpype.base.utils.workspace.os.chdir") as mock_chdir, patch(
        "mlpype.base.utils.workspace._find_all_py_files", return_value=extra_files_expected
    ) as mock_find_all_py, patch("mlpype.base.utils.workspace.sys") as mock_sys, patch(
        "mlpype.base.utils.workspace.importlib.import_module", side_effect=libs
    ) as mock_import:
        result = _create_temporary_workspace(target_workspace, extra_files, main_lib)

        mock_chdir.assert_called_once_with(target_workspace)
        mock_sys.path.append.assert_has_calls(
            [
                call(str(target_workspace)),
                call(os.path.join(".", "a.py")),
                call(os.path.join(".", "example/b.py")),
            ],
            any_order=True,
        )

        mock_find_all_py.assert_called_once_with(extra_files)
        mock_import.assert_has_calls([call("a"), call("example.b")])

        assert main_lib.__dict__ == {
            "var1": v1,
            "var2": v2,
            "var3": v3,
            "var_new_2": a_lib.__dict__["var_new_2"],
            "var_new_3": example_b_lib.__dict__["var_new_3"],
        }

        assert result == {"var_new_2", "var_new_3"}


def test_switch_workspace():
    path = Path(".")
    extra_files = ["a.py"]

    with patch("mlpype.base.utils.workspace._create_temporary_workspace") as mock_create, patch(
        "mlpype.base.utils.workspace._reset_workspace"
    ) as mock_reset, patch("mlpype.base.utils.workspace.importlib.import_module") as mock_importlib:
        cwd = os.getcwd()
        c_sys_path = sys.path.copy()

        with switch_workspace(path, extra_files):
            pass

        mock_create.assert_called_once_with(path, extra_files, mock_importlib.return_value)
        mock_reset.assert_called_once_with(cwd, c_sys_path, mock_importlib.return_value, mock_create.return_value)
        mock_importlib.assert_called_once_with("__main__")


def _is_dir_in_sys_path(path: Path) -> bool:
    return str(path.absolute()) in [str(Path(p).absolute()) for p in sys.path]


def test_switch_workspace_integration():
    path = Path(__file__).parent / "dummy_folder"
    extra_files = ["a.py"]

    cwd = os.getcwd()
    c_sys_path = sys.path.copy()
    main_lib = importlib.import_module("__main__")

    try:
        assert not _is_dir_in_sys_path(path)
        assert "probably_not_known_form_of_hello" not in main_lib.__dict__
        with switch_workspace(path, extra_files):
            assert os.getcwd() != cwd
            assert Path(os.getcwd()).absolute() == path.absolute()

            # make sure the new directory is added to path, temporarily
            assert _is_dir_in_sys_path(path)

            # make sure the 'hello' function is imported correctly
            # see dummy_folder/a.py for the function
            assert "probably_not_known_form_of_hello" in main_lib.__dict__
            result = main_lib.__dict__["probably_not_known_form_of_hello"]()
            assert result == "Hello!"

        assert os.getcwd() == cwd
        assert sys.path == c_sys_path
        assert not _is_dir_in_sys_path(path)
        assert "probably_not_known_form_of_hello" not in main_lib.__dict__
    finally:
        sys.path = c_sys_path
        os.chdir(cwd)


def test_find_all_py_files_files():
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        files = [tmp_dir / "a.py", tmp_dir / "ignored.txt"]
        all_files = ["a.py", "ignored.txt", "b.py"]
        for f in all_files:
            _make_dummy_file(tmp_dir / f)

        result = _find_all_py_files(files)

        assert result == [str(tmp_dir / "a.py")]


def test_find_all_py_files_directories():
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        files = [tmp_dir / "a.py", tmp_dir / "ignored.txt", tmp_dir / "b"]
        all_files = ["a.py", "ignored.txt", "b/b.py", "b/table.csv", "c/c.py", "b/b-2.py"]
        for f in all_files:
            _make_dummy_file(tmp_dir / f)

        result = _find_all_py_files(files)
        result.sort()
        expected = ["a.py"]
        expected = [str(tmp_dir / f) for f in expected]
        expected.sort()

        assert result == expected


def test_find_all_py_files_assert():
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        missing_file = tmp_dir / "definitely not there"
        with pytest_assert(AssertionError, f"`{str(missing_file)}` not found!"):
            assert not missing_file.exists()
            _find_all_py_files([missing_file])


def _make_dummy_file(f: Path):
    f.parent.mkdir(exist_ok=True, parents=True)
    with open(f, "w") as f_pointer:
        f_pointer.write("text!")
