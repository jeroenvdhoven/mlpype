from unittest.mock import patch

from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser


class Test_JoblibSerialiser:
    def test_save(self):
        with patch("mlpype.base.serialiser.joblib_serialiser.dump") as mock_dump:
            serialiser = JoblibSerialiser()
            obj = [None]
            file = "some file"
            serialiser.serialise(obj, file)

            mock_dump.assert_called_once_with(obj, file)

    def test_load(self):
        with patch("mlpype.base.serialiser.joblib_serialiser.load") as mock_load:
            serialiser = JoblibSerialiser()
            file = "some file"
            result = serialiser.deserialise(file)

            mock_load.assert_called_once_with(file)
            assert mock_load.return_value == result
