from unittest.mock import MagicMock

from pype.spark.data import SparkReadSource


class Test_SparkReadSource:
    def test(self):
        mock_spark = MagicMock()

        file = "/a/path"
        formats = "a format"
        options = {"a": 3, "b": 4}

        source = SparkReadSource(mock_spark, file, formats, options)
        result = source.read()

        mock_format = mock_spark.read.format
        mock_load = mock_format.return_value.load
        mock_format.assert_called_once_with(formats)
        mock_load.assert_called_once_with(file, a=3, b=4)
        assert result == mock_load.return_value
