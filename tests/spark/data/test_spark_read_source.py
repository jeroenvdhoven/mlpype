from unittest.mock import MagicMock, patch

from mlpype.spark.data import SparkReadSource


class Test_SparkReadSource:
    def test(self):
        mock_spark = MagicMock()

        file = "/a/path"
        formats = "a format"
        options = {"a": 3, "b": 4}

        with patch("mlpype.spark.data.spark_read.guarantee_spark") as mock_guarantee:
            source = SparkReadSource(file, formats, options, spark_session=mock_spark)
            result = source.read()

        mock_guarantee.assert_called_once_with(mock_spark)
        mock_guarantee_spark = mock_guarantee.return_value

        mock_format = mock_guarantee_spark.read.format
        mock_load = mock_format.return_value.load
        mock_format.assert_called_once_with(formats)
        mock_load.assert_called_once_with(file, a=3, b=4)
        assert result == mock_load.return_value
