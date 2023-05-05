# we test shared functions through LocalLogger

from unittest.mock import call, patch

from mlpype.base.logger.local_logger import LocalLogger


class Test_local_logger:
    def test_log_metrics(self):
        ws = 5
        logger = LocalLogger(ws)
        metrics = {"a": 2, "b4": 9}

        with patch("mlpype.base.logger.local_logger.print") as mock_print:
            logger._log_metrics(metrics)

            mock_print.assert_has_calls(
                [
                    call("a    : 2"),
                    call("b4   : 9"),
                ]
            )

    def test_log_parameters(self):
        ws = 5
        logger = LocalLogger(ws)
        parameters = {"a": 2, "b4": 9}

        with patch("mlpype.base.logger.local_logger.print") as mock_print:
            logger.log_parameters(parameters)

            mock_print.assert_has_calls(
                [
                    call("a    : 2"),
                    call("b4   : 9"),
                ]
            )

    def test_log_file(self):
        # just shouldn't break.
        logger = LocalLogger()
        logger.log_file("")
