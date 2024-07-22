"""ExperimentLogger records the results of an experiment.

This allows you to integrate with different reporting tools, like MLflow.
"""
from .experiment_logger import ExperimentLogger
from .local_logger import LocalLogger
