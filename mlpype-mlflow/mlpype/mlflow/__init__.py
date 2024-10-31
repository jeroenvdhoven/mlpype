"""Provides tools for using MLpype with MLflow.

At the moment, MLflow is the recommended logging tool for MLpype. This package provides a few tools:

- An MlflowLogger: An MLpype logger that uses MLflow. Set this up and you don't have to worry about how to\
    log your model.
- A quick function to load an Inferencer from an experiment stored in MLflow. You can find this in
    `mlpype.mlflow.deploy`.
- A mlflow compatible version of a Model that should work with any MLpype model. It extends mlflow's PythonModel.
    This should allow you to load any model that has been stored in MLflow using mlflow's logic. You don't
    need to do anything to use this except use the MLflowLogger. Once registered, the model can loaded using
    mlflow's `from mlflow.pyfunc import load_model`.
"""
from . import deploy, logger
