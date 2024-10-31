"""Provides tools for using Sklearn with MLpype.

This provides a few tools:

1. An SklearnModel: An MLpype model that uses Sklearn. This is easily extended to most models in Sklearn.
This can be found in `mlpype.sklearn.model`. This module also contains pre-loaded models from Sklearn.
2. Type checkers for Numpy and Pandas data.

MLpype base pipelines/pipes should already be compatible with Sklearn transformers, so you can use them directly.
StandardScaler for instance will work with MLpype.
"""
from . import data, model, pipeline
