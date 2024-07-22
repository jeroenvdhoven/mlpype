"""Provides tools for using Sklearn with MLpype.

This provides 3 main tools:

- An SklearnModel: An MLpype model that uses Sklearn. This is easily extended to most models in Sklearn.
- Type checkers for Sklearn data

MLpype base pipelines/pipes should already be compatible with Sklearn, so you can use them directly.
"""
from . import data, model, pipeline
