"""Provides tools for evaluating mlpype models.

Evaluators are used to evaluate models in mlpype. The Evaluator class itself can be provided
a dictionary of {name: function} to evaluate a model. Evaluators are run against
all datasets of an experiment.
"""
from .base_evaluator import BaseEvaluator
from .evaluator import Evaluator
from .plot import BasePlotter, Plotter
