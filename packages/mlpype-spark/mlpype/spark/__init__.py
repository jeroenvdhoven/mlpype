"""Provides tools for using Spark with MLpype.

Important to note is that serialisation and evaluation work differently in Spark.

- Evaluation works slightly different, not taking a dictionary of functions but a JavaEvaluator.
- Serialisation has to take the SparkSession into account and serialise the trained Pipeline and Model properly.
"""
from . import data, evaluate, model, pipeline, serialiser
