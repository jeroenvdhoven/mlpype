"""The core of the mlpype library: run a standardised ML experiment with the given parameters.

The Experiment defines everything you need to run a standardised ML experiment. It will then do
the following:

- Load data
- Fit the pipeline
- Transform data
- Fit the model
- Evaluate the model
- Log the results and artifacts
"""
from .experiment import Experiment
