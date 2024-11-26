"""Creates a fastapi app for a saved mlpype experiment.

This assumes you've run at least one of the other examples first, e.g. one
of the sklearn models. The `outputs` folder is used as the basis for hosting a model.

Run this using `uvicorn examples.fastapi.host_model:app --reload`
"""
from pathlib import Path

from mlpype.fastapi.deploy import mlpypeApp

app = mlpypeApp("example_model", Path("outputs")).create_app()
