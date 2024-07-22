"""Creates a fastapi app for a saved mlpype experiment."""
# Run this using `uvicorn examples.fastapi.host_model:app --reload`
from pathlib import Path

from mlpype.fastapi.deploy import mlpypeApp

app = mlpypeApp("example_model", Path("outputs")).create_app()
