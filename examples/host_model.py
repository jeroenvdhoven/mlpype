# Run this using `uvicorn examples.host_model:app --reload`
from pathlib import Path

from pype.base.deploy.app import PypeApp

app = PypeApp("example_model", Path("outputs")).create_app()
