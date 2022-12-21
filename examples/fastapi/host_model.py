# Run this using `uvicorn examples.fastapi.host_model:app --reload`
from pathlib import Path

from pype.fastapi.deploy import PypeApp

app = PypeApp("example_model", Path("outputs")).create_app()
