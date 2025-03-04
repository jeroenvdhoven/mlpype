"""Provides an importable extension to inject an FastAPI app into your wheel file for MLpype models."""
from pathlib import Path

from mlpype.base.deploy.wheel import WheelExtension

FastApiExtension = WheelExtension(
    "fastapi", [(Path(__file__).parent / "wheel" / "helpers.py", ["load_app"])], ["mlpype.fastapi"]
)
