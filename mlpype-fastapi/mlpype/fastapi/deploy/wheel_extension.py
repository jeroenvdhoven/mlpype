from pathlib import Path

from mlpype.base.deploy.wheel import WheelExtension

FastApiExtension = WheelExtension(
    "fastapi", [(Path(__file__).parent / "wheel" / "helpers.py", ["load_app"])], ["mlpype.fastapi"]
)
