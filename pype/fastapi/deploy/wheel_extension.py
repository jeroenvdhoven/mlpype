from pathlib import Path

from pype.base.deploy.wheel import WheelExtension

FastApiExtension = WheelExtension(
    "fastapi", [(Path(__file__).parent / "wheel" / "helpers.py", ["load_app"])], ["pype.fastapi"]
)
