"""Make a wheel file out of an output folder.

Run this using `python -m examples.fastapi.wheel_with_fastapi`


This example assumes you've already run on of the other examples.
The sklearn examples are good candidates for this.

The expected output folder is `outputs` in the base repository


You can convert the model into a fastapi wheel. To do this, use:
1. WheelBuilder from `mlpype.base.deploy.wheel`
2. Add the `FastApiExtension` from `mlpype.fastapi.deploy`

This will create a wheel file of your model that you can install. Once installed, you can
create a FastAPI app with:
```
from example_model import load_app
ml_app = load_app()
```
If you store this file as app.py, you can then be run using `uvicorn app:ml_app`

If you installed mlpype in developer mode (from inside the cloned git repo), you may experience
problems with mlpype dependencies in your new env due to those dependencies being installed in dev mode while
training the model. Just install them as normal. This shouldn't happen with any models trained in production.
"""

from pathlib import Path

from mlpype.base.deploy.wheel import WheelBuilder
from mlpype.fastapi.deploy import FastApiExtension

folder = Path(".").absolute()
model_folder = folder / "outputs"
assert model_folder.is_dir(), f"Please make sure {model_folder} exists."
output_wheel_file = folder / "wheel_output"
print(f"Reading model from {model_folder}. Storing result in {output_wheel_file}")

builder = WheelBuilder(
    model_folder=model_folder,
    model_name="example_model",
    version="0.0.1",
    output_wheel_file=output_wheel_file,
    extensions=[FastApiExtension],
)

builder.build()

"""
Now you can install your model in a different environment using
pip install ./wheel_output/<name of the .whl file>
"""
