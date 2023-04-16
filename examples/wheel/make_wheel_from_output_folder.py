"""Make a wheel file out of an output folder.

This example assumes you've already run on of the other examples.
The sklearn examples are good candidates for this.

The expected output folder is `outputs` in the base repository

Run this using python -m examples.wheel.make_wheel_from_output_folder
"""

from pathlib import Path

from pype.base.deploy.wheel import WheelBuilder

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
)

builder.build()

"""
Now you can install your model in a different environment using
pip install ./wheel_output/<name of the .whl file>
"""
