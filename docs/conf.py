# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

project = "MLpype"
copyright = "2024, Jeroen van den Hoven"
author = "Jeroen van den Hoven"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
]
# Path setup.
root_path = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_path))
autoapi_dirs = []  # location to parse for API reference
for f in root_path.iterdir():
    if f.name.startswith("mlpype-") and f.is_dir():
        autoapi_dirs.append(str(f))
        sys.path.insert(0, str(f / "mlpype"))

autoapi_ignore = ["*/wheel/helpers/*"]
autoapi_python_use_implicit_namespaces = True
autoapi_add_toctree_entry = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
