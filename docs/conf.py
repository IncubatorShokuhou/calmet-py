"""Sphinx configuration for the PyCALMET user guide."""

project = "PyCALMET"
author = "PyCALMET contributors"
extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence"]
source_suffix = {".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "requirements.txt"]
html_theme = "alabaster"
html_title = "PyCALMET User Guide"
