# PyCALMET User Guide

PyCALMET turns the CALMET preprocessing sequence into a typed Python package.
This guide explains how to install it, how the NumPy and JAX backends differ,
and what each processing stage expects. It is written in the same spirit as
the scikit-learn user guide: a short narrative, a minimal example, and a
pointer to the public API rather than a parameter dump.

The repository `README.md` remains the parameter reference. This guide is the
place to start when wiring a new domain or choosing an array backend.

## Contents

1. [Installation](install.md) — base install, the `gpu` extra, and development setup
2. [Array backends](backends.md) — NumPy by default, optional JAX, devices, and dtypes
3. [End-to-end pipeline](pipeline.md) — `run_pipeline` and the files the stages exchange
4. [CALWRF](calwrf.md) — WRF NetCDF to CALMET meteorology
5. [TERREL](terrel.md) — terrain elevation on the CALMET grid
6. [CTGPROC](ctgproc.md) — categorical land use to fractional hits
7. [MAKEGEO](makegeo.md) — geophysical fields from terrain and land use
8. [CALMET](calmet.md) — diagnostic wind, mixing height, and the final dataset

```{toctree}
:maxdepth: 2
:caption: User Guide

install
backends
pipeline
calwrf
terrel
ctgproc
makegeo
calmet
```

The fenced `toctree` above is read by Sphinx. On GitHub, use the numbered list.

## Build the HTML guide

From the repository root, with the documentation extra tools installed:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. The HTML build is gitignored.
