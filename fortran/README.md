# Fortran reference sources

This directory contains the original Fortran routines used to validate the
NumPy implementation. They are reference material and are not imported,
compiled, or executed when installing or running `pycalmet`.

- `calwrf/`: WRF-to-CALMET preprocessing sources.
- `terrel/`: terrain preprocessing sources.
- `ctgproc/`: land-use preprocessing sources.
- `makegeo/`: geophysical-field preprocessing sources.
- `calmet/`: diagnostic meteorological model sources.
- `reference/`: small numerical oracle programs used to derive regression
  expectations.
- `lulc/` and `utilities/`: retained standalone Fortran utilities.

Legacy run scripts, NCL post-processing programs, compiler wrappers, generated
executables, and runtime outputs are intentionally not retained. Preserve
fixed-form column layout and established include-file names when editing these
sources.
