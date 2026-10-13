# Repository Guidelines

## Project Structure & Module Organization

This repository packages a CALMET meteorological-processing workflow. `calmet_mpi/calmet_mpi/` contains CALMET source and runners; `code/code/` contains CALWRF source; `CTGPROC/CTGPROC/` and `MAKEGEO/MAKEGEO/` prepare land-use and geophysical inputs; `LULC/LULC/` contains conversion utilities. Regional scripts live in `script/script/`, helper binaries in `bin/bin/`, and bundled dependencies in `LIBRARIES/LIBRARIES/`. Treat archives, executables, model data, logs, and generated `.dat` files as artifacts unless explicitly targeted.

## Build, Test, and Development Commands

Run commands from the component directory because scripts use relative paths and environment variables.

- `./build_gnu.sh` builds CALWRF, TERREL, CTGPROC, MAKEGEO, and CALMET with
  GNU Fortran; NetCDF-Fortran must provide `nf-config`.
- `cd CTGPROC/CTGPROC && csh ctgproc.csh` generates `ctgproc.inp` and runs land-cover preprocessing.
- `cd MAKEGEO/MAKEGEO && csh makegeo.csh` creates the geophysical dataset after `terrel_root` and `ctgproc_root` are set.
- `cd calmet_mpi/calmet_mpi && csh calmet.csh` builds the CALMET input and runs the model; set `calwrfout_root` and `makegeo_root` first.
- `./run_zhejiang.sh --wrf PATH --strm-dir DIR --start YYYYMMDDHH --end YYYYMMDDHH`
  runs the parameterized Zhejiang workflow.

## Coding Style & Naming Conventions

Preserve fixed-form Fortran layout: comments in column 1, continuation and column widths unchanged, and established uppercase identifiers. Use descriptive shell names such as `autorun_calmet_<region>.csh`. Keep C-shell syntax and avoid broad formatting changes in legacy sources.

## Testing Guidelines

There is no repository-wide automated test suite or coverage target. Smoke-test the changed component with a small known input, require a zero exit status, and inspect its `.log`, `.lst`, and QA outputs (for example `qa.met`, `qa.geo`, or `qa.ctg`). Compare generated grids and record counts against a known-good run; do not commit transient outputs.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout, so use concise imperative subjects, optionally scoped: `CALWRF: correct precipitation rollover`. Keep source, configuration, and regenerated artifacts in separate commits. Pull requests should state the affected processing stage and region, required environment variables/data, exact validation command, and before/after output evidence. Link the issue and call out any binary or large-data changes explicitly.

## Configuration & Data Safety

Do not commit machine-specific absolute paths, credentials, or licensed input datasets. Scripts remove and recreate local products; run them only inside the intended component working directory and verify all root variables before execution.
