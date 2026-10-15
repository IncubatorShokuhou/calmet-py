# Repository guidelines

## Structure

Runtime Python code lives in `src/pycalmet/`. Tests live in `tests/`, the
runnable real-data case in `examples/zhejiang/`, original terrain data in
`data/terrain/`, original land-use data in `data/landuse/`, Python utilities
in `tools/`, and reference-only Fortran sources in `fortran/`. Large bundled
binary sources such as `data/landuse/usgs-glcc/EAUSGS1.IMG` use Git LFS;
`EAUSGS2.IMG` is not part of the Zhejiang workflow and must not be added merely
as an unused companion file.

Do not add generated executables, model outputs, caches, NCL programs, regional
run wrappers, or machine-specific paths to the repository.

## Development

Use Python 3.10 or newer:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Run the bundled smoke workflow from the repository root:

```bash
./examples/zhejiang/run.sh
```

## Changes

Keep public APIs typed and document new behavior in both `README.md` and
`README.zh-CN.md` when it affects users. Add focused tests for behavior
changes. Preserve fixed-form columns and include-file names in `fortran/`;
those files are numerical references and are not part of the runtime package.

Do not commit secrets, licensed inputs, full operational WRF datasets, or
generated model products.
