# Repository guidelines

## Structure

Runtime Python code lives in `src/pycalmet/`. Tests live in `tests/`, the
runnable real-data case in `examples/zhejiang/`, SRTM3 data in
`terrain/srtm3/`, Python utilities in `tools/`, and reference-only Fortran
sources in `fortran/`.

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
./examples/zhejiang/run.sh \
  --wrf examples/zhejiang \
  --geo examples/zhejiang/geo.nc \
  --start 2018031800 --end 2018031801 \
  --skip-static --output output/zhejiang.nc
```

## Changes

Keep public APIs typed and document new behavior in both `README.md` and
`README.zh-CN.md` when it affects users. Add focused tests for behavior
changes. Preserve fixed-form columns and include-file names in `fortran/`;
those files are numerical references and are not part of the runtime package.

Do not commit secrets, licensed inputs, full operational WRF datasets, or
generated model products.
