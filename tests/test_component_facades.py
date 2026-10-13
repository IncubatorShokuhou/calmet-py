from pathlib import Path
from types import SimpleNamespace

import pycalmet
from pycalmet.config import CalmetConfig, CtgprocConfig, MakegeoConfig, TerrelConfig


def test_public_terrel_facade_returns_output_path(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        "pycalmet._terrel.run",
        lambda config: captured.append(config) or SimpleNamespace(output=config.output),
    )
    output = tmp_path / "terrain.nc"

    result = pycalmet.terrel(
        SRTM_DIR=tmp_path / "srtm",
        OUTPUT=output,
        GRID=pycalmet.GridConfig(1, 1, 1.0, 1.0),
    )

    assert result == output
    assert isinstance(result, Path)
    assert isinstance(captured[0], TerrelConfig)
    from pycalmet.terrel import run

    assert run(captured[0]) == output
    assert callable(pycalmet.terrel)


def test_public_ctrproc_spelling_and_ctgproc_alias_return_output_path(
    monkeypatch, tmp_path
):
    captured = []
    monkeypatch.setattr(
        "pycalmet._ctgproc.run",
        lambda config: captured.append(config) or SimpleNamespace(output=config.output),
    )
    grid = pycalmet.GridConfig(1, 1, 1.0, 1.0)
    output = tmp_path / "landuse.nc"

    result = pycalmet.ctrproc(
        {"LaNdUsE_FiLeS": [], "OuTpUt": output, "GrId": grid}
    )
    compatible = pycalmet.ctgproc(
        LANDUSE_FILES=[], OUTPUT=tmp_path / "compat.nc", GRID=grid
    )

    assert result == output
    assert compatible == tmp_path / "compat.nc"
    assert all(isinstance(config, CtgprocConfig) for config in captured)
    from pycalmet.ctgproc import run as compatibility_run
    from pycalmet.ctrproc import run

    assert run(captured[0]) == output
    assert compatibility_run(captured[0]) == output
    assert callable(pycalmet.ctrproc)
    assert callable(pycalmet.ctgproc)


def test_public_makegeo_facade_returns_primary_output_path(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        "pycalmet._makegeo.run",
        lambda config: captured.append(config) or SimpleNamespace(output=config.output),
    )
    output = tmp_path / "geo.nc"

    result = pycalmet.makegeo(
        TeRrAiN=tmp_path / "terrain.nc",
        LANDUSE=tmp_path / "landuse.nc",
        OUTPUT=output,
        GRID=pycalmet.GridConfig(1, 1, 1.0, 1.0),
    )

    assert result == output
    assert isinstance(result, Path)
    assert isinstance(captured[0], MakegeoConfig)
    from pycalmet.makegeo import run

    assert run(captured[0]) == output
    assert callable(pycalmet.makegeo)


def test_public_calmet_component_module_preserves_package_function(
    monkeypatch, tmp_path
):
    captured = []
    monkeypatch.setattr(
        "pycalmet._calmet.run",
        lambda config: captured.append(config) or Path(config.output),
    )
    config = CalmetConfig(
        meteorology=tmp_path / "calwrf.nc",
        geophysical=tmp_path / "geo.nc",
        output=tmp_path / "calmet.nc",
        start="2018010100",
        end="2018010101",
        grid=pycalmet.GridConfig(1, 1, 1.0, 1.0),
    )

    result = pycalmet.calmet(config)

    assert result == tmp_path / "calmet.nc"
    assert isinstance(result, Path)
    assert captured == [config]
    from pycalmet.calmet import run

    assert run(config) == tmp_path / "calmet.nc"
    assert callable(pycalmet.calmet)
