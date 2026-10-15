from pathlib import Path
from shutil import copy2

from tools import generate_pipeline_api


def test_pipeline_api_generator_bootstraps_without_generated_module(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "src"
    package = source / "pycalmet"
    package.mkdir(parents=True)
    project_package = Path(__file__).parents[1] / "src" / "pycalmet"
    for name in ("api.py", "config.py", "parameters.py", "_path_safety.py",
                 "_parameter_docs.py", "datetime_utils.py"):
        copy2(project_package / name, package / name)

    # This minimal source package intentionally has neither __init__.py nor
    # _pipeline_api.py.  Rendering must therefore avoid importing the real
    # package facade, whose pipeline module consumes the generated target.
    monkeypatch.setattr(generate_pipeline_api, "SOURCE", source)

    assert generate_pipeline_api.render() == (
        project_package / "_pipeline_api.py"
    ).read_text(encoding="utf-8")
