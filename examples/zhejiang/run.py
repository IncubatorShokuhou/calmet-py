"""Run the complete Zhejiang example from the bundled source datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeAlias

import pycalmet


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG: Final[Path] = ROOT / "examples" / "zhejiang" / "config"
OUTPUT: Final[Path] = ROOT / "output" / "zhejiang"
StageRunner: TypeAlias = Callable[[Path], Path]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stages: tuple[tuple[str, StageRunner, Path], ...] = (
        ("CALWRF", pycalmet.calwrf, CONFIG / "calwrf.inp"),
        ("TERREL", pycalmet.terrel, CONFIG / "terrel.inp"),
        ("CTGPROC", pycalmet.ctrproc, CONFIG / "ctgproc.inp"),
        ("MAKEGEO", pycalmet.makegeo, CONFIG / "makegeo.inp"),
        ("CALMET", pycalmet.calmet, CONFIG / "calmet.inp"),
    )
    for index, (name, function, control) in enumerate(stages, start=1):
        print(f"[{index}/5] {name}: {control}", flush=True)
        product = function(control)
        print(f"      -> {product}", flush=True)


if __name__ == "__main__":
    main()
