"""Run the complete Zhejiang example from the bundled source datasets."""

from pathlib import Path

import pycalmet


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples" / "zhejiang" / "config"
OUTPUT = ROOT / "output" / "zhejiang"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stages = (
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
