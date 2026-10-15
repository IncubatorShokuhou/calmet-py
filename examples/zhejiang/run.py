"""Run the complete Zhejiang example as one end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pycalmet


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG: Final[Path] = ROOT / "examples" / "zhejiang" / "config"
OUTPUT: Final[Path] = ROOT / "output" / "zhejiang"


def main() -> None:
    """Run all five stages and retain only the final CALMET dataset."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result: Path = pycalmet.run_pipeline(
        CONFIG,
        output=OUTPUT / "calmet.nc",
    )
    print(f"CALMET -> {result}", flush=True)


if __name__ == "__main__":
    main()
