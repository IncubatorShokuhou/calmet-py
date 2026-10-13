"""Pure-Python/NumPy CALMET workflow.

The public functions accept either their configuration dataclass or a mapping
whose case-insensitive keys use descriptive dataclass fields or legacy
control/list-file variable names.  :func:`readlst` returns a kwargs-ready
mapping for an existing control or list file.
"""
from __future__ import annotations

# Load the public component modules before binding the package-level functions.
# This keeps ``pycalmet.<stage>(...)`` callable after Python imports the
# same-named component modules.
from . import calmet as _calmet_component
from . import calwrf as _calwrf_component
from . import ctgproc as _ctgproc_component
from . import ctrproc as _ctrproc_component
from . import makegeo as _makegeo_component
from . import readlst as _readlst_component
from . import terrel as _terrel_component
from .api import calmet, calwrf, ctgproc, ctrproc, makegeo, terrel
from .aux_data import AuxiliaryCloudHour,AuxiliaryData,AuxiliaryHeader
from .config import (CalmetConfig,CalwrfConfig,CtgprocConfig,CtgprocSource,
                     GridConfig,MakegeoConfig,TerrelConfig,TerrelSource)
from .readlst import readlst
from .datums import DATUMS,DatumDefinition,datum_definition,datum_shift
from .prog_data import ProgData,ProgRecord
from . import legacy

__all__ = [
    "calwrf", "terrel", "ctrproc", "ctgproc", "makegeo", "calmet", "readlst",
    "GridConfig", "CalwrfConfig", "TerrelConfig", "TerrelSource", "CtgprocConfig", "CtgprocSource",
    "MakegeoConfig", "CalmetConfig",
    "DATUMS", "DatumDefinition", "datum_definition", "datum_shift",
    "ProgData", "ProgRecord",
    "AuxiliaryData", "AuxiliaryHeader", "AuxiliaryCloudHour", "legacy",
]

__version__ = "0.1.1"
