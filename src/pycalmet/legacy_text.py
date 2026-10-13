"""Text formatting shared by legacy CALPUFF-family interchange files."""

from __future__ import annotations

import math

import numpy as np


def _gfortran_real(value: float) -> str:
    """Format one default REAL as used by gfortran list-directed output."""
    number=float(np.float32(value))
    magnitude=abs(number)
    if magnitude==0.:
        return '-0.00000000' if np.signbit(number) else '0.00000000'
    if magnitude<.1 or magnitude>=1.e9:
        return f'{number:.8E}'
    digits=math.floor(math.log10(magnitude))+1
    return f'{number:.{max(0,9-digits)}f}'


def gfortran_real_pair(first: float, second: float) -> str:
    """Reproduce GNU Fortran ``WRITE(unit,*)`` for two default REALs."""
    def signed(value):
        text=_gfortran_real(value)
        return text if text.startswith('-') else ' '+text

    left=('  '+signed(first)).ljust(17)
    return (left+'  '+signed(second)).ljust(34)+'\n'
