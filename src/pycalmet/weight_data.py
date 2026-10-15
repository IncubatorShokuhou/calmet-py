"""CALMET WT.DAT station-versus-prognostic weighting grids."""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np


def _numbers(line):
    return re.findall(r'[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?',line)


def read_weight_data(path,nx,ny,z_centers_m):
    """Read RDWT's formatted fine-grid weights, returned as ``(z,y,x)``."""
    lines=Path(path).read_text(encoding='ascii',errors='strict').splitlines()
    if len(lines)<6: raise ValueError('WT.DAT is truncated')
    index=3
    fine=_numbers(lines[index][15:]); index+=1
    coarse=_numbers(lines[index][15:]); index+=3
    if len(fine)!=5 or len(coarse)!=5: raise ValueError('WT.DAT grid headers are invalid')
    nx_file,ny_file=int(fine[2]),int(fine[3])
    if (nx_file,ny_file)!=(nx,ny): raise ValueError('WT.DAT fine grid does not match CALMET grid')
    centers=np.asarray(z_centers_m,dtype=np.float32)
    output=np.empty((centers.size,ny,nx),dtype=np.float32)
    for level,wanted in enumerate(centers):
        if index>=len(lines): raise ValueError('WT.DAT is missing a height group')
        found=float(_numbers(lines[index])[-1]); index+=2
        if not np.isclose(found,wanted,rtol=1.e-6,atol=1.e-5):
            raise ValueError('WT.DAT levels do not match CALMET cell centers')
        rows=[]
        for _ in range(ny):
            values=[float(item.replace('D','E').replace('d','e')) for item in _numbers(lines[index][6:])]
            index+=1
            if len(values)!=nx: raise ValueError('WT.DAT row width does not match CALMET grid')
            rows.append(values)
        output[level]=np.asarray(rows[::-1],dtype=np.float32); index+=1
    if np.any((output<0.)|(output>1.)): raise ValueError('WT.DAT weights must lie between zero and one')
    return output
