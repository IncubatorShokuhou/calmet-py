#!/usr/bin/env python3
"""Compare PyCALMET NetCDF or legacy output with a GNU MAKEGEO GEO.DAT."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pycalmet.legacy_geo import GeoReader


NAMES=('terrain_height','dominant_landuse','roughness_length','albedo',
       'bowen_ratio','soil_heat_flux_parameter','anthropogenic_heat_flux',
       'leaf_area_index')


def _fields(path):
    path=Path(path)
    if path.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(path) as dataset:
            return {name:np.asarray(dataset[name][:]) for name in NAMES}
    data=GeoReader(path).read()
    return dict(zip(NAMES,(data.terrain,*data.fields),strict=True))


def _legacy_data_lines(path):
    lines=Path(path).read_bytes().splitlines(keepends=True)
    try:
        comments=int(lines[1])
        index=2+comments
        projection=lines[index][:8].strip().upper(); index+=2
        if projection in {b'LCC',b'LAZA',b'TTM'}: index+=1
        index+=3  # datum, grid geometry, horizontal/vertical units
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid GEO.DAT header in {path}') from error
    return lines[index:]


def compare_data_lines(candidate,reference):
    wanted=_legacy_data_lines(reference); got=_legacy_data_lines(candidate)
    if len(got)!=len(wanted):
        raise ValueError(f'GEO.DAT data line count differs: {len(got)} != {len(wanted)}')
    for index,(actual,expected) in enumerate(zip(got,wanted,strict=True),1):
        if actual!=expected:
            raise ValueError(f'GEO.DAT data line {index} differs byte-for-byte')
    return len(got)


def compare(candidate,reference,*,exact_records=False):
    wanted=_fields(reference); got=_fields(candidate); rows=[]
    legacy=Path(candidate).suffix.lower() not in {'.nc','.nc4','.cdf'}
    for name in NAMES:
        if got[name].shape!=wanted[name].shape:
            raise ValueError(f'{name} grid shape differs')
        delta=np.abs(got[name].astype(np.float64)-wanted[name].astype(np.float64))
        tolerance=(0. if legacy else (5.1e-5 if name=='roughness_length' else
                                     0. if name=='dominant_landuse' else .0051))
        rows.append((name,int(np.count_nonzero(delta<=tolerance)),delta.size,
                     float(delta.mean()),float(delta.max(initial=0.)),tolerance))
    if exact_records and legacy and Path(reference).suffix.lower() not in {'.nc','.nc4','.cdf'}:
        compare_data_lines(candidate,reference)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',required=True)
    parser.add_argument('--fortran',required=True)
    parser.add_argument('--numeric-only',action='store_true',
                        help='skip byte-for-byte comparison of legacy GEO.DAT data records')
    args=parser.parse_args()
    try: rows=compare(args.python,args.fortran,exact_records=not args.numeric_only)
    except ValueError as error: raise SystemExit(str(error)) from error
    failed=False
    for name,equal,total,mean,maximum,tolerance in rows:
        print(f'{name}: within={equal/total:.6%} mean={mean:.9g} '
              f'max={maximum:.9g} tolerance={tolerance:g}')
        failed|=equal!=total
    if failed: raise SystemExit('MAKEGEO regression exceeds formatted precision')


if __name__=='__main__':
    main()
