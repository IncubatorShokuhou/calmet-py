#!/usr/bin/env python3
"""Compare PyCALMET NetCDF or legacy land use with CTGPROC LULC.DAT."""
import argparse
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

def read_legacy(path, nx, ny, categories=38):
    lines=open(path,encoding="ascii").read().splitlines()[12:]
    result=np.empty((categories,ny,nx),dtype=np.float32)
    for line in lines:
        values=np.fromstring(line[:10+6*categories],sep=" ")
        result[:,int(values[1])-1,int(values[0])-1]=values[2:]
    return result

def read_candidate(path,nx,ny,categories):
    if Path(path).suffix.lower() not in {'.nc','.nc4','.cdf'}:
        return read_legacy(path,nx,ny,categories)
    with Dataset(path) as dataset:
        actual=np.asarray(dataset['landuse_fraction'][:])
    if actual.shape!=(categories,ny,nx):
        raise ValueError('CTGPROC candidate grid does not match reference')
    return actual

def _landuse_data_lines(path):
    lines=Path(path).read_bytes().splitlines(keepends=True)
    try:
        comments=int(lines[1])
        index=2+comments+1  # file type follows the comment block
        projection=lines[index][:8].strip().upper(); index+=2
        if projection in {b'LCC',b'LAZA',b'TTM'}: index+=1
        index+=3  # datum, grid geometry and units; retain category declaration
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid CTGPROC header in {path}') from error
    return lines[index:]

def compare_data_lines(candidate,reference):
    wanted=_landuse_data_lines(reference); got=_landuse_data_lines(candidate)
    if len(got)!=len(wanted):
        raise ValueError(f'CTGPROC data line count differs: {len(got)} != {len(wanted)}')
    for index,(actual,expected) in enumerate(zip(got,wanted,strict=True),1):
        if actual!=expected:
            raise ValueError(f'CTGPROC data line {index} differs byte-for-byte')
    return len(got)

def compare(candidate,reference,nx,ny,categories=38,*,exact_records=False):
    expected=read_legacy(reference,nx,ny,categories)
    actual=read_candidate(candidate,nx,ny,categories)
    expected_totals=expected.sum(axis=0)
    if not np.any(expected_totals>0.):
        raise ValueError('CTGPROC reference contains no land-use hits')
    if np.any(~np.isfinite(expected)):
        raise ValueError('CTGPROC reference contains non-finite fractions')
    actual_totals=actual.sum(axis=0)
    populated=expected_totals>0.
    if not np.array_equal(actual_totals>0.,populated):
        raise ValueError('CTGPROC populated-cell mask differs from reference')
    delta=np.abs(actual-expected)
    dominant=float(np.mean(np.argmax(actual,axis=0)[populated]==
                           np.argmax(expected,axis=0)[populated]))
    precision=float(np.mean(delta[:,populated]<=0.00051))
    if (exact_records and Path(candidate).suffix.lower() not in {'.nc','.nc4','.cdf'} and
            Path(reference).suffix.lower() not in {'.nc','.nc4','.cdf'}):
        compare_data_lines(candidate,reference)
    return precision,dominant,float(delta.mean()),float(delta.max())

def main():
    p=argparse.ArgumentParser(); p.add_argument("--python",required=True); p.add_argument("--fortran",required=True)
    p.add_argument('--nx',type=int,default=610); p.add_argument('--ny',type=int,default=499)
    p.add_argument('--categories',type=int,default=38)
    p.add_argument('--numeric-only',action='store_true',
                   help='skip byte-for-byte comparison of legacy land-use records')
    args=p.parse_args()
    try:
        precision,dominant,mean,maximum=compare(
            args.python,args.fortran,args.nx,args.ny,args.categories,
            exact_records=not args.numeric_only)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"formatted-precision components: {precision:.6%}")
    print(f"dominant-category agreement: {dominant:.6%}")
    print(f"mean/max absolute difference: {mean:.9f} / {maximum:.9f}")
    if precision != 1. or dominant != 1.:
        raise SystemExit("CTGPROC regression exceeds formatted precision")

if __name__ == "__main__": main()
