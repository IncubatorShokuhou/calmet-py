#!/usr/bin/env python3
"""Compare every gridded cell in two formatted CALWRF M3D files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalmet.legacy_m3d import M3DReader


def _bitwise_equal(wanted,got):
    wanted=np.asarray(wanted)
    got=np.asarray(got)
    if wanted.shape!=got.shape or wanted.dtype!=got.dtype:
        return False
    return np.array_equal(wanted.view(f'V{wanted.dtype.itemsize}'),
                          got.view(f'V{got.dtype.itemsize}'))


def _data_stream(path):
    """Open *path* and position it at the first hourly surface record."""
    stream=Path(path).open('rb',buffering=1024*1024)
    try:
        stream.readline()
        comments=int(stream.readline())
        for _ in range(comments): stream.readline()
        stream.readline()  # output switches
        stream.readline()  # source-grid definition
        stream.readline()  # physics options
        period=stream.readline().split()
        nx,ny,nz=map(int,period[2:5])
        stream.readline()  # extraction bounds
        for _ in range(nz+nx*ny): stream.readline()
        return stream,nx,ny,nz
    except Exception:
        stream.close()
        raise


def compare_record_bytes(reference_path,candidate_path,*,hours):
    """Require byte-identical fixed-width hourly records after the static header."""
    reference,nx,ny,nz=_data_stream(reference_path)
    candidate,cx,cy,cz=_data_stream(candidate_path)
    try:
        if (cx,cy,cz)!=(nx,ny,nz):
            raise ValueError('M3D byte comparison grid dimensions differ')
        count=int(hours)*nx*ny*(nz+1)
        for index in range(count):
            wanted=reference.readline(); got=candidate.readline()
            if got!=wanted:
                raise ValueError(f'M3D fixed-width record {index+1} differs byte-for-byte')
        return count
    finally:
        reference.close(); candidate.close()


def compare(reference_path,candidate_path,*,hours=None,exact_records=False):
    reference_header,reference_cells=M3DReader(reference_path).cells()
    candidate_header,candidate_cells=M3DReader(candidate_path).cells()
    scalar_fields=('projection','latitude_origin','longitude_origin',
                   'standard_parallel_1','standard_parallel_2','x_origin_km',
                   'y_origin_km','spacing_km','nx','ny','nz','start',
                   'source_nx','source_ny','source_nz','i_start','j_start','k_start')
    for name in scalar_fields:
        wanted=getattr(reference_header,name)
        got=getattr(candidate_header,name)
        if got!=wanted:
            raise ValueError(f'M3D header {name} differs: {got!r} != {wanted!r}')
    for name in ('sigma','latitude','longitude','terrain','landuse'):
        wanted=getattr(reference_header,name)
        got=getattr(candidate_header,name)
        if not _bitwise_equal(wanted,got):
            raise ValueError(f'M3D static field {name} differs')
    count=(min(reference_header.nhours,candidate_header.nhours)
           if hours is None else int(hours))
    if count<1 or count>reference_header.nhours or count>candidate_header.nhours:
        raise ValueError('requested M3D hour count is outside either file')
    if exact_records:
        compare_record_bytes(reference_path,candidate_path,hours=count)
    totals={'surface':0,'levels':0,'water':0}
    differences={name:0 for name in totals}
    maxima={name:0. for name in totals}
    for index in range(count*reference_header.nx*reference_header.ny):
        try:
            wanted=next(reference_cells); got=next(candidate_cells)
        except StopIteration as error:
            raise ValueError('M3D file ends before the requested cells') from error
        if (got.date,got.i,got.j)!=(wanted.date,wanted.i,wanted.j):
            raise ValueError(
                f'M3D cell {index+1} differs: {(got.date,got.i,got.j)!r} != '
                f'{(wanted.date,wanted.i,wanted.j)!r}')
        for name in totals:
            wanted_values=getattr(wanted,name)
            got_values=getattr(got,name)
            if got_values.shape!=wanted_values.shape:
                raise ValueError(f'M3D {name} shape differs at cell {index+1}')
            equal=(got_values.view(np.uint32)==wanted_values.view(np.uint32))
            totals[name]+=equal.size
            differences[name]+=int(equal.size-np.count_nonzero(equal))
            delta=np.abs(got_values.astype(np.float64)-
                         wanted_values.astype(np.float64))
            maxima[name]=max(maxima[name],float(delta.max(initial=0.)))
    return {name:(totals[name]-differences[name],totals[name],maxima[name])
            for name in totals}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--hours',type=int)
    parser.add_argument('--numeric-only',action='store_true',
                        help='skip byte-for-byte comparison of fixed-width hourly records')
    args=parser.parse_args()
    results=compare(args.reference,args.candidate,hours=args.hours,
                    exact_records=not args.numeric_only)
    failed=False
    for name,(equal,total,maximum) in results.items():
        agreement=equal/total
        print(f'{name:8s} bitwise={agreement:.6%} ({equal}/{total}) '
              f'max={maximum:.9g}')
        failed|=equal!=total
    if failed:
        raise SystemExit('CALWRF M3D regression mismatch')


if __name__=='__main__':
    main()
