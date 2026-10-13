#!/usr/bin/env python3
"""Compare every gridded record in two formatted CALWRF M2D files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalmet.legacy_m2d import M2DReader


def _data_stream(path):
    """Open *path* at the first M2D variable record."""
    stream=Path(path).open('rb',buffering=1024*1024)
    try:
        stream.readline()
        comments=int(stream.readline())
        comment_lines=[stream.readline() for _ in range(comments)]
        variables=int(comment_lines[1].split()[0])
        stream.readline()  # output switches
        stream.readline()  # source-grid definition
        stream.readline()  # physics options
        period=stream.readline().split()
        nx,ny,nz=map(int,period[2:5])
        stream.readline()  # extraction bounds
        for _ in range(nz+nx*ny): stream.readline()
        return stream,nx,ny,nz,variables
    except Exception:
        stream.close()
        raise


def compare_record_bytes(reference_path,candidate_path,*,hours):
    """Require byte-identical variable labels and fixed-width grid rows."""
    reference,nx,ny,nz,variables=_data_stream(reference_path)
    candidate,cx,cy,cz,cvariables=_data_stream(candidate_path)
    try:
        if (cx,cy,cz,cvariables)!=(nx,ny,nz,variables):
            raise ValueError('M2D byte comparison dimensions or variable count differ')
        lines=int(hours)*variables*(1+ny*((nx+7)//8))
        for index in range(lines):
            wanted=reference.readline(); got=candidate.readline()
            if got!=wanted:
                raise ValueError(f'M2D fixed-width record line {index+1} differs byte-for-byte')
        return lines
    finally:
        reference.close(); candidate.close()


def compare(reference_path,candidate_path,*,hours=None,exact_records=False):
    reference_header,reference_records=M2DReader(reference_path).records()
    candidate_header,candidate_records=M2DReader(candidate_path).records()
    header_fields=('nx','ny','nz','start','variables','source_nx','source_ny',
                   'source_nz','i_start','j_start','k_start')
    for name in header_fields:
        wanted=getattr(reference_header,name)
        got=getattr(candidate_header,name)
        if got!=wanted:
            raise ValueError(f'M2D header {name} differs: {got!r} != {wanted!r}')
    count=(min(reference_header.nhours,candidate_header.nhours)
           if hours is None else int(hours))
    if count<1 or count>reference_header.nhours or count>candidate_header.nhours:
        raise ValueError('requested M2D hour count is outside either file')
    if exact_records:
        compare_record_bytes(reference_path,candidate_path,hours=count)
    rows=[]
    total=count*len(reference_header.variables)
    for index in range(total):
        try:
            wanted=next(reference_records); got=next(candidate_records)
        except StopIteration as error:
            raise ValueError('M2D file ends before the requested records') from error
        if (got.date,got.name)!=(wanted.date,wanted.name):
            raise ValueError(
                f'M2D record {index+1} differs: {(got.date,got.name)!r} != '
                f'{(wanted.date,wanted.name)!r}')
        if got.values.shape!=wanted.values.shape:
            raise ValueError(f'M2D {got.name} grid shape differs')
        equal=got.values.view(np.uint32)==wanted.values.view(np.uint32)
        agreement=float(np.mean(equal))
        delta=np.abs(got.values.astype(np.float64)-wanted.values.astype(np.float64))
        rows.append((index//len(reference_header.variables)+1,got.name,
                     agreement,float(delta.max(initial=0.))))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--hours',type=int)
    parser.add_argument('--numeric-only',action='store_true',
                        help='skip byte-for-byte comparison of fixed-width hourly records')
    args=parser.parse_args()
    rows=compare(args.reference,args.candidate,hours=args.hours,
                 exact_records=not args.numeric_only)
    failed=False
    for hour,name,agreement,maximum in rows:
        print(f'hour={hour:4d} {name:8s} bitwise={agreement:.6%} max={maximum:.9g}')
        failed|=agreement!=1.
    if failed:
        raise SystemExit('CALWRF M2D regression mismatch')


if __name__=='__main__':
    main()
