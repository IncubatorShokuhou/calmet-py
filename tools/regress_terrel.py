#!/usr/bin/env python3
"""Compare every terrain cell in two legacy TERREL outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalmet._makegeo import _read_terrain
from pycalmet.control import terrel_config


def _terrain_data_lines(path):
    lines=Path(path).read_bytes().splitlines(keepends=True)
    try:
        comments=int(lines[1])
        index=2+comments
        projection=lines[index][:8].strip().upper(); index+=2
        if projection in {b'LCC',b'LAZA',b'TTM'}: index+=1
        index+=4  # datum, grid geometry, units and record ordering
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid TERREL header in {path}') from error
    return lines[index:]


def compare_data_lines(candidate,reference):
    wanted=_terrain_data_lines(reference); got=_terrain_data_lines(candidate)
    if len(got)!=len(wanted):
        raise ValueError(f'TERREL data line count differs: {len(got)} != {len(wanted)}')
    for index,(actual,expected) in enumerate(zip(got,wanted,strict=True),1):
        if actual!=expected:
            raise ValueError(f'TERREL data line {index} differs byte-for-byte')
    return len(got)


def compare(control,reference,candidate,*,exact_records=False):
    grid=terrel_config(control).grid
    wanted=_read_terrain(reference,grid)
    got=_read_terrain(candidate,grid)
    equal=got.view(np.uint64)==wanted.view(np.uint64)
    delta=np.abs(got-wanted)
    if exact_records:
        compare_data_lines(candidate,reference)
    return int(np.count_nonzero(equal)),equal.size,float(delta.max(initial=0.))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--numeric-only',action='store_true',
                        help='skip byte-for-byte comparison of legacy terrain records')
    args=parser.parse_args()
    equal,total,maximum=compare(args.control,args.reference,args.candidate,
                                exact_records=not args.numeric_only)
    print(f'terrain bitwise={equal/total:.6%} ({equal}/{total}) max={maximum:.9g}')
    if equal!=total:
        raise SystemExit('TERREL regression mismatch')


if __name__=='__main__':
    main()
