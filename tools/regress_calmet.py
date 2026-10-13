#!/usr/bin/env python3
"""Compare every field in one period of two legacy CALMET.DAT files."""
from __future__ import annotations

import argparse
from datetime import datetime,timedelta
import struct

import numpy as np

from pycalmet.igf_data import InitialGuessCalmetData
from pycalmet.legacy_io import FortranSequentialReader


def _timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace('_','T'))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f'invalid ending time: {value}') from error


def _period_payloads(path,ending):
    fields=[]
    with FortranSequentialReader(path) as reader:
        while (payload:=reader.read_record()) is not None:
            if len(payload)<24: continue
            try: label=payload[:8].decode('ascii').rstrip()
            except UnicodeDecodeError: continue
            if not (label.startswith(('U-LEV','V-LEV','WFACE','T-LEV')) or
                    label in {'IPGT','USTAR','ZI','EL','WSTAR','RMM','TEMPK',
                              'RHO','QSW','IRH','IPCODE'}):
                continue
            _begin,_begin_seconds,end_code,end_seconds=struct.unpack_from('<4i',payload,8)
            text=str(end_code)
            if len(text)!=9: continue
            end=datetime(int(text[:4]),1,1)+timedelta(
                days=int(text[4:7])-1,hours=int(text[7:]),seconds=end_seconds)
            if end==ending: fields.append(payload)
    return fields


def compare_record_bytes(reference_path,candidate_path,ending):
    wanted=_period_payloads(reference_path,ending)
    got=_period_payloads(candidate_path,ending)
    if len(got)!=len(wanted):
        raise ValueError(f'CALMET period record count differs: {len(got)} != {len(wanted)}')
    for index,(actual,expected) in enumerate(zip(got,wanted,strict=True),1):
        if actual!=expected:
            raise ValueError(f'CALMET period record {index} differs byte-for-byte')
    return len(got)


def compare(reference_path,candidate_path,ending,*,require_bitwise=False,
            exact_records=False):
    options=dict(retain_fields=True,field_endings=(ending,),
                 record_endings=(ending,))
    reference=InitialGuessCalmetData(reference_path,**options)
    candidate=InitialGuessCalmetData(candidate_path,**options)
    if reference.grid!=candidate.grid:
        raise ValueError('CALMET grids differ')
    if not np.array_equal(reference.z_faces_m,candidate.z_faces_m):
        raise ValueError('CALMET vertical faces differ')
    reference_begin,expected=reference.fields_at_end(ending)
    candidate_begin,actual=candidate.fields_at_end(ending)
    if reference_begin!=candidate_begin:
        raise ValueError('CALMET period beginnings differ')
    if tuple(expected)!=tuple(actual):
        raise ValueError('CALMET field labels or order differ')
    failed=False; rows=[]
    for label,wanted in expected.items():
        got=np.asarray(actual[label]); wanted=np.asarray(wanted)
        if got.shape!=wanted.shape:
            raise ValueError(f'{label} shapes differ: {got.shape} != {wanted.shape}')
        if got.dtype.kind in 'iu':
            agreement=float(np.mean(got==wanted))
            delta=np.abs(got.astype(np.int64)-wanted.astype(np.int64))
            maximum=int(np.max(delta))
            location=np.unravel_index(int(np.argmax(delta)),delta.shape)
            rows.append((label,'integer',agreement,maximum,location))
            failed|=agreement!=1.
        else:
            if not np.isfinite(got).all() or not np.isfinite(wanted).all():
                raise ValueError(f'{label} contains non-finite values')
            delta=np.abs(got.astype(np.float64)-wanted.astype(np.float64))
            bitwise=float(np.mean(got.view(np.uint32)==wanted.view(np.uint32)))
            location=np.unravel_index(int(np.argmax(delta)),delta.shape)
            rows.append((label,'float',float(delta.mean()),float(delta.max()),
                         bitwise,location))
            failed|=require_bitwise and bitwise!=1.
    if exact_records:
        compare_record_bytes(reference_path,candidate_path,ending)
    return rows,failed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--ending',required=True,type=_timestamp)
    parser.add_argument('--require-bitwise',action='store_true')
    parser.add_argument('--numeric-only',action='store_true',
                        help='skip byte-for-byte comparison of the selected period records')
    args=parser.parse_args()
    rows,failed=compare(args.reference,args.candidate,args.ending,
                        require_bitwise=args.require_bitwise,
                        exact_records=not args.numeric_only)
    for row in rows:
        if row[1]=='integer':
            print(f'{row[0]:8s} integer agreement={row[2]:.6%} max={row[3]} '
                  f'at={row[4]}')
        else:
            print(f'{row[0]:8s} mae={row[2]:.9g} max={row[3]:.9g} '
                  f'bitwise={row[4]:.6%} at={row[5]}')
    if failed:
        raise SystemExit('CALMET regression mismatch')


if __name__=='__main__':
    main()
