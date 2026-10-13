#!/usr/bin/env python3
"""Compare cloud-water records in two CALMET.AUX 1.1 files."""
from __future__ import annotations

import argparse
import struct

import numpy as np

from pycalmet import AuxiliaryData
from pycalmet.legacy_io import FortranSequentialReader


def _hour_payloads(path,maximum_hours):
    hours=[]
    with FortranSequentialReader(path) as reader:
        reader.read_record()
        count=struct.unpack('<i',reader.read_record())[0]
        for _ in range(count): reader.read_record()
        run=reader.read_record(); nz=struct.unpack_from('<i',run,60)[0]
        for _ in range(3): reader.read_record()  # ZFACE and variable tables
        while maximum_hours is None or len(hours)<maximum_hours:
            first=reader.read_record()
            if first is None: break
            records=[first]
            # Three 2-D variables: first timestamp already read, then count/data;
            # subsequent fields repeat the timestamp.  The 3-D field has one
            # timestamp followed by count/data pairs for every vertical layer.
            for field in range(3):
                if field: records.append(reader.read_record())
                records.extend((reader.read_record(),reader.read_record()))
            records.append(reader.read_record())
            for _ in range(nz):
                records.extend((reader.read_record(),reader.read_record()))
            if any(record is None for record in records):
                raise ValueError('CALMET.AUX ends inside an hourly record group')
            hours.append(records)
    return hours


def compare_record_bytes(reference_path,candidate_path,maximum_hours):
    wanted=_hour_payloads(reference_path,maximum_hours)
    got=_hour_payloads(candidate_path,maximum_hours)
    if len(got)!=len(wanted):
        raise ValueError(f'CALMET.AUX hour count differs: {len(got)} != {len(wanted)}')
    count=0
    for hour,(actual,expected) in enumerate(zip(got,wanted,strict=True),1):
        if len(actual)!=len(expected):
            raise ValueError(f'CALMET.AUX hour {hour} record count differs')
        for index,(left,right) in enumerate(zip(actual,expected,strict=True),1):
            if left!=right:
                raise ValueError(
                    f'CALMET.AUX hour {hour} record {index} differs byte-for-byte')
            count+=1
    return count


def compare(reference_path,candidate_path,*,maximum_hours=None,require_bitwise=False,
            exact_records=False):
    rows=[]; failed=False; hour_index=0
    with AuxiliaryData(reference_path) as reference, AuxiliaryData(candidate_path) as candidate:
        expected_grid=(reference.header.nx,reference.header.ny,reference.header.nz)
        actual_grid=(candidate.header.nx,candidate.header.ny,candidate.header.nz)
        if actual_grid!=expected_grid:
            raise ValueError(f'CALMET.AUX grids differ: {actual_grid} != {expected_grid}')
        if not np.array_equal(reference.header.z_faces_m,candidate.header.z_faces_m):
            raise ValueError('CALMET.AUX vertical faces differ')
        while maximum_hours is None or hour_index<maximum_hours:
            expected=reference.read_hour(); actual=candidate.read_hour()
            if expected is None or actual is None:
                if expected is not None or actual is not None:
                    raise ValueError('CALMET.AUX files contain different hour counts')
                break
            expected_stamp=(expected.begin_code,expected.begin_seconds,
                            expected.end_code,expected.end_seconds)
            actual_stamp=(actual.begin_code,actual.begin_seconds,
                          actual.end_code,actual.end_seconds)
            if actual_stamp!=expected_stamp:
                raise ValueError(f'CALMET.AUX timestamps differ: {actual_stamp} != {expected_stamp}')
            fields=(
                ('CLDMRUP',expected.cloud_water_above_g_kg,actual.cloud_water_above_g_kg),
                ('ZCLDBOT',expected.cloud_bottom_above_m,actual.cloud_bottom_above_m),
                ('ZCLDTOP',expected.cloud_top_above_m,actual.cloud_top_above_m),
                ('CLDMR3D',expected.cloud_water_3d_g_kg,actual.cloud_water_3d_g_kg),
            )
            for label,wanted,got in fields:
                delta=np.abs(got.astype(np.float64)-wanted.astype(np.float64))
                agreement=float(np.mean(got.view(np.uint32)==wanted.view(np.uint32)))
                rows.append((hour_index+1,label,float(delta.mean()),float(delta.max()),agreement))
                failed|=require_bitwise and agreement!=1.
            hour_index+=1
    if exact_records:
        compare_record_bytes(reference_path,candidate_path,maximum_hours)
    return rows,failed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--hours',type=int)
    parser.add_argument('--require-bitwise',action='store_true')
    parser.add_argument('--numeric-only',action='store_true',
                        help='skip byte-for-byte comparison of hourly AUX records')
    args=parser.parse_args()
    if args.hours is not None and args.hours<1:
        parser.error('--hours must be positive')
    rows,failed=compare(args.reference,args.candidate,maximum_hours=args.hours,
                        require_bitwise=args.require_bitwise,
                        exact_records=not args.numeric_only)
    for hour,label,mean,maximum,agreement in rows:
        print(f'hour={hour:4d} {label:8s} mae={mean:.9g} max={maximum:.9g} '
              f'bitwise={agreement:.6%}')
    if failed:
        raise SystemExit('CALMET.AUX regression mismatch')


if __name__=='__main__':
    main()
