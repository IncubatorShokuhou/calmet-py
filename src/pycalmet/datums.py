"""CALPUFF/NIMA datum definitions and vectorized datum transformations.

The constants reproduce the 21 February 2003 NIMA table embedded in the
legacy COORDLIB blockdat.crd. Radii and shifts are metres; inv_f is the
inverse flattening. Duplicate IDN and CAC source rows are represented once.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np

from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS, Transformer

# Date stamp returned by the legacy COORDLIB ``NIMADATE`` routine.  It is
# part of every TERREL/CTGPROC/MAKEGEO formatted output datum record.
NIMA_DATABASE_DATE = '02-21-2003'


class DatumDefinition(NamedTuple):
    """Ellipsoid and datum-to-WGS84 translation used by COORDLIB."""

    radius: float
    inv_f: float
    dx: float
    dy: float
    dz: float


DATUMS: dict[str, DatumDefinition] = {
    'WGS-84': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'WGS-96': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'WGS-G': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'WGS-72': DatumDefinition(6378135.0, 298.26, 0.0, 0.0, 0.0),
    'NWS-84': DatumDefinition(6370000.0, 1e+20, 0.0, 0.0, 0.0),
    'ESR-S': DatumDefinition(6370997.0, 1e+20, 0.0, 0.0, 0.0),
    'ADI-M': DatumDefinition(6378249.0, 293.465, -166.0, -15.0, 204.0),
    'ADI-E': DatumDefinition(6378249.0, 293.465, -118.0, -14.0, 218.0),
    'ADI-F': DatumDefinition(6378249.0, 293.465, -134.0, -2.0, 210.0),
    'ADI-A': DatumDefinition(6378249.0, 293.465, -165.0, -11.0, 206.0),
    'ADI-C': DatumDefinition(6378249.0, 293.465, -123.0, -20.0, 220.0),
    'ADI-D': DatumDefinition(6378249.0, 293.465, -128.0, -18.0, 224.0),
    'ADI-B': DatumDefinition(6378249.0, 293.465, -161.0, -14.0, 205.0),
    'AFG': DatumDefinition(6378245.0, 298.3, -43.0, -163.0, 45.0),
    'ARF-M': DatumDefinition(6378249.0, 293.465, -143.0, -90.0, -294.0),
    'ARF-A': DatumDefinition(6378249.0, 293.465, -138.0, -105.0, -289.0),
    'ARF-H': DatumDefinition(6378249.0, 293.465, -153.0, -5.0, -292.0),
    'ARF-B': DatumDefinition(6378249.0, 293.465, -125.0, -108.0, -295.0),
    'ARF-C': DatumDefinition(6378249.0, 293.465, -161.0, -73.0, -317.0),
    'ARF-D': DatumDefinition(6378249.0, 293.465, -134.0, -105.0, -295.0),
    'ARF-E': DatumDefinition(6378249.0, 293.465, -169.0, -19.0, -278.0),
    'ARF-F': DatumDefinition(6378249.0, 293.465, -147.0, -74.0, -283.0),
    'ARF-G': DatumDefinition(6378249.0, 293.465, -142.0, -96.0, -293.0),
    'ARS-M': DatumDefinition(6378249.0, 293.465, -160.0, -6.0, -302.0),
    'ARS-A': DatumDefinition(6378249.0, 293.465, -157.0, -2.0, -299.0),
    'ARS-B': DatumDefinition(6378249.0, 293.465, -175.0, -23.0, -303.0),
    'PHA': DatumDefinition(6378249.0, 293.465, -79.0, -129.0, 145.0),
    'BID': DatumDefinition(6378388.0, 297.0, -173.0, 253.0, 27.0),
    'CAP': DatumDefinition(6378249.0, 293.465, -136.0, -108.0, -292.0),
    'CGE': DatumDefinition(6378249.0, 293.465, -263.0, 6.0, 431.0),
    'DAL': DatumDefinition(6378249.0, 293.465, -83.0, 37.0, 124.0),
    'EUR-F': DatumDefinition(6378388.0, 297.0, -130.0, -117.0, -151.0),
    'EUR-T': DatumDefinition(6378388.0, 297.0, -112.0, -77.0, -145.0),
    'LEH': DatumDefinition(6378249.0, 293.465, -130.0, 29.0, 364.0),
    'LIB': DatumDefinition(6378249.0, 293.465, -90.0, 40.0, 88.0),
    'MAS': DatumDefinition(6377397.0, 299.1528, 639.0, 405.0, 60.0),
    'MER': DatumDefinition(6378249.0, 293.465, 31.0, 146.0, 47.0),
    'MIN-A': DatumDefinition(6378249.0, 293.465, -81.0, -84.0, 115.0),
    'MIN-B': DatumDefinition(6378249.0, 293.465, -92.0, -93.0, 122.0),
    'MPO': DatumDefinition(6378249.0, 293.465, -74.0, -130.0, 42.0),
    'NSD': DatumDefinition(6378249.0, 293.465, -186.0, -93.0, 310.0),
    'OEG': DatumDefinition(6378200.0, 298.3, -130.0, 110.0, -13.0),
    'PTB': DatumDefinition(6378249.0, 293.465, -106.0, -129.0, 165.0),
    'PTN': DatumDefinition(6378249.0, 293.465, -148.0, 51.0, -291.0),
    'SCK': DatumDefinition(6377484.0, 299.1528, 616.0, 97.0, -251.0),
    'SRL': DatumDefinition(6378249.0, 293.465, -88.0, 4.0, 101.0),
    'TAN': DatumDefinition(6378388.0, 297.0, -189.0, -242.0, -91.0),
    'VOI': DatumDefinition(6378249.0, 293.465, -73.0, -247.0, 227.0),
    'VOR': DatumDefinition(6378249.0, 293.465, -123.0, -206.0, 219.0),
    'AIN-A': DatumDefinition(6378388.0, 297.0, -150.0, -250.0, -1.0),
    'AIN-B': DatumDefinition(6378388.0, 297.0, -143.0, -236.0, 7.0),
    'BUR': DatumDefinition(6377397.0, 299.1528, -384.0, 664.0, -48.0),
    'BAT': DatumDefinition(6377397.0, 299.1528, -377.0, 681.0, -50.0),
    'EUR-H': DatumDefinition(6378388.0, 297.0, -117.0, -132.0, -164.0),
    'EUR-S': DatumDefinition(6378388.0, 297.0, -103.0, -106.0, -141.0),
    'GSE': DatumDefinition(6377397.0, 299.1528, -403.0, 684.0, 41.0),
    'HEN': DatumDefinition(6378388.0, 297.0, -333.0, -222.0, 114.0),
    'HKD': DatumDefinition(6378388.0, 297.0, -156.0, -271.0, -189.0),
    'HTN': DatumDefinition(6378388.0, 297.0, -637.0, -549.0, -203.0),
    'IND-B': DatumDefinition(6377276.0, 300.8017, 282.0, 726.0, 254.0),
    'IND-I': DatumDefinition(6377301.0, 300.8017, 295.0, 736.0, 257.0),
    'IND-P': DatumDefinition(6377310.0, 300.8017, 283.0, 682.0, 231.0),
    'INF-A': DatumDefinition(6377276.0, 300.8017, 217.0, 823.0, 299.0),
    'ING-A': DatumDefinition(6377276.0, 300.8017, 198.0, 881.0, 317.0),
    'ING-B': DatumDefinition(6377276.0, 300.8017, 182.0, 915.0, 344.0),
    'INH-A': DatumDefinition(6377276.0, 300.8017, 209.0, 818.0, 290.0),
    'INH-A1': DatumDefinition(6377276.0, 300.8017, 210.0, 814.0, 289.0),
    'IDN': DatumDefinition(6378160.0, 298.247, -24.0, -15.0, 5.0),
    'KAN': DatumDefinition(6377276.0, 300.8017, -97.0, 787.0, 86.0),
    'KEA': DatumDefinition(6377304.0, 300.8017, -11.0, 851.0, 5.0),
    'KGS': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'NAH-A': DatumDefinition(6378249.0, 293.465, -247.0, -148.0, 369.0),
    'NAH-B': DatumDefinition(6378249.0, 293.465, -249.0, -156.0, 381.0),
    'NAH-C': DatumDefinition(6378249.0, 293.465, -243.0, -192.0, 477.0),
    'FAH': DatumDefinition(6378249.0, 293.465, -346.0, -1.0, 224.0),
    'PUK': DatumDefinition(6378245.0, 298.3, 28.0, -130.0, -95.0),
    'QAT': DatumDefinition(6378388.0, 297.0, -128.0, -283.0, 22.0),
    'SOA': DatumDefinition(6378155.0, 298.3, 7.0, -10.0, -26.0),
    'TIL': DatumDefinition(6377299.0, 300.8017, -679.0, 669.0, -48.0),
    'TOY-M': DatumDefinition(6377397.0, 299.1528, -148.0, 507.0, 685.0),
    'TOY-A': DatumDefinition(6377397.0, 299.1528, -148.0, 507.0, 685.0),
    'TOY-C': DatumDefinition(6377397.0, 299.1528, -158.0, 507.0, 676.0),
    'TOY-B': DatumDefinition(6377397.0, 299.1528, -146.0, 507.0, 687.0),
    'TOY-B1': DatumDefinition(6377397.0, 299.1528, -147.0, 506.0, 687.0),
    'AUA': DatumDefinition(6378160.0, 298.25, -133.0, -48.0, 148.0),
    'AUG': DatumDefinition(6378160.0, 298.25, -134.0, -48.0, 149.0),
    'EST': DatumDefinition(6377397.0, 299.1528, 374.0, 150.0, 588.0),
    'EUR-M': DatumDefinition(6378388.0, 297.0, -87.0, -98.0, -121.0),
    'EUR-A': DatumDefinition(6378388.0, 297.0, -87.0, -96.0, -120.0),
    'EUR-E': DatumDefinition(6378388.0, 297.0, -104.0, -101.0, -140.0),
    'EUR-G': DatumDefinition(6378388.0, 297.0, -86.0, -96.0, -120.0),
    'EUR-K': DatumDefinition(6378388.0, 297.0, -86.0, -96.0, -120.0),
    'EUR-B': DatumDefinition(6378388.0, 297.0, -84.0, -95.0, -130.0),
    'EUR-I': DatumDefinition(6378388.0, 297.0, -97.0, -103.0, -120.0),
    'EUR-J': DatumDefinition(6378388.0, 297.0, -97.0, -88.0, -135.0),
    'EUR-L': DatumDefinition(6378388.0, 297.0, -107.0, -88.0, -149.0),
    'EUR-C': DatumDefinition(6378388.0, 297.0, -87.0, -95.0, -120.0),
    'EUR-D': DatumDefinition(6378388.0, 297.0, -84.0, -107.0, -120.0),
    'EUS': DatumDefinition(6378388.0, 297.0, -86.0, -98.0, -119.0),
    'HER': DatumDefinition(6377397.0, 299.1528, 682.0, -203.0, 480.0),
    'IRL': DatumDefinition(6377340.0, 299.325, 506.0, -122.0, 611.0),
    'OGB-M': DatumDefinition(6377563.0, 299.325, 375.0, -111.0, 431.0),
    'OGB-A': DatumDefinition(6377563.0, 299.325, 371.0, -112.0, 434.0),
    'OGB-B': DatumDefinition(6377563.0, 299.325, 371.0, -111.0, 434.0),
    'OGB-C': DatumDefinition(6377563.0, 299.325, 384.0, -111.0, 425.0),
    'OGB-D': DatumDefinition(6377563.0, 299.325, 370.0, -108.0, 434.0),
    'MOD': DatumDefinition(6378388.0, 297.0, -225.0, -65.0, 9.0),
    'SPK-A': DatumDefinition(6378245.0, 298.3, 28.0, -121.0, -77.0),
    'SPK-B': DatumDefinition(6378245.0, 298.3, 23.0, -124.0, -82.0),
    'SPK-C': DatumDefinition(6378245.0, 298.3, 26.0, -121.0, -78.0),
    'SPK-D': DatumDefinition(6378245.0, 298.3, 24.0, -124.0, -82.0),
    'SPK-E': DatumDefinition(6378245.0, 298.3, 15.0, -130.0, -84.0),
    'SPK-F': DatumDefinition(6378245.0, 298.3, 24.0, -130.0, -92.0),
    'SPK-G': DatumDefinition(6378245.0, 298.3, 28.0, -121.0, -77.0),
    'CCD': DatumDefinition(6377397.0, 299.1528, 589.0, 76.0, 480.0),
    'CAC': DatumDefinition(6378206.0, 294.9787, -2.0, 151.0, 181.0),
    'NAS-C': DatumDefinition(6378206.0, 294.9787, -8.0, 160.0, 176.0),
    'NAS-B': DatumDefinition(6378206.0, 294.9787, -8.0, 159.0, 175.0),
    'NAS-A': DatumDefinition(6378206.0, 294.9787, -9.0, 161.0, 179.0),
    'NAS-D': DatumDefinition(6378206.0, 294.9787, -5.0, 135.0, 172.0),
    'NAS-V': DatumDefinition(6378206.0, 294.9787, -2.0, 152.0, 149.0),
    'NAS-W': DatumDefinition(6378206.0, 294.9787, 2.0, 204.0, 105.0),
    'NAS-Q': DatumDefinition(6378206.0, 294.9787, -4.0, 154.0, 178.0),
    'NAS-R': DatumDefinition(6378206.0, 294.9787, 1.0, 140.0, 165.0),
    'NAS-E': DatumDefinition(6378206.0, 294.9787, -10.0, 158.0, 187.0),
    'NAS-F': DatumDefinition(6378206.0, 294.9787, -7.0, 162.0, 188.0),
    'NAS-G': DatumDefinition(6378206.0, 294.9787, -22.0, 160.0, 190.0),
    'NAS-H': DatumDefinition(6378206.0, 294.9787, -9.0, 157.0, 184.0),
    'NAS-I': DatumDefinition(6378206.0, 294.9787, 4.0, 159.0, 188.0),
    'NAS-J': DatumDefinition(6378206.0, 294.9787, -7.0, 139.0, 181.0),
    'NAS-O': DatumDefinition(6378206.0, 294.9787, 0.0, 125.0, 201.0),
    'NAS-P': DatumDefinition(6378206.0, 294.9787, -3.0, 142.0, 183.0),
    'NAS-N': DatumDefinition(6378206.0, 294.9787, 0.0, 125.0, 194.0),
    'NAS-T': DatumDefinition(6378206.0, 294.9787, -9.0, 152.0, 178.0),
    'NAS-U': DatumDefinition(6378206.0, 294.9787, 11.0, 114.0, 195.0),
    'NAS-L': DatumDefinition(6378206.0, 294.9787, -12.0, 130.0, 190.0),
    'NAR-A': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'NAR-E': DatumDefinition(6378137.0, 298.2572, -2.0, 0.0, 4.0),
    'NAR-B': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'NAR-C': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'NAR-H': DatumDefinition(6378137.0, 298.2572, 1.0, 1.0, -1.0),
    'NAR-D': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'BOO': DatumDefinition(6378388.0, 297.0, 307.0, 304.0, -318.0),
    'CAI': DatumDefinition(6378388.0, 297.0, -148.0, 136.0, 90.0),
    'CHU': DatumDefinition(6378388.0, 297.0, -134.0, 229.0, -29.0),
    'COA': DatumDefinition(6378388.0, 297.0, -206.0, 172.0, -6.0),
    'PRP-M': DatumDefinition(6378388.0, 297.0, -288.0, 175.0, -376.0),
    'PRP-A': DatumDefinition(6378388.0, 297.0, -270.0, 188.0, -388.0),
    'PRP-B': DatumDefinition(6378388.0, 297.0, -270.0, 183.0, -390.0),
    'PRP-C': DatumDefinition(6378388.0, 297.0, -305.0, 243.0, -442.0),
    'PRP-D': DatumDefinition(6378388.0, 297.0, -282.0, 169.0, -371.0),
    'PRP-E': DatumDefinition(6378388.0, 297.0, -278.0, 171.0, -367.0),
    'PRP-F': DatumDefinition(6378388.0, 297.0, -298.0, 159.0, -369.0),
    'PRP-G': DatumDefinition(6378388.0, 297.0, -279.0, 175.0, -379.0),
    'PRP-H': DatumDefinition(6378388.0, 297.0, -295.0, 173.0, -371.0),
    'HIT': DatumDefinition(6378388.0, 297.0, 16.0, 196.0, 93.0),
    'SAN-M': DatumDefinition(6378160.0, 298.25, -57.0, 1.0, -41.0),
    'SAN-A': DatumDefinition(6378160.0, 298.25, -62.0, -1.0, -37.0),
    'SAN-B': DatumDefinition(6378160.0, 298.25, -61.0, 2.0, -48.0),
    'SAN-C': DatumDefinition(6378160.0, 298.25, -60.0, -2.0, -41.0),
    'SAN-D': DatumDefinition(6378160.0, 298.25, -75.0, -1.0, -44.0),
    'SAN-E': DatumDefinition(6378160.0, 298.25, -44.0, 6.0, -36.0),
    'SAN-F': DatumDefinition(6378160.0, 298.25, -48.0, 3.0, -44.0),
    'SAN-J': DatumDefinition(6378160.0, 298.25, -47.0, 26.0, -42.0),
    'SAN-G': DatumDefinition(6378160.0, 298.25, -53.0, 3.0, -47.0),
    'SAN-H': DatumDefinition(6378160.0, 298.25, -61.0, 2.0, -33.0),
    'SAN-I': DatumDefinition(6378160.0, 298.25, -58.0, 0.0, -44.0),
    'SAN-K': DatumDefinition(6378160.0, 298.25, -45.0, 12.0, -33.0),
    'SAN-L': DatumDefinition(6378160.0, 298.25, -45.0, 8.0, -33.0),
    'SIR': DatumDefinition(6378137.0, 298.2572, 0.0, 0.0, 0.0),
    'YAC': DatumDefinition(6378388.0, 297.0, -155.0, 171.0, 37.0),
    'ZAN': DatumDefinition(6378388.0, 297.0, -265.0, 120.0, -358.0),
    'AIA': DatumDefinition(6378249.0, 293.465, -270.0, 13.0, 62.0),
    'ASC': DatumDefinition(6378388.0, 297.0, -205.0, 107.0, 53.0),
    'SHB': DatumDefinition(6378388.0, 297.0, -320.0, 550.0, -494.0),
    'BER': DatumDefinition(6378206.0, 294.9787, -73.0, 213.0, 296.0),
    'DID': DatumDefinition(6378249.0, 293.465, 260.0, 12.0, -147.0),
    'FOT': DatumDefinition(6378249.0, 293.465, -7.0, 215.0, 225.0),
    'GRA': DatumDefinition(6378388.0, 297.0, -104.0, 167.0, -38.0),
    'HJO': DatumDefinition(6378388.0, 297.0, -73.0, 46.0, -86.0),
    'ISG': DatumDefinition(6378388.0, 297.0, -794.0, 119.0, -298.0),
    'LCF': DatumDefinition(6378206.0, 294.9787, 42.0, 124.0, 147.0),
    'ASM': DatumDefinition(6378249.0, 293.465, 174.0, 359.0, 365.0),
    'NAP': DatumDefinition(6378388.0, 297.0, -10.0, 375.0, 165.0),
    'FLO': DatumDefinition(6378388.0, 297.0, -425.0, -169.0, 81.0),
    'PLN': DatumDefinition(6378388.0, 297.0, -307.0, -92.0, 127.0),
    'POS': DatumDefinition(6378388.0, 297.0, -499.0, -249.0, 314.0),
    'PUR': DatumDefinition(6378206.0, 294.9787, 11.0, 72.0, -101.0),
    'QUO': DatumDefinition(6378388.0, 297.0, 164.0, 138.0, -189.0),
    'SAO': DatumDefinition(6378388.0, 297.0, -203.0, 141.0, 53.0),
    'SAP': DatumDefinition(6378388.0, 297.0, -355.0, 21.0, 72.0),
    'SGM': DatumDefinition(6378388.0, 297.0, -289.0, -124.0, 60.0),
    'TDC': DatumDefinition(6378388.0, 297.0, -632.0, 438.0, -609.0),
    'ANO': DatumDefinition(6378160.0, 298.25, -491.0, -22.0, 435.0),
    'GAA': DatumDefinition(6378388.0, 297.0, -133.0, -321.0, 50.0),
    'IST': DatumDefinition(6378388.0, 297.0, 208.0, -435.0, -229.0),
    'KEG': DatumDefinition(6378388.0, 297.0, 145.0, -187.0, 103.0),
    'MIK': DatumDefinition(6378249.0, 293.465, 41.0, -220.0, -134.0),
    'REU': DatumDefinition(6378388.0, 297.0, 94.0, -948.0, -1262.0),
    'AMA': DatumDefinition(6378206.0, 294.9787, -115.0, 118.0, 426.0),
    'ATF': DatumDefinition(6378388.0, 297.0, 145.0, 75.0, -272.0),
    'TRN': DatumDefinition(6378388.0, 297.0, 114.0, -116.0, -333.0),
    'ASQ': DatumDefinition(6378388.0, 297.0, 124.0, -234.0, -25.0),
    'IBE': DatumDefinition(6378388.0, 297.0, -127.0, -769.0, 472.0),
    'CAZ': DatumDefinition(6378388.0, 297.0, -104.0, -129.0, 239.0),
    'CAO': DatumDefinition(6378388.0, 297.0, 298.0, -304.0, -375.0),
    'CHI': DatumDefinition(6378388.0, 297.0, 175.0, -38.0, 113.0),
    'GIZ': DatumDefinition(6378388.0, 297.0, 230.0, -199.0, -752.0),
    'EAS': DatumDefinition(6378388.0, 297.0, 211.0, 147.0, 111.0),
    'GEO': DatumDefinition(6378388.0, 297.0, 84.0, -22.0, 209.0),
    'GUA': DatumDefinition(6378206.0, 294.9787, -100.0, -248.0, 259.0),
    'DOB': DatumDefinition(6378388.0, 297.0, 252.0, -209.0, -751.0),
    'JOH': DatumDefinition(6378388.0, 297.0, 189.0, -79.0, -202.0),
    'KUS': DatumDefinition(6378388.0, 297.0, 647.0, 1777.0, -1124.0),
    'LUZ-A': DatumDefinition(6378206.0, 294.9787, -133.0, -77.0, -51.0),
    'LUZ-B': DatumDefinition(6378206.0, 294.9787, -133.0, -79.0, -72.0),
    'MID': DatumDefinition(6378388.0, 297.0, 912.0, -58.0, 1227.0),
    'OHA-M': DatumDefinition(6378206.0, 294.9787, 61.0, -285.0, -181.0),
    'OHA-A': DatumDefinition(6378206.0, 294.9787, 89.0, -279.0, -183.0),
    'OHA-B': DatumDefinition(6378206.0, 294.9787, 45.0, -290.0, -172.0),
    'OHA-C': DatumDefinition(6378206.0, 294.9787, 65.0, -290.0, -190.0),
    'OHA-D': DatumDefinition(6378206.0, 294.9787, 58.0, -283.0, -182.0),
    'OHI-M': DatumDefinition(6378388.0, 297.0, 201.0, -228.0, -346.0),
    'OHI-A': DatumDefinition(6378388.0, 297.0, 229.0, -222.0, -348.0),
    'OHI-B': DatumDefinition(6378388.0, 297.0, 185.0, -233.0, -337.0),
    'OHI-C': DatumDefinition(6378388.0, 297.0, 205.0, -233.0, -355.0),
    'OHI-D': DatumDefinition(6378388.0, 297.0, 198.0, -226.0, -347.0),
    'PIT': DatumDefinition(6378388.0, 297.0, 185.0, 165.0, 42.0),
    'SAE': DatumDefinition(6378388.0, 297.0, 170.0, 42.0, 84.0),
    'MVS': DatumDefinition(6378249.0, 293.465, 51.0, 391.0, -36.0),
    'ENW': DatumDefinition(6378270.0, 297.0, 102.0, 52.0, -38.0),
    'WAK': DatumDefinition(6378388.0, 297.0, 276.0, -57.0, 149.0),
}


def normalize_datum(datum: str) -> str:
    """Normalize accepted CALPUFF spelling variants to a NIMA datum ID."""
    normalized = datum.strip().upper().replace("_", "-")
    aliases = {"WGS84": "WGS-84", "WGS96": "WGS-96", "WGS72": "WGS-72",
               "ESR-R": "ESR-S"}
    return aliases.get(normalized, normalized)


def datum_definition(datum: str) -> DatumDefinition:
    """Return a datum definition, raising a useful error for unknown IDs."""
    normalized = normalize_datum(datum)
    try:
        return DATUMS[normalized]
    except KeyError as error:
        raise ValueError(f"unknown CALPUFF datum {datum!r}") from error


def datum_proj4_parameters(datum: str) -> str:
    """Return the PROJ ellipsoid and Helmert parameters for a CALPUFF datum.

    CALPUFF's WGS-72 correction is the standard legacy WGS-72 to WGS-84
    seven-parameter operation.  PROJ expresses rotations in arc-seconds and
    scale in parts per million, matching the NIMA definition used here.
    """
    normalized = normalize_datum(datum)
    definition = datum_definition(normalized)
    if definition.inv_f >= 1.0e20:
        return f"+R={definition.radius:.15g}"
    if normalized == "WGS-72":
        return (f"+a={definition.radius:.15g} +rf={definition.inv_f:.15g} "
                "+towgs84=0,0,4.5,0,0,0.554,0.219")
    return (f"+a={definition.radius:.15g} +rf={definition.inv_f:.15g} "
            f"+towgs84={definition.dx:g},{definition.dy:g},{definition.dz:g}")


@lru_cache(maxsize=None)
def _transformation_crs(datum: str) -> CRS:
    """Build a three-dimensional CRS used only for datum transformations."""
    normalized = normalize_datum(datum)
    definition = datum_definition(normalized)
    # CALPUFF's spherical earth choices define projection geometry.  They do
    # not define a geodetic transformation, so retain their geographic
    # coordinates while PROJ transforms any datum on the other side.
    parameters = (datum_proj4_parameters("WGS-84")
                  if definition.inv_f >= 1.0e20
                  else datum_proj4_parameters(normalized))
    return CRS.from_proj4(f"+proj=longlat {parameters} +type=crs").to_3d()


@lru_cache(maxsize=None)
def _datum_transformer(source: str, target: str) -> Transformer:
    return Transformer.from_crs(
        _transformation_crs(source),
        _transformation_crs(target),
        always_xy=True,
        allow_ballpark=False,
    )


def datum_shift(longitude, latitude, height=0.0, *, source="WGS-84", target="WGS-84"):
    """Transform CALPUFF datum coordinates with PROJ in float64 precision."""
    source_name, target_name = normalize_datum(source), normalize_datum(target)
    datum_definition(source_name)
    datum_definition(target_name)
    lon, lat, z = np.broadcast_arrays(
        np.asarray(longitude, dtype=np.float64),
        np.asarray(latitude, dtype=np.float64),
        np.asarray(height, dtype=np.float64),
    )
    if source_name == target_name:
        return lon.copy(), lat.copy(), z.copy()
    transformed = _datum_transformer(source_name, target_name).transform(
        lon.ravel(), lat.ravel(), z.ravel(), errcheck=True)
    shifted_lon = np.asarray(transformed[0], dtype=np.float64).reshape(lon.shape)
    shifted_lat = np.asarray(transformed[1], dtype=np.float64).reshape(lat.shape)
    shifted_height = np.asarray(transformed[2], dtype=np.float64).reshape(z.shape)
    # PROJ normalizes geographic longitudes.  CALPUFF accepts continuous
    # longitude branches around the antimeridian, so restore the branch of
    # each source coordinate after applying the geodetic transformation.
    shifted_lon += 360.0 * np.rint((lon - shifted_lon) / 360.0)
    return shifted_lon, shifted_lat, shifted_height
