"""Fixed legacy lookup tables used by CTGPROC."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True,slots=True)
class Nlcd92Spec:
    nx: int
    ny: int
    x_origin_km: float
    y_origin_km: float


@dataclass(frozen=True,slots=True)
class ModisSpec:
    nx: int=43200
    ny: int=21600
    x_origin: float=-180.+1./240.
    y_origin: float=90.-1./240.
    spacing: float=1./120.


MODIS_SPEC=ModisSpec()


NLCD92_SPECS={
    'alabama':Nlcd92Spec(11864,19175,696.27,1386.3),
    'arizona':Nlcd92Spec(20036,23328,-1747.23,1701.78),
    'arkansas':Nlcd92Spec(14869,13461,122.34,1510.8),
    'california_north':Nlcd92Spec(14971,20260,-2361.9,2455.83),
    'california_south':Nlcd92Spec(23851,20260,-2361.9,1849.83),
    'colorado':Nlcd92Spec(21418,16915,-1146.81,2074.02),
    'connecticut':Nlcd92Spec(5248,5264,1831.26,2368.23),
    'delaware':Nlcd92Spec(3300,5170,1703.49,2055.69),
    'florida':Nlcd92Spec(27314,23393,783.81,969.66),
    'georgia':Nlcd92Spec(17074,17648,926.91,1417.05),
    'idaho':Nlcd92Spec(17567,28387,-1715.97,3060.15),
    'illinois':Nlcd92Spec(11835,21096,378.03,2201.31),
    'indiana':Nlcd92Spec(9184,15757,686.85,2139.03),
    'iowa':Nlcd92Spec(17834,11712,-52.62,2288.88),
    'kansas':Nlcd92Spec(21829,11788,-532.68,1903.92),
    'kentucky':Nlcd92Spec(21849,11189,569.07,1849.05),
    'louisiana':Nlcd92Spec(17234,14956,181.38,1115.49),
    'maine':Nlcd92Spec(11408,17729,1923.78,3016.02),
    'maryland':Nlcd92Spec(13575,7014,1396.05,2038.29),
    'massachusetts':Nlcd92Spec(10672,6312,1826.46,2483.43),
    'michigan':Nlcd92Spec(22795,23944,428.49,2835.39),
    'minnesota':Nlcd92Spec(19430,21780,-92.4,2931.27),
    'mississippi':Nlcd92Spec(11058,18639,407.97,1361.55),
    'missouri':Nlcd92Spec(19665,17045,18.6,1964.1),
    'montana':Nlcd92Spec(30100,19094,-1498.26,3044.73),
    'nebraska':Nlcd92Spec(24363,12153,-671.73,2250.99),
    'nevada':Nlcd92Spec(18756,28903,-2037.84,2358.6),
    'new_hampshire':Nlcd92Spec(5364,10237,1879.38,2734.83),
    'new_jersey':Nlcd92Spec(4297,9932,1721.79,2242.86),
    'new_mexico':Nlcd92Spec(20583,21277,-1233.9,1629.78),
    'new_york':Nlcd92Spec(23005,17455,1317.21,2663.82),
    'north_carolina':Nlcd92Spec(26378,12094,1047.69,1699.53),
    'north_dakota':Nlcd92Spec(19386,12143,-624.21,2913.75),
    'ohio':Nlcd92Spec(12371,15385,922.17,2251.08),
    'oklahoma':Nlcd92Spec(25428,13368,-621.21,1573.8),
    'oregon':Nlcd92Spec(23924,20221,-2301.36,2907.51),
    'pennsylvania':Nlcd92Spec(17370,11105,1261.38,2295.15),
    'rhode_island':Nlcd92Spec(2463,3507,1968.93,2374.11),
    'south_carolina':Nlcd92Spec(15620,11915,1138.29,1458.54),
    'south_dakota':Nlcd92Spec(20585,13774,-652.56,2577.51),
    'tennessee':Nlcd92Spec(26006,9396,498.66,1613.76),
    'texas_n':Nlcd92Spec(28285,16717,-664.32,1518.24),
    'texas_se':Nlcd92Spec(20078,24577,-364.86,1047.9),
    'texas_sw':Nlcd92Spec(22199,20435,-1000.65,1047.9),
    'utah':Nlcd92Spec(16563,20729,-1582.08,2250.99),
    'vermont':Nlcd92Spec(5659,10140,1758.93,2707.02),
    'virginia':Nlcd92Spec(23630,13032,1088.7,1967.1),
    'washington_dc':Nlcd92Spec(662,784,1610.19,1936.68),
    'washington':Nlcd92Spec(19975,14795,-2144.46,3177.72),
    'west_virginia':Nlcd92Spec(13265,13240,1154.67,2064.6),
    'wisconsin':Nlcd92Spec(17414,18513,241.98,2715.09),
    'wyoming':Nlcd92Spec(20575,17088,-1250.67,2539.89),
}
