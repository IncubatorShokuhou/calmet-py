"""Readers for the big-endian GSHHS polygon formats used by CTGPROC."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO,Iterator

import numpy as np

from .config import GridConfig
from .datums import datum_shift
from .projections import project_geographic,unproject_geographic


@dataclass(frozen=True,slots=True)
class GshhsHeader:
    """Normalized BREADH fields for one GSHHS polygon."""

    identifier: int
    point_count: int
    level: int
    west_microdegrees: int
    east_microdegrees: int
    south_microdegrees: int
    north_microdegrees: int
    area_tenth_km2: int
    version: int
    crosses_greenwich: bool
    source: int
    river_lake: bool = False
    full_area_tenth_km2: int = 0
    container: int = -1
    ancestor: int = -1


@dataclass(frozen=True,slots=True)
class GshhsPolygon:
    """One decoded polygon with east longitude and north latitude degrees."""

    header: GshhsHeader
    longitude: np.ndarray
    latitude: np.ndarray


def boundary_side(x,y,xmax,xmin,ymax,ymin):
    """Vectorized transcription of COASTEXT ISIDE (west/north/east/south)."""
    x_array,y_array=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    west=np.abs(x_array-xmin); east=np.abs(x_array-xmax)
    south=np.abs(y_array-ymin); north=np.abs(y_array-ymax)
    side=np.empty(x_array.shape,dtype=np.int8)
    vertical=(x_array>xmin)&(x_array<xmax)&((y_array<=ymin)|(y_array>=ymax))
    side[vertical]=np.where(south[vertical]<north[vertical],4,2)
    horizontal=(~vertical)&(y_array>ymin)&(y_array<ymax)&((x_array<=xmin)|(x_array>=xmax))
    side[horizontal]=np.where(east[horizontal]<west[horizontal],3,1)
    remaining=~(vertical|horizontal)
    inside=((x_array>=xmin)&(x_array<=xmax)&(y_array>=ymin)&(y_array<=ymax))
    choose_horizontal=np.where(inside,np.minimum(south,north)>np.minimum(east,west),
                               np.minimum(south,north)<np.minimum(east,west))
    use_horizontal=remaining&choose_horizontal
    side[use_horizontal]=np.where(east[use_horizontal]<west[use_horizontal],3,1)
    use_vertical=remaining&~choose_horizontal
    side[use_vertical]=np.where(south[use_vertical]<north[use_vertical],4,2)
    return int(side) if not side.shape else side


def _append_corner(points,side1,side2,bounds):
    """PCORNER, including its consecutive duplicate cancellation."""
    xmin,xmax,ymin,ymax=bounds
    product=side1*side2
    corners={2:(xmin,ymax),6:(xmax,ymax),12:(xmax,ymin),4:(xmin,ymin)}
    if side1==side2 or product not in corners:
        raise ValueError(f'invalid coastline side transition {side1}->{side2}')
    corner=corners[product]
    if points and points[-1]==corner:
        points.pop()
    else:
        points.append(corner)


def clip_coast_polygon(longitude,latitude,x,y,kind,geographic_bounds,
                       projected_bounds,polygon_bounds=None):
    """Reproduce COASTEXT's segment retention and directional corner closure.

    Bounds use ``(xmin, xmax, ymin, ymax)``. Geographic coordinates determine
    whether a vertex is retained; projected coordinates are written to BLN.
    """
    lon,lat,px,py=np.broadcast_arrays(*(np.asarray(value,dtype=float)
                                        for value in (longitude,latitude,x,y)))
    if lon.ndim!=1 or lon.size==0: raise ValueError('coastline polygon must be a non-empty vector')
    gxmin,gxmax,gymin,gymax=geographic_bounds
    xmin,xmax,ymin,ymax=projected_bounds
    inside=(lon>=gxmin)&(lon<=gxmax)&(lat>=gymin)&(lat<=gymax)
    points=[]; sides_seen=set(); good=False; starts_inside=False; first=True
    previous=None; start_side=end_side=exit_side=None; exit_point=None; exit_index=None
    for index,(is_inside,xp,yp,xg,yg) in enumerate(zip(inside,px,py,lon,lat)):
        point=(float(xp),float(yp))
        if is_inside:
            if index==0: starts_inside=True
            if not good:
                good=True
                if first:
                    first=False
                    if not starts_inside and previous is not None: points.append(previous)
                    start_side=boundary_side(xp,yp,xmax,xmin,ymax,ymin)
                elif previous is not None:
                    points.append(previous)
            points.append(point)
        elif good:
            end_side=boundary_side(xp,yp,xmax,xmin,ymax,ymin)
            exit_point=point; exit_index=len(points); exit_side=boundary_side(
                xg,yg,gxmax,gxmin,gymax,gymin)
            points.append(point); good=False
        else:
            previous=point
            next_side=boundary_side(xg,yg,gxmax,gxmin,gymax,gymin)
            if not first and next_side!=end_side:
                _append_corner(points,next_side,end_side,projected_bounds)
                end_side=next_side
            else:
                end_side=next_side
            sides_seen.add(end_side)
        previous=point

    if first:
        bounds=(np.min(lon),np.max(lon),np.min(lat),np.max(lat)) if polygon_bounds is None else polygon_bounds
        surrounds=(gxmax<=bounds[1] and gxmin>=bounds[0] and
                   gymax<=bounds[3] and gymin>=bounds[2] and sides_seen=={1,2,3,4})
        if not surrounds: return None
        points=[]; direction=1 if kind%2==0 else -1
        side=1
        if direction<0: side=1
        for _ in range(5):
            next_side=((side-1+direction)%4)+1
            _append_corner(points,side,next_side,projected_bounds)
            side=next_side
        return np.asarray(points,dtype=float)

    if not starts_inside and exit_index is not None:
        points=points[:exit_index+1]
        end_side=exit_side
        difference=0.
        if start_side==end_side:
            if start_side==1: difference=exit_point[1]-points[0][1]
            elif start_side==2: difference=exit_point[0]-points[0][0]
            elif start_side==3: difference=points[0][1]-exit_point[1]
            else: difference=points[0][0]-exit_point[0]
        if kind in {1,3} and difference<0.:
            next_side=4 if end_side==1 else end_side-1
            _append_corner(points,end_side,next_side,projected_bounds); end_side=next_side
        elif kind in {2,4} and difference>0.:
            next_side=1 if end_side==4 else end_side+1
            _append_corner(points,end_side,next_side,projected_bounds); end_side=next_side
        direction=-1 if kind in {1,3} else 1
        while start_side!=end_side:
            next_side=((end_side-1+direction)%4)+1
            _append_corner(points,end_side,next_side,projected_bounds)
            end_side=next_side
    if points and points[-1]!=points[0]: points.append(points[0])
    return np.asarray(points,dtype=float) if len(points)>=3 else None


def _packed_header(values: tuple[int,...],version: int) -> GshhsHeader:
    identifier,count,flag,west,east,south,north,area,*extra=values
    return GshhsHeader(
        identifier,count,flag&255,west,east,south,north,area,version,
        bool((flag>>16)&1),(flag>>24)&1,bool((flag>>25)&1),
        *(tuple(extra)+(0,-1,-1))[:3],
    )


def _read_header(stream: BinaryIO,version: int | None) -> tuple[GshhsHeader,int]:
    """Read one header, detecting the historical layout on the first call."""
    start=stream.tell(); prefix=stream.read(12)
    if not prefix:
        raise EOFError
    if len(prefix)!=12:
        raise ValueError('truncated GSHHS polygon header')
    identifier,count,third=struct.unpack('>3i',prefix)
    if count<0:
        raise ValueError(f'negative GSHHS point count for polygon {identifier}')
    packed_version=(third>>8)&255
    stream.seek(start)
    if version is None:
        if packed_version>=10: version=10
        elif packed_version in {4,5}: version=packed_version
        else:
            candidate=stream.read(40)
            if len(candidate)==40 and struct.unpack('>9i2h',candidate)[8]==3:
                version=3
            else:
                version=2
            stream.seek(start)

    if version>=10:
        payload=stream.read(44)
        if len(payload)!=44: raise ValueError('truncated GSHHS v2 polygon header')
        values=struct.unpack('>11i',payload)
        actual=(values[2]>>8)&255
        if actual<10: raise ValueError('inconsistent GSHHS v2 version flag')
        return _packed_header(values,actual),version
    if version in {4,5}:
        payload=stream.read(32)
        if len(payload)!=32: raise ValueError('truncated GSHHS v1.4 polygon header')
        values=struct.unpack('>8i',payload)
        actual=(values[2]>>8)&255
        if actual not in {4,5}: raise ValueError('inconsistent GSHHS v1.4 version flag')
        return _packed_header(values,actual),version
    if version==3:
        payload=stream.read(40)
        if len(payload)!=40: raise ValueError('truncated GSHHS v1.3 polygon header')
        identifier,count,level,west,east,south,north,area,actual,greenwich,source=struct.unpack(
            '>9i2h',payload)
        if actual!=3: raise ValueError('inconsistent GSHHS v1.3 version field')
        return GshhsHeader(identifier,count,level,west,east,south,north,area,
                           actual,bool(greenwich),source),version
    payload=stream.read(36)
    if len(payload)!=36: raise ValueError('truncated legacy GSHHS polygon header')
    identifier,count,level,west,east,south,north,area,greenwich,source=struct.unpack(
        '>8i2h',payload)
    return GshhsHeader(identifier,count,level,west,east,south,north,area,
                       3,bool(greenwich),source),version


def iter_gshhs(path: str | Path) -> Iterator[GshhsPolygon]:
    """Yield all polygons while enforcing one consistent file version."""
    version=None
    with Path(path).open('rb') as stream:
        while True:
            try:
                header,version=_read_header(stream,version)
            except EOFError:
                return
            payload=stream.read(header.point_count*8)
            if len(payload)!=header.point_count*8:
                raise ValueError(f'truncated GSHHS points for polygon {header.identifier}')
            points=np.frombuffer(payload,dtype='>i4').reshape(header.point_count,2)
            yield GshhsPolygon(
                header,
                points[:,0].astype(np.float64)/1.e6,
                points[:,1].astype(np.float64)/1.e6,
            )


def _longitude_branch(longitude,center):
    values=np.asarray(longitude,dtype=float)
    return (values-center+180.)%360.-180.+center


def extract_gshhs(source: str | Path,output: str | Path,grid: GridConfig,
                  *,source_datum='WGS-84',wdbii_datum='WGS-72') -> int:
    """Extract COASTEXT-compatible projected polygons and write a BLN file."""
    x0,y0=grid.x_origin_km,grid.y_origin_km
    x1=x0+grid.nx*grid.dx_km; y1=y0+grid.ny*grid.dy_km
    width=(x1-x0)*.01; height=(y1-y0)*.01
    projected_bounds=(x0-width,x1+width,y0-height,y1+height)
    corner_x=np.array([x0,x0,x1,x1]); corner_y=np.array([y0,y1,y0,y1])
    corner_lon,corner_lat=unproject_geographic(corner_x,corner_y,grid,
                                               grid.datum,units='km')
    center=float(np.mean(corner_lon)); corner_lon=_longitude_branch(corner_lon,center)
    mean_lat=.5*(float(np.min(corner_lat))+float(np.max(corner_lat)))
    longitude_margin=width/107.9*np.cos(np.radians(mean_lat))
    latitude_margin=height/107.9
    geographic_bounds=(float(np.min(corner_lon))-longitude_margin,
                       float(np.max(corner_lon))+longitude_margin,
                       float(np.min(corner_lat))-latitude_margin,
                       float(np.max(corner_lat))+latitude_margin)
    records=[]
    for polygon in iter_gshhs(source):
        header=polygon.header
        if header.point_count==0 or header.area_tenth_km2==0 or header.level not in {1,2,3,4}:
            continue
        # COASTEXT recognizes source zero as WDBII, but the released code
        # fixes LWDBII=.FALSE. and consumes those points without retaining
        # the polygon.  DWDBII remains a parsed legacy control for fidelity.
        if header.source==0:
            continue
        polygon_datum=source_datum
        source_lon=_longitude_branch(polygon.longitude,center)
        output_lon,output_lat,_=datum_shift(source_lon,polygon.latitude,
                                            source=polygon_datum,target=grid.datum)
        output_lon=_longitude_branch(output_lon,center)
        gx0,gx1,gy0,gy1=geographic_bounds
        polygon_bounds=(float(np.min(output_lon)),float(np.max(output_lon)),
                        float(np.min(output_lat)),float(np.max(output_lat)))
        if gx1<polygon_bounds[0] or gx0>polygon_bounds[1] or gy1<polygon_bounds[2] or gy0>polygon_bounds[3]:
            continue
        px,py=project_geographic(source_lon,polygon.latitude,grid,polygon_datum,
                                 units='km')
        points=clip_coast_polygon(output_lon,output_lat,px,py,header.level,
                                  geographic_bounds,projected_bounds,polygon_bounds)
        if points is None: continue
        surrounds=(len(points)==5 and np.array_equal(points[0],points[-1]) and
                   np.isin(points[:-1,0],projected_bounds[:2]).all() and
                   np.isin(points[:-1,1],projected_bounds[2:]).all())
        if surrounds and records:
            previous=records[-1]
            records[-1]=(header.level,*previous[1:])
            continue
        records.append((header.level,header.source+1,header.identifier,points))
    if len(records)>1 and len(records[0][3])==5 and records[1][0]-records[0][0]!=1:
        first=records[0]; records[0]=(records[1][0]-1,*first[1:])
    destination=Path(output); destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w',encoding='ascii',newline='\n') as stream:
        for kind,source_id,identifier,points in records:
            stream.write(f'{len(points):7d},{int(kind%2==0):1d},"{kind:d},{source_id:d},{identifier:6d}"\n')
            clipped_x=np.clip(points[:,0],x0-.0001,x1+.0001)
            clipped_y=np.clip(points[:,1],y0-.0001,y1+.0001)
            for x_value,y_value in zip(clipped_x,clipped_y):
                stream.write(f'{x_value:12.6f},{y_value:12.6f}\n')
    return len(records)
