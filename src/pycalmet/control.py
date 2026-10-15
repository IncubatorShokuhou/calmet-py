"""Readers for legacy CALPUFF-style control files and list-file echoes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from datetime import datetime,timedelta
from pathlib import Path
import re
import shlex
from typing import Any, Callable, Final, Literal, Sequence, overload

from .config import (CalmetConfig,CalwrfConfig,CtgprocConfig,CtgprocSource,
                     GridConfig,MakegeoConfig,TerrelConfig,TerrelSource)
from .datetime_utils import (exact_integer,format_timezone_label,parse_calendar,
                             parse_date,parse_timestamp,parse_timezone_label)

_ASSIGNMENT: Final[re.Pattern[str]] = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", re.DOTALL
)
_ACTIVE_ASSIGNMENT: Final[re.Pattern[str]] = re.compile(
    r"!(?=\s*[A-Za-z][A-Za-z0-9_]*\s*=)(.*?)!", re.DOTALL
)


def assignments(path: str | Path) -> dict[str, list[str]]:
    """Return active ``! NAME=VALUE !`` assignments, preserving repeats.

    Asterisks delimit inactive/default entries in these control files and are
    intentionally ignored.  List files contain an echoed copy of the same
    active records, so the same reader handles both formats.

    Args:
        path: Legacy control or listing file to scan as ASCII text.

    Returns:
        Uppercase variable names mapped to all active value strings in file
        order.
    """
    text=Path(path).read_text(encoding="ascii",errors="replace")
    found: defaultdict[str,list[str]]=defaultdict(list)
    # Start a delimited block only when the opening bang is followed by an
    # assignment.  CALMET templates and listings sometimes contain ordinary
    # prose introduced by a lone ``!``; pairing that marker with the next bang
    # would otherwise consume a valid assignment and desynchronise the rest of
    # the file.
    for delimited in _ACTIVE_ASSIGNMENT.finditer(text):
        match=_ASSIGNMENT.match(delimited.group(1))
        if match and match.group(1).upper()!="END":
            found[match.group(1).upper()].append(match.group(2).strip())
    return dict(found)


def _case_insensitive_child(parent: Path, name: str) -> Path | None:
    """Find a file in one directory without requiring matching case.

    Args:
        parent: Directory to inspect.
        name: Requested child filename.

    Returns:
        The exact or case-insensitively matched file, or ``None`` when the
        directory or file does not exist.
    """
    exact=parent/name
    if exact.is_file(): return exact
    if not parent.is_dir(): return None
    folded=name.casefold()
    return next((item for item in parent.iterdir() if item.is_file() and item.name.casefold()==folded),None)


def _control_source(path: str | Path) -> Path:
    """Resolve a control file directly or through a processor listing.

    ``.lst`` and ``.log`` files are searched for their recorded ``.inp``
    basename. A complete CALMET listing that embeds active assignments is
    accepted as its own source.

    Args:
        path: Control file or processor list/log file.

    Returns:
        Path containing the complete control assignments to parse.

    Raises:
        ValueError: If a list is empty, records a failed partial run, or does
            not identify or embed a usable control file.
    """
    source=Path(path)
    if source.suffix.casefold() not in {'.lst','.log'}: return source
    text=source.read_text(encoding='ascii',errors='replace')
    if not text.strip():
        raise ValueError(f'list file is empty: {source}')
    lines=text.splitlines()
    candidates: list[str] = []
    for index,line in enumerate(lines):
        if 'control file used' in line.casefold():
            candidates.extend(item.strip() for item in lines[index+1:index+5] if item.strip())
            break
    candidates.extend(line.strip() for line in lines[:12] if line.strip().casefold().endswith('.inp'))
    # Full CALMET listings do not echo the delimited input records.  Their
    # INPUT FILES table does, however, record the control-file basename.
    for line in lines:
        match=re.search(r'\b([^\s/\\]+\.inp)\b',line,re.I)
        if match: candidates.append(match.group(1))
    for candidate in candidates:
        resolved=_case_insensitive_child(source.parent,Path(candidate).name)
        if resolved is not None: return resolved
    # A processor can leave a partial listing after READIN fails.  Although
    # such a file contains some echoed assignments, it is not a complete
    # control source (often the INPUT FILES group has not yet been emitted).
    if re.search(r'ERROR IN SUBR\.\s*READIN|Error in input data',text,re.I):
        raise ValueError(f'list file is a truncated failed run: {source}')
    # CALMET list files echo the full delimited control records and need no
    # separate source file. Other summaries cannot reconstruct every option.
    if assignments(source): return source
    raise ValueError(f'cannot locate the control file recorded by {source}')


def _one(
    values: dict[str, list[str]], name: str, default: Any = None
) -> Any:
    """Read the last active value for one case-insensitive control name.

    Args:
        values: Assignment mapping returned by :func:`assignments`.
        name: Control variable name.
        default: Value returned when the variable has no active assignment.

    Returns:
        The final assigned string or ``default``.
    """
    items=values.get(name.upper())
    return default if not items else items[-1]


def _number(value: str) -> int | float:
    """Parse integer, real, or Fortran double-exponent text.

    Args:
        value: Numeric control-field text.

    Returns:
        An integer when the token has no decimal/exponent marker, otherwise a
        floating-point value.

    Raises:
        ValueError: If ``value`` is not valid numeric text.
    """
    token=value.strip().replace("D","E").replace("d","e")
    return float(token) if any(character in token for character in ".Ee") else int(token)


def _integer_number(value: str, label: str) -> int:
    """Parse an integer control field without truncating a decimal value.

    Args:
        value: Numeric control-field value.
        label: Variable name included in an invalid-value error.

    Returns:
        The exactly represented integer.

    Raises:
        ValueError: If ``value`` is nonnumeric, non-finite, or not an exact
            integer.
    """
    return exact_integer(value,label)


def _logical(value: str) -> bool:
    """Parse a Fortran logical token.

    Args:
        value: ``T``/``F``, ``.TRUE.``/``.FALSE.``, ``TRUE``/``FALSE``, or
            zero/one text.

    Returns:
        The represented Boolean value.

    Raises:
        ValueError: If the token is not a recognized Fortran logical.
    """
    token=value.strip().upper()
    if token in {'T','.TRUE.','TRUE','1'}: return True
    if token in {'F','.FALSE.','FALSE','0'}: return False
    raise ValueError(f'invalid Fortran logical value: {value!r}')


def _angle(value: str) -> float:
    """Parse a signed angle with an optional compass direction.

    Args:
        value: Numeric text containing at most one ``N``, ``S``, ``E``, or
            ``W`` suffix or marker.

    Returns:
        Signed decimal degrees; east/west values are wrapped once into the
        conventional longitude range.

    Raises:
        ValueError: If the text is nonnumeric or has multiple directions.
    """
    token=re.sub(r"\s+","",str(value))
    directions=re.findall(r"[NSEW]",token,re.I)
    if len(directions)>1:
        raise ValueError(f"invalid directional angle: {value!r}")
    direction=directions[0].upper() if directions else ''
    if direction: token=re.sub(r"[NSEW]","",token,flags=re.I)
    try: number=float(token)
    except ValueError as error:
        raise ValueError(f"invalid directional angle: {value!r}") from error
    if direction in {"S","W"}: number=-abs(number)
    elif direction in {"N","E"}: number=abs(number)
    if direction in {"E","W"}:
        if number>180.: number-=360.
        elif number < -180.: number+=360.
    return number


def _path(parent: Path, value: str | Path) -> Path:
    """Resolve a control-file path and honor case-insensitive legacy names.

    Args:
        parent: Directory containing the control file.
        value: Absolute path or path relative to ``parent``.

    Returns:
        The resolved candidate, with the actual on-disk filename substituted
        when only its letter case differs.
    """
    candidate=Path(str(value).strip())
    resolved=candidate if candidate.is_absolute() else parent/candidate
    if resolved.exists(): return resolved
    matched=_case_insensitive_child(resolved.parent,resolved.name)
    return resolved if matched is None else matched


def _station_rows(
    values: dict[str, list[str]], prefix: str, count: int, fields: int
) -> tuple[tuple[str | int | float, ...], ...]:
    """Decode an optional CALMET Group 7, 8, or 9 station table.

    Args:
        values: Parsed control assignments.
        prefix: Station-row prefix: ``SS``, ``US``, or ``PS``.
        count: Declared number of indexed station rows.
        fields: Required number of fields in each row.

    Returns:
        Station tuples in one-based index order, or an empty tuple when no
        rows are present.

    Raises:
        ValueError: If row indices are incomplete, a row has the wrong width,
            or a numeric field is invalid.
    """
    present=[index for index in range(1,count+1) if values.get(f'{prefix}{index}')]
    if not present: return ()
    if present!=list(range(1,count+1)):
        raise ValueError(f'{prefix} station rows must cover 1..{count}')
    rows: list[tuple[str | int | float, ...]] = []
    for index in present:
        tokens=shlex.split(_one(values,f'{prefix}{index}'))
        if len(tokens)!=fields:
            raise ValueError(f'{prefix}{index} station row must contain {fields} fields')
        rows.append((tokens[0],int(_number(tokens[1])),
                     *(float(_number(item)) for item in tokens[2:])))
    return tuple(rows)


def grid_from_assignments(values: dict[str,list[str]]) -> GridConfig:
    """Build a grid from common TERREL, CTGPROC, MAKEGEO, or CALMET keys.

    Args:
        values: Parsed active control assignments.

    Returns:
        Cartesian grid geometry with distances in kilometres and angles in
        decimal degrees.

    Raises:
        ValueError: If ``NX``, ``NY``, or ``DGRIDKM`` is absent, or a grid
            value is invalid.
    """
    required=("NX","NY","DGRIDKM")
    missing=[name for name in required if _one(values,name) is None]
    if missing: raise ValueError("control file lacks grid fields: "+", ".join(missing))
    return GridConfig(
        nx=int(_number(_one(values,"NX"))),ny=int(_number(_one(values,"NY"))),
        dx_km=float(_number(_one(values,"DGRIDKM"))),dy_km=float(_number(_one(values,"DGRIDKM"))),
        projection=str(_one(values,"PMAP","UTM")).upper(),
        lat0=_angle(_one(values,"RLAT0","0")),lon0=_angle(_one(values,"RLON0","0")),
        lat1=_angle(_one(values,"RLAT1",_one(values,"XLAT1","0"))),
        lat2=_angle(_one(values,"RLAT2",_one(values,"XLAT2","0"))),
        x_origin_km=float(_number(_one(values,"XORIGKM",_one(values,"XREFKM","0")))),
        y_origin_km=float(_number(_one(values,"YORIGKM",_one(values,"YREFKM","0")))),
        false_easting_km=float(_number(_one(values,"FEAST","0"))),
        false_northing_km=float(_number(_one(values,"FNORTH","0"))),
        ttm_scale=float(_number(_one(values,"TMSCALE","1"))),
        utm_zone=int(_number(_one(values,"IUTMZN","50"))),
        hemisphere=str(_one(values,"UTMHEM","N")).upper(),datum=str(_one(values,"DATUM","WGS-84")),
    )


def _require_projection_grid(
    values: dict[str, list[str]],
    program: str,
    x_name: str,
    y_name: str,
    latitude_1: str = 'RLAT1',
    latitude_2: str = 'RLAT2',
) -> None:
    """Check the projection-dependent mandatory control fields.

    Args:
        values: Parsed control assignments, modified to include ``PMAP=UTM``
            when the projection is omitted.
        program: Program name used in validation errors.
        x_name: Required x-origin variable name.
        y_name: Required y-origin variable name.
        latitude_1: First-standard-latitude variable for LCC or PS.
        latitude_2: Second-standard-latitude variable for LCC.

    Raises:
        ValueError: If a field required by the selected projection is absent.
    """
    values.setdefault('PMAP',['UTM'])
    projection=str(_one(values,'PMAP')).upper()
    required=[x_name,y_name]
    if projection=='UTM':
        required.append('IUTMZN')
    else:
        required.extend(('RLAT0','RLON0'))
        if projection in {'LCC','PS'}: required.append(latitude_1)
        if projection=='LCC': required.append(latitude_2)
    missing=[name for name in required if _one(values,name) is None]
    if missing:
        raise ValueError(f'{program} control lacks mandatory grid fields: '+', '.join(missing))


def calwrf_config(path: str | Path, **overrides: Any) -> CalwrfConfig:
    """Read CALWRF's compact line-oriented control file.

    Negative or zero start/end selectors request the full timestamp range
    available across the configured WRF files. Explicit keyword overrides are
    applied after parsing; overriding the primary output disables inherited
    side outputs unless those paths are also overridden.

    Args:
        path: CALWRF control file or a listing that identifies it.
        **overrides: Canonical :class:`CalwrfConfig` fields replacing parsed
            values.

    Returns:
        Validated CALWRF configuration with paths resolved relative to the
        control file.

    Raises:
        ValueError: If the control is empty or truncated, its ranges or file
            count are invalid, or requested WRF timestamps cannot be found.
    """
    source=_control_source(path)
    raw_lines=source.read_text(encoding="ascii").splitlines()
    if not raw_lines:
        raise ValueError("CALWRF control file is empty")
    # GETSTR terminates string records at their first blank, while Fortran's
    # list-directed numeric reads accept ``!`` comments.  Treat wholly
    # commented records as inactive as well; this supports the commented
    # alternate WRF paths shipped with the repository.
    records: list[str] = []
    for line in raw_lines[1:]:
        stripped=line.strip()
        if not stripped or stripped.startswith('!'):
            continue
        records.append(stripped.split('!',1)[0].strip())
    # TITLE is the only A-format record.  Unlike GETSTR records it may contain
    # spaces, so retain its text independently from the tokenised tail.
    title=raw_lines[0][:132].rstrip()
    lines=[title,*records]
    if len(lines)<8: raise ValueError("CALWRF control file is truncated")
    count=int(lines[6]); files=lines[7:7+count]
    if len(files)!=count: raise ValueError("CALWRF file count exceeds supplied paths")
    configured_output=_path(source.parent,lines[2])
    canonical_output=(configured_output if configured_output.suffix.casefold()=='.nc'
                      else configured_output.with_suffix('.nc'))
    legacy_output=(None if configured_output.suffix.casefold()=='.nc'
                   else configured_output)
    bounds=[int(item.strip()) for item in lines[3].split(',')]
    if len(bounds)!=6: raise ValueError('CALWRF I/J/K selection must contain six integers')
    def selected(first: int, last: int) -> tuple[int, int] | None:
        """Convert a one-based inclusive bound pair to its optional form.

        Args:
            first: First selected one-based index.
            last: Last selected one-based index.

        Returns:
            ``None`` when either bound is below one, otherwise ``(first,
            last)``.
        """
        return None if first<1 or last<1 else (first,last)
    create_surface=True
    surface_output=configured_output.with_suffix('.m2d')
    if len(lines)>7+count:
        create_surface=int(_number(lines[7+count]))==1
        if create_surface and len(lines)>8+count:
            surface_output=_path(source.parent,lines[8+count])
    wrf_files=[_path(source.parent,item) for item in files]
    requested_files=overrides.get('wrf_files',wrf_files)
    requested_files=([requested_files]
                     if isinstance(requested_files,(str,Path)) else requested_files)
    raw_start,raw_end=lines[4],lines[5]

    def all_available(token: str) -> bool:
        """Return whether a CALWRF date token requests all available times.

        Args:
            token: Raw start or end date record.

        Returns:
            ``True`` only when the record is an integer below one.
        """
        try:
            return int(token.strip())<1
        except ValueError:
            return False

    if all_available(raw_start) or all_available(raw_end):
        from netCDF4 import Dataset,chartostring
        available: list[datetime] = []
        for wrf_path in requested_files:
            with Dataset(wrf_path) as dataset:
                if 'Times' not in dataset.variables:
                    raise ValueError(f'WRF input has no Times variable: {wrf_path}')
                values=chartostring(dataset.variables['Times'][:]).tolist()
                if not isinstance(values,list): values=[values]
                available.extend(parse_timestamp(str(value)) for value in values)
        if not available:
            raise ValueError('WRF inputs contain no dated records')
        control_start,control_end=min(available),max(available)
    else:
        control_start=parse_timestamp(raw_start)
        control_end=parse_timestamp(raw_end)

    data: dict[str, Any] = dict(wrf_files=wrf_files,start=control_start,end=control_end,
              output=canonical_output,create_surface=create_surface,
              title=title,listing_output=_path(source.parent,lines[1]),
              legacy_output=legacy_output,surface_output=(surface_output if create_surface else None),
              i_range=selected(*bounds[0:2]),j_range=selected(*bounds[2:4]),
              k_range=selected(*bounds[4:6]))
    if 'output' in overrides:
        # A caller redirecting the canonical NetCDF product must not retain
        # hidden side outputs from the control file.  Apart from being
        # surprising, a short audit run could overwrite its full M3D oracle.
        for side_output in ('legacy_output','surface_output','listing_output'):
            if side_output not in overrides:
                data[side_output]=None
    data.update(overrides); return CalwrfConfig(**data)


def terrel_config(path: str | Path, **overrides: Any) -> TerrelConfig:
    """Read a TERREL control file into typed terrain-processing settings.

    Args:
        path: TERREL control file or a listing that identifies it.
        **overrides: Canonical :class:`TerrelConfig` fields replacing parsed
            values.

    Returns:
        Validated TERREL configuration, including typed terrain sources,
        projection grid, receptor settings, and optional QA outputs.

    Raises:
        ValueError: If required grid fields, declared source counts, polar
            dimensions, receptor controls, replacement codes, or percentages
            are invalid.
    """
    source=_control_source(path); values=assignments(source)
    _require_projection_grid(values,'TERREL','XREFKM','YREFKM')
    lowercase_files=_logical(_one(values,'LCFILES','T'))
    def terrel_path(value: str) -> Path:
        """Resolve a TERREL filename using its ``LCFILES`` case policy.

        Args:
            value: Filename from the control file.

        Returns:
            Path relative to the control directory, case-normalized according
            to ``LCFILES`` and matched to an existing file when possible.
        """
        spelling=str(value).lower() if lowercase_files else str(value).upper()
        return _path(source.parent,spelling)
    order=('USGS90','USGS30','ARM3','3CD','DMDF','CDED','SRTM1','SRTM3',
           'GTOPO30','USGSLA','NZGEN','GEN','GEOTIFF')
    datum_keys={'USGS90':'DUSGS90','USGS30':'DUSGS30','ARM3':'DARM3','3CD':'D3CD',
                'DMDF':'DDMDF','CDED':'DCDED','SRTM1':'DSRTM1','SRTM3':'DSRTM3',
                'GTOPO30':'DGTOPO30','USGSLA':'DUSGSLA','NZGEN':'DNZGEN',
                'GEN':'DGEN','GEOTIFF':'DGEOTIFF'}
    defaults={'USGS90':'WGS-72','USGS30':'NAS-C','ARM3':'NAS-C','3CD':'WGS-72',
              'DMDF':'NAS-C','CDED':'NAR-C','SRTM1':'WGS-96','SRTM3':'WGS-96',
              'GTOPO30':'WGS-84','USGSLA':'ESR-S','NZGEN':'WGS-84',
              'GEN':'WGS-84','GEOTIFF':'WGS-84'}
    sources: list[TerrelSource] = []
    for kind in order:
        for item in values.get(kind,[]):
            sources.append(TerrelSource(terrel_path(item),kind,
                str(_one(values,datum_keys[kind],defaults[kind]))))
    declared_sources=int(_number(_one(values,'NTDF','0')))
    if declared_sources!=len(sources):
        raise ValueError(
            f'TERREL NTDF declares {declared_sources} files but {len(sources)} are active')
    grid=grid_from_assignments(values)
    model=int(_number(_one(values,'IMODEL','1')))
    grid.grid_type=int(_number(_one(values,'IGRID','1')))
    if model == 3:
        # READCF forces ISC-polar onto a polar grid before SPAN is called.
        grid.grid_type=3
    if model == 1 and grid.grid_type != 1:
        raise ValueError('TERREL CALMET output requires IGRID=1 corner reference')
    if grid.grid_type == 2:
        # Internally all Cartesian calculations use the SW cell corner (XLLK/YLLK).
        # IGRID=2 supplies XREFKM/YREFKM at the centre of cell (1,1).
        grid.x_origin_km -= 0.5 * grid.dx_km
        grid.y_origin_km -= 0.5 * grid.dy_km
    # The control dictionary names are DISKM/ANGDEG.  Retain the shorter
    # spellings accepted by early PyCALMET releases as compatibility aliases.
    grid.ring_distances_km=tuple(float(_number(value)) for name in ('DISKM','DISK')
                                 for value in values.get(name,[]))
    grid.ray_angles_deg=tuple(float(_number(value)) for name in ('ANGDEG','ANG')
                              for value in values.get(name,[]))
    grid.polar_processing=int(_number(_one(values,'IPROC','2')))
    if grid.grid_type==3:
        nrings=int(_number(_one(values,'NRING',str(len(grid.ring_distances_km)))))
        nrays=int(_number(_one(values,'NRAYS',str(len(grid.ray_angles_deg)))))
        if len(grid.ring_distances_km)!=nrings or len(grid.ray_angles_deg)!=nrays:
            raise ValueError('TERREL polar DISK/ANG counts do not match NRING/NRAYS')
        grid.nx,grid.ny=nrings,nrays
    receptors=_logical(str(_one(values,'LXY','F')))
    receptor_input=_one(values,'XYINP') or 'XYINP.DAT'
    receptor_output=_one(values,'XYOUT') or 'XYOUT.DAT'
    receptor_columns=int(_number(_one(values,'NXYCOL','4')))
    interpolate_receptors=_logical(str(_one(values,'LINTXY','F')))
    receptor_radius=float(_number(_one(values,'XYRADKM','0')))
    if receptors and receptor_columns not in {2,4}:
        raise ValueError('TERREL NXYCOL must be 2 or 4')
    if receptors and (not receptor_input or not receptor_output):
        raise ValueError('TERREL LXY=T requires XYINP and XYOUT')
    if receptors and not interpolate_receptors and receptor_radius <= 0.0:
        raise ValueError('TERREL peak receptor processing requires positive XYRADKM')
    continuation=_logical(str(_one(values,'LPREV','F')))
    previous_save=_one(values,'PREVFIL') or 'PREV.SAV'
    save_output=_one(values,'SAVEFIL','TERREL.SAV')
    sheet_method=int(_number(_one(values,'MSHEET','0')))
    if sheet_method not in {0,1}:
        raise ValueError('TERREL MSHEET must be 0 or 1')
    def pair(
        name: str,
        defaults: Sequence[int | float],
        integer: bool = False,
    ) -> tuple[int | float, ...]:
        """Parse a two-value TERREL replacement record and add defaults.

        Traditional controls specify the first two source classes; defaults
        supply the remaining three classes used by the typed configuration.

        Args:
            name: Control variable to read.
            defaults: Five-value fallback sequence.
            integer: Convert all returned values to integers when true.

        Returns:
            Five integer or floating-point replacement values.

        Raises:
            ValueError: If the active record does not contain two values.
        """
        raw=str(_one(values,name,','.join(str(value) for value in defaults[:2])))
        parsed=tuple(_number(token) for token in raw.replace(',', ' ').split())
        if len(parsed)!=2:
            raise ValueError(f'TERREL {name} must contain two values')
        result=parsed+tuple(defaults[2:])
        return tuple(int(value) for value in result) if integer else tuple(float(value) for value in result)
    noise_replacement=pair('INOISEREP',(0,0,0,0,0),True)
    noise_minimum=pair('ZNOISE',(0.,1.,1.,1.,1.))
    missing_replacement=pair('ITERREP',(3,0,0,0,0),True)
    default_elevation=pair('TERDEF',(0.,0.,0.,0.,0.))
    if any(value not in {0,1,2,3} for value in noise_replacement):
        raise ValueError('TERREL INOISEREP values must be in the range 0..3')
    if any(value not in {0,1,2,3} for value in missing_replacement):
        raise ValueError('TERREL ITERREP values must be in the range 0..3')
    coastal=_logical(str(_one(values,'LCOAST','F')))
    read_bln=_logical(str(_one(values,'LBLNREAD','F')))
    raw_echo=_logical(str(_one(values,'LRAWECHO','F')))
    raw_echo_value=_one(values,'RAWECHO','RAWECHO.DAT')
    threshold=int(_number(_one(values,'ITHRES','75')))
    if not 0<=threshold<=100:
        raise ValueError('TERREL ITHRES must be in the range 0..100')
    data: dict[str, Any] = dict(srtm_dir=source.parent,output=terrel_path(_one(values,"OUTFIL","TERR.DAT")),
              grid=grid,datum=str(_one(values,"DATUM","WGS-84")),
              sampling=("direct" if sheet_method==1 else "legacy"),
              terrain_files=[item.path for item in sources],terrain_sources=tuple(sources),
              void_fill=_logical(_one(values,'LVOIDFIL','F')),
              cell_radius_km=float(_number(_one(values,'CELLRADKM','0'))),
              model=model,
              processing_method=grid.polar_processing,
              receptor_input=(terrel_path(receptor_input) if receptors else None),
              receptor_output=(terrel_path(receptor_output) if receptors else None),
              receptor_columns=receptor_columns,
              interpolate_receptors=interpolate_receptors,
              receptor_radius_km=receptor_radius,
              previous_save=(terrel_path(previous_save) if continuation else None),
              save_output=terrel_path(save_output),
              coastline_file=(terrel_path(_one(values,'COASTBLN','COAST.BLN'))
                              if coastal else None),
              gshhs_file=(terrel_path(_one(values,'GSHHSIN','GSHHS_F.B'))
                          if coastal and not read_bln else None),
              coastline_source_datum=str(_one(values,'DWVS','WGS-84')),
              coastline_wdbii_datum=str(_one(values,'DWDBII','WGS-72')),
              noise_replacement=noise_replacement,
              noise_minimum_m=noise_minimum,
              missing_replacement=missing_replacement,
              default_elevation_m=default_elevation,
              plot_output=terrel_path(_one(values,'PLTFIL','QATERR.GRD')),
              raw_echo_output=(terrel_path(raw_echo_value) if raw_echo else None),
              qa_hit_threshold_percent=threshold)
    if 'output' in overrides:
        # Keep an isolated audit/run isolated.  These optional products are
        # otherwise inherited from the legacy control and can silently
        # overwrite the reference files beside it.
        for name in ('save_output','plot_output','raw_echo_output'):
            if name not in overrides:
                data[name]=None
    if 'terrain_files' in overrides and 'terrain_sources' not in overrides:
        requested=overrides['terrain_files']
        files=((requested,) if isinstance(requested,(str,Path)) else tuple(requested))
        overrides['terrain_files']=files
        metadata={(item.input_type,item.input_datum) for item in sources}
        if len(files)==len(sources):
            data['terrain_sources']=tuple(
                TerrelSource(path,item.input_type,item.input_datum)
                for path,item in zip(files,sources))
        elif len(metadata)==1:
            kind,input_datum=next(iter(metadata))
            data['terrain_sources']=tuple(
                TerrelSource(path,kind,input_datum) for path in files)
        else:
            # An untyped replacement cannot be associated safely with a
            # heterogeneous legacy source list.  Let the SRTM file path use
            # the ordinary untyped TERREL route instead of retaining stale
            # control-file sources.
            data['terrain_sources']=()
    data.update(overrides); return TerrelConfig(**data)


def ctgproc_config(path: str | Path, **overrides: Any) -> CtgprocConfig:
    """Read a CTGPROC control file into typed land-use settings.

    Args:
        path: CTGPROC control file or a listing that identifies it.
        **overrides: Canonical :class:`CtgprocConfig` fields replacing parsed
            values.

    Returns:
        Validated CTGPROC configuration with source-specific datum and mesh
        metadata.

    Raises:
        ValueError: If required grid fields, source selections, declared
            counts, or legacy land-use type/region codes are invalid.
    """
    source=_control_source(path); values=assignments(source)
    _require_projection_grid(values,'CTGPROC','XREFKM','YREFKM')
    lowercase_files=_logical(_one(values,'LCFILES','T'))
    def ctgproc_path(value: str) -> Path:
        """Resolve a CTGPROC filename using its ``LCFILES`` case policy.

        Args:
            value: Filename from the control file.

        Returns:
            Path relative to the control directory, case-normalized according
            to ``LCFILES`` and matched to an existing file when possible.
        """
        spelling=str(value).lower() if lowercase_files else str(value).upper()
        return _path(source.parent,spelling)
    order=("GLC2K","UMDGLC","CTG","NZGEN","GLAZNA","GLAZSA","GLAZEU",
           "GLAZAS","GLAZAF","GLAZAP","NLCD92","NLCDTF","NLCD01","CORINE","MODIS","GEN")
    declared_sources=int(_number(_one(values,'NDBF','0')))
    if declared_sources<0:
        raise ValueError('CTGPROC NDBF cannot be negative')
    kinds=[name for name in order if name in values]
    def datum(kind: str) -> str:
        """Return the configured or traditional datum for an input type.

        Args:
            kind: CTGPROC land-use source type.

        Returns:
            Datum code selected by the corresponding control variable.
        """
        return (str(_one(values,'DCTG','NAS-C')) if kind=='CTG' else
                str(_one(values,'DNZGEN','WGS-84')) if kind=='NZGEN' else
                str(_one(values,'DNLCD','NAR-C')) if kind.startswith('NLCD') else
                str(_one(values,'DCORINE','WGS-84')) if kind=='CORINE' else
                str(_one(values,'DGLC2K','WGS-84')) if kind=='GLC2K' else
                str(_one(values,'DUMDGLC','WGS-84')) if kind=='UMDGLC' else
                str(_one(values,'DMODIS','ESR-S')) if kind=='MODIS' else
                str(_one(values,'DUSGSLA','ESR-S')) if kind.startswith('GLAZ') else
                str(_one(values,'DATUM','WGS-84')))
    def mesh(kind: str) -> int:
        """Return the mesh subdivision count for an input type.

        Args:
            kind: CTGPROC land-use source type.

        Returns:
            Integer CTG-, CORINE-, or general-source mesh density.
        """
        return int(_number(_one(values,'MESHCTG','1') if kind=='CTG' else
                           _one(values,'MESHCORN','1') if kind=='CORINE' else
                           _one(values,'MESHGLAZ','1')))
    sources: tuple[CtgprocSource, ...]
    if declared_sources==0:
        legacy_types={1:'CTG',3:'NZGEN',4:'NLCD92',5:'NLCD01',6:'NLCDTF',
                      7:'CORINE',8:'GLC2K',9:'UMDGLC',10:'MODIS',11:'GEN'}
        lulc=int(_number(_one(values,'LULC','1')))
        if lulc==2:
            region=int(_number(_one(values,'IGLAZR','0')))
            if region not in range(1,7):
                raise ValueError('CTGPROC legacy LULC=2 requires IGLAZR from 1 through 6')
            kind=('GLAZNA','GLAZSA','GLAZEU','GLAZAS','GLAZAF','GLAZAP')[region-1]
        else:
            try:
                kind=legacy_types[lulc]
            except KeyError as error:
                raise ValueError(f'CTGPROC legacy LULC must be from 1 through 11, got {lulc}') from error
        sources=(CtgprocSource(ctgproc_path(_one(values,'LUINDAT','luin.dat')),
                               kind,datum(kind),mesh(kind)),)
        kinds=[kind]
    else:
        if not kinds:
            raise ValueError('control file must select at least one supported land-use input type')
        sources=tuple(CtgprocSource(ctgproc_path(item),kind,datum(kind),mesh(kind))
                      for kind in kinds for item in values[kind])
        if declared_sources!=len(sources):
            raise ValueError(
                f'CTGPROC NDBF declares {declared_sources} files but {len(sources)} are active')
    kind=kinds[0]; input_datum=datum(kind)
    coastal=_logical(str(_one(values,'LCOAST','F')))
    read_bln=_logical(str(_one(values,'LBLNREAD','F')))
    continuation=_logical(str(_one(values,'LPREV','F')))
    previous_value=_one(values,'PREVDAT') or 'PREV.DAT'
    data: dict[str, Any] = dict(landuse_files=[item.path for item in sources],landuse_sources=sources,
              output=ctgproc_path(_one(values,"LUDAT","LU.DAT")),grid=grid_from_assignments(values),
              input_type=kind,mesh_density=sources[0].mesh_density,
              input_datum=input_datum,
              coastline_file=(ctgproc_path(_one(values,'COASTBLN','COAST.BLN'))
                              if coastal else None),
              gshhs_file=(ctgproc_path(_one(values,'GSHHSIN') or 'GSHHS_F.B')
                          if coastal and not read_bln else None),
              coastline_source_datum=str(_one(values,'DWVS','WGS-84')),
              coastline_wdbii_datum=str(_one(values,'DWDBII','WGS-72')),
              coastline_swap=coastal and _logical(str(_one(values,'LMARSWAP','F'))),
              fill_empty_ocean=coastal and _logical(str(_one(values,'LMARFILL','T'))),
              ocean_category=int(_number(_one(values,'IOCEAN','55'))),
              previous_landuse=(ctgproc_path(previous_value) if continuation else None),
              final_output=_logical(str(_one(values,'LFINAL','T'))),
              snow_processing=_logical(str(_one(values,'LSNOW','F'))),
              qa_cell_output=_logical(str(_one(values,'LQACELL','F'))),
              qa_hit_threshold_percent=int(_number(_one(values,'ITHRESH','75'))))
    source_overrides={'landuse_files','input_type','input_datum','mesh_density'}&overrides.keys()
    if source_overrides and 'landuse_sources' not in overrides:
        requested=overrides.get('landuse_files',data['landuse_files'])
        files=((requested,) if isinstance(requested,(str,Path)) else tuple(requested))
        overrides['landuse_files']=files
        if len(files)==len(sources):
            data['landuse_sources']=tuple(
                CtgprocSource(
                    path,str(overrides.get('input_type',item.input_type)),
                    str(overrides.get('input_datum',item.input_datum)),
                    int(overrides.get('mesh_density',item.mesh_density)))
                for path,item in zip(files,sources))
        else:
            input_type=str(overrides.get('input_type',data['input_type']))
            input_datum=str(overrides.get('input_datum',data['input_datum']))
            mesh_density=int(overrides.get('mesh_density',data['mesh_density']))
            data['landuse_sources']=tuple(
                CtgprocSource(path,input_type,input_datum,mesh_density) for path in files)
    data.update(overrides); return CtgprocConfig(**data)


def makegeo_config(path: str | Path, **overrides: Any) -> MakegeoConfig:
    """Read a MAKEGEO control file into typed geophysical settings.

    Args:
        path: MAKEGEO control file or a listing that identifies it.
        **overrides: Canonical :class:`MakegeoConfig` fields replacing parsed
            values.

    Returns:
        Validated MAKEGEO configuration with surface-property tables,
        category mappings, optional snow controls, and resolved file paths.

    Raises:
        TypeError: If snow processing is enabled and a required snow date is
            absent or has an unsupported type.
        ValueError: If the control is empty, required grid fields or record
            widths are invalid, declared category counts disagree, the output
            format is unsupported, or snow tables are inconsistent.
    """
    source=_control_source(path); values=assignments(source)
    lines=source.read_text(encoding='ascii',errors='replace').splitlines()
    if not lines: raise ValueError(f'MAKEGEO control file is empty: {source}')
    # READIN uses an A80 record: leading blanks are data, and characters
    # beyond column 80 are discarded by the CHARACTER*80 destination.
    title=(lines[1][:80] if lines[0][:11]=='MAKEGEO.INP' and len(lines)>1
           else lines[0][:80])
    _require_projection_grid(values,'MAKEGEO','XREFKM','YREFKM')
    lowercase_files=_logical(_one(values,'LCFILES','T'))
    def makegeo_path(value: str) -> Path:
        """Resolve a MAKEGEO filename using its ``LCFILES`` case policy.

        Args:
            value: Filename from the control file.

        Returns:
            Path relative to the control directory, case-normalized according
            to ``LCFILES`` and matched to an existing file when possible.
        """
        spelling=str(value).lower() if lowercase_files else str(value).upper()
        return _path(source.parent,spelling)
    properties: dict[
        int, tuple[float, float, float, float, float, float, int]
    ] = {}
    for record in values.get('X',[]):
        parts=[item.strip() for item in record.split(',')]
        if len(parts)!=8: raise ValueError(f'invalid MAKEGEO X record: {record!r}')
        properties[int(_number(parts[0]))] = (
            float(_number(parts[1])), float(_number(parts[2])),
            float(_number(parts[3])), float(_number(parts[4])),
            float(_number(parts[5])), float(_number(parts[6])),
            int(_number(parts[7])),
        )
    snow_properties: dict[
        int, tuple[float, float, float, float, float, float, int]
    ] = {}
    for record in values.get('XS',[]):
        parts=[item.strip() for item in record.split(',')]
        if len(parts)!=8: raise ValueError(f'invalid MAKEGEO XS record: {record!r}')
        snow_properties[int(_number(parts[0]))] = (
            float(_number(parts[1])), float(_number(parts[2])),
            float(_number(parts[3])), float(_number(parts[4])),
            float(_number(parts[5])), float(_number(parts[6])),
            int(_number(parts[7])),
        )
    snow_age_parameters: dict[int, tuple[int, float]] = {}
    for record in values.get('XF',[]):
        parts=[item.strip() for item in record.split(',')]
        if len(parts)!=3: raise ValueError(f'invalid MAKEGEO XF record: {record!r}')
        snow_age_parameters[int(_number(parts[0]))]=(
            int(_number(parts[1])),float(_number(parts[2])))
    output_categories=[category for record in values.get('OUTCAT',[])
                       for item in record.split(',') if item.strip()
                       for category in (int(_number(item)),) if category!=0]
    water_categories=[int(_number(item)) for item in values.get('IWAT',[])]
    splits: list[tuple[int, int, float]] = []
    for record in values.get('XSPLIT',[]):
        parts=[item.strip() for item in record.split(',')]
        if len(parts)!=3: raise ValueError(f'invalid MAKEGEO XSPLIT record: {record!r}')
        splits.append((int(_number(parts[0])),int(_number(parts[1])),float(_number(parts[2]))))
    if properties and len(properties)!=int(_number(_one(values,'NINCAT',str(len(properties))))):
        raise ValueError('MAKEGEO NINCAT does not match the number of X records')
    if output_categories and len(output_categories)!=int(_number(_one(values,'NOUTCAT',str(len(output_categories))))):
        raise ValueError('MAKEGEO NOUTCAT does not match the number of OUTCAT entries')
    if water_categories and len(water_categories)!=int(_number(_one(values,'NUMWAT',str(len(water_categories))))):
        raise ValueError('MAKEGEO NUMWAT does not match the number of IWAT entries')
    if len({source for source,_,_ in splits})!=int(_number(_one(values,'NSPLIT','0'))):
        raise ValueError('MAKEGEO NSPLIT does not match the number of split source categories')
    snow_processing=_logical(_one(values,'LSNOW','F'))
    output_format=int(_number(_one(values,'IFMTGEO','1')))
    if output_format!=1:
        raise ValueError('MAKEGEO only supports IFMTGEO=1')
    if snow_processing:
        if set(snow_properties)!=set(properties):
            raise ValueError('MAKEGEO XS snow categories must match X categories')
        if set(snow_age_parameters)!=set(properties):
            raise ValueError('MAKEGEO XF snow categories must match X categories')
        for category in properties:
            if snow_properties[category][-1]!=properties[category][-1]:
                raise ValueError('MAKEGEO snow and no-snow output category maps must match')
    ixqa=int(_number(_one(values,'IXQA','0')))
    iyqa=int(_number(_one(values,'IYQA','0')))
    data: dict[str, Any] = dict(terrain=makegeo_path(_one(values,"TERRDAT","TERR.DAT")),
              landuse=makegeo_path(_one(values,"LUDAT","LU.DAT")),
              output=makegeo_path(_one(values,"GEODAT","GEO.DAT")),grid=grid_from_assignments(values),
              water_threshold=float(_number(_one(values,"CFRACT","0.5"))),
              missing_category=int(_number(_one(values,"IMISS","55"))),
              minimum_coverage=float(_number(_one(values,"FLUMIN","0.96"))),
              secondary_landuse=(makegeo_path(_one(values,"LU2DAT") or 'LU2.DAT')
                                  if _logical(_one(values,"LLU2","F")) else None),
              use_terrain=_logical(_one(values,"LTERR","T")),input_properties=properties,
              water_categories=water_categories or (51,52,53,54,55),
              output_categories=output_categories or (10,20,-20,30,40,51,54,55,60,61,62,70,80,90),
              output_water_min=int(_number(_one(values,"IWAT1","50"))),
              output_water_max=int(_number(_one(values,"IWAT2","55"))),category_splits=tuple(splits),
              qa_cell=((ixqa,iyqa) if ixqa>0 and iyqa>0 else None),
              snow_processing=snow_processing,
              snow_data=(makegeo_path(_one(values,'SNOWDAT','SNOW.DAT'))
                         if snow_processing else None),
              snow_start_date=(parse_date(_one(values,'IDATEBEG'),
                                          'MAKEGEO IDATEBEG')
                               if snow_processing else None),
              snow_end_date=(parse_date(_one(values,'IDATEEND'),
                                        'MAKEGEO IDATEEND')
                             if snow_processing else None),
              snow_properties=snow_properties,
              snow_age_parameters=snow_age_parameters,
              snow_roughness_method=int(_number(_one(values,'MSRL','1'))),
              snow_height_scale_m=float(_number(_one(values,'HSCL','10.0'))),
              snow_albedo_method=int(_number(_one(values,'MSAL','1'))),
              snow_depth_minimum_m=float(_number(_one(values,'SDPMIN','0.01'))),
              landuse_plot_output=makegeo_path(_one(values,'LUGRD','QALUSE.GRD')),
              terrain_plot_output=makegeo_path(_one(values,'TEGRD','QATERR.GRD')),
              color_map_output=makegeo_path(
                  str(Path(_one(values,"GEODAT","GEO.DAT")).with_name('luse.clr'))),
              image_mode=int(_number(_one(values,'IMAGE','0'))),title=title)
    if 'output' in overrides:
        for name in ('landuse_plot_output','terrain_plot_output','color_map_output'):
            if name not in overrides:
                data[name]=None
    data.update(overrides); return MakegeoConfig(**data)


def _date(values: dict[str, list[str]], prefix: str) -> datetime:
    """Parse a CALMET begin or end timestamp from control assignments.

    Args:
        values: Parsed CALMET control assignments.
        prefix: ``"IB"`` for the beginning or ``"IE"`` for the ending
            component names.

    Returns:
        A naive, second-precision timestamp. CALMET's hour-24 boundary form is
        normalized to the next calendar day.

    Raises:
        ValueError: If a required component is absent, is not an exact
            integer, or produces an invalid calendar value.
    """
    fields=[_integer_number(_one(values,prefix+suffix),prefix+suffix)
            for suffix in ("YR","MO","DY","HR")]
    seconds=_integer_number(_one(values,prefix+"SEC","0"),prefix+"SEC")
    try:
        return parse_calendar(
            fields[0], fields[1], fields[2], fields[3], seconds
        )
    except ValueError as error:
        if 'invalid hour' in str(error):
            raise ValueError(f'{prefix}HR must be between 0 and 24') from error
        raise


def _calmet_version(source: Path) -> str:
    """Read the fixed-width CALMET control-file version field.

    Args:
        source: CALMET control file.

    Returns:
        The stripped version stored in columns 17 through 32, or an empty
        string when the header is shorter than 32 characters.

    Raises:
        ValueError: If the control file is empty.
    """
    lines=source.read_text(encoding='ascii',errors='replace').splitlines()
    if not lines: raise ValueError(f'CALMET control file is empty: {source}')
    return lines[0][16:32].strip() if len(lines[0])>=32 else ''


def _calmet_period(
    version: str, values: dict[str, list[str]]
) -> tuple[datetime, datetime, int, str]:
    """Decode CALMET's version-dependent run period and base time zone.

    Version 2.1 and 2.2 controls contain explicit beginning and ending fields
    plus ``ABTZ``. Older controls provide the first hour-ending timestamp,
    duration ``IRLG``, and positive-west ``IBTZ``.

    Args:
        version: Fixed-width control-file version string.
        values: Parsed CALMET assignments.

    Returns:
        ``(start, end, utc_offset_hours, timezone_label)`` using naive,
        second-precision datetimes.

    Raises:
        ValueError: If required period fields are absent or dates, duration,
            timezone, or version-specific fields are invalid.
    """
    if version in {'2.1','2.2'}:
        if _one(values,'ABTZ') is None:
            raise ValueError(f'CALMET {version} control requires explicit ABTZ')
        try:
            base_timezone,timezone=parse_timezone_label(
                ''.join(str(_one(values,"ABTZ")).split()).upper())
        except (TypeError,ValueError) as error:
            if 'whole-hour' in str(error):
                raise ValueError(
                    'CALMET base_timezone only supports whole-hour ABTZ values') from error
            raise ValueError(
                f"unsupported CALMET timezone: {_one(values, 'ABTZ')!r}") from error
        return _date(values,"IB"),_date(values,"IE"),base_timezone,timezone

    # Before 2.1, IB* is the first hour-ending time, IRLG is the number of
    # hours, and IBTZ uses CALMET's positive-west convention.  READCF derives
    # explicit output bounds by subtracting one hour at the beginning.
    explicit=('IBSEC','IEYR','IEMO','IEDY','IEHR','IESEC')
    if any(name in values for name in explicit):
        raise ValueError('pre-2.1 CALMET controls cannot contain explicit beginning/ending times')
    hour_ending=_date(values,'IB')
    length=_integer_number(_one(values,'IRLG'),'IRLG')
    if length<=0: raise ValueError('IRLG must be positive')
    begin=hour_ending-timedelta(hours=1)
    end=begin+timedelta(hours=length)
    base_timezone=_integer_number(_one(values,'IBTZ'),'IBTZ')
    timezone=format_timezone_label(base_timezone)
    return begin,end,base_timezone,timezone


def calmet_config(path: str | Path, **overrides: Any) -> CalmetConfig:
    """Read a CALMET control file into the complete typed configuration.

    The loader supports both current and pre-2.1/2.2 period and cloud-control
    conventions, resolves input paths beside the control file, expands
    numbered input series, and assembles station, barrier, and lake-breeze
    records.

    Args:
        path: CALMET control file or a listing that identifies or embeds it.
        **overrides: Canonical :class:`CalmetConfig` fields replacing parsed
            values.

    Returns:
        Complete validated CALMET configuration.

    Raises:
        ValueError: If a required control value is absent, or grid geometry,
            time controls, vertical arrays, declared
            counts, station tables, barriers, lake-breeze data, or versioned
            cloud settings are invalid or inconsistent.
    """
    source=_control_source(path); values=assignments(source)
    lowercase_files=_logical(_one(values,'LCFILES','T'))
    def calmet_path(value: str) -> Path:
        """Resolve a CALMET filename using its ``LCFILES`` case policy.

        Args:
            value: Filename from the control file.

        Returns:
            Path relative to the control directory, case-normalized according
            to ``LCFILES`` and matched to an existing file when possible.
        """
        spelling=str(value).lower() if lowercase_files else str(value).upper()
        return _path(source.parent,spelling)
    def numbered_files(key: str, count: int, stem: str) -> tuple[Path,...]:
        """Resolve an explicitly named or default-numbered file series.

        Args:
            key: Repeated control variable containing explicit filenames.
            count: Declared number of files.
            stem: Default basename prefix for omitted filenames.

        Returns:
            Exactly ``count`` resolved paths, with missing trailing names
            filled as ``<stem><index>.dat``.

        Raises:
            ValueError: If more explicit filenames are present than declared.
        """
        explicit=list(values.get(key,()))
        if len(explicit)>count:
            raise ValueError(f'{key} provides more files than its declared count')
        names=explicit+[f'{stem}{index}.dat' for index in range(len(explicit)+1,count+1)]
        return tuple(calmet_path(name) for name in names)
    # CALMET spells the standard-parallel controls XLAT1/XLAT2, unlike the
    # preprocessing programs' RLAT1/RLAT2.
    _require_projection_grid(values,'CALMET','XORIGKM','YORIGKM','XLAT1','XLAT2')
    grid=grid_from_assignments(values)
    version=_calmet_version(source)
    faces=[float(_number(item)) for item in re.split(r"\s*,\s*",_one(values,"ZFACE")) if item]
    declared_levels=int(_number(_one(values,'NZ',str(len(faces)-1))))
    if declared_levels!=len(faces)-1:
        raise ValueError('CALMET NZ does not match the number of ZFACE layers')
    start,end,base_timezone,base_timezone_label=_calmet_period(version,values)
    smoothing_text=_one(values,"NSMTH",','.join(['2']+['4']*(len(faces)-2)))
    smoothing=[int(_number(item)) for item in re.split(r"\s*,\s*",smoothing_text) if item]
    nintr=[int(_number(item)) for item in re.split(r"\s*,\s*",_one(values,'NINTR2',','.join(['99']*(len(faces)-1)))) if item]
    fextr=[float(_number(item)) for item in re.split(r"\s*,\s*",_one(values,'FEXTR2',','.join(['0']*(len(faces)-1)))) if item]
    wind_bias=[float(_number(item)) for item in re.split(
        r"\s*,\s*",_one(values,'BIAS',','.join(['0']*(len(faces)-1)))) if item]
    def integer_vector(name: str, count: int, default: int=0) -> tuple[int,...]:
        """Read a comma/whitespace-delimited fixed-length integer vector.

        Args:
            name: CALMET control variable to read.
            count: Number of repeated default values when the field is absent.
            default: Default value repeated ``count`` times.

        Returns:
            Parsed integer values. Length validation is performed later with
            the rest of the CALMET configuration.
        """
        text=str(_one(values,name,','.join([str(default)]*count)))
        return tuple(int(_number(item)) for item in re.split(r'[\s,]+',text.strip()) if item)
    print_uv_levels=integer_vector('IUVOUT',len(faces)-1)
    print_w_levels=integer_vector('IWOUT',len(faces)-1)
    print_temperature_levels=integer_vector('ITOUT',len(faces)-1)
    print_surface_fields=tuple(int(_number(_one(values,name,'0'))) for name in
        ('STABILITY','USTAR','MONIN','MIXHT','WSTAR','PRECIP','SENSHEAT','CONVZI'))
    wind_debug_fields=tuple(int(_number(_one(values,f'IPR{index}','0')))
                            for index in range(9))
    barrier_columns: list[list[float]] = []
    for name in ('XBBAR','YBBAR','XEBAR','YEBAR'):
        barrier_columns.append([float(_number(item)) for item in re.split(r"\s*,\s*",_one(values,name,'')) if item])
    barrier_count=int(_number(_one(values,'NBAR','0')))
    if barrier_count and any(len(column)!=barrier_count for column in barrier_columns):
        raise ValueError('NBAR does not match XBBAR/YBBAR/XEBAR/YEBAR lengths')
    barriers=tuple(zip(*barrier_columns)) if barrier_count else ()
    lake_enabled=_logical(_one(values,'LLBREZE','F'))
    lake_count=int(_number(_one(values,'NBOX','0')))
    @overload
    def lake_values(name: str, integer: Literal[True]) -> list[int]:
        """Type the integer form of a repeated lake-breeze control array.

        Args:
            name: Lake-breeze control variable to collect.
            integer: Literal selecting integer conversion.

        Returns:
            Flattened integer values from all active assignments.
        """
        ...

    @overload
    def lake_values(name: str, integer: Literal[False] = False) -> list[float]:
        """Type the floating-point form of a lake-breeze control array.

        Args:
            name: Lake-breeze control variable to collect.
            integer: Literal selecting floating-point conversion.

        Returns:
            Flattened floating-point values from all active assignments.
        """
        ...

    def lake_values(
        name: str, integer: bool = False
    ) -> list[int] | list[float]:
        """Collect repeated lake-breeze array assignments.

        Args:
            name: Lake-breeze control variable to collect.
            integer: Return integers instead of floating-point values.

        Returns:
            Flattened numeric values from all active assignments.
        """
        parsed=[_number(item) for value in values.get(name,())
                for item in re.split(r'[\s,]+',value.strip()) if item]
        return [int(item) for item in parsed] if integer else [float(item) for item in parsed]
    lake_columns=[lake_values(name) for name in
                  ('XG1','XG2','YG1','YG2','XBCST','YBCST','XECST','YECST')]
    lake_station_counts=lake_values('NLB',integer=True)
    lake_station_ids=lake_values('METBXID',integer=True)
    if lake_enabled:
        if lake_count<1 or any(len(column)!=lake_count for column in lake_columns):
            raise ValueError('LLBREZE requires NBOX matching all lake-breeze coordinate arrays')
        if len(lake_station_counts)!=lake_count or sum(lake_station_counts)!=len(lake_station_ids):
            raise ValueError('lake-breeze NLB counts do not match NBOX/METBXID')
        lake_region_list: list[
            tuple[
                float, float, float, float,
                float, float, float, float,
                tuple[int, ...],
            ]
        ] = []
        station_offset = 0
        for index,count in enumerate(lake_station_counts):
            if count<1: raise ValueError('each lake-breeze region requires at least one station')
            coordinates=[column[index] for column in lake_columns]
            for offset,origin in ((4,grid.x_origin_km),(5,grid.y_origin_km),
                                  (6,grid.x_origin_km),(7,grid.y_origin_km)):
                coordinates[offset]=(coordinates[offset]-origin)*1000.
            selected=tuple(lake_station_ids[station_offset:station_offset+count])
            station_offset+=count
            lake_region_list.append((
                coordinates[0], coordinates[1], coordinates[2], coordinates[3],
                coordinates[4], coordinates[5], coordinates[6], coordinates[7],
                selected,
            ))
        lake_regions = tuple(lake_region_list)
    else:
        lake_regions=()
    surface_count=int(_number(_one(values,'NSSTA','0')))
    precipitation_count=int(_number(_one(values,'NPSTA','0')))
    surface_control_stations=_station_rows(values,'SS',surface_count,6)
    upper_control_stations=_station_rows(values,'US',int(_number(_one(values,'NUSTA','0'))),5)
    precipitation_control_stations=(_station_rows(values,'PS',precipitation_count,4)
                                    if precipitation_count>0 else ())
    declared_upper=int(_number(_one(values,'NUSTA','0')))
    upper_air_files=numbered_files('UPDAT',declared_upper,'up')
    declared_sea=int(_number(_one(values,'NOWSTA','0')))
    sea_files=numbered_files('SEADAT',declared_sea,'sea')
    iprog=int(_number(_one(values,'IPROG','0')))
    meteorology_files: tuple[Path, ...]
    if iprog in {1,2}:
        meteorology_files=(calmet_path(_one(values,'PRGDAT','PROG.DAT')),)
    else:
        declared_meteorology=int(_number(_one(values,'NM3D','0')))
        legacy_mm4=_one(values,'MM4DAT')
        if declared_meteorology==0 and legacy_mm4 is not None:
            # READFN preserves the pre-multiple-file protocol by promoting
            # MM4DAT to M3DDAT(1) and resetting NM3D to one.
            meteorology_files=(calmet_path(legacy_mm4),)
        elif declared_meteorology==0:
            # A diagnostic CALMET run can be driven entirely by SURF.DAT and
            # UP*.DAT.  Legacy control templates may still contain an active
            # M3DDAT label, but READFN consumes no filenames when NM3D is zero.
            if iprog!=0:
                raise ValueError('CALMET prognostic mode requires positive NM3D')
            meteorology_files=()
        else:
            meteorology_files=numbered_files('M3DDAT',declared_meteorology,'mm5')
    declared_initial_guess=int(_number(_one(values,'NIGF','0')))
    initial_guess_files=numbered_files('IGFDAT',declared_initial_guess,'igf')
    if version=='2.2':
        if 'MCLOUD' not in values or 'ICLDOUT' not in values:
            raise ValueError('CALMET 2.2 controls require MCLOUD and ICLDOUT')
        cloud_method=int(_number(_one(values,'MCLOUD')))
        cloud_output=int(_number(_one(values,'ICLDOUT','0')))
    else:
        # Pre-2.2 controls used one ICLOUD selector: 0/1 meant observed
        # clouds without/with OUTCLD, and 2..4 map directly to MCLOUD.
        if 'ICLOUD' not in values:
            raise ValueError('pre-2.2 CALMET controls require ICLOUD')
        legacy_cloud=int(_number(_one(values,'ICLOUD')))
        cloud_method=1 if legacy_cloud in {0,1} else legacy_cloud
        cloud_output=int(legacy_cloud==1)
    output_format=int(_number(_one(values,"IFORMO","1")))
    write_coast_distance=_logical(_one(values,"LDBCST","F"))
    prognostic_step_seconds=_integer_number(
        _one(values,'ISTEPPGS','0'),'ISTEPPGS')
    if prognostic_step_seconds==0:
        prognostic_step_seconds=3600*_integer_number(
            _one(values,'ISTEPPG','1'),'ISTEPPG')
    data: dict[str, Any] = dict(meteorology=(None if not meteorology_files else
              meteorology_files[0] if len(meteorology_files)==1 else meteorology_files),
              geophysical=calmet_path(_one(values,"GEODAT","GEO.DAT")),
              output=calmet_path(_one(values,"METDAT","CALMET.DAT")),
              start=start,end=end,grid=grid,z_faces_m=faces,
              regulatory_mode=int(_number(_one(values,'MREG','-1'))),
              control_lines=tuple(source.read_text(encoding='ascii',errors='replace').splitlines()),
              base_timezone=base_timezone,base_timezone_label=base_timezone_label,
              terrain_radius_km=float(_number(_one(values,"TERRAD","15"))),
              run_type=int(_number(_one(values,"IRTYPE","1"))),
              save_output=_logical(_one(values,"LSAVE","T")),
              legacy_output_format=output_format,
              mesopac_output=(calmet_path(_one(values,"PACDAT","PACOUT.DAT"))
                              if output_format==2 else None),
              distance_to_coast_output=(calmet_path(_one(values,"DCSTGD","DCST.GRD"))
                                        if write_coast_distance else None),
              listing_output=calmet_path(_one(values,"METLST","CALMET.LST")),
              print_fields=_logical(_one(values,'LPRINT','F')),
              print_interval_hours=_integer_number(
                  _one(values,'IPRINF','1'),'IPRINF'),
              print_uv_levels=print_uv_levels,
              print_w_levels=print_w_levels,
              print_temperature_levels=print_temperature_levels,
              print_surface_fields=print_surface_fields,
              micrometeorology_debug=_logical(_one(values,'LDB','F')),
              debug_first_step=_integer_number(_one(values,'NN1','1'),'NN1'),
              debug_last_step=_integer_number(_one(values,'NN2','1'),'NN2'),
              wind_debug_output=int(_number(_one(values,'IOUTD','0'))),
              wind_debug_levels=int(_number(_one(values,'NZPRN2','1'))),
              wind_debug_fields=wind_debug_fields,
              final_wind_debug_output=calmet_path(_one(values,'TSTOUT','TEST.OUT')),
              kinematic_wind_debug_output=calmet_path(_one(values,'TSTKIN','TEST.KIN')),
              froude_wind_debug_output=calmet_path(_one(values,'TSTFRD','TEST.FRD')),
              slope_wind_debug_output=calmet_path(_one(values,'TSTSLP','TEST.SLP')),
              wind_print_output=calmet_path(_one(values,'TSTPRT','TEST.PRT')),
              test_mode=int(_number(_one(values,"ITEST","2"))),
              timestep_seconds=_integer_number(
                  _one(values,"NSECDT","3600"),'NSECDT'),
              prognostic_timestep_seconds=prognostic_step_seconds,
              cloud_method=cloud_method,
              cloud_data=calmet_path(_one(values,"CLDDAT","CLOUD.DAT")),
              cloud_data_format=int(_number(_one(values,"IFORMC","2"))),
              cloud_output=cloud_output,
              surface_data=calmet_path(_one(values,"SRFDAT","SURF.DAT")),
              surface_data_format=int(_number(_one(values,"IFORMS","2"))),
              surface_control_stations=surface_control_stations,
              upper_air_files=upper_air_files,
              upper_control_stations=upper_control_stations,
              sea_files=sea_files,
              diagnostic_data=calmet_path(_one(values,"DIADAT","DIAG.DAT")),
              diagnostic_surface_temperature=int(_number(_one(values,"IDIOPT1","0"))),
              diagnostic_lapse_rate=int(_number(_one(values,"IDIOPT2","0"))),
              diagnostic_initial_wind=int(_number(_one(values,"IDIOPT3","0"))),
              diagnostic_surface_wind=int(_number(_one(values,"IDIOPT4","0"))),
              diagnostic_upper_wind=int(_number(_one(values,"IDIOPT5","0"))),
              initial_guess_upper_station=int(_number(_one(values,"IUPWND","-1"))),
              initial_guess_height_range_m=tuple(float(_number(item)) for item in
                  re.split(r"[\s,]+",_one(values,"ZUPWND","1.,1000.").strip()) if item),
              use_initial_guess_calmet=int(_number(_one(values,"IGFMET","0"))),
              initial_guess_calmet_files=initial_guess_files,
              weight_data=calmet_path(_one(values,"WTDAT","WT.DAT")),
              precipitation_station_count=int(_number(_one(values,"NPSTA","0"))),
              precipitation_data=calmet_path(_one(values,"PRCDAT","PRECIP.DAT")),
              precipitation_data_format=int(_number(_one(values,"IFORMP","2"))),
              precipitation_control_stations=precipitation_control_stations,
              noobs=int(_number(_one(values,"NOOBS","0"))),iprog=iprog,
              irhprog=int(_number(_one(values,"IRHPROG","0"))),itprog=int(_number(_one(values,"ITPROG","0"))),
              imixh=int(_number(_one(values,"IMIXH","1"))),iavet=int(_number(_one(values,"IAVET","1"))),
              iavezi=int(_number(_one(values,"IAVEZI","1"))),
              mixing_wind_level=int(_number(_one(values,"ILEVZI","1"))),
              mnmdav=int(_number(_one(values,"MNMDAV","1"))),
              half_average_angle=float(_number(_one(values,"HAFANG","30"))),
              include_w_temperature=_logical(_one(values,"LCALGRD","T")))
    data.update(constb=float(_number(_one(values,"CONSTB","1.41"))),
                conste=float(_number(_one(values,"CONSTE","0.15"))),
                constn=float(_number(_one(values,"CONSTN","2400"))),
                constw=float(_number(_one(values,"CONSTW","0.16"))),
                coriolis=float(_number(_one(values,"FCORIOL","999"))),
                minimum_lapse_rate=float(_number(_one(values,"DPTMIN","0.001"))),
                mixing_lapse_depth_m=float(_number(_one(values,"DZZI","200"))),
                minimum_mixing_height_m=float(_number(_one(values,"ZIMIN","50"))),
                maximum_mixing_height_m=float(_number(_one(values,"ZIMAX","3000"))),
                minimum_water_mixing_height_m=float(_number(_one(values,"ZIMINW","50"))),
                maximum_water_mixing_height_m=float(_number(_one(values,"ZIMAXW","3000"))),
                land_flux_threshold=float(_number(_one(values,"THRESHL","0"))),
                water_flux_threshold=float(_number(_one(values,"THRESHW",".05"))),
                water_temperature_mode=int(_number(_one(values,"ITWPROG","0"))),
                prognostic_ocean_category=int(_number(_one(values,"ILUOC3D","16"))),
                overwater_flux_method=int(_number(_one(values,"ICOARE","10"))),
                coastal_shelf_length_km=float(_number(_one(values,"DSHELF","0"))),
                warm_layer=int(_number(_one(values,"IWARM","0"))),
                cool_skin=int(_number(_one(values,"ICOOL","0"))),
                convective_relaxation=int(_number(_one(values,"IZICRLX","1"))),
                convective_relaxation_time_s=float(_number(_one(values,"TZICRLX","800"))),
                precipitation_interpolation=int(_number(_one(values,"NFLAGP","2"))),
                precipitation_radius_km=float(_number(_one(values,"SIGMAP","100"))),
                precipitation_cutoff=float(_number(_one(values,"CUTP","0.01"))),
                turbidity_coefficient_1=float(_number(_one(values,"HA1","990"))),
                turbidity_coefficient_2=float(_number(_one(values,"HA2","-30"))),
                cloudiness_coefficient_1=float(_number(_one(values,"HB1","-0.75"))),
                cloudiness_coefficient_2=float(_number(_one(values,"HB2","3.4"))),
                net_radiation_coefficient_1=float(_number(_one(values,"HC1","5.31e-13"))),
                net_radiation_coefficient_2=float(_number(_one(values,"HC2","60"))),
                net_radiation_coefficient_3=float(_number(_one(values,"HC3","0.12"))),
                surface_temperature_option=int(_number(_one(values,"ISURFT","-1"))),
                upper_temperature_option=int(_number(_one(values,"IUPT","-1"))),
                lapse_height_m=float(_number(_one(values,"ZUPT","200"))),
                temperature_interpolation_power=int(_number(_one(values,"IRAD","1"))),
                temperature_radius_km=float(_number(_one(values,"TRADKM","500"))),
                temperature_station_limit=int(_number(_one(values,"NUMTS","5"))),
                water_temperature_gradient_below=float(_number(_one(values,"TGDEFB","-.0098"))),
                water_temperature_gradient_above=float(_number(_one(values,"TGDEFA","-.0045"))),
                water_temperature_category_min=int(_number(_one(values,"JWAT1","999"))),
                water_temperature_category_max=int(_number(_one(values,"JWAT2","999"))))
    data.update(iwfcod=int(_number(_one(values,"IWFCOD","1"))),
                wind_extrapolation=int(_number(_one(values,"IEXTRP","-4"))),
                upper_air_separation_cells=float(_number(_one(values,'RMIN2','4'))),
                extrapolate_calm=int(_number(_one(values,'ICALM','0'))),
                wind_bias=tuple(wind_bias),
                varying_radius=_logical(_one(values,"LVARY","F")),
                minimum_station_radius_km=float(_number(_one(values,"RMIN","0.1"))),
                maximum_surface_radius_km=float(_number(_one(values,"RMAX1","30"))),
                maximum_upper_radius_km=float(_number(_one(values,"RMAX2","30"))),
                maximum_water_radius_km=float(_number(_one(values,"RMAX3","50"))),
                surface_background_radius_km=float(_number(_one(values,"R1","1"))),
                upper_background_radius_km=float(_number(_one(values,"R2","1"))),
                prognostic_radius_km=float(_number(_one(values,"RPROG","0"))),
                stations_per_level=tuple(nintr),wind_extrapolation_factors=tuple(fextr),
                barrier_count=barrier_count,barrier_top_level=int(_number(_one(values,"KBAR",str(len(faces)-1)))),
                barriers=barriers,
                lake_breeze_regions=lake_regions,
                ifradj=int(_number(_one(values,"IFRADJ","1"))),ikine=int(_number(_one(values,"IKINE","0"))),
                iobr=int(_number(_one(values,"IOBR","0"))),islope=int(_number(_one(values,"ISLOPE","1"))),
                critical_froude=float(_number(_one(values,"CRITFN","1"))),smoothing_passes=smoothing,
                divergence_limit=float(_number(_one(values,"DIVLIM","5e-6"))),
                divergence_iterations=int(_number(_one(values,"NITER","50"))),
                kinematic_alpha=float(_number(_one(values,"ALPHA","0.1"))))
    data.update(overrides); return CalmetConfig(**data)


def readlst(path: str | Path, component: str | None = None) -> dict[str, Any]:
    """Decode a legacy control/list file into kwargs accepted by the public API.

    The returned keys are the descriptive configuration field names.  Nested
    grid settings remain a :class:`GridConfig` instance, so the result can be
    passed directly as ``pycalmet.<component>(**pycalmet.readlst(path))``.
    ``component`` is normally inferred from the recorded assignments or path;
    it can be one of ``calwrf``, ``terrel``, ``ctrproc``/``ctgproc``,
    ``makegeo``, or ``calmet`` when a generically named file is ambiguous.

    Args:
        path: Control file or processor listing to decode.
        component: Optional workflow-stage name. ``None`` infers the stage
            from assignments and the source path.

    Returns:
        Canonically named keyword arguments, including a resolved
        :class:`GridConfig` where the selected stage uses one.

    Raises:
        ValueError: If the source control cannot be found, the component
            cannot be inferred, or ``component`` is unknown.
    """
    source = _control_source(path)
    if component is None:
        values = assignments(source)
        if {"METDAT", "IBYR"} <= values.keys() or "M3DDAT" in values:
            component = "calmet"
        elif "TERRDAT" in values or "NINCAT" in values:
            component = "makegeo"
        elif "NDBF" in values or any(name in values for name in
                ("GLAZNA", "GLAZSA", "GLAZEU", "GLAZAS", "GLAZAF", "GLAZAP")):
            component = "ctgproc"
        elif "NTDF" in values or "OUTFIL" in values:
            component = "terrel"
        else:
            hint = f"{source.parent.name}/{source.name}".casefold()
            component = next((name for name in
                ("calwrf", "terrel", "ctgproc", "makegeo", "calmet")
                if name in hint), None)
        if component is None:
            raise ValueError(
                f"cannot infer component for {path}; pass component explicitly"
            )
    key = str(component).casefold().replace("-", "").replace("_", "")
    aliases = {"calwrf":"calwrf", "terrel":"terrel", "ctrproc":"ctgproc",
               "ctgproc":"ctgproc", "makegeo":"makegeo", "calmet":"calmet"}
    try:
        selected = aliases[key]
    except KeyError as error:
        raise ValueError(f"unknown CALMET workflow component: {component!r}") from error
    loaders: dict[str, Callable[..., Any]] = {
        "calwrf": calwrf_config,
        "terrel": terrel_config,
        "ctgproc": ctgproc_config,
        "makegeo": makegeo_config,
        "calmet": calmet_config,
    }
    config: Any = loaders[selected](path)
    return {field.name:getattr(config, field.name) for field in fields(config)}
