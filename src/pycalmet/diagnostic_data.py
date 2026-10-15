"""Sequential scalar records from legacy CALMET DIAG.DAT files."""
from __future__ import annotations

from typing import TypeAlias

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


Float64Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    """One optional set of PREPDI diagnostic overrides.

    Attributes:
        surface_temperature_k: Domain diagnostic surface temperature in
            kelvin, or ``None`` when that record type was not requested.
        lapse_rate_k_m: Domain temperature lapse rate in kelvin per metre, or
            ``None``. DIAG.DAT stores this value in kelvin per kilometre.
        initial_u_m_s: Initial grid-wide U wind component in metres per second,
            or ``None``.
        initial_v_m_s: Initial grid-wide V wind component in metres per second,
            or ``None``.
        surface_u_m_s: Float64 array of shape ``(surface_station,)`` containing
            surface-station U components in metres per second. Missing values
            are encoded as 9999; ``None`` means surface winds were not read.
        surface_v_m_s: Surface-station V components with the same representation
            as ``surface_u_m_s``.
        upper_u_m_s: Float64 array of shape ``(level, upper_station)`` containing
            upper-air U components in metres per second. Missing values are
            encoded as 9999; ``None`` means upper winds were not read.
        upper_v_m_s: Upper-air V components with the same representation as
            ``upper_u_m_s``.
    """
    surface_temperature_k: float | None = None
    lapse_rate_k_m: float | None = None
    initial_u_m_s: float | None = None
    initial_v_m_s: float | None = None
    surface_u_m_s: Float64Array | None = None
    surface_v_m_s: Float64Array | None = None
    upper_u_m_s: Float64Array | None = None
    upper_v_m_s: Float64Array | None = None


def _fixed_value(line: str, label: str, width: int) -> float:
    """Read the value following the ten columns skipped by formats 1010/1030.

    Args:
        line: Fixed-column DIAG.DAT record text.
        label: Record description used in validation errors.
        width: Number of characters occupied by the numeric field.

    Returns:
        Parsed floating-point value from columns 11 through ``10 + width``.

    Raises:
        ValueError: If the record is too short or the field is not numeric.
    """
    if len(line) < 10+width:
        raise ValueError(f'DIAG.DAT {label} record is shorter than ten columns')
    try:
        return float(line[10:10+width])
    except ValueError as error:
        raise ValueError(f'invalid DIAG.DAT {label} record') from error


class DiagnosticData:
    """Index supported PREPDI DIAG.DAT records by CALMET interval start.

    Attributes:
        path: Path to the formatted DIAG.DAT file.
        records: Mapping from interval-start datetimes to
            :class:`DiagnosticRecord` objects.
    """

    def __init__(
        self,
        path: str | Path,
        start: datetime,
        count: int,
        timestep_seconds: int,
        *,
        surface_temperature: bool=False,
        lapse_rate: bool=False,
        initial_wind: bool=False,
        surface_wind: bool=False,
        upper_wind: bool=False,
        surface_wind_names: Sequence[object]=(),
        upper_wind_names: Sequence[object]=(),
        levels: int=0,
        iextrp: int=-1,
    ) -> None:
        """Read the requested sequence of diagnostic records.

        Args:
            path: Path to formatted DIAG.DAT text.
            start: Beginning datetime assigned to the first record group.
            count: Number of timestep record groups to read.
            timestep_seconds: Seconds between consecutive record keys.
            surface_temperature: Whether each group contains a domain surface
                temperature scalar.
            lapse_rate: Whether each group contains a lapse-rate scalar.
            initial_wind: Whether each group contains initial U and V scalars.
            surface_wind: Whether surface-wind blocks should be consumed even
                when ``surface_wind_names`` is empty.
            upper_wind: Whether upper-air wind blocks should be consumed even
                when ``upper_wind_names`` is empty.
            surface_wind_names: Expected surface-station names. Names are
                normalized to uppercase four-character identifiers.
            upper_wind_names: Expected upper-air station names, normalized in
                the same way.
            levels: Number of CALMET layers in each upper-air block.
            iextrp: Surface-wind extrapolation option. A positive value permits
                the first upper-air level unless the station is also a surface
                station.

        Raises:
            OSError: If the input file cannot be read.
            UnicodeDecodeError: If the input is not ASCII.
            ValueError: If a requested record is missing or malformed, station
                identifiers are ambiguous, or ``levels`` is negative.
        """
        self.path: Path=Path(path)
        lines=iter(self.path.read_text(encoding='ascii',errors='strict').splitlines())

        def take(label: str,width: int) -> float:
            """Read one fixed-width scalar from the next input line.

            Args:
                label: Record description used in error messages.
                width: Width of the numeric field after the ten-column prefix.

            Returns:
                Parsed scalar value.

            Raises:
                ValueError: If the file ends early or the field is invalid.
            """
            try: return _fixed_value(next(lines),label,width)
            except StopIteration as error:
                raise ValueError(f'DIAG.DAT ends before {label} record') from error

        def line(label: str) -> str:
            """Return the next raw line from DIAG.DAT.

            Args:
                label: Record description used when reporting early EOF.

            Returns:
                The next line without its newline terminator.

            Raises:
                ValueError: If the file ends before the requested line.
            """
            try: return next(lines)
            except StopIteration as error:
                raise ValueError(f'DIAG.DAT ends before {label} record') from error

        def names(values: Sequence[object],label: str) -> tuple[str,...]:
            """Normalize and validate a station-name sequence.

            Args:
                values: Station identifiers accepted by ``str``.
                label: Station group name used in validation errors.

            Returns:
                Tuple of unique uppercase names truncated to four characters.

            Raises:
                ValueError: If two inputs collapse to the same A4 identifier.
            """
            normalized=tuple(str(item).strip().upper()[:4] for item in values)
            if len(set(normalized))!=len(normalized):
                raise ValueError(f'DIAG.DAT {label} station names are not unique in A4 format')
            return normalized

        surface_names=names(surface_wind_names,'surface')
        upper_names=names(upper_wind_names,'upper-air')
        read_surface_wind=bool(surface_wind or surface_names)
        read_upper_wind=bool(upper_wind or upper_names)
        nz=int(levels)
        if nz<0: raise ValueError('DIAG.DAT level count must be non-negative')

        def fields(text: str,start: int,number: int,label: str) -> list[float]:
            """Decode consecutive five-character numeric fields.

            Args:
                text: Fixed-column record text.
                start: Zero-based character offset of the first field.
                number: Number of fields to decode.
                label: Record description used in validation errors.

            Returns:
                List of ``number`` floating-point values.

            Raises:
                ValueError: If the record is too short or a field is not
                    numeric.
            """
            if len(text)<start+5*number:
                raise ValueError(f'DIAG.DAT {label} record is too short')
            try: return [float(text[start+5*i:start+5*(i+1)]) for i in range(number)]
            except ValueError as error: raise ValueError(f'invalid DIAG.DAT {label} record') from error

        def surface_block() -> tuple[Float64Array,Float64Array]:
            """Read one LAST-terminated surface-wind station block.

            Returns:
                Tuple ``(u, v)`` of float64 arrays with shape
                ``(surface_station,)`` and units of metres per second. Missing
                component values are encoded as 9999.

            Raises:
                ValueError: If a row is truncated, malformed, or names an
                    unconfigured station.
            """
            u=np.full(len(surface_names),9999.); v=np.full(len(surface_names),9999.)
            lookup={name:index for index,name in enumerate(surface_names)}
            while True:
                text=line('surface wind')
                if len(text)<19: raise ValueError('DIAG.DAT surface wind record is too short')
                name=text[15:19].strip().upper()
                if name=='LAST': break
                if not name: continue
                if name not in lookup: raise ValueError(f'unrecognized DIAG.DAT surface station {name!r}')
                _,u_value,v_value=fields(text,20,3,'surface wind')
                index=lookup[name]
                u[index]=9999. if u_value==999. else u_value
                v[index]=9999. if v_value==999. else v_value
            return u,v

        def upper_block() -> tuple[Float64Array,Float64Array]:
            """Read one LAST-terminated upper-air wind station block.

            Returns:
                Tuple ``(u, v)`` of float64 arrays with shape
                ``(level, upper_station)`` and units of metres per second.
                Missing components are encoded as 9999.

            Raises:
                ValueError: If a primary or continuation row is truncated,
                    malformed, or names an unconfigured station.
            """
            u=np.full((nz,len(upper_names)),9999.); v=np.full_like(u,9999.)
            lookup={name:index for index,name in enumerate(upper_names)}
            total=2*nz+1; first=min(total,11)
            while True:
                text=line('upper-air wind')
                if len(text)<17: raise ValueError('DIAG.DAT upper-air wind record is too short')
                name=text[13:17].strip().upper()
                values=fields(text,18,first,'upper-air wind')
                if total>11:
                    values+=fields(line('upper-air wind continuation'),20,total-11,
                                   'upper-air wind continuation')
                if name=='LAST': break
                if name not in lookup: raise ValueError(f'unrecognized DIAG.DAT upper-air station {name!r}')
                index=lookup[name]
                for level in range(nz):
                    u_value,v_value=values[1+2*level:3+2*level]
                    if level==0 and (iextrp<=0 or name in surface_names): continue
                    if u_value!=999. and v_value!=999.:
                        u[level,index]=u_value; v[level,index]=v_value
            return u,v

        records: dict[datetime,DiagnosticRecord]={}
        for index in range(int(count)):
            temperature: float | None
            gamma: float | None
            u: float | None
            v: float | None
            surface_u: Float64Array | None
            surface_v: Float64Array | None
            upper_u: Float64Array | None
            upper_v: Float64Array | None
            temperature=take('surface temperature',6) if surface_temperature else None
            # GAMDIA is read in K/km and converted to K/m by PREPDI.
            gamma=take('lapse rate',5)/1000. if lapse_rate else None
            u=take('initial U wind',5) if initial_wind else None
            v=take('initial V wind',5) if initial_wind else None
            surface_u,surface_v=(surface_block() if read_surface_wind else (None,None))
            upper_u,upper_v=(upper_block() if read_upper_wind else (None,None))
            when=start+timedelta(seconds=index*timestep_seconds)
            records[when]=DiagnosticRecord(temperature,gamma,u,v,
                surface_u,surface_v,upper_u,upper_v)
        self.records: dict[datetime,DiagnosticRecord]=records

    def at_begin(self, when: datetime) -> DiagnosticRecord:
        """Return the diagnostic record at an exact interval beginning.

        Args:
            when: Interval-start datetime used while indexing the file.

        Returns:
            The corresponding :class:`DiagnosticRecord`.

        Raises:
            ValueError: If no diagnostic record begins at ``when``.
        """
        try: return self.records[when]
        except KeyError as error:
            raise ValueError(f'DIAG.DAT has no record for {when}') from error
