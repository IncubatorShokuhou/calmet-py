"""Legacy CALMET line-printer grid formatting."""
from __future__ import annotations

from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Callable, Iterable, Sequence, TextIO, cast

import numpy as np
import numpy.typing as npt

from .calmet_numerics import convective_velocity_scale, pgt_stability
from .calmet_solver import HourFields, Step1Fields
from .config import CalmetConfig
from .datetime_utils import format_ordinal_code
from .surface_data import SurfaceRecord, SurfaceStation
from .upper_air import Sounding, UpperAirStation


def _coded_time(value: datetime) -> int:
    """Encode a listing timestamp as CALMET ``YYYYJJJHH``.

    Args:
        value: Datetime to encode.

    Returns:
        Integer ordinal-date code.

    Raises:
        ValueError: If ``value`` cannot be represented.
    """
    return format_ordinal_code(value)


def format_surface_input_debug(
    record: SurfaceRecord,
    stations: Iterable[SurfaceStation],
) -> str:
    """Reproduce COMP formats 108/109 for one raw SURF.DAT record.

    Args:
        record: Surface record whose eight station fields are one-dimensional.
        stations: Station metadata in the same order as the record arrays.

    Returns:
        Newline-terminated listing text containing interval timestamps and one
        row per station. Units are metres per second, degrees, hundreds of
        feet, cloud tenths, kelvin, percent, and millibars as labeled.

    Raises:
        ValueError: If any observation vector does not have shape
            ``(len(stations),)`` or a timestamp cannot be encoded.
    """
    station_values: tuple[SurfaceStation, ...] = tuple(stations)
    count: int = len(station_values)
    fields: tuple[npt.NDArray[np.generic], ...] = (
        record.wind_speed,record.wind_direction,record.ceiling_hundreds_ft,
        record.cloud_tenths,record.temperature_k,record.relative_humidity,
        record.pressure_mb,record.precipitation_code)
    if any(np.asarray(values).shape!=(count,) for values in fields):
        raise ValueError('surface debug record does not match station count')
    lines: list[str] = [
        (f"\n SURFACE MET. INPUT DATA -- YYYYJJJHH: {_coded_time(record.begin):10d}   "
         f"seconds: {record.begin.minute*60+record.begin.second:5d}  to -- YYYYJJJHH: "
         f"{_coded_time(record.end):10d}   seconds: "
         f"{record.end.minute*60+record.end.second:5d}"),
        (" Station ID      WS       WD     Ceil Ht   Cloud Cover    Temp    "
         "Rel Hum    Pres   Precip Code"),
        ("                (m/s)   (deg)   (100s FT)    (tenths)    (deg K)"
         "    (%)      (mb)"),
    ]
    for index,station in enumerate(station_values):
        lines.append(
            f" {int(station.identifier):9d}    {float(record.wind_speed[index]):7.2f}  "
            f"{float(record.wind_direction[index]):6.1f}     "
            f"{int(record.ceiling_hundreds_ft[index]):4d}        "
            f"{int(record.cloud_tenths[index]):4d}      "
            f"{float(record.temperature_k[index]):7.2f}    "
            f"{int(record.relative_humidity[index]):4d}    "
            f"{float(record.pressure_mb[index]):7.2f}     "
            f"{int(record.precipitation_code[index]):4d}")
    return "\n".join(lines)+"\n"


def format_no_upper_prepdi_debug(step: Step1Fields, ending: datetime) -> str:
    """Format ``NOOBS=1`` PREPDI diagnostics, including unavailable data.

    Args:
        step: Step-one fields containing retained diagnostic station names,
            U/V winds, and optional domain surface temperature.
        ending: Ending datetime printed in the diagnostic heading.

    Returns:
        Newline-terminated listing block. Station winds are printed in metres
        per second and temperature, when available, in kelvin.

    Raises:
        ValueError: If retained U/V station arrays do not match the number of
            station names.
    """
    names: tuple[str, ...] = tuple(step.debug_station_names)
    u: npt.NDArray[np.floating] = np.asarray(step.debug_station_u)
    v: npt.NDArray[np.floating] = np.asarray(step.debug_station_v)
    if u.shape!=(len(names),) or v.shape!=(len(names),):
        raise ValueError('PREPDI debug winds do not match station count')
    day=ending.timetuple().tm_yday
    second=ending.minute*60+ending.second
    temperature: str = (
        "  Domain scale surf. temperature (TINF) = N/A (not available)"
        if step.debug_domain_temperature is None else
        (f"  Domain scale surf. temperature (TINF) = "
         f"{step.debug_domain_temperature:10.2f} (deg. K)"))
    lines: list[str] = [
        (f"\n  Diagnostic wind field data -- Year: {ending.year:4d}   "
         f"Julian day: {day:3d}   Hour: {ending.hour:2d}   Second: {second:4d}"),
        temperature,
        "  Domain-scale temp. lapse rate (GAMMA) is not yet  defined",
        "  Domain-scale U-component of wind (UM) is not yet  defined",
        "  Domain-scale V-component of wind (VM) is not yet  defined",
        (f"\n  Surface station data for wind field module -- Year: {ending.year:4d}   "
         f"Julian day: {day:3d}   Hour: {ending.hour:2d}   Second: {second:4d}"),
        "     Station  Station    U-CMPT    V-CMPT",
        "      Name     Number     (m/s)     (m/s)",
    ]
    for index,(name,u_value,v_value) in enumerate(zip(names,u,v),1):
        lines.append(f"      {str(name):4.4s}     {index:4d}     {u_value:7.2f}   {v_value:7.2f}")
    lines.append("  upper air data not yet defined...")
    return "\n".join(lines)+"\n"


def format_upper_air_input_debug(
    station_number: int,
    sounding: Sounding,
    convergence_degrees: npt.ArrayLike,
    layer_u: npt.ArrayLike,
    layer_v: npt.ArrayLike,
    temperature_faces: npt.ArrayLike | None = None,
) -> str:
    """Reproduce COMP formats 133--136 for one raw UP.DAT sounding.

    Args:
        station_number: One-based station number printed in each source row.
        sounding: Upper-air sounding containing pressure, height, temperature,
            direction, and speed arrays.
        convergence_degrees: Map convergence passed to ``sounding.converted``.
        layer_u: Cell-averaged U wind by model layer, in metres per second.
        layer_v: Cell-averaged V wind by model layer, in metres per second.
        temperature_faces: Optional face temperature profile in kelvin.

    Returns:
        Newline-terminated listing block. Raw profile columns use millibars,
        metres local ground level, kelvin, and metres per second.

    Raises:
        ValueError: If raw sounding arrays have inconsistent shapes or sounding
            conversion fails.
    """
    height: npt.NDArray[np.float32]
    u: npt.NDArray[np.float32]
    v: npt.NDArray[np.float32]
    height,u,v=sounding.converted(convergence_degrees)
    count: int = len(height)
    fields: tuple[npt.NDArray[np.generic], ...] = (
        sounding.pressure_mb,sounding.temperature_c,u,v)
    if any(np.asarray(values).shape!=(count,) for values in fields):
        raise ValueError('upper-air debug sounding fields do not match')
    timestamp=sounding.timestamp
    lines: list[str] = [
        (f"\n  UPPER AIR DATA  --  Year:{timestamp.year:4d}  Julian Day:"
         f"{timestamp.timetuple().tm_yday:3d}  Hour:{timestamp.hour:2d} (GMT)"),
        "",
        "  STA  PRES   HEIGHT    TEMP      U       V",
        "       (mb)   (m LGL)   (K)     (m/s)   (m/s)",
    ]
    temperature: npt.NDArray[np.float32] = np.asarray(
        sounding.temperature_c,dtype=np.float32)
    temperature=np.where(temperature<np.float32(999.89),
                         temperature+np.float32(273.15),temperature)
    for pressure,z,temp,u_value,v_value in zip(
            sounding.pressure_mb,height,temperature,u,v):
        lines.append(f" {int(station_number):3d}  {pressure:6.1f}  {z:7.1f}  "
                     f"{temp:6.2f}  {u_value:6.2f}  {v_value:6.2f}")
    def profile(label: str, values: npt.ArrayLike) -> list[str]:
        """Format a labeled profile in chunks of ten values.

        Args:
            label: Text printed before the first profile row.
            values: One-dimensional values formatted with three decimals.

        Returns:
            List of listing lines without trailing newline characters.
        """
        value_array: npt.NDArray[np.number] = np.asarray(values)
        chunks: list[str] = []
        for begin in range(0,value_array.size,10):
            prefix=f" {label:<20} = " if begin==0 else " "*24
            chunks.append(prefix+''.join(
                f"{value:10.3f}" for value in value_array[begin:begin+10]))
        return chunks
    lines.extend(profile('U -- cell averaged',layer_u))
    lines.extend(profile('V -- cell averaged',layer_v))
    if temperature_faces is not None:
        lines.extend(profile('T -- cell face',temperature_faces))
    return "\n".join(lines)+"\n"


def format_full_prepdi_debug(
    step: Step1Fields,
    ending: datetime,
    lapse_rate: npt.ArrayLike,
) -> str:
    """Reproduce COMP formats 152 and 154--159 for ``NOOBS=0``.

    Args:
        step: Step-one fields containing retained domain, surface-station, and
            upper-air diagnostics.
        ending: Ending datetime printed in the diagnostic headings.
        lapse_rate: Grid lapse-rate array in kelvin per metre; element ``[0, 0]``
            is printed.

    Returns:
        Newline-terminated PREPDI listing block. Wind components use metres per
        second and the optional domain temperature uses kelvin.

    Raises:
        ValueError: If required domain wind scalars are absent or retained
            station wind arrays do not match their names.
        IndexError: If ``lapse_rate`` has no ``[0, 0]`` element.
    """
    if step.debug_domain_u is None or step.debug_domain_v is None:
        raise ValueError('complete PREPDI debug wind scalars are required')
    names: tuple[str, ...] = tuple(step.debug_station_names)
    u: npt.NDArray[np.floating] = np.asarray(
        cast(npt.ArrayLike,step.debug_station_u))
    v: npt.NDArray[np.floating] = np.asarray(
        cast(npt.ArrayLike,step.debug_station_v))
    if u.shape!=(len(names),) or v.shape!=(len(names),):
        raise ValueError('PREPDI debug winds do not match station count')
    day=ending.timetuple().tm_yday
    second=ending.minute*60+ending.second
    temperature: str = (
        "  Domain scale surf. temperature (TINF) = N/A (not available)"
        if step.debug_domain_temperature is None else
        (f"  Domain scale surf. temperature (TINF) = "
         f"{step.debug_domain_temperature:10.2f} (deg. K)"))
    lines: list[str] = [
        (f"\n  Diagnostic wind field data -- Year: {ending.year:4d}   "
         f"Julian day: {day:3d}   Hour: {ending.hour:2d}   Second: {second:4d}"),
        ("  Domain-scale temp. lapse rate at point (1,1)(GAMMA) = "
         f"{float(np.asarray(lapse_rate)[0,0]):10.5f} (deg. K/m)"),
        temperature,
        (f"  Domain-scale U-component of wind (UM) = "
         f"{step.debug_domain_u:10.3f} (m/s)"),
        (f"  Domain-scale V-component of wind (VM) = "
         f"{step.debug_domain_v:10.3f} (m/s)"),
        (f"\n  Surface station data for wind field module -- Year: {ending.year:4d}   "
         f"Julian day: {day:3d}   Hour: {ending.hour:2d}   Second: {second:4d}"),
        "     Station  Station    U-CMPT    V-CMPT",
        "      Name     Number     (m/s)     (m/s)",
    ]
    for index,(name,u_value,v_value) in enumerate(zip(names,u,v),1):
        lines.append(f"      {str(name):4.4s}     {index:4d}     {u_value:7.2f}   {v_value:7.2f}")
    for station_number,item in enumerate(step.debug_upper_data,1):
        station,_,_,upper_u,upper_v=cast(
            "tuple[UpperAirStation, tuple[Sounding, ...], float, "
            "npt.NDArray[np.float32], npt.NDArray[np.float32]]",item)
        lines.extend((
            (f"\n  Upper air station: {str(station.name):4.4s}   Number: "
             f"{station_number:4d}   Year: {ending.year:4d}   Julian day: {day:3d}"
             f"   Hour: {ending.hour:2d}   Second: {second:4d}"),
            "        Level   U-CMPT   V-CMPT",
            "                 (m/s)    (m/s)",
        ))
        for level,(u_value,v_value) in enumerate(zip(upper_u,upper_v),1):
            lines.append(f"       {level:4d}    {u_value:7.2f}  {v_value:7.2f}")
    return "\n".join(lines)+"\n"


def format_grid(
    values: npt.ArrayLike,
    message: str,
    *,
    significant_digits: int = 5,
    begin: datetime | None = None,
    timestep_seconds: int = 3600,
) -> str:
    """Reproduce CALMET ``OUT/WRT/WRT2`` for one two-dimensional field.

    Args:
        values: Nonempty two-dimensional numeric field of shape ``(ny, nx)``.
        message: Heading text, truncated to the listing's 70-character field.
        significant_digits: Printed significant-digit mode from 1 through 5;
            this also selects values per output row.
        begin: Optional interval beginning printed above the grid.
        timestep_seconds: Seconds added to ``begin`` for the printed interval
            ending.

    Returns:
        Newline-terminated line-printer representation. Values are converted to
        float32 for scaling and integer rounding. An all-zero field produces a
        suppression message instead of grid rows.

    Raises:
        ValueError: If ``significant_digits`` is outside 1--5, ``values`` is not
            two-dimensional, or the array is empty.
    """
    if significant_digits not in range(1, 6):
        raise ValueError("significant_digits must be from 1 through 5")
    data: npt.NDArray[np.number] = np.asarray(values)
    if data.ndim!=2:
        raise ValueError("line-printer fields must be two-dimensional")
    ny,nx=data.shape
    width=(3,3,3,4,5)[significant_digits-1]
    columns=(40,40,30,25,20)[significant_digits-1]
    lines: list[str] = ["", "", f" {message:<70}"[:71]]
    if begin is not None:
        end=begin+timedelta(seconds=timestep_seconds)
        lines.extend(("", _period_line(begin,end), ""))
    maximum=float(np.max(data)); minimum=float(np.min(data))
    if minimum==0. and maximum==0.:
        lines.append(" GRID NOT PRINTED -- all values zero")
        return "\n".join(lines)+"\n"
    magnitude=max(maximum,abs(minimum))
    exponent=int(np.log10(np.float32(magnitude)))
    if magnitude<1.: exponent-=1
    exponent-=significant_digits-(1 if significant_digits==1 else 2)
    scale=np.float32(10.)**np.int32(-exponent)
    lines.extend((f" Multiply all values by 10 ** {exponent:3d}", ""))
    for first in range(0,nx,columns):
        last=min(first+columns,nx)
        for row in range(ny-1,-1,-1):
            selected: npt.NDArray[np.float32] = np.asarray(
                data[row,first:last],dtype=np.float32)
            signs: npt.NDArray[np.str_] = np.where(selected<0.,'-','+')
            rounded: npt.NDArray[np.int64] = np.where(
                selected<0.,-(selected*scale-np.float32(.5)),
                selected*scale+np.float32(.5)).astype(np.int64)
            lines.append(f" {row+1:3d} I"+''.join(f"{value:{width}d} " for value in rounded))
            lines.append("     I"+''.join(f"{sign:>{width}} " for sign in signs))
        count=last-first
        underline=(significant_digits+2 if significant_digits==1
                   else significant_digits+1)*count-1
        lines.append("     "+'-'*underline)
        lines.append("      "+''.join(f"{index:{width}d} "
                                      for index in range(first+1,last+1)))
    return "\n".join(lines)+"\n"


def _period_line(begin: datetime, end: datetime) -> str:
    """Format one CALMET listing interval line.

    Args:
        begin: Interval beginning datetime.
        end: Interval ending datetime.

    Returns:
        Single line containing calendar date, hour, and seconds after the hour
        for both bounds.
    """
    return (f" FROM: year: {begin.year:4d}  month: {begin.month:2d}  day: {begin.day:2d}  "
            f"hour: {begin.hour:2d}  second: {begin.minute*60+begin.second:4d}  TO: year: "
            f"{end.year:4d}  month: {end.month:2d}  day: {end.day:2d}  hour: {end.hour:2d}  "
            f"second: {end.minute*60+end.second:4d}")


class CalmetListingWriter:
    """Append Input Group 3 selected fields to a CALMET listing.

    Attributes:
        stream: Open ASCII output stream, or ``None`` after closing.
        config: CALMET configuration controlling print switches and cadence.
        landuse: Integer-like land-use array with shape ``(ny, nx)``.
        water_range: Inclusive land-use category range treated as water.
        first_hour: First whole-hour timestamp seen, used to number simulation
            hours, or ``None`` before the first append.
    """

    stream: TextIO | None
    config: CalmetConfig
    landuse: npt.NDArray[np.integer]
    water_range: tuple[int, int]
    first_hour: datetime | None

    def __init__(
        self,
        path: str | PathLike[str],
        config: CalmetConfig,
        landuse: npt.ArrayLike,
        water_range: Sequence[int],
    ) -> None:
        """Open a listing destination and retain its print configuration.

        Args:
            path: Destination ASCII file; missing parent directories are
                created.
            config: CALMET configuration with listing and debug switches.
            landuse: Integer-like ``(ny, nx)`` land-use categories.
            water_range: Inclusive ``(minimum, maximum)`` water codes.

        Raises:
            OSError: If the destination directory or file cannot be created.

        Notes:
            Construction truncates an existing destination file.
        """
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        self.stream=target.open('w',encoding='ascii')
        self.config=config
        self.landuse=np.asarray(landuse)
        self.water_range=cast(tuple[int,int],tuple(water_range))
        self.first_hour=None

    def append(
        self,
        hour: HourFields,
        solar_elevation_sine: npt.ArrayLike,
    ) -> None:
        """Append enabled debug and line-printer fields for one timestep.

        Args:
            hour: Solver fields for one timestamp. Printed grids have shape
                ``(ny, nx)`` and retain the units named in their headings.
            solar_elevation_sine: ``(ny, nx)`` sine of solar elevation used for
                stability classification and point diagnostics.

        Raises:
            ValueError: If an enabled field is not two-dimensional or cannot be
                formatted with its configured precision.
            RuntimeError: If debug output requires retained source fields that
                are absent.
            OSError: If writing or flushing the listing fails.

        Notes:
            Micrometeorology diagnostics follow their own configured step
            window. Ordinary grids are written only on whole hours satisfying
            ``print_interval_hours``. The stream is flushed after output.
        """
        config=self.config
        timestamp=hour.timestamp
        current_hour=timestamp.replace(minute=0,second=0,microsecond=0)
        if self.first_hour is None:
            self.first_hour=current_hour
        nh=int((current_hour-self.first_hour).total_seconds()//3600)+1
        if (config.micrometeorology_debug and
                config.debug_first_step<=nh<=config.debug_last_step):
            self._append_noobs_debug(hour,solar_elevation_sine)
        if (nh%config.print_interval_hours or timestamp.minute or timestamp.second):
            return
        duration=config.timestep_seconds
        def write(
            values: npt.ArrayLike,
            message: str,
            digits: int = 5,
        ) -> None:
            """Format and write one grid using the current interval.

            Args:
                values: Two-dimensional grid values.
                message: Listing heading including physical units.
                digits: Significant-digit mode from 1 through 5.

            Raises:
                ValueError: If the grid or precision is invalid.
                OSError: If writing to the listing fails.

            Notes:
                The current hour timestamp and configured timestep determine the
                interval printed above the field.
            """
            stream=cast(TextIO,self.stream)
            stream.write(format_grid(
                values,message,significant_digits=digits,begin=timestamp,
                timestep_seconds=duration))
        for level,enabled in enumerate(config.print_uv_levels):
            if enabled:
                write(hour.wind.u[level],f'U-component (m/s) -- Level: {level+1:3d}')
                write(hour.wind.v[level],f'V-component (m/s) -- Level: {level+1:3d}')
        for level,enabled in enumerate(config.print_w_levels):
            if enabled:
                write(hour.wind.w[level+1],f'W-component (m/s) -- Level: {level+1:3d}')
        step=hour.step1; surface=hour.surface
        switches=config.print_surface_fields
        if switches[0]:
            stability: npt.NDArray[np.int8] = pgt_stability(
                hour.wind.u[0],hour.wind.v[0],solar_elevation_sine,
                step.cloud_fraction,step.ceiling_height,self.landuse,
                surface.monin_obukhov_length,water_range=self.water_range)
            write(stability,'PGT stability class',2)
        averaging=('After averaging' if config.iavezi else 'No averaging')
        fields: tuple[tuple[Callable[[], npt.ArrayLike], str], ...] = (
            (lambda:surface.friction_velocity,'Friction velocity (m/s)'),
            (lambda:surface.monin_obukhov_length,'Monin-Obukhov length (m)'),
            (lambda:surface.mixing_height,f'Mixing height (m) -- {averaging}'),
            (lambda:self._convective_velocity(hour),'Convective velocity scale (m/s)'),
            (lambda:step.precipitation_rate,'Precipitation rate (mm/hr)'),
            (lambda:surface.sensible_heat_flux,'Surface sensible heat flux (W/m**2)'),
            (lambda:surface.convective_mixing_height,
             f'Convective mixing height (m) -- {averaging}'),
        )
        for enabled,(values,message) in zip(switches[1:],fields):
            if enabled:
                write(values(),message)
        if config.include_w_temperature or config.itprog>=1:
            for level,enabled in enumerate(config.print_temperature_levels):
                if enabled:
                    write(hour.temperature[level],f'Temperature (K) -- Level: {level+1:3d}')
        cast(TextIO,self.stream).flush()

    def _append_noobs_debug(
        self,
        hour: HourFields,
        solar_elevation_sine: npt.ArrayLike,
    ) -> None:
        """Write COMP LDB diagnostics for the configured observation mode.

        Args:
            hour: Solver fields with any retained SURF.DAT, UP.DAT, PREPDI, and
                unaveraged micrometeorology diagnostics.
            solar_elevation_sine: Grid sine of solar elevation; point ``[0, 0]``
                is printed.

        Raises:
            RuntimeError: If an observation mode requires a retained surface
                debug record that is absent.
            ValueError: If retained station/profile fields are inconsistent or
                a grid cannot be line-printer formatted.
            IndexError: If required point or grid arrays are empty.
            OSError: If writing or flushing the listing fails.

        Notes:
            The method writes no-observation placeholders for ``NOOBS=2``,
            surface diagnostics for ``NOOBS=1``, and full upper-air/PREPDI
            diagnostics for ``NOOBS=0`` before common surface parameters.
        """
        timestamp=hour.timestamp
        surface=hour.surface
        sine=float(np.asarray(solar_elevation_sine)[0,0])
        stream=cast(TextIO,self.stream)
        stream.write(
            f"\n  Computed solar elevation angle data at point 1,1 -- Year: "
            f"{timestamp.year:4d}  Julian day: {timestamp.timetuple().tm_yday:3d}"
            f"   Hour: {timestamp.hour:2d}\n\n"
            f"   Sine (solar elev. angle)  {sine:7.5f}\n")
        if self.config.noobs==2:
            stream.write(
                " No observation mode\n"
                " upper air data not yet defined...\n"
                " surface   data not yet defined...\n")
        else:
            if hour.step1.debug_surface_record is None:
                raise RuntimeError('missing retained SURF.DAT debug record')
            stream.write(format_surface_input_debug(
                hour.step1.debug_surface_record,
                hour.step1.debug_surface_stations))
            ending=timestamp+timedelta(seconds=self.config.timestep_seconds)
            if self.config.noobs==0:
                from .upper_air import temperature_faces
                for station_number,item in enumerate(
                        hour.step1.debug_upper_data,1):
                    _,soundings,convergence,layer_u,layer_v=cast(
                        "tuple[UpperAirStation, tuple[Sounding, ...], float, "
                        "npt.NDArray[np.float32], npt.NDArray[np.float32]]",
                        item)
                    for sounding in soundings:
                        face_temperature: npt.NDArray[np.float32] | None = (
                            temperature_faces(
                                sounding,self.config.z_faces_m)+np.float32(273.15)
                            if self.config.include_w_temperature else None)
                        stream.write(format_upper_air_input_debug(
                            station_number,sounding,convergence,layer_u,layer_v,
                            face_temperature))
                stream.write(format_full_prepdi_debug(
                    hour.step1,ending,hour.wind.lapse_rate))
            else:
                stream.write(format_no_upper_prepdi_debug(hour.step1,ending))
        stream.write(
            f"\n  Computed surface parameters -- Year: {timestamp.year:4d}   "
            f"Julian day: {timestamp.timetuple().tm_yday:3d}   "
            f"Hour: {timestamp.hour:2d}   Second: "
            f"{timestamp.minute*60+timestamp.second:4d}   At point (1,1):\n"
            f"       Air Density            {surface.density[0,0]:6.3f}    (kg/m**3)\n"
            f"       Short-Wave Radiation   {surface.shortwave_flux[0,0]:7.2f}    (W/m**2)\n")
        switches=self.config.print_surface_fields
        if switches[3] and surface.unaveraged_mixing_height is not None:
            stream.write(format_grid(surface.unaveraged_mixing_height,
                'Mixing height (m) -- Before averaging ',
                begin=timestamp,timestep_seconds=self.config.timestep_seconds))
        if switches[7] and surface.unaveraged_convective_mixing_height is not None:
            stream.write(format_grid(surface.unaveraged_convective_mixing_height,
                'Convective mixing height (m) -- Before averaging',
                begin=timestamp,timestep_seconds=self.config.timestep_seconds))
        stream.write(format_grid(surface.temperature_jump,
            'Temperature jump at top of mixed layer (deg. K)',
            begin=timestamp,timestep_seconds=self.config.timestep_seconds))
        stream.flush()

    def _convective_velocity(
        self,
        hour: HourFields,
    ) -> npt.NDArray[np.float32]:
        """Compute the listing's land/water convective velocity field.

        Args:
            hour: Solver fields for one timestamp.

        Returns:
            Float32 ``(ny, nx)`` convective velocity scale in metres per second.
            Retained water values replace the land formula on water categories.

        Raises:
            ValueError: If input fields cannot be broadcast to the land-use
                grid.
        """
        surface=hour.surface
        value: npt.NDArray[np.float32] = convective_velocity_scale(
            surface.convective_mixing_height,surface.sensible_heat_flux,
            hour.temperature[0],surface.density,self.landuse,
            water_range=self.water_range)
        if surface.water_convective_velocity_scale is not None:
            water: npt.NDArray[np.bool_] = (
                (self.landuse>=self.water_range[0])&
                (self.landuse<=self.water_range[1]))
            value=np.where(water,surface.water_convective_velocity_scale,value)
        return np.asarray(value,dtype=np.float32)

    def close(self) -> None:
        """Flush and close the listing stream; repeated calls are harmless."""
        if self.stream is not None:
            self.stream.close()
            self.stream=None

    def __enter__(self) -> CalmetListingWriter:
        """Return this open writer for context-manager use.

        Returns:
            The :class:`CalmetListingWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the listing on context-manager exit.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.

        Returns:
            ``None``; exceptions are not suppressed.
        """
        self.close()
