"""Legacy wind-module disk debug output."""
from __future__ import annotations

from collections.abc import Iterator,Sequence
from datetime import datetime
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import TextIO,TypeAlias,cast

import numpy as np
import numpy.typing as npt

from .calmet_solver import HourFields


FilePath: TypeAlias=str | PathLike[str]
NumericArray: TypeAlias=npt.NDArray[np.generic]
RealScalar: TypeAlias=int | float | np.integer | np.floating


def _fortran_exponential(value: RealScalar,digits: int,width: int) -> str:
    """Format a scalar like a fixed-width Fortran ``E`` field.

    Args:
        value: Numeric scalar to format.
        digits: Number of digits after the mantissa decimal point.
        width: Minimum output field width.

    Returns:
        Right-justified exponential string with a two-digit signed exponent.

    Raises:
        TypeError: If ``value`` is not convertible to ``float``.
        ValueError: If formatting controls or a NaN value cannot be converted
            by the exponent calculation.
        OverflowError: If an infinite value cannot be converted to an integer
            exponent.
    """
    value=float(value)
    if value==0.:
        exponent=0; mantissa=0.
    else:
        exponent=int(np.floor(np.log10(abs(value))))+1
        mantissa=value/(10.**exponent)
        if abs(round(mantissa,digits))>=1.:
            mantissa/=10.; exponent+=1
    return f'{mantissa:.{digits}f}E{exponent:+03d}'.rjust(width)


class FinalWindDebugWriter:
    """Reproduce ``OUTFIL`` records written to ``TEST.OUT`` for ``IPR8``.

    Attributes:
        stream: Open ASCII output stream, or ``None`` after closing.
        nx: Number of horizontal columns.
        ny: Number of horizontal rows.
        nz: Number of wind layers.
        dx_km: Grid-cell width in kilometres.
        dy_km: Grid-cell height in kilometres.
        first_hour: First whole-hour timestamp seen, or ``None`` before output.
    """

    stream: TextIO | None
    nx: int
    ny: int
    nz: int
    dx_km: float
    dy_km: float
    first_hour: datetime | None

    def __init__(
        self,
        path: FilePath,
        nx: int,
        ny: int,
        nz: int,
        dx_km: float,
        dy_km: float,
    ) -> None:
        """Open TEST.OUT and store its fixed grid metadata.

        Args:
            path: Destination ASCII path; missing parent directories are
                created.
            nx: Number of grid columns.
            ny: Number of grid rows.
            nz: Number of horizontal-wind layers.
            dx_km: Grid-cell width in kilometres.
            dy_km: Grid-cell height in kilometres.

        Raises:
            OSError: If the destination directory or file cannot be created.

        Notes:
            Construction truncates an existing destination file.
        """
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        self.stream=target.open('w',encoding='ascii')
        self.nx=int(nx); self.ny=int(ny); self.nz=int(nz)
        self.dx_km=float(dx_km); self.dy_km=float(dy_km)
        self.first_hour=None

    @staticmethod
    def _rows(
        values: npt.ArrayLike,
        format_spec: str,
        width: int,
    ) -> Iterator[str]:
        """Yield fixed-size chunks from each row of an array.

        Args:
            values: Array-like object whose first axis is iterated as rows.
            format_spec: Python scalar format specification for each value.
            width: Maximum number of values emitted per text line.

        Yields:
            Concatenated formatted strings without newline terminators.

        Raises:
            ValueError: If ``width`` is zero or a value cannot be formatted
                with ``format_spec``.
        """
        array: NumericArray=np.asarray(values)
        for row in array:
            for first in range(0,row.size,width):
                yield ''.join(format(value,format_spec) for value in row[first:first+width])

    def append(self,hour: HourFields) -> None:
        """Append final U/V/W fields from one solver hour.

        Args:
            hour: Solver output providing ``timestamp`` and wind arrays.

        Raises:
            ValueError: If wind arrays do not match the configured grid.
            OSError: If writing or flushing TEST.OUT fails.

        Notes:
            This delegates to :meth:`append_fields` and writes no other solver
            fields.
        """
        self.append_fields(hour.timestamp,hour.wind.u,hour.wind.v,hour.wind.w)

    def append_fields(
        self,
        timestamp: datetime,
        u: npt.ArrayLike,
        v: npt.ArrayLike,
        w: npt.ArrayLike,
    ) -> None:
        """Append one timestamp and its final wind arrays to TEST.OUT.

        Args:
            timestamp: Model timestamp used to derive one-based simulation hour.
            u: Horizontal U wind with shape ``(nz, ny, nx)`` in metres per
                second.
            v: Horizontal V wind with shape ``(nz, ny, nx)`` in metres per
                second.
            w: Face vertical velocity with shape ``(nz + 1, ny, nx)`` in metres
                per second.

        Raises:
            ValueError: If U, V, or W has the wrong shape.
            OSError: If writing or flushing the output stream fails.

        Notes:
            U/V use fixed-point fields and W uses exponential fields. Data are
            flattened level-major, row-major without numerical conversion.
        """
        current=timestamp.replace(minute=0,second=0,microsecond=0)
        if self.first_hour is None: self.first_hour=current
        nh=int((current-self.first_hour).total_seconds()//3600)+1
        u_values: NumericArray=np.asarray(u)
        v_values: NumericArray=np.asarray(v)
        w_values: NumericArray=np.asarray(w)
        if (u_values.shape!=(self.nz,self.ny,self.nx)
                or v_values.shape!=u_values.shape):
            raise ValueError('TEST.OUT U/V fields do not match the configured grid')
        if w_values.shape!=(self.nz+1,self.ny,self.nx):
            raise ValueError('TEST.OUT W fields do not match the configured grid')
        stream=cast(TextIO,self.stream)
        stream.write(f'{float((nh-1)*100):5.0f}.\n')
        stream.write(f'{self.nx:4d}{self.ny:4d}{self.nz:4d}\n')
        stream.write(f'{self.dx_km:3.0f}.{self.dy_km:3.0f}.\n')
        for line in self._rows(u_values.reshape(-1,self.nx),'7.2f',59): stream.write(line+'\n')
        for line in self._rows(v_values.reshape(-1,self.nx),'7.2f',59): stream.write(line+'\n')
        for line in self._rows(w_values.reshape(-1,self.nx),'8.1E',59): stream.write(line+'\n')
        stream.flush()

    def close(self) -> None:
        """Flush and close TEST.OUT; repeated calls are harmless."""
        if self.stream is not None:
            self.stream.close(); self.stream=None

    def __enter__(self) -> FinalWindDebugWriter:
        """Return this open writer for context-manager use.

        Returns:
            The :class:`FinalWindDebugWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close TEST.OUT on context-manager exit.

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


def format_wind_number_map(values: npt.ArrayLike) -> str:
    """Reproduce the legacy ``WNDLPT`` two-digit scaled number map.

    Args:
        values: Nonempty two-dimensional numeric field. Only the first 36 rows
            and 72 columns are printed.

    Returns:
        Newline-terminated number-map text. Nonzero fields are float32-scaled
        so rounded integers normally occupy two digits; all-zero fields return
        the legacy suppression message.

    Raises:
        ValueError: If ``values`` is not two-dimensional or is empty.
    """
    data: npt.NDArray[np.float32]=np.asarray(values,dtype=np.float32)
    if data.ndim!=2: raise ValueError('WNDLPT fields must be two-dimensional')
    maximum=float(np.max(np.abs(data)))
    if maximum==0.:
        return "\n         FIELD CONTAINS ALL ZEROS - PRINTING SUPPRESSED\n"
    factor: np.float32
    if maximum<10.:
        factor=next(np.float32(10.)**power for power in range(1,21)
                    if int((np.float32(10.)**power)*maximum)>=10)
    elif maximum>=100.:
        factor=next(np.float32(10.)**(-power) for power in range(1,21)
                    if int((np.float32(10.)**(-power))*maximum)<100)
    else:
        factor=np.float32(1.)
    scaled: npt.NDArray[np.float32]=np.asarray(data*factor,dtype=np.float32)
    mapped: NumericArray=np.where(
        scaled>0.,np.trunc(scaled+np.float32(.5)),
        np.where(scaled<0.,np.trunc(scaled-np.float32(.5)),0.)).astype(int)
    lines: list[str]=[]; ny=min(data.shape[0],36)
    for first in range(0,min(data.shape[1],72),36):
        last=min(first+36,data.shape[1],72)
        lines.append('')
        lines.append('       '+''.join(f'{index:3d}' for index in range(first+1,last+1)))
        lines.append('      I'+'--+'*(last-first))
        for row in range(ny-1,-1,-1):
            lines.append(f'   {row+1:2d} I'+''.join(
                f'{value:3d}' for value in mapped[row,first:last]))
    scale_text=_fortran_exponential(factor,1,9)
    lines.extend(('',f'         ARRAY HAS BEEN SCALED BY{scale_text}  FOR PRINTING'))
    return '\n'.join(lines)+'\n'


class WindPrintWriter:
    """Write requested wind-module diagnostic maps to ``TEST.PRT``.

    Attributes:
        stream: Open ASCII output stream, or ``None`` after closing.
        fields: Tuple of wind-module print switches.
        levels: Optional maximum number of levels for selected products.
        first_hour: First whole-hour timestamp seen, or ``None`` before output.
    """

    stream: TextIO | None
    fields: tuple[bool | int,...]
    levels: int | None
    first_hour: datetime | None

    def __init__(
        self,
        path: FilePath,
        fields: Sequence[bool | int],
        levels: int | None=None,
    ) -> None:
        """Open TEST.PRT and retain its field switches.

        Args:
            path: Destination ASCII path; missing parent directories are
                created.
            fields: Sequence of at least nine truth-like print switches using
                CALMET's wind-debug switch order.
            levels: Optional maximum number of levels printed for speed,
                direction, and divergence products.

        Raises:
            OSError: If the destination directory or file cannot be created.

        Notes:
            Construction truncates an existing destination file. Switch count
            and level validity are consumed when :meth:`append` is called.
        """
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        self.stream=target.open('w',encoding='ascii')
        self.fields=tuple(fields); self.levels=levels; self.first_hour=None

    def _windpr(
        self,
        u: NumericArray,
        v: NumericArray,
        w: NumericArray,
        include_w: bool,
    ) -> None:
        """Write component number maps for every horizontal wind level.

        Args:
            u: Array-like U wind with shape ``(level, ny, nx)`` in metres per
                second.
            v: Array-like V wind with the same shape and units as ``u``.
            w: Face W velocity with shape ``(level + 1, ny, nx)`` when
                ``include_w`` is true; ignored otherwise.
            include_w: Whether to print interior W faces before U/V maps.

        Raises:
            IndexError: If requested component levels are missing.
            ValueError: If a selected map is not a nonempty 2-D field.
            OSError: If writing TEST.PRT fails.

        Notes:
            Each component is passed through :func:`format_wind_number_map`.
        """
        stream=cast(TextIO,self.stream)
        for level in range(u.shape[0]):
            if include_w and level:
                stream.write(
                    f'\n     Z-COMPONENT OF WIND (W) BETWEEN LEVELS{level:4d} & {level+1:2d}'
                    '     (M/SEC)\n     '+'-'*59+'\n')
                stream.write(format_wind_number_map(w[level]))
            stream.write(
                f'\n     X-COMPONENT OF WIND (U) AT LEVEL = {level+1:4d}     (M/SEC)\n'
                '     '+'-'*49+'\n')
            stream.write(format_wind_number_map(u[level]))
            stream.write(
                f'\n     Y-COMPONENT OF WIND (V) AT LEVEL = {level+1:4d}     (M/SEC)\n'
                '     '+'-'*49+'\n')
            stream.write(format_wind_number_map(v[level]))

    def _speed_direction(
        self,
        u: npt.ArrayLike,
        v: npt.ArrayLike,
    ) -> None:
        """Print final horizontal-wind speed and meteorological direction.

        Args:
            u: Array-like U wind with shape ``(level, ny, nx)`` in metres per
                second.
            v: Array-like V wind with the same shape and units as ``u``.

        Raises:
            IndexError: If V has fewer requested levels than U.
            ValueError: If a selected level is not a nonempty 2-D field.
            OSError: If writing TEST.PRT fails.

        Notes:
            Calm directions are printed as zero; non-calm northerly flow is
            printed as 360 degrees.
        """
        u_values: NumericArray=np.asarray(u)
        v_values: NumericArray=np.asarray(v)
        levels=(u_values.shape[0] if self.levels is None
                else min(int(self.levels),u_values.shape[0]))
        stream=cast(TextIO,self.stream)
        for level in range(levels):
            speed=cast(
                npt.NDArray[np.floating],
                np.hypot(u_values[level],v_values[level]))
            direction: NumericArray=np.mod(
                270.-np.rad2deg(
                    np.arctan2(v_values[level],u_values[level])),360.)
            direction=np.where(
                speed<1.e-5,0.,np.where(direction==0.,360.,direction))
            stream.write(
                f'\n\n     WIND SPEED (M/S) AT LEVEL = {level+1:4d}\n'
                '     '+'-'*26+'\n')
            stream.write(format_wind_number_map(speed))
            stream.write(
                f'\n\n     WIND DIRECTION (DEGREES) AT LEVEL = {level+1:4d}\n'
                '     '+'-'*34+'\n')
            stream.write(format_wind_number_map(direction))

    def append(self,hour: HourFields) -> None:
        """Append all enabled wind-stage diagnostics for one solver hour.

        Args:
            hour: Solver output containing final winds, divergence, timestamp,
                and any retained intermediate wind stages requested by
                ``fields``.

        Raises:
            IndexError: If ``fields`` has fewer than nine switches or selected
                arrays lack required levels.
            RuntimeError: If an enabled intermediate stage was not retained.
            TypeError: If a selected product requires ``levels`` but it is
                ``None`` or not integer-compatible.
            ValueError: If a selected map is not a nonempty 2-D field.
            OSError: If writing or flushing TEST.PRT fails.

        Notes:
            Enabled sections are written in legacy TEST.PRT order: intermediate
            stages, final wind, final divergence, then speed and direction.
        """
        current=hour.timestamp.replace(minute=0,second=0,microsecond=0)
        if self.first_hour is None: self.first_hour=current
        nh=int((current-self.first_hour).total_seconds()//3600)+1
        stream=cast(TextIO,self.stream)
        stages: tuple[tuple[int,str,int,str,bool],...]=(
                (5,'KINEMATIC EFFECTS',17,'kinematic_stage',True),
                (6,'FROUDE NUMBER EFFECTS',21,'froude_stage',False),
                (7,'SLOPE EFFECTS',13,'slope_stage',False))
        for index,title,dashes,attribute,include_w in stages:
            if self.fields[index]:
                stage=getattr(hour.wind,attribute)
                if stage is None: raise RuntimeError(f'missing retained wind stage {attribute}')
                stream.write(f'\n          {title}\n          '+dashes*'-'+'\n')
                stage_values=cast(
                    tuple[NumericArray,NumericArray,NumericArray],stage)
                self._windpr(*stage_values,include_w=include_w)
        if self.fields[0]:
            stage=hour.wind.interpolated_stage
            if stage is None: raise RuntimeError('missing retained interpolated wind stage')
            stream.write('           INTERPOLATED WIND FIELD\n           '+23*'-'+'\n')
            self._windpr(*stage,np.empty(0),include_w=False)
        if self.fields[1]:
            stage=hour.wind.terrain_adjusted_stage
            if stage is None: raise RuntimeError('missing retained terrain-adjusted wind stage')
            stream.write('\n          TERRAIN ADJUSTED SURFACE WINDS\n          '+30*'-'+'\n')
            self._windpr(*stage,np.empty(0),include_w=False)
        if self.fields[2]:
            stage=hour.wind.initial_stage
            if stage is None: raise RuntimeError('missing retained initial wind stage')
            u,v,w,divergence=stage
            stream.write('\n          SMOOTHED WIND AND INITIAL DIVERGENCE FIELDS\n'
                              '          '+43*'-'+'\n')
            self._windpr(u,v,w,include_w=True)
            maximum=float(np.max(np.abs(divergence)))
            stream.write('\n\n                    MAXIMUM DIVERGENCE = '+
                              _fortran_exponential(maximum,4,12)+'\n')
            for level in range(min(
                    int(cast(int,self.levels)),divergence.shape[0])):
                stream.write(
                    f'\n\n     DIVERGENCE AT LEVEL = {level+1:4d}\n'
                    '     '+'-'*24+'\n')
                stream.write(format_wind_number_map(divergence[level]))
        if self.fields[8]:
            time=float((nh-1)*100)
            stream.write(
                f'          FINAL WIND FIELD AT TIME = {time:6.0f}.'
                f'     (SIMULATION HOUR NO.{nh:3d} )\n          '+63*'-'+'\n')
            self._windpr(hour.wind.u,hour.wind.v,hour.wind.w,include_w=True)
        levels=hour.wind.u.shape[0] if self.levels is None else min(
            int(self.levels),hour.wind.u.shape[0])
        if self.fields[4]:
            stream.write('\n          FINAL DIVERGENCE FIELD AT TIME =\n')
            for level in range(levels):
                stream.write(
                    f'\n\n     DIVERGENCE AT LEVEL = {level+1:4d}\n'
                    '     '+'-'*24+'\n')
                stream.write(format_wind_number_map(hour.wind.divergence[level]))
        if self.fields[3]:
            time=float((nh-1)*100)
            stream.write(
                f'\n          FINAL WIND SPEED AND DIRECTION AT TIME = {time:7.0f}.'
                f'     (SIMULATION HOUR NO.{nh:3d} )\n          '+72*'-'+'\n')
            self._speed_direction(hour.wind.u,hour.wind.v)
        stream.flush()

    def close(self) -> None:
        """Flush and close TEST.PRT; repeated calls are harmless."""
        if self.stream is not None: self.stream.close(); self.stream=None

    def __enter__(self) -> WindPrintWriter:
        """Return this open writer for context-manager use.

        Returns:
            The :class:`WindPrintWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close TEST.PRT on context-manager exit.

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
