"""Shared parsing and formatting for CALMET date/time fields."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import re


def exact_integer(value: object, label: str = "value") -> int:
    """Return an exactly integral value without silently truncating it.

    Decimal parsing accepts the exponent notation used by Fortran controls
    while preserving whether the supplied value has a fractional part.
    """
    if isinstance(value, bool):
        return int(value)
    try:
        number = Decimal(str(value).strip().replace("D", "E").replace("d", "e"))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError(f"{label} must be an integer: {value!r}") from error
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{label} must be an integer: {value!r}")
    return int(number)


def expand_year(value: int | str) -> int:
    """Expand a CALMET year using :mod:`datetime`'s standard rules.

    Two-digit years intentionally follow ``strptime(..., "%y")`` instead of
    maintaining a separate project-specific century pivot.
    """
    year = exact_integer(value, "year")
    pattern = "%y" if 0 <= year <= 99 else "%Y"
    width = 2 if pattern == "%y" else 4
    return datetime.strptime(f"{year:0{width}d}", pattern).year


def parse_calendar(year: int | str, month: int | str, day: int | str,
                   hour: int | str = 0, seconds: int | str = 0) -> datetime:
    """Parse separate calendar fields and normalize legacy clock boundaries.

    CALMET files sometimes represent the following boundary as hour 24 or as
    3600 seconds.  :class:`datetime` validates the calendar date and
    :class:`timedelta` performs that boundary normalization.
    """
    calendar_year = expand_year(year)
    calendar_month = exact_integer(month, "month")
    calendar_day = exact_integer(day, "day")
    clock_hour = exact_integer(hour, "hour")
    clock_seconds = exact_integer(seconds, "seconds")
    if not 0 <= clock_hour <= 24:
        raise ValueError(f"invalid hour {hour!r}")
    if not 0 <= clock_seconds <= 3600:
        raise ValueError(f"invalid seconds {seconds!r}")
    text = f"{calendar_year:04d}-{calendar_month:02d}-{calendar_day:02d}"
    try:
        base = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(
            f"invalid calendar date {year!r}/{month!r}/{day!r}"
        ) from error
    return base + timedelta(hours=clock_hour, seconds=clock_seconds)


def parse_naive_datetime(value: datetime | str,
                         label: str = "timestamp") -> datetime:
    """Return a second-resolution, timezone-naive :class:`datetime`."""
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        text = value.rstrip("\x00 ")
        for pattern in (
            "%Y-%m-%d_%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y%m%d%H",
        ):
            try:
                result = datetime.strptime(text, pattern)
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"invalid {label} {value!r}")
    else:
        raise TypeError(f"{label} must be a datetime or timestamp string")
    if result.tzinfo is not None or result.utcoffset() is not None:
        raise ValueError(f"{label} must not include a timezone")
    if result.microsecond:
        raise ValueError(f"{label} must have whole-second precision")
    return result


def parse_date(value: date | datetime | str, label: str = "date") -> date:
    """Parse an unambiguous calendar date without discarding a time part."""
    if isinstance(value, datetime):
        if (value.tzinfo is not None or value.utcoffset() is not None or
                value.time() != datetime.min.time()):
            raise ValueError(f"{label} datetime must be naive midnight")
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a date, datetime, or date string")
    text = value.strip()
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            result = datetime.strptime(text, pattern).date()
        except ValueError:
            continue
        if result.strftime(pattern) == text:
            return result
    raise ValueError(f"invalid {label} {value!r}")


def parse_timezone_label(value: str) -> tuple[int, str]:
    """Return CALMET's positive-west offset and a canonical ``UTC+HHMM`` label."""
    if not isinstance(value, str):
        raise TypeError("timezone must be a UTC offset string")
    text = value.strip()
    if re.fullmatch(r"UTC[+-]\d{4}", text) is None:
        raise ValueError(f"invalid timezone {value!r}; expected UTC+HHMM or UTC-HHMM")
    try:
        parsed = datetime.strptime(text, "UTC%z")
    except ValueError as error:
        raise ValueError(f"invalid timezone {value!r}") from error
    offset = parsed.utcoffset()
    if offset is None:
        raise ValueError(f"invalid timezone {value!r}")
    total_seconds = offset.total_seconds()
    if total_seconds % 3600:
        raise ValueError("CALMET timezone must use a whole-hour UTC offset")
    seconds = exact_integer(total_seconds, "timezone offset seconds")
    positive_west = -(seconds // 3600)
    return positive_west,format_timezone_label(positive_west)


def format_timezone_label(hours_west: object) -> str:
    """Format CALMET positive-west whole hours as a canonical UTC label."""
    hours=exact_integer(hours_west,"CALMET timezone hours")
    if not -12<=hours<=12:
        raise ValueError("CALMET timezone hours must be between -12 and 12")
    zone=timezone(timedelta(hours=-hours))
    return datetime(2000,1,1,tzinfo=zone).strftime("UTC%z")


def parse_ordinal(year: int | str, day: int | str, hour: int | str,
                  seconds: int | str = 0) -> datetime:
    ordinal_day = exact_integer(day, "ordinal day")
    clock_hour = exact_integer(hour, "hour")
    clock_seconds = exact_integer(seconds, "seconds")
    if not 0 <= clock_hour <= 24:
        raise ValueError(f"invalid hour {hour!r}")
    if not 0 <= clock_seconds <= 3600:
        raise ValueError(f"invalid seconds {seconds!r}")
    date_text = f"{expand_year(year):04d}{ordinal_day:03d}"
    base = datetime.strptime(date_text, "%Y%j")
    if base.strftime("%Y%j") != date_text:
        raise ValueError(f"invalid ordinal day {day!r}")
    return base + timedelta(hours=clock_hour, seconds=clock_seconds)


def parse_ordinal_code(code: int | str, seconds: int | str = 0) -> datetime:
    text = str(code).strip()
    if not text.isdecimal():
        raise ValueError(f"invalid CALMET coded time {code!r}")
    # Integer fields cannot retain the leading zeros in YYJJJHH for years
    # 2000--2009.  Restore that fixed-width representation before parsing.
    if not isinstance(code, str) and len(text) < 7:
        text = text.zfill(7)
    patterns = {7: "%y%j%H", 9: "%Y%j%H"}
    hour_ending_24 = text.endswith("24")
    parse_text = text[:-2] + "00" if hour_ending_24 else text
    try:
        pattern = patterns[len(text)]
        base = datetime.strptime(parse_text, pattern)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid CALMET coded time {code!r}") from error
    if base.strftime(pattern) != parse_text:
        raise ValueError(f"invalid CALMET coded time {code!r}")
    if hour_ending_24:
        base += timedelta(days=1)
    clock_seconds = exact_integer(seconds, "seconds")
    if not 0 <= clock_seconds <= 3600:
        raise ValueError(f"invalid seconds {seconds!r}")
    return base + timedelta(seconds=clock_seconds)


def format_ordinal_code(value: datetime) -> int:
    return int(value.strftime("%Y%j%H"))


def parse_timestamp(value: str) -> datetime:
    return parse_naive_datetime(value)
