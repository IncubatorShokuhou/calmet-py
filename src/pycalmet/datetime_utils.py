"""Shared parsing and formatting for CALMET date/time fields."""
from __future__ import annotations

from datetime import datetime, timedelta


def expand_year(value: int | str) -> int:
    year = int(value)
    if 0 <= year < 50:
        return 2000 + year
    if 50 <= year < 100:
        return 1900 + year
    return year


def parse_ordinal(year: int | str, day: int | str, hour: int | str,
                  seconds: int | str = 0) -> datetime:
    base = datetime.strptime(f"{expand_year(year):04d} {int(day):03d}", "%Y %j")
    return base + timedelta(hours=int(hour), seconds=int(seconds))


def parse_ordinal_code(code: int | str, seconds: int | str = 0) -> datetime:
    text = str(int(code))
    if len(text) not in (7, 9):
        raise ValueError(f"invalid CALMET coded time {code!r}")
    year_width = len(text) - 5
    return parse_ordinal(text[:year_width], text[year_width:-2], text[-2:], seconds)


def format_ordinal_code(value: datetime) -> int:
    return int(value.strftime("%Y%j%H"))


def parse_timestamp(value: str) -> datetime:
    text = value.rstrip("\x00 ")
    for pattern in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            pass
    raise ValueError(f"invalid timestamp {value!r}")
