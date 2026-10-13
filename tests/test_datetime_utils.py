from datetime import datetime
import pytest
from pycalmet.datetime_utils import format_ordinal_code,parse_ordinal_code,parse_timestamp


def test_timestamp_formats_use_datetime_parser():
    expected=datetime(2018,3,18,1)
    assert parse_timestamp('2018-03-18_01:00:00') == expected
    assert parse_timestamp('2018-03-18 01:00:00') == expected
    assert parse_timestamp('2018031801') == expected


def test_ordinal_codes_round_trip_with_leap_days_and_two_digit_years():
    leap=datetime(2020,2,29,23)
    assert parse_ordinal_code(format_ordinal_code(leap)) == leap
    assert parse_ordinal_code(1807701) == datetime(2018,3,18,1)


def test_invalid_timestamp_is_rejected():
    with pytest.raises(ValueError): parse_timestamp('2018-02-30_00:00:00')
