from datetime import datetime
import pytest
from pycalmet.datetime_utils import (exact_integer,expand_year,
                                     format_ordinal_code,format_timezone_label,
                                     parse_calendar,parse_ordinal,
                                     parse_ordinal_code,parse_timestamp,
                                     parse_timezone_label)


def test_timestamp_formats_use_datetime_parser():
    expected=datetime(2018,3,18,1)
    assert parse_timestamp('2018-03-18_01:00:00') == expected
    assert parse_timestamp('2018-03-18 01:00:00') == expected
    assert parse_timestamp('2018-03-18T01:00:00') == expected
    assert parse_timestamp('2018031801') == expected


def test_ordinal_codes_round_trip_with_leap_days_and_two_digit_years():
    leap=datetime(2020,2,29,23)
    assert parse_ordinal_code(format_ordinal_code(leap)) == leap
    assert parse_ordinal_code(1807701) == datetime(2018,3,18,1)


def test_two_digit_years_use_datetime_standard_century_window():
    assert expand_year(68) == 2068
    assert expand_year(69) == 1969
    assert parse_ordinal(68,1,0) == datetime(2068,1,1)
    assert parse_calendar(18,3,18) == datetime(2018,3,18)


def test_two_digit_ordinal_code_keeps_fixed_width_leading_zero_year():
    expected=datetime(2000,1,1,1)
    assert parse_ordinal_code('0000101') == expected
    assert parse_ordinal_code(101) == expected


def test_ordinal_code_normalizes_legacy_hour_ending_24():
    assert parse_ordinal_code(1800124) == datetime(2018,1,2)
    assert parse_ordinal(2018,1,24) == datetime(2018,1,2)
    with pytest.raises(ValueError,match='hour'):
        parse_ordinal(2018,1,25)


def test_ordinal_parser_rejects_day_366_in_non_leap_year():
    with pytest.raises(ValueError):
        parse_ordinal(2019,366,0)


@pytest.mark.parametrize('arguments',[(2018,77.5,1),(2018,77,1.5),
                                      (2018,77,1,.5)])
def test_ordinal_fields_reject_fractional_values_instead_of_truncating(arguments):
    with pytest.raises(ValueError):
        parse_ordinal(*arguments)


@pytest.mark.parametrize('arguments',[(2018,77,-1),(2018,77,25),
                                      (2018,77,1,-1),(2018,77,1,3601)])
def test_ordinal_fields_reject_out_of_range_clock_values(arguments):
    with pytest.raises(ValueError):
        parse_ordinal(*arguments)


def test_invalid_timestamp_is_rejected():
    with pytest.raises(ValueError): parse_timestamp('2018-02-30_00:00:00')


@pytest.mark.parametrize('value',('1.5','NaN','Infinity',object()))
def test_exact_integer_rejects_nonintegral_or_nonfinite_values(value):
    with pytest.raises(ValueError,match='must be an integer'):
        exact_integer(value,'sample')


def test_exact_integer_accepts_integral_fortran_exponents():
    assert exact_integer('9D2','sample')==900
    assert exact_integer('900.0','sample')==900


@pytest.mark.parametrize('hours',(-12,-8,0,8,12))
def test_timezone_labels_round_trip_through_datetime(hours):
    label=format_timezone_label(hours)
    parsed,canonical=parse_timezone_label(label)
    assert (parsed,canonical)==(hours,label)


@pytest.mark.parametrize('label',(
    'UTC+010000.5','UTC+01:00','utc+0100','UTC+010','UTC +0100',
))
def test_timezone_parser_rejects_non_calmet_label_formats(label):
    with pytest.raises(ValueError,match='invalid timezone'):
        parse_timezone_label(label)


def test_timezone_parser_rejects_fractional_hour_offsets():
    with pytest.raises(ValueError,match='whole-hour'):
        parse_timezone_label('UTC+0130')


@pytest.mark.parametrize('value',(12.5,13,-13))
def test_timezone_formatter_rejects_fractional_or_out_of_range_hours(value):
    with pytest.raises(ValueError):
        format_timezone_label(value)
