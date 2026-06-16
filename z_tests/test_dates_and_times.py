import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Python Basics - Dates and Times"))
from dates_and_times import (
    add_business_days,
    is_trading_day,
    next_trading_day,
    parse_timestamp,
    trading_days_between,
    year_fraction,
)


def test_parse_iso_and_us_formats():
    assert parse_timestamp("2024-03-15").date() == date(2024, 3, 15)
    assert parse_timestamp("03/15/2024").date() == date(2024, 3, 15)
    assert parse_timestamp("20240315").date() == date(2024, 3, 15)


def test_parse_rejects_garbage():
    try:
        parse_timestamp("not a date")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_is_trading_day_skips_weekend():
    assert is_trading_day("2024-03-15")  # Friday
    assert not is_trading_day("2024-03-16")  # Saturday
    assert not is_trading_day("2024-03-17")  # Sunday


def test_is_trading_day_skips_holiday():
    assert not is_trading_day("2024-01-01", holidays=["2024-01-01"])


def test_trading_days_between_excludes_weekends():
    # Mon 2024-03-11 -> Fri 2024-03-15 is 4 trading days (Tue..Fri).
    assert trading_days_between("2024-03-11", "2024-03-15") == 4


def test_trading_days_between_respects_holidays():
    base = trading_days_between("2024-01-01", "2024-03-15")
    with_hol = trading_days_between("2024-01-01", "2024-03-15", holidays=["2024-01-15"])
    assert with_hol == base - 1


def test_trading_days_between_non_positive_interval():
    assert trading_days_between("2024-03-15", "2024-03-15") == 0
    assert trading_days_between("2024-03-15", "2024-03-10") == 0


def test_add_business_days_t_plus_two():
    # Friday + 2 business days -> Tuesday.
    assert add_business_days("2024-03-15", 2) == date(2024, 3, 19)


def test_add_business_days_backward():
    # Monday - 1 business day -> Friday.
    assert add_business_days("2024-03-18", -1) == date(2024, 3, 15)


def test_add_business_days_zero_is_identity():
    assert add_business_days("2024-03-16", 0) == date(2024, 3, 16)


def test_next_trading_day_jumps_weekend():
    assert next_trading_day("2024-03-15") == date(2024, 3, 18)


def test_year_fraction_conventions():
    assert abs(year_fraction("2024-01-01", "2024-07-01", "ACT/365") - 182 / 365) < 1e-12
    assert abs(year_fraction("2024-01-01", "2024-07-01", "ACT/360") - 182 / 360) < 1e-12
    assert abs(year_fraction("2024-01-01", "2024-07-01", "30/360") - 0.5) < 1e-12


def test_year_fraction_unknown_basis():
    try:
        year_fraction("2024-01-01", "2024-07-01", "ACT/ACT")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
