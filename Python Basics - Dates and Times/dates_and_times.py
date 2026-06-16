"""
Dates and Times for Finance
---------------------------
Markets run on a calendar, not a clock. Interest accrues over *days*, options
expire on a *date*, and a backtest that miscounts trading days will quietly
misreport every annualised number it produces. Python's ``datetime`` module is
the right tool, but finance adds two wrinkles the standard library does not know
about: weekends/holidays are not trading days, and "a year" can mean 360, 365,
or actual days depending on the instrument.

This module is a hands-on tour of the pieces you actually need:

* parsing the timestamp formats data vendors hand you,
* counting and stepping over **trading days** (skipping weekends and holidays),
* and the **day-count conventions** (ACT/365, ACT/360, 30/360) that turn a pair
  of dates into the "year fraction" every rate and discount formula expects.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Optional

# Common formats vendors emit, tried in order.
_KNOWN_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y",
    "%d-%b-%Y",
    "%Y%m%d",
)


def parse_timestamp(value: str) -> datetime:
    """Parse a timestamp string, trying several common vendor formats.

    Args:
        value: A date/datetime string such as ``"2024-03-15"`` or ``"03/15/2024"``.

    Returns:
        A ``datetime``. Date-only inputs come back at midnight.

    Raises:
        ValueError: If none of the known formats match.
    """
    text = value.strip()
    for fmt in _KNOWN_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised timestamp format: {value!r}")


def _as_date(value) -> date:
    """Coerce a ``date``/``datetime``/string into a plain ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return parse_timestamp(str(value)).date()


def is_trading_day(day, holidays: Optional[Iterable] = None) -> bool:
    """True when *day* is a weekday and not in the holiday set.

    Args:
        day: A date, datetime, or parseable string.
        holidays: Optional iterable of market holidays.
    """
    d = _as_date(day)
    holiday_set = {_as_date(h) for h in holidays} if holidays else set()
    return d.weekday() < 5 and d not in holiday_set


def next_trading_day(day, holidays: Optional[Iterable] = None) -> date:
    """Return the first trading day strictly after *day*."""
    d = _as_date(day) + timedelta(days=1)
    while not is_trading_day(d, holidays):
        d += timedelta(days=1)
    return d


def trading_days_between(start, end, holidays: Optional[Iterable] = None) -> int:
    """Count trading days in the half-open interval ``(start, end]``.

    Weekends and any supplied holidays are skipped. The start date itself is
    excluded and the end date included, so consecutive calls chain cleanly.

    Args:
        start: Interval start (exclusive).
        end: Interval end (inclusive).
        holidays: Optional iterable of market holidays.
    """
    s, e = _as_date(start), _as_date(end)
    if e <= s:
        return 0
    holiday_set = {_as_date(h) for h in holidays} if holidays else set()
    count = 0
    d = s + timedelta(days=1)
    while d <= e:
        if d.weekday() < 5 and d not in holiday_set:
            count += 1
        d += timedelta(days=1)
    return count


def add_business_days(start, n: int, holidays: Optional[Iterable] = None) -> date:
    """Step *n* trading days forward (n > 0) or backward (n < 0) from *start*."""
    d = _as_date(start)
    if n == 0:
        return d
    step = 1 if n > 0 else -1
    remaining = abs(n)
    while remaining > 0:
        d += timedelta(days=step)
        if is_trading_day(d, holidays):
            remaining -= 1
    return d


def year_fraction(start, end, basis: str = "ACT/365") -> float:
    """Convert two dates into a year fraction under a day-count convention.

    The same number of calendar days means different things to different
    instruments. Money-market rates usually quote ACT/360; many bonds use
    30/360; most equity-vol work uses ACT/365.

    Args:
        start: Period start date.
        end: Period end date.
        basis: One of ``"ACT/365"``, ``"ACT/360"`` or ``"30/360"``.

    Returns:
        The fraction of a year between the two dates.
    """
    s, e = _as_date(start), _as_date(end)
    convention = basis.upper().replace(" ", "")

    if convention in ("ACT/365", "ACT/365F"):
        return (e - s).days / 365.0
    if convention == "ACT/360":
        return (e - s).days / 360.0
    if convention == "30/360":
        # US (NASD) 30/360: clamp day-of-month to 30 with the standard rule.
        d1 = min(s.day, 30)
        d2 = 30 if (d1 == 30 and e.day == 31) else e.day
        days = 360 * (e.year - s.year) + 30 * (e.month - s.month) + (d2 - d1)
        return days / 360.0
    raise ValueError(f"Unknown day-count basis: {basis!r}")


if __name__ == "__main__":
    print("Dates and Times for Finance")
    print("=" * 40)

    ts = parse_timestamp("2024-03-15")
    print(f"Parsed '2024-03-15'      -> {ts.date()} ({ts.strftime('%A')})")
    print(f"Parsed '03/15/2024'      -> {parse_timestamp('03/15/2024').date()}")

    # US market holidays in early 2024 for the example.
    holidays = ["2024-01-01", "2024-01-15", "2024-02-19"]
    start, end = "2024-01-01", "2024-03-15"
    td = trading_days_between(start, end, holidays)
    print(f"\nTrading days {start} -> {end}: {td}")
    print(f"  (raw calendar days: {(_as_date(end) - _as_date(start)).days})")

    settle = add_business_days("2024-03-15", 2)  # T+2 settlement
    print(f"\nT+2 settlement from 2024-03-15 -> {settle} ({settle.strftime('%A')})")

    print("\nYear fraction, 2024-01-01 to 2024-07-01:")
    for basis in ("ACT/365", "ACT/360", "30/360"):
        yf = year_fraction("2024-01-01", "2024-07-01", basis)
        print(f"  {basis:<8} -> {yf:.5f}")
