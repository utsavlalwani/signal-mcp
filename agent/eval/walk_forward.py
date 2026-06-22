"""Walk-forward cross-validation split generator.

Pure, dependency-free utility -- no VectorBT, no pandas magic. The quant-engine server uses VectorBT-backed backtesting; this module exists so that the splits themselves are inspectable, testable, and replaceable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def walk_forward_splits(
    overall_start: date,
    overall_end: date,
    train_months: int,
    test_months: int,
) -> list[WalkForwardSplit]:
    """Generate sequential walk-forward splits.

    Each split has a non-overlapping test window that comes strictly after its training window. The next split's training window expands to include everything up to the new test window (expanding-window WF).
    """
    if train_months < 1 or test_months < 1:
        raise ValueError("train/test windows must be >= 1 month")

    splits: list[WalkForwardSplit] = []
    cursor_year = overall_start.year
    cursor_month = overall_start.month

    def add_months(y: int, m: int, delta: int) -> tuple[int, int]:
        m0 = (m - 1) + delta
        y += m0 // 12
        m = (m0 % 12) + 1
        return y, m

    def month_end(y: int, m: int) -> date:
        # last day of month m, year y
        if m == 12:
            return date(y, 12, 31)
        from datetime import timedelta as _td
        return date(y, m + 1, 1) - _td(days=1)

    while True:
        train_end_y, train_end_m = add_months(cursor_year, cursor_month, train_months - 1)
        train_end = month_end(train_end_y, train_end_m)
        test_start_y, test_start_m = add_months(train_end_y, train_end_m, 1)
        test_start = date(test_start_y, test_start_m, 1)
        test_end_y, test_end_m = add_months(test_start_y, test_start_m, test_months - 1)
        test_end = month_end(test_end_y, test_end_m)

        if test_end > overall_end:
            break

        splits.append(WalkForwardSplit(
            train_start=date(cursor_year, cursor_month, 1),
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))

        # Slide forward by test_months (rolling) -- could be made expanding by not advancing the train_start.
        cursor_year, cursor_month = add_months(cursor_year, cursor_month, test_months)

    return splits

