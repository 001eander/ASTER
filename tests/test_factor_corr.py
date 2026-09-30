"""``quant.factor_lib.correlation`` 的单元测试：行为相关性口径。

全部用确定性合成值长表，不触网、不读真实数据。覆盖：完全正 / 负相关、
逐日截面样本数下限、绝对值取最大、空库。
"""
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable

import polars as pl
import pytest

from quant.factor_lib.correlation import (
    cross_section_corr,
    max_library_corr,
)

START: dt.date = dt.date(2024, 1, 1)


def _panel(
    n_instruments: int,
    n_days: int,
    value_fn: Callable[[int, int], float],
) -> pl.DataFrame:
    """构造 ``(date, instrument, value)`` 长表，``value_fn(day, instrument)`` 决定取值。"""
    rows: list[dict[str, object]] = []
    for day in range(n_days):
        for index in range(n_instruments):
            rows.append(
                {
                    "date": START + dt.timedelta(days=day),
                    "instrument": f"{600000 + index:06d}.SH",
                    "value": value_fn(day, index),
                }
            )
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "instrument": pl.String, "value": pl.Float64},
    )


def _signal(day: int, index: int) -> float:
    """确定性的截面变化序列（非随机），逐日独立。"""
    return math.sin(1.7 * index + 0.3 * day) + 0.05 * (index % 7)


class TestCrossSectionCorr:
    def test_perfect_positive_is_one(self) -> None:
        library = _panel(40, 6, _signal)
        new = library.with_columns((2.0 * pl.col("value") + 3.0).alias("value"))

        assert cross_section_corr(new, library) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_is_minus_one(self) -> None:
        library = _panel(40, 6, _signal)
        new = library.with_columns((-3.0 * pl.col("value")).alias("value"))

        assert cross_section_corr(new, library) == pytest.approx(-1.0, abs=1e-9)

    def test_days_below_min_count_are_skipped(self) -> None:
        library = _panel(10, 6, _signal)
        new = library.with_columns((2.0 * pl.col("value")).alias("value"))

        # 逐日只有 10 只证券，低于默认下限 30，全部日子跳过 → None。
        assert cross_section_corr(new, library) is None
        # 放宽下限后可得有限相关。
        relaxed = cross_section_corr(new, library, min_count=5)
        assert relaxed is not None and math.isfinite(relaxed)

    def test_no_overlap_returns_none(self) -> None:
        library = _panel(40, 6, _signal)
        shifted = library.with_columns(
            (pl.col("date") + pl.duration(days=100)).alias("date")
        )

        assert cross_section_corr(shifted, library) is None


class TestMaxLibraryCorr:
    def test_uses_absolute_value_of_negative_corr(self) -> None:
        library = _panel(40, 6, _signal)
        new = library.with_columns((2.0 * pl.col("value")).alias("value"))

        report = max_library_corr(new, {"neg": library.with_columns((-pl.col("value")).alias("value"))})

        assert report.per_factor["neg"] == pytest.approx(-1.0, abs=1e-9)
        assert report.max_corr == pytest.approx(1.0, abs=1e-9)

    def test_empty_library_gives_none(self) -> None:
        new = _panel(40, 6, _signal)

        report = max_library_corr(new, {})

        assert report.per_factor == {}
        assert report.max_corr is None

    def test_all_unavailable_gives_none(self) -> None:
        library = _panel(10, 6, _signal)

        report = max_library_corr(library, {"tiny": library})

        assert report.per_factor == {"tiny": None}
        assert report.max_corr is None
