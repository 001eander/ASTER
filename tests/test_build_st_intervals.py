"""scripts/build_st_intervals_from_csmar.py 的单元测试。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from scripts.build_st_intervals_from_csmar import build_st_intervals_from_trdsta

#: 测试日期窗口：2026-06-29 ~ 2026-07-08，横跨 MAIN_ST_UNIFY_DATE（2026-07-06）。
START = date(2026, 6, 29)


def _rows(
    instrument: str,
    bands: list[float | None],
    trdsta: list[int] | None = None,
    start: date = START,
) -> list[dict[str, object]]:
    """按每日「涨停幅度」造行：ref_limit_up = ref_pre_close × (1 + band)，band 为 None 时 ref 缺省。日期只取交易日（跳过周末）。"""
    days: list[date] = []
    cursor = start
    while len(days) < len(bands):
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    out: list[dict[str, object]] = []
    for day, band in zip(days, bands):
        pre_close = 10.0
        out.append(
            {
                "instrument": instrument,
                "date": day,
                "trdsta": (trdsta[len(out)] if trdsta is not None else 1),
                "ref_limit_up": (round(pre_close * (1 + band), 2) if band is not None else None),
                "ref_pre_close": (pre_close if band is not None else None),
            }
        )
    return out


@pytest.fixture
def reference() -> pl.DataFrame:
    rows = (
        # 主板：trdsta 一直为 1，但中段执行 5% 带（戴帽未标注）→ 判 ST。
        _rows("600001.SH", [0.10, 0.10, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        # 主板：trdsta 一直为 2，但后段执行 10% 带（摘帽未同步，603822 案例）→ 后段判非 ST。
        + _rows("600002.SH", [0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], trdsta=[2] * 10)
        # 主板：全程 5% 带，但区间不得越过切换日 2026-07-06（此后强制非 ST）。
        + _rows("000001.SZ", [0.05] * 10, trdsta=[2] * 10)
        # 创业板 / 北交所：即使按 5% 执行也不入表（非主板）。
        + _rows("300001.SZ", [0.05] * 10, trdsta=[2] * 10)
        + _rows("830001.BJ", [0.05] * 10, trdsta=[2] * 10)
        # 主板：ref 缺失的日子回退 trdsta 判定（2=ST，4=S 不算）；切换日前的窗口。
        + _rows("600003.SH", [None, None, None, None], trdsta=[2, 2, 4, 1])
        # 主板：切换日后即使 ref 缺失且 trdsta=2，也强制非 ST。
        + _rows(
            "600005.SH",
            [None, None, None, None],
            trdsta=[2, 2, 2, 2],
            start=date(2026, 7, 6),
        )
        # 主板：无涨跌幅限制日（CSMAR 哨兵 99999.99）判非 ST。
        + _rows("600004.SH", [0.05, 0.05], trdsta=[2, 2])
        + [
            {
                "instrument": "600004.SH",
                "date": START + timedelta(days=2),
                "trdsta": 2,
                "ref_limit_up": 99999.99,
                "ref_pre_close": 10.0,
            }
        ]
    )
    return pl.DataFrame(rows).cast(
        {
            "instrument": pl.String,
            "date": pl.Date,
            "trdsta": pl.Int64,
            "ref_limit_up": pl.Float64,
            "ref_pre_close": pl.Float64,
        }
    )


def test_band_detection_overrides_trdsta(reference: pl.DataFrame) -> None:
    out = build_st_intervals_from_trdsta(reference)
    seg = out.filter(pl.col("instrument") == "600001.SH")
    assert seg.height == 1
    row = seg.row(0, named=True)
    assert row["start_date"] == date(2026, 7, 1)
    assert row["end_date"] == date(2026, 7, 2)


def test_delisted_st_sync(reference: pl.DataFrame) -> None:
    out = build_st_intervals_from_trdsta(reference)
    seg = out.filter(pl.col("instrument") == "600002.SH")
    assert seg.height == 1
    row = seg.row(0, named=True)
    assert row["start_date"] == date(2026, 6, 29)
    assert row["end_date"] == date(2026, 6, 30)


def test_st_band_ends_at_unify_date(reference: pl.DataFrame) -> None:
    out = build_st_intervals_from_trdsta(reference)
    seg = out.filter(pl.col("instrument") == "000001.SZ")
    assert seg.height == 1
    row = seg.row(0, named=True)
    assert row["start_date"] == date(2026, 6, 29)
    # 切换日后强制非 ST：区间在切换日前最后一个有效日终止，不再出现 null。
    assert row["end_date"] == date(2026, 7, 3)


def test_board_filter(reference: pl.DataFrame) -> None:
    out = build_st_intervals_from_trdsta(reference)
    assert out.filter(pl.col("instrument") == "300001.SZ").height == 0
    assert out.filter(pl.col("instrument") == "830001.BJ").height == 0


def test_fallback_to_trdsta_and_sentinel(reference: pl.DataFrame) -> None:
    out = build_st_intervals_from_trdsta(reference)
    seg = out.filter(pl.col("instrument") == "600003.SH")
    assert seg.height == 1
    row = seg.row(0, named=True)
    assert row["start_date"] == date(2026, 6, 29)
    assert row["end_date"] == date(2026, 6, 30)

    seg4 = out.filter(pl.col("instrument") == "600004.SH")
    assert seg4.height == 1
    row4 = seg4.row(0, named=True)
    assert row4["start_date"] == date(2026, 6, 29)
    assert row4["end_date"] == date(2026, 6, 30)


def test_forced_non_st_after_unify_date(reference: pl.DataFrame) -> None:
    out = build_st_intervals_from_trdsta(reference)
    assert out.filter(pl.col("instrument") == "600005.SH").height == 0


def test_empty_input() -> None:
    empty = pl.DataFrame(
        schema={
            "instrument": pl.String,
            "date": pl.Date,
            "trdsta": pl.Int64,
            "ref_limit_up": pl.Float64,
            "ref_pre_close": pl.Float64,
        }
    )
    out = build_st_intervals_from_trdsta(empty)
    assert out.height == 0
    assert list(out.columns) == ["instrument", "start_date", "end_date"]
