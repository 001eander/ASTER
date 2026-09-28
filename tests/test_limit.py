"""``quant.data.limit`` 单元测试。

全部用例通过构造 polars 表 + monkeypatch 曾用名接口完成，不触网。
"""
from __future__ import annotations

import datetime as dt
import logging
from decimal import ROUND_HALF_UP, Decimal

import polars as pl
import pytest

import quant.data.limit as limit
from quant.data.limit import (
    ST_INTERVALS,
    build_st_intervals,
    compute_limits,
    is_st_name,
    limit_ratio,
    st_intervals_from_name_history,
)
from quant.data.schema import DAILY_BARS, INSTRUMENT_INFO, board_of, check_daily_bars

D1 = dt.date(2020, 8, 21)  # 创业板改革前（周五）
D2 = dt.date(2020, 8, 24)  # 改革生效日（周一）
D3 = dt.date(2020, 8, 25)

BOARD_BY_INSTRUMENT = {
    "600001.SH": "main",
    "300001.SZ": "cyb",
    "688001.SH": "kcb",
    "920001.BJ": "bj",
}


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def _bars(records: list[tuple[str, dt.date, float]]) -> pl.DataFrame:
    """records: (instrument, date, close)，其余价格列用收盘价填充。"""
    n = len(records)
    closes = [r[2] for r in records]
    return (
        pl.DataFrame(
            {
                "date": [r[1] for r in records],
                "instrument": [r[0] for r in records],
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "vwap": closes,
                "volume": [1.0] * n,
                "amount": [1.0] * n,
                "adjfactor": [1.0] * n,
                "limit_up": [None] * n,
                "limit_down": [None] * n,
            }
        )
        .cast(DAILY_BARS)
        .sort(["instrument", "date"])
    )


def _instruments(
    rows: list[tuple[str, str, dt.date | None]],
) -> pl.DataFrame:
    """rows: (instrument, name, list_date)。"""
    return pl.DataFrame(
        {
            "instrument": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "board": [board_of(r[0]) for r in rows],
            "list_date": [r[2] for r in rows],
            "delist_date": [None] * len(rows),
        }
    ).cast(INSTRUMENT_INFO)


def _intervals(rows: list[tuple[str, dt.date, dt.date | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": [r[0] for r in rows],
            "start_date": [r[1] for r in rows],
            "end_date": [r[2] for r in rows],
        },
        schema=ST_INTERVALS,
    )


def _expected(prev_close: float, ratio: float, direction: int) -> float:
    factor = Decimal(1) + Decimal(direction) * Decimal(str(ratio))
    return float(
        (Decimal(str(prev_close)) * factor).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )


# ---------------------------------------------------------------------------
# limit_ratio 规则表
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("board", "day", "is_st", "expected"),
    [
        ("main", dt.date(2015, 1, 5), False, 0.10),
        ("main", dt.date(2015, 1, 5), True, 0.05),
        ("main", dt.date(2026, 1, 5), True, 0.05),
        # 创业板改革前
        ("cyb", dt.date(2020, 8, 21), False, 0.10),
        ("cyb", dt.date(2020, 8, 21), True, 0.05),
        # 创业板改革生效日当天即 20%，含 ST
        ("cyb", dt.date(2020, 8, 24), False, 0.20),
        ("cyb", dt.date(2020, 8, 24), True, 0.20),
        ("cyb", dt.date(2026, 1, 5), True, 0.20),
        # 科创板恒 20%，含 ST
        ("kcb", dt.date(2019, 7, 22), False, 0.20),
        ("kcb", dt.date(2015, 1, 5), True, 0.20),
        ("kcb", dt.date(2026, 1, 5), True, 0.20),
        # 北交所恒 30%，含 ST
        ("bj", dt.date(2021, 11, 15), False, 0.30),
        ("bj", dt.date(2015, 1, 5), True, 0.30),
        ("bj", dt.date(2026, 1, 5), True, 0.30),
    ],
)
def test_limit_ratio_rule_table(
    board: str, day: dt.date, is_st: bool, expected: float
) -> None:
    assert limit_ratio(board, day, is_st) == expected


def test_limit_ratio_cyb_reform_boundary_one_day_apart() -> None:
    before = dt.date(2020, 8, 21)
    after = dt.date(2020, 8, 24)
    assert limit_ratio("cyb", before, False) == 0.10
    assert limit_ratio("cyb", after, False) == 0.20
    assert limit_ratio("cyb", before, True) == 0.05
    assert limit_ratio("cyb", after, True) == 0.20


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ST海虹", True),
        ("*ST海虹", True),
        ("st海虹", True),
        ("贵州茅台", False),
        ("", False),
        (None, False),
    ],
)
def test_is_st_name(name: str | None, expected: bool) -> None:
    assert is_st_name(name) is expected


# ---------------------------------------------------------------------------
# ROUND_HALF_UP 舍入
# ---------------------------------------------------------------------------


def test_round_half_up_classic_cases() -> None:
    bars = _bars(
        [
            ("600001.SH", D1, 5.255),
            ("600001.SH", D2, 7.0),
            ("600002.SH", D1, 5.245),
            ("600002.SH", D2, 7.0),
        ]
    )
    info = _instruments(
        [("600001.SH", "甲", dt.date(2000, 1, 1)), ("600002.SH", "乙", dt.date(2000, 1, 1))]
    )
    out = compute_limits(bars, info, _intervals([]))
    got = {
        (row["instrument"], row["date"]): (row["limit_up"], row["limit_down"])
        for row in out.iter_rows(named=True)
    }
    assert got[("600001.SH", D2)] == (5.78, 4.73)
    assert got[("600002.SH", D2)] == (5.77, 4.72)


def test_round_half_up_not_bankers_rounding() -> None:
    # 5.255 * 1.10 = 5.7805 -> HALF_UP 得 5.78；银行家舍入会得到 5.78 或 5.77 视表示而定。
    assert limit._limit_up(5.255, 0.10) == 5.78
    assert limit._limit_up(5.245, 0.10) == 5.77
    assert limit._limit_down(5.255, 0.10) == 4.73
    assert limit._limit_down(5.245, 0.10) == 4.72
    assert limit._limit_up(None, 0.10) is None
    assert limit._limit_up(5.0, None) is None


# ---------------------------------------------------------------------------
# 无前视与首行 null
# ---------------------------------------------------------------------------


def test_limits_use_prev_close_not_same_day() -> None:
    bars = _bars(
        [
            ("600001.SH", D1, 10.0),
            ("600001.SH", D2, 100.0),  # 当日巨幅波动，不应影响当日 limit
            ("600001.SH", D3, 20.0),
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    out = compute_limits(bars, info, _intervals([]))
    got = out.select(["date", "limit_up", "limit_down"]).rows()
    assert got == [
        (D1, None, None),
        (D2, 11.00, 9.00),  # 依赖 D1 收盘 10.0
        (D3, 110.00, 90.00),  # 依赖 D2 收盘 100.0，与 D3 收盘 20 无关
    ]


def test_first_row_null_per_instrument() -> None:
    bars = _bars(
        [
            ("600001.SH", D1, 10.0),
            ("600001.SH", D2, 11.0),
            ("300001.SZ", D1, 50.0),
            ("300001.SZ", D2, 55.0),
        ]
    )
    info = _instruments(
        [("600001.SH", "甲", dt.date(2000, 1, 1)), ("300001.SZ", "乙", dt.date(2010, 1, 1))]
    )
    out = compute_limits(bars, info, _intervals([]))
    first = out.group_by("instrument", maintain_order=True).first()
    assert first["limit_up"].to_list() == [None, None]
    assert first["limit_down"].to_list() == [None, None]


# ---------------------------------------------------------------------------
# ST 区间解析
# ---------------------------------------------------------------------------


def test_st_intervals_from_name_history_merges_adjacent_and_open_ended() -> None:
    history = pl.DataFrame(
        {
            "date": [
                dt.date(2010, 1, 1),
                dt.date(2013, 5, 1),
                dt.date(2015, 6, 1),
                dt.date(2016, 7, 1),
                dt.date(2019, 1, 1),
            ],
            "name": ["正常股", "ST正常", "*ST正常", "正常股", "*ST再次"],
        }
    )
    out = st_intervals_from_name_history(history, "600001.SH")
    assert out.rows() == [
        ("600001.SH", dt.date(2013, 5, 1), dt.date(2016, 6, 30)),
        ("600001.SH", dt.date(2019, 1, 1), None),
    ]


def test_st_intervals_from_name_history_handles_unsorted_and_missing() -> None:
    history = pl.DataFrame(
        {
            "date": [dt.date(2016, 7, 1), dt.date(2013, 5, 1), dt.date(2010, 1, 1)],
            "name": ["正常股", "ST正常", "正常股"],
        }
    )
    out = st_intervals_from_name_history(history, "600001.SH")
    assert out.rows() == [("600001.SH", dt.date(2013, 5, 1), dt.date(2016, 6, 30))]


def test_st_intervals_from_name_history_empty() -> None:
    out = st_intervals_from_name_history(pl.DataFrame(schema=limit.NAME_HISTORY), "600001.SH")
    assert out.height == 0
    assert out.schema == ST_INTERVALS


def test_normalize_name_history_without_dates_returns_empty() -> None:
    # 复现 akshare 1.18.97 的实际返回：只有 index / name，无日期。
    raw = pl.DataFrame({"index": [1, 2], "name": ["贵州茅台", "贵州茅台"]})
    assert limit._normalize_name_history(raw).height == 0


def test_normalize_name_history_with_dates() -> None:
    raw = pl.DataFrame(
        {"变更日期": ["2013-05-01", "2019-01-01"], "名称": ["ST甲", "甲"]}
    )
    out = limit._normalize_name_history(raw)
    assert out.rows() == [(dt.date(2013, 5, 1), "ST甲"), (dt.date(2019, 1, 1), "甲")]


# ---------------------------------------------------------------------------
# build_st_intervals
# ---------------------------------------------------------------------------


class _StubSource:
    def __init__(self, info: pl.DataFrame) -> None:
        self._info = info

    def instrument_info(self) -> pl.DataFrame:
        return self._info


def test_build_st_intervals_fallback_current_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 曾用名接口无日期 -> 退化方案
    monkeypatch.setattr(limit, "_fetch_name_history", lambda instrument: limit._empty_name_history())
    info = _instruments(
        [
            ("600001.SH", "*ST甲", dt.date(2010, 1, 1)),
            ("300001.SZ", "乙", dt.date(2011, 5, 5)),
            ("688001.SH", "ST丙", None),
        ]
    )
    out = build_st_intervals(_StubSource(info), list(BOARD_BY_INSTRUMENT))
    assert out.rows() == [
        ("600001.SH", dt.date(2010, 1, 1), None),
        ("688001.SH", limit.EARLIEST_TRADE_DATE, None),
    ]


def test_build_st_intervals_dated_path(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "600001.SH": pl.DataFrame(
            {
                "date": [dt.date(2013, 5, 1), dt.date(2016, 7, 1)],
                "name": ["ST甲", "甲"],
            }
        ),
        "300001.SZ": pl.DataFrame(
            {"date": [dt.date(2015, 1, 1)], "name": ["乙"]}
        ),
    }

    def fake(instrument: str) -> pl.DataFrame:
        return histories[instrument]

    monkeypatch.setattr(limit, "_fetch_name_history", fake)
    info = _instruments(
        [("600001.SH", "甲", dt.date(2000, 1, 1)), ("300001.SZ", "乙", dt.date(2000, 1, 1))]
    )
    out = build_st_intervals(_StubSource(info), ["600001.SH", "300001.SZ"])
    assert out.rows() == [("600001.SH", dt.date(2013, 5, 1), dt.date(2016, 6, 30))]


def test_build_st_intervals_single_failure_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake(instrument: str) -> pl.DataFrame:
        if instrument == "688001.SH":
            raise RuntimeError("网络抖动")
        return pl.DataFrame(
            {"date": [dt.date(2015, 1, 1)], "name": ["ST票"]}
        )

    monkeypatch.setattr(limit, "_fetch_name_history", fake)
    info = _instruments(
        [
            ("600001.SH", "甲", dt.date(2000, 1, 1)),
            ("688001.SH", "乙", dt.date(2000, 1, 1)),
            ("300001.SZ", "丙", dt.date(2000, 1, 1)),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="quant.data.limit"):
        out = build_st_intervals(
            _StubSource(info), ["600001.SH", "688001.SH", "300001.SZ"]
        )
    assert out.rows() == [
        ("300001.SZ", dt.date(2015, 1, 1), None),
        ("600001.SH", dt.date(2015, 1, 1), None),
    ]
    assert any("688001.SH" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# compute_limits 端到端
# ---------------------------------------------------------------------------


def test_compute_limits_end_to_end_all_boards() -> None:
    records: list[tuple[str, dt.date, float]] = []
    for instrument in BOARD_BY_INSTRUMENT:
        records += [
            (instrument, D1, 10.0),
            (instrument, D2, 20.0),
            (instrument, D3, 30.0),
        ]
    bars = _bars(records)
    info = _instruments([(i, f"票{i}", dt.date(2000, 1, 1)) for i in BOARD_BY_INSTRUMENT])
    out = compute_limits(bars, info, _intervals([]))
    check_daily_bars(out)

    got = {
        (row["instrument"], row["date"]): (row["limit_up"], row["limit_down"])
        for row in out.iter_rows(named=True)
    }
    # D2 的前收是 10.0
    assert got[("600001.SH", D2)] == (11.00, 9.00)
    assert got[("300001.SZ", D2)] == (12.00, 8.00)
    assert got[("688001.SH", D2)] == (12.00, 8.00)
    assert got[("920001.BJ", D2)] == (13.00, 7.00)
    # D3 的前收是 20.0
    assert got[("600001.SH", D3)] == (22.00, 18.00)
    assert got[("300001.SZ", D3)] == (24.00, 16.00)
    assert got[("920001.BJ", D3)] == (26.00, 14.00)


def test_compute_limits_applies_st_and_cyb_reform() -> None:
    records: list[tuple[str, dt.date, float]] = []
    for instrument in BOARD_BY_INSTRUMENT:
        records += [(instrument, D1, 10.0), (instrument, D2, 20.0)]
    bars = _bars(records)
    info = _instruments([(i, f"票{i}", dt.date(2000, 1, 1)) for i in BOARD_BY_INSTRUMENT])
    # 主板 / 创业板从 D2 起戴帽，覆盖到至今
    intervals = _intervals(
        [("600001.SH", D2, None), ("300001.SZ", D2, None)]
    )
    out = compute_limits(bars, info, intervals)
    got = {
        (row["instrument"], row["date"]): (row["limit_up"], row["limit_down"])
        for row in out.iter_rows(named=True)
    }
    assert got[("600001.SH", D2)] == (10.50, 9.50)  # 主板 ST 5%
    assert got[("300001.SZ", D2)] == (12.00, 8.00)  # 改革后创业板 ST 仍 20%
    assert got[("688001.SH", D2)] == (12.00, 8.00)  # 科创板 20%
    assert got[("920001.BJ", D2)] == (13.00, 7.00)  # 北交所 30%


def test_compute_limits_st_interval_closed_boundary() -> None:
    bars = _bars(
        [("600001.SH", D1, 10.0), ("600001.SH", D2, 20.0), ("600001.SH", D3, 30.0)]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    intervals = _intervals([("600001.SH", D2, D2)])  # 仅 D2 一天
    out = compute_limits(bars, info, intervals)
    got = {row["date"]: row["limit_up"] for row in out.iter_rows(named=True)}
    assert got[D2] == 10.50  # 前收 10.0 * 1.05，D2 当天是 ST
    assert got[D3] == 22.00  # 前收 20.0 * 1.10，D3 已摘帽


def test_compute_limits_vectorized_matches_rule_function() -> None:
    records: list[tuple[str, dt.date, float]] = []
    for instrument in BOARD_BY_INSTRUMENT:
        records += [
            (instrument, dt.date(2020, 8, 20), 10.0),
            (instrument, D1, 10.0),
            (instrument, D2, 20.0),
            (instrument, D3, 30.0),
        ]
    bars = _bars(records)
    info = _instruments([(i, f"票{i}", dt.date(2000, 1, 1)) for i in BOARD_BY_INSTRUMENT])
    for intervals in (
        _intervals([]),
        _intervals([(i, dt.date(1900, 1, 1), None) for i in BOARD_BY_INSTRUMENT]),
    ):
        out = compute_limits(bars, info, intervals)
        st_all = intervals.height > 0
        prev_map = {
            (row["instrument"], row["date"]): row["prev"]
            for row in bars.with_columns(
                pl.col("close").shift(1).over("instrument", order_by="date").alias("prev")
            ).iter_rows(named=True)
        }
        for row in out.iter_rows(named=True):
            prev_close = prev_map[(row["instrument"], row["date"])]
            if prev_close is None:
                assert row["limit_up"] is None
                assert row["limit_down"] is None
                continue
            ratio = limit_ratio(
                BOARD_BY_INSTRUMENT[row["instrument"]], row["date"], st_all
            )
            assert row["limit_up"] == _expected(prev_close, ratio, 1)
            assert row["limit_down"] == _expected(prev_close, ratio, -1)


def test_compute_limits_empty_bars() -> None:
    out = compute_limits(
        pl.DataFrame(schema=DAILY_BARS),
        _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))]),
        _intervals([]),
    )
    assert out.height == 0
    assert out.schema == DAILY_BARS
