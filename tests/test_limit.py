"""``quant.data.limit`` 单元测试。

全部用例通过构造 polars 表 + monkeypatch 曾用名接口完成，不触网。
"""
from __future__ import annotations

import datetime as dt
import logging
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

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
from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INSTRUMENT_INFO,
    board_of,
    check_daily_bars,
)

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
    return _bars_adj([(instrument, day, close, 1.0) for instrument, day, close in records])


def _bars_adj(records: list[tuple[str, dt.date, float, float]]) -> pl.DataFrame:
    """records: (instrument, date, close, adjfactor)，其余价格列用收盘价填充。"""
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
                "adjfactor": [r[3] for r in records],
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


def _actions(rows: list[tuple[str, dt.date, float, float]]) -> pl.DataFrame:
    """rows: (instrument, date, cash_per_share, share_per_share)。"""
    return pl.DataFrame(
        {
            "instrument": [r[0] for r in rows],
            "date": [r[1] for r in rows],
            "cash_per_share": [r[2] for r in rows],
            "share_per_share": [r[3] for r in rows],
        },
        schema=CORPORATE_ACTIONS,
    )


def _expected(prev_close: float, ratio: float, direction: int, board: str = "main") -> float:
    factor = Decimal(1) + Decimal(direction) * Decimal(str(ratio))
    rounding = ROUND_HALF_UP
    if board == "bj":
        # 北交所：涨停向下取整、跌停向上取整（issue #60，CSMAR 实证）
        rounding = ROUND_FLOOR if direction > 0 else ROUND_CEILING
    return float(
        (Decimal(str(prev_close)) * factor).quantize(Decimal("0.01"), rounding=rounding)
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
        # 主板 ST 于 2026-07-06 起并入 10%（交易所规则变更，issue #59 实证）
        ("main", dt.date(2026, 7, 3), True, 0.05),
        ("main", dt.date(2026, 7, 6), True, 0.10),
        ("main", dt.date(2026, 9, 28), True, 0.10),
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


def test_limit_ratio_main_st_unify_boundary_one_day_apart() -> None:
    before = dt.date(2026, 7, 3)
    after = dt.date(2026, 7, 6)
    assert limit_ratio("main", before, True) == 0.05
    assert limit_ratio("main", after, True) == 0.10
    # 非 ST 始终 10%，不受切换影响
    assert limit_ratio("main", before, False) == 0.10
    assert limit_ratio("main", after, False) == 0.10


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
    assert limit._limit_up(5.255, 0.10, "main") == 5.78
    assert limit._limit_up(5.245, 0.10, "main") == 5.77
    assert limit._limit_down(5.255, 0.10, "main") == 4.73
    assert limit._limit_down(5.245, 0.10, "main") == 4.72
    assert limit._limit_up(None, 0.10, "main") is None
    assert limit._limit_up(5.0, None, "main") is None


def test_bj_rounds_limit_up_floor_and_limit_down_ceil() -> None:
    # 北交所口径（CSMAR 25.5 万行实证，issue #60）：涨停向下取整、跌停向上取整。
    # 8.52 × 1.3 = 11.076 -> 11.07（四舍五入会给 11.08）
    # 8.52 × 0.7 = 5.964  -> 5.97（四舍五入会给 5.96）
    assert limit._limit_up(8.52, 0.30, "bj") == 11.07
    assert limit._limit_down(8.52, 0.30, "bj") == 5.97
    # 沪深保持四舍五入
    assert limit._limit_up(8.52, 0.30, "kcb") == 11.08
    assert limit._limit_down(8.52, 0.30, "kcb") == 5.96


def test_compute_limits_bj_floor_ceil_end_to_end() -> None:
    bars = _bars(
        [
            ("920001.BJ", D1, 8.52),
            ("920001.BJ", D2, 8.6),
        ]
    )
    info = _instruments([("920001.BJ", "北", dt.date(2022, 1, 1))])
    out = compute_limits(bars, info, _intervals([]))
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    assert got[D2] == (11.07, 5.97)


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
            board = BOARD_BY_INSTRUMENT[row["instrument"]]
            assert row["limit_up"] == _expected(prev_close, ratio, 1, board)
            assert row["limit_down"] == _expected(prev_close, ratio, -1, board)


def test_compute_limits_empty_bars() -> None:
    out = compute_limits(
        pl.DataFrame(schema=DAILY_BARS),
        _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))]),
        _intervals([]),
    )
    assert out.height == 0
    assert out.schema == DAILY_BARS


# ---------------------------------------------------------------------------
# 除权日基准价（issue #60）
#
# 除权除息日（corporate_actions 有记录的 (instrument, date)）基准价取除权参考价，
# 由后复权因子跳变隐含：ref = prev_close × prev_adj / adj，四舍五入到分后再按
# 板块规则算涨跌停。以下用例的 adj 按真实事件口径构造：现金分红 C 时
# adj = prev_close / (prev_close - C)，送转 S 时 adj = 1 + S，两者叠加
# adj = prev_close / ((prev_close - C) / (1 + S))。
# ---------------------------------------------------------------------------

#: 现金分红 0.5 元、前收 10.0 时的后复权因子（ref = 9.5）。
_ADJ_CASH_05_ON_10: float = 10.0 / 9.5


def test_ex_date_cash_dividend_uses_reference_price() -> None:
    bars = _bars_adj(
        [
            ("600001.SH", D1, 10.0, 1.0),
            ("600001.SH", D2, 10.2, _ADJ_CASH_05_ON_10),
            ("600001.SH", D3, 10.5, _ADJ_CASH_05_ON_10),
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    actions = _actions([("600001.SH", D2, 0.5, 0.0)])
    out = compute_limits(bars, info, _intervals([]), actions)
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    # D2 是除息日：基准价 9.5（10.0 - 0.5），limit = 9.5 × (1 ± 10%)
    assert got[D1] == (None, None)
    assert got[D2] == (10.45, 8.55)
    # D3 非除权日：基准价回到前收 10.2
    assert got[D3] == (11.22, 9.18)


def test_ex_date_share_dividend_uses_reference_price() -> None:
    bars = _bars_adj(
        [
            ("600001.SH", D1, 20.0, 1.0),
            ("600001.SH", D2, 13.5, 1.5),  # 10 转 5：ref = 20 / 1.5
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    actions = _actions([("600001.SH", D2, 0.0, 0.5)])
    out = compute_limits(bars, info, _intervals([]), actions)
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    # ref = 13.3333… → 取整 13.33；limit = 13.33 × (1 ± 10%)
    assert got[D2] == (14.66, 12.00)


def test_ex_date_cash_and_share_uses_reference_price() -> None:
    bars = _bars_adj(
        [
            ("600001.SH", D1, 10.0, 1.0),
            ("600001.SH", D2, 6.5, 10.0 / (9.5 / 1.5)),  # 派 0.5 + 10 转 5
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    actions = _actions([("600001.SH", D2, 0.5, 0.5)])
    out = compute_limits(bars, info, _intervals([]), actions)
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    # ref = (10 - 0.5) / 1.5 = 6.3333… → 取整 6.33；limit = 6.33 × (1 ± 10%)
    assert got[D2] == (6.96, 5.70)


def test_ex_date_reference_price_rounds_half_up_to_cent() -> None:
    # 构造隐含 ref 恰好落在半分边界：ref = 5.245（adj = 5.3 / 5.245）。
    bars = _bars_adj(
        [
            ("600001.SH", D1, 5.3, 1.0),
            ("600001.SH", D2, 5.5, 5.3 / 5.245),
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    actions = _actions([("600001.SH", D2, 0.055, 0.0)])
    out = compute_limits(bars, info, _intervals([]), actions)
    got = {row["date"]: row["limit_up"] for row in out.iter_rows(named=True)}
    # ref 半升取整为 5.25（银行家舍入会给 5.24），limit_up = 5.25 × 1.1 = 5.78
    assert got[D2] == 5.78


def test_adj_jump_without_corporate_action_keeps_prev_close() -> None:
    # 因子跳变但无公司行为记录：不触发除权基准，沿用前收（退化保护）。
    bars = _bars_adj(
        [
            ("600001.SH", D1, 10.0, 1.0),
            ("600001.SH", D2, 10.0, 2.0),
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    out = compute_limits(bars, info, _intervals([]), _actions([]))
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    assert got[D2] == (11.00, 9.00)


def test_corporate_action_with_flat_factor_keeps_prev_close() -> None:
    # 有公司行为记录但因子未动（如重整转增给债权人，交易所不调整参考价）：
    # 隐含 ref = 前收，行为与修复前一致。
    bars = _bars_adj(
        [
            ("600001.SH", D1, 10.0, 1.0),
            ("600001.SH", D2, 10.1, 1.0),
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    actions = _actions([("600001.SH", D2, 0.5, 0.1)])
    out = compute_limits(bars, info, _intervals([]), actions)
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    assert got[D2] == (11.00, 9.00)


def test_ex_date_on_first_bar_stays_null() -> None:
    bars = _bars_adj([("600001.SH", D1, 10.0, 1.0)])
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    actions = _actions([("600001.SH", D1, 0.5, 0.0)])
    out = compute_limits(bars, info, _intervals([]), actions)
    assert out["limit_up"].to_list() == [None]
    assert out["limit_down"].to_list() == [None]


def test_compute_limits_without_actions_unchanged() -> None:
    # 缺省 corporate_actions=None：与三参数旧调用行为一致。
    bars = _bars_adj(
        [
            ("600001.SH", D1, 10.0, 1.0),
            ("600001.SH", D2, 10.2, _ADJ_CASH_05_ON_10),
        ]
    )
    info = _instruments([("600001.SH", "甲", dt.date(2000, 1, 1))])
    out = compute_limits(bars, info, _intervals([]))
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    assert got[D2] == (11.00, 9.00)


def test_ex_date_reference_price_applies_board_ratio() -> None:
    # 除权日与板块 / ST 规则叠加：创业板改革后 20%。
    bars = _bars_adj(
        [
            ("300001.SZ", D1, 10.0, 1.0),
            ("300001.SZ", D2, 9.6, _ADJ_CASH_05_ON_10),
        ]
    )
    info = _instruments([("300001.SZ", "乙", dt.date(2010, 1, 1))])
    actions = _actions([("300001.SZ", D2, 0.5, 0.0)])
    out = compute_limits(bars, info, _intervals([]), actions)
    got = {row["date"]: (row["limit_up"], row["limit_down"]) for row in out.iter_rows(named=True)}
    # ref 9.5 × (1 ± 20%)
    assert got[D2] == (11.40, 7.60)
