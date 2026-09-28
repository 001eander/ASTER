"""``quant.data.source.akshare`` 单元测试。

全部用例通过 monkeypatch 替换 akshare 函数，不触网。
akshare 的返回类型是 ``pandas.DataFrame``，这里用 polars 构造数据后 ``to_pandas()``
还原成同类对象喂给被测代码，避免在项目代码里 import pandas。
"""
from __future__ import annotations

import datetime as dt

# akshare 的返回类型就是 pandas.DataFrame，测试需要构造同形状的输入。
# 生产代码不 import pandas，这里仅在测试夹具里使用。
import pandas as pd
import polars as pl
import pytest

import quant.data.source.akshare as ak_source
from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    check_schema,
)
from quant.data.source.akshare import AkshareSource, parse_dividend_text
from quant.data.source.base import DataSource


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source, "REQUEST_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(ak_source, "RETRY_BASE_DELAY_SECONDS", 0.0)


def _pandas(df: pl.DataFrame) -> object:
    return pd.DataFrame(df.to_dict(as_series=False))


def _raise(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("网络不可用")


def _boom_em(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("东方财富不可用")


# ---------------------------------------------------------------------------
# daily_bars
# ---------------------------------------------------------------------------


def _em_raw() -> object:
    return _pandas(
        pl.DataFrame(
            {
                "日期": [dt.date(2026, 9, 1), dt.date(2026, 9, 2)],
                "开盘": [10.0, 11.0],
                "收盘": [11.0, 12.0],
                "最高": [11.5, 12.5],
                "最低": [9.5, 10.5],
                "成交量": [100.0, 0.0],  # 手
                "成交额": [105_000.0, 0.0],  # 元
            }
        )
    )


def _em_hfq() -> object:
    return _pandas(
        pl.DataFrame(
            {
                "日期": [dt.date(2026, 9, 1), dt.date(2026, 9, 2)],
                "开盘": [20.0, 22.0],
                "收盘": [22.0, 24.0],  # 因子恒为 2
                "最高": [23.0, 25.0],
                "最低": [19.0, 21.0],
                "成交量": [100.0, 0.0],
                "成交额": [210_000.0, 0.0],
            }
        )
    )


def test_daily_bars_em_maps_units_factor_and_vwap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", lambda **kw: _em_raw() if kw["adjust"] == "" else _em_hfq())
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", _raise)

    out = AkshareSource().daily_bars(["600519.SH"], dt.date(2026, 9, 1), dt.date(2026, 9, 2))

    check_schema(out, DAILY_BARS)
    assert out["instrument"].to_list() == ["600519.SH", "600519.SH"]
    assert out["volume"].to_list() == [10_000.0, 0.0]  # 手 -> 股
    assert out["vwap"].to_list() == [10.5, None]
    assert out["adjfactor"].to_list() == [2.0, 2.0]
    assert out["limit_up"].to_list() == [None, None]
    assert out["limit_down"].to_list() == [None, None]


def test_daily_bars_stop_day_rows_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    # 原始数据只有 9/1 与 9/3，停牌日 9/2 不应被补行。
    raw = _pandas(
        pl.DataFrame(
            {
                "日期": [dt.date(2026, 9, 1), dt.date(2026, 9, 3)],
                "开盘": [10.0, 11.0],
                "收盘": [11.0, 12.0],
                "最高": [11.5, 12.5],
                "最低": [9.5, 10.5],
                "成交量": [100.0, 100.0],
                "成交额": [105_000.0, 115_000.0],
            }
        )
    )
    hfq = _pandas(
        pl.DataFrame(
            {
                "日期": [dt.date(2026, 9, 1), dt.date(2026, 9, 3)],
                "收盘": [22.0, 24.0],
            }
        )
    )
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", lambda **kw: raw if kw["adjust"] == "" else hfq)
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", _raise)

    out = AkshareSource().daily_bars(["600519.SH"], dt.date(2026, 9, 1), dt.date(2026, 9, 3))

    assert out["date"].to_list() == [dt.date(2026, 9, 1), dt.date(2026, 9, 3)]


def test_daily_bars_sorted_by_instrument_date(monkeypatch: pytest.MonkeyPatch) -> None:
    def em(**kw: object) -> object:
        if kw["symbol"] == "600519":
            return _pandas(
                pl.DataFrame(
                    {
                        "日期": [dt.date(2026, 9, 2), dt.date(2026, 9, 1)],
                        "开盘": [11.0, 10.0],
                        "收盘": [12.0, 11.0],
                        "最高": [12.5, 11.5],
                        "最低": [10.5, 9.5],
                        "成交量": [100.0, 100.0],
                        "成交额": [120_000.0, 110_000.0],
                    }
                )
            )
        return _pandas(
            pl.DataFrame(
                {
                    "日期": [dt.date(2026, 9, 1)],
                    "开盘": [5.0],
                    "收盘": [6.0],
                    "最高": [6.5],
                    "最低": [4.5],
                    "成交量": [200.0],
                    "成交额": [120_000.0],
                }
            )
        )

    def em_hfq(**kw: object) -> object:
        if kw["symbol"] == "600519":
            return _pandas(
                pl.DataFrame(
                    {
                        "日期": [dt.date(2026, 9, 2), dt.date(2026, 9, 1)],
                        "收盘": [24.0, 22.0],
                    }
                )
            )
        return _pandas(pl.DataFrame({"日期": [dt.date(2026, 9, 1)], "收盘": [12.0]}))

    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", lambda **kw: em_hfq(**kw) if kw["adjust"] == "hfq" else em(**kw))
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", _raise)

    out = AkshareSource().daily_bars(
        ["600519.SH", "000001.SZ"], dt.date(2026, 9, 1), dt.date(2026, 9, 2)
    )
    assert out.select(["instrument", "date"]).rows() == [
        ("000001.SZ", dt.date(2026, 9, 1)),
        ("600519.SH", dt.date(2026, 9, 1)),
        ("600519.SH", dt.date(2026, 9, 2)),
    ]


def test_daily_bars_falls_back_to_sina(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", _boom_em)

    def sina(symbol: str, adjust: str = "", **_kw: object) -> object:
        if adjust == "hfq-factor":
            return _pandas(
                pl.DataFrame(
                    {
                        "date": [dt.date(1900, 1, 1), dt.date(2026, 9, 1)],
                        "hfq_factor": [1.0, 2.0],
                    }
                )
            )
        return _pandas(
            pl.DataFrame(
                {
                    "date": [dt.date(2026, 9, 1)],
                    "open": [10.0],
                    "high": [11.5],
                    "low": [9.5],
                    "close": [11.0],
                    "volume": [10_000.0],
                    "amount": [105_000.0],
                    "outstanding_share": [1e9],
                    "turnover": [0.001],
                }
            )
        )

    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", sina)

    out = AkshareSource().daily_bars(["000001.SZ"], dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    check_schema(out, DAILY_BARS)
    assert out["adjfactor"].to_list() == [2.0]
    assert out["volume"].to_list() == [10_000.0]  # 新浪已是股，不再换算
    assert out["vwap"].to_list() == [10.5]


def test_daily_bars_falls_back_to_tx_for_bj(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", _boom_em)

    def tx(symbol: str, adjust: str = "", **_kw: object) -> object:
        close = 24.0 if adjust == "hfq" else 12.0
        return _pandas(
            pl.DataFrame(
                {
                    "date": [dt.date(2026, 9, 1)],
                    "open": [11.0],
                    "close": [close],
                    "high": [12.5],
                    "low": [10.5],
                    "volume": [1_000_000.0],
                    "turnover": [0.01],
                    "amount": [12_000_000.0],
                }
            )
        )

    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist_tx", tx)

    out = AkshareSource().daily_bars(["920001.BJ"], dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    check_schema(out, DAILY_BARS)
    assert out["instrument"].to_list() == ["920001.BJ"]
    assert out["volume"].to_list() == [1_000_000.0]
    assert out["adjfactor"].to_list() == [2.0]


def test_daily_bars_empty_when_all_sources_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", lambda **_kw: pd.DataFrame())
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", lambda **_kw: pd.DataFrame())
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist_tx", lambda **_kw: pd.DataFrame())

    out = AkshareSource().daily_bars(["600519.SH"], dt.date(2026, 9, 1), dt.date(2026, 9, 2))
    assert out.height == 0
    check_schema(out, DAILY_BARS)


# ---------------------------------------------------------------------------
# trade_calendar
# ---------------------------------------------------------------------------


def test_trade_calendar_marks_open(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _pandas(
        pl.DataFrame(
            {"trade_date": [dt.date(2026, 9, 1), dt.date(2026, 9, 3)]}
        )
    )
    monkeypatch.setattr(ak_source.ak, "tool_trade_date_hist_sina", lambda: raw)

    out = AkshareSource().trade_calendar(dt.date(2026, 9, 1), dt.date(2026, 9, 4))
    check_schema(out, TRADE_CALENDAR)
    assert out.rows() == [
        (dt.date(2026, 9, 1), True),
        (dt.date(2026, 9, 2), False),
        (dt.date(2026, 9, 3), True),
        (dt.date(2026, 9, 4), False),
    ]


# ---------------------------------------------------------------------------
# corporate_actions
# ---------------------------------------------------------------------------


def test_corporate_actions_sina_per_share_and_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _pandas(
        pl.DataFrame(
            {
                "公告日期": [dt.date(2026, 4, 1)] * 3,
                "送股": [0.0, 1.0, 0.0],
                "转增": [3.0, 2.0, 0.0],
                "派息": [21.78, 5.0, 9.0],
                "进度": ["实施", "实施", "预案"],
                "除权除息日": [dt.date(2026, 4, 22), dt.date(2026, 4, 22), None],
                "股权登记日": [dt.date(2026, 4, 21)] * 3,
                "红股上市日": [None, None, None],
            }
        )
    )
    monkeypatch.setattr(ak_source.ak, "stock_history_dividend_detail", lambda **_kw: raw)

    out = AkshareSource().corporate_actions(
        ["300750.SZ"], dt.date(2026, 1, 1), dt.date(2026, 12, 31)
    )
    check_schema(out, CORPORATE_ACTIONS)
    # 同一天两条实施记录合并：派息 (21.78+5)/10，送转 (3+1+2)/10
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["date"] == dt.date(2026, 4, 22)
    assert row["instrument"] == "300750.SZ"
    assert row["cash_per_share"] == pytest.approx(2.678)
    assert row["share_per_share"] == pytest.approx(0.6)


def test_corporate_actions_out_of_range_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _pandas(
        pl.DataFrame(
            {
                "公告日期": [dt.date(2025, 1, 1)],
                "送股": [0.0],
                "转增": [0.0],
                "派息": [10.0],
                "进度": ["实施"],
                "除权除息日": [dt.date(2025, 5, 1)],
                "股权登记日": [dt.date(2025, 4, 30)],
                "红股上市日": [None],
            }
        )
    )
    monkeypatch.setattr(ak_source.ak, "stock_history_dividend_detail", lambda **_kw: raw)

    out = AkshareSource().corporate_actions(
        ["600519.SH"], dt.date(2026, 1, 1), dt.date(2026, 12, 31)
    )
    assert out.height == 0
    check_schema(out, CORPORATE_ACTIONS)


def test_corporate_actions_falls_back_to_em(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source.ak, "stock_history_dividend_detail", _raise)
    raw = _pandas(
        pl.DataFrame(
            {
                "报告期": [dt.date(2026, 4, 1)],
                "除权除息日": [dt.date(2026, 4, 22)],
                "现金分红-现金分红比例": [21.78],
                "现金分红-现金分红比例描述": ["10派21.78元(含税)"],
                "送转股份-送股比例": [1.0],
                "送转股份-转股比例": [2.0],
                "方案进度": ["实施分配"],
            }
        )
    )
    monkeypatch.setattr(ak_source.ak, "stock_fhps_detail_em", lambda **_kw: raw)

    out = AkshareSource().corporate_actions(
        ["300750.SZ"], dt.date(2026, 1, 1), dt.date(2026, 12, 31)
    )
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["date"] == dt.date(2026, 4, 22)
    assert row["cash_per_share"] == pytest.approx(2.178)
    assert row["share_per_share"] == pytest.approx(0.3)


def test_parse_dividend_text() -> None:
    assert parse_dividend_text("10转1.00派6.00元(含税,免税后4.80元)") == (0.6, 0.1)
    assert parse_dividend_text("10派3元") == (0.3, None)
    assert parse_dividend_text("10送2转3") == (None, 0.5)
    assert parse_dividend_text("不分配不转增") == (None, None)


# ---------------------------------------------------------------------------
# instrument_info
# ---------------------------------------------------------------------------


def test_instrument_info_normalizes_and_boards(monkeypatch: pytest.MonkeyPatch) -> None:
    sh_main = _pandas(
        pl.DataFrame(
            {
                "证券代码": ["600000", "688981"],
                "证券简称": ["浦发银行", "中芯国际"],
                "证券全称": ["x", "y"],
                "公司代码": ["a", "b"],
                "公司全称": ["c", "d"],
                "上市日期": ["1999-11-10", "2020-07-16"],
            }
        )
    )
    sh_kcb = _pandas(
        pl.DataFrame(
            {
                "证券代码": ["689009"],
                "证券简称": ["九号公司"],
                "上市日期": ["2020-10-29"],
            }
        )
    )
    sz = _pandas(
        pl.DataFrame(
            {
                "板块": ["主板"],
                "A股代码": ["000001"],
                "A股简称": ["平安银行"],
                "A股上市日期": ["1991-04-03"],
                "A股总股本": ["1"],
                "A股流通股本": ["1"],
                "所属行业": ["J"],
            }
        )
    )
    bj = _pandas(
        pl.DataFrame({"证券代码": ["920001"], "证券简称": ["纬达光电"], "上市日期": ["2022-12-27"]})
    )
    sh_delist = _pandas(
        pl.DataFrame(
            {
                "公司代码": ["600001"],
                "公司简称": ["邯郸钢铁"],
                "上市日期": ["1998-01-22"],
                "暂停上市日期": ["2009-12-29"],
            }
        )
    )
    sz_delist = _pandas(
        pl.DataFrame(
            {
                "证券代码": ["000003"],
                "证券简称": ["PT金田A"],
                "上市日期": ["1991-01-14"],
                "终止上市日期": ["2002-06-14"],
            }
        )
    )

    monkeypatch.setattr(
        ak_source.ak,
        "stock_info_sh_name_code",
        lambda symbol: {"主板A股": sh_main, "科创板": sh_kcb}[symbol],
    )
    monkeypatch.setattr(ak_source.ak, "stock_info_sz_name_code", lambda **_kw: sz)
    monkeypatch.setattr(ak_source.ak, "stock_info_bj_name_code", lambda: bj)
    monkeypatch.setattr(ak_source.ak, "stock_info_sh_delist", lambda: sh_delist)
    monkeypatch.setattr(ak_source.ak, "stock_info_sz_delist", lambda **_kw: sz_delist)

    out = AkshareSource().instrument_info()
    check_schema(out, INSTRUMENT_INFO)
    got = {row[0]: row for row in out.rows()}
    assert got["600000.SH"][2] == "main"
    assert got["688981.SH"][2] == "kcb"
    assert got["689009.SH"][2] == "kcb"
    assert got["000001.SZ"][2] == "main"
    assert got["920001.BJ"][2] == "bj"
    assert got["600001.SH"][4] == dt.date(2009, 12, 29)
    assert got["000003.SZ"][4] == dt.date(2002, 6, 14)
    assert got["920001.BJ"][4] is None


def test_instrument_info_skips_unparseable_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _pandas(
        pl.DataFrame(
            {
                "证券代码": ["900942", "600000"],  # 900942 是 B 股，无法归一化
                "证券简称": ["黄山B股", "浦发银行"],
                "上市日期": ["1996-11-22", "1999-11-10"],
            }
        )
    )
    empty = pd.DataFrame()
    monkeypatch.setattr(ak_source.ak, "stock_info_sh_name_code", lambda **_kw: bad)
    monkeypatch.setattr(ak_source.ak, "stock_info_sz_name_code", lambda **_kw: empty)
    monkeypatch.setattr(ak_source.ak, "stock_info_bj_name_code", lambda: empty)
    monkeypatch.setattr(ak_source.ak, "stock_info_sh_delist", lambda: empty)
    monkeypatch.setattr(ak_source.ak, "stock_info_sz_delist", lambda **_kw: empty)

    out = AkshareSource().instrument_info()
    assert out["instrument"].to_list() == ["600000.SH"]


# ---------------------------------------------------------------------------
# 重试与协议
# ---------------------------------------------------------------------------


def test_retry_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky(**_kw: object) -> object:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("临时失败")
        return _pandas(
            pl.DataFrame({"trade_date": [dt.date(2026, 9, 1)]})
        )

    monkeypatch.setattr(ak_source.ak, "tool_trade_date_hist_sina", flaky)
    out = AkshareSource().trade_calendar(dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    assert calls["n"] == 3
    assert out["is_open"].to_list() == [True]


def test_retry_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source.ak, "tool_trade_date_hist_sina", _raise)
    with pytest.raises(RuntimeError, match="重试"):
        AkshareSource().trade_calendar(dt.date(2026, 9, 1), dt.date(2026, 9, 1))


def test_satisfies_protocol() -> None:
    assert isinstance(AkshareSource(), DataSource)
# ---------------------------------------------------------------------------
# 东财熔断
# ---------------------------------------------------------------------------


def _sina_raw() -> object:
    return _pandas(
        pl.DataFrame(
            {
                "date": [dt.date(2026, 9, 1)],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "volume": [1000.0],
                "amount": [10_500.0],
            }
        )
    )


def _sina_factor() -> object:
    return _pandas(
        pl.DataFrame(
            {"date": [dt.date(2026, 9, 1)], "hfq_factor": [2.0]}
        )
    )


def test_em_circuit_breaker_skips_em_after_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    em_calls = {"n": 0}

    def counting_boom(**_kw: object) -> object:
        em_calls["n"] += 1
        raise RuntimeError("东财不可用")

    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", counting_boom)
    monkeypatch.setattr(
        ak_source.ak,
        "stock_zh_a_daily",
        lambda **kw: _sina_factor() if kw.get("adjust") == "hfq-factor" else _sina_raw(),
    )

    source = AkshareSource()
    instruments = [f"60000{i}.SH" for i in range(8)]
    out = source.daily_bars(instruments, dt.date(2026, 9, 1), dt.date(2026, 9, 1))

    assert out.height == 8
    # 熔断后不再请求东财：底层调用数 = 阈值 × 单票重试次数，而不是逐票持续请求
    assert em_calls["n"] == ak_source.EM_CIRCUIT_THRESHOLD * ak_source.RETRY_TIMES
    assert source._em_consecutive_failures == ak_source.EM_CIRCUIT_THRESHOLD


def test_em_circuit_breaker_resets_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    em_calls = {"n": 0}

    def flaky_em(**kw: object) -> object:
        em_calls["n"] += 1
        if em_calls["n"] <= 2:
            raise RuntimeError("东财抖动")
        return _em_raw() if kw["adjust"] == "" else _em_hfq()

    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", flaky_em)
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", _raise)

    source = AkshareSource()
    instruments = [f"60000{i}.SH" for i in range(3)]
    out = source.daily_bars(instruments, dt.date(2026, 9, 1), dt.date(2026, 9, 2))

    # 前两只走回退，第三只东财恢复后直连成功，熔断计数复位
    assert source._em_consecutive_failures == 0
    assert out.filter(pl.col("instrument") == "600002.SH").height == 2
# ---------------------------------------------------------------------------
# 沪深第三回退：腾讯兜底
# ---------------------------------------------------------------------------


def _tx_raw() -> object:
    return _pandas(
        pl.DataFrame(
            {
                "date": [dt.date(2026, 9, 1)],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "volume": [1000.0],
                "amount": [10_500.0],
            }
        )
    )


def _tx_hfq() -> object:
    return _pandas(
        pl.DataFrame(
            {
                "date": [dt.date(2026, 9, 1)],
                "close": [21.0],
            }
        )
    )


def test_daily_bars_falls_back_to_tx_when_em_and_sina_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", _boom_em)
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", _raise)
    monkeypatch.setattr(
        ak_source.ak,
        "stock_zh_a_hist_tx",
        lambda **kw: _tx_raw() if kw["adjust"] == "" else _tx_hfq(),
    )

    out = AkshareSource().daily_bars(["600519.SH"], dt.date(2026, 9, 1), dt.date(2026, 9, 1))

    check_schema(out, DAILY_BARS)
    assert out.height == 1
    assert out["adjfactor"].to_list() == [2.0]


def test_sina_circuit_breaker_skips_sina_after_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sina_calls = {"n": 0}

    def counting_boom(**_kw: object) -> object:
        sina_calls["n"] += 1
        raise RuntimeError("新浪被限流")

    monkeypatch.setattr(ak_source.ak, "stock_zh_a_hist", _boom_em)
    monkeypatch.setattr(ak_source.ak, "stock_zh_a_daily", counting_boom)
    monkeypatch.setattr(
        ak_source.ak,
        "stock_zh_a_hist_tx",
        lambda **kw: _tx_raw() if kw["adjust"] == "" else _tx_hfq(),
    )

    source = AkshareSource()
    instruments = [f"60001{i}.SH" for i in range(8)]
    out = source.daily_bars(instruments, dt.date(2026, 9, 1), dt.date(2026, 9, 1))

    assert out.height == 8
    # 东财与新浪都被熔断：新浪底层调用停在阈值，后续票直连腾讯
    assert sina_calls["n"] == ak_source.CIRCUIT_THRESHOLD * ak_source.RETRY_TIMES
