"""``quant.data.source.csmar`` 单元测试。

用 ``tmp_path`` 合成迷你 CSMAR 目录（若干小 csv），不依赖本机真实解压包。
覆盖字段映射、Markettype 过滤、交易所交叉校验、逐日复权因子、``factor_scale``、
日历并集、``instrument_info`` 合并与 schema 校验。
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    HISTORY_START,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    check_daily_bars,
    check_schema,
)
from quant.data.source.base import DataSource
from quant.data.source.csmar import CsmarSource

# ---------------------------------------------------------------------------
# 迷你 CSMAR 夹具
# ---------------------------------------------------------------------------

_DALYR_HEADER = [
    "Stkcd", "Trddt", "Opnprc", "Hiprc", "Loprc", "Clsprc", "Dnshrtrd",
    "Dnvaltrd", "Markettype", "Trdsta", "LimitUp", "LimitDown", "PreClosePrice",
]


def _write_csv(
    path: Path, header: list[str], rows: list[list[str]], *, bom: bool = False
) -> None:
    encoding = "utf-8-sig" if bom else "utf-8"
    lines = [",".join(f'"{cell}"' for cell in header)]
    lines += [",".join(f'"{cell}"' for cell in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding=encoding)


def _dalyr_rows() -> list[list[str]]:
    """日线样本：含正常票、停牌零量行、B 股行、代码段与 Markettype 矛盾行。"""
    return [
        # 600519.SH：9/30 零成交应被剔除，10/08 保留
        ["600519", "2021-09-29", "10.0", "11.0", "9.0", "10.5", "1000", "10500",
         "1", "1", "11.55", "9.45", "10.40"],
        ["600519", "2021-09-30", "10.5", "11.0", "10.0", "10.8", "0", "0",
         "1", "1", "11.55", "9.45", "10.50"],
        ["600519", "2021-10-08", "20.0", "22.0", "19.0", "21.0", "2000", "42000",
         "1", "1", "23.10", "18.90", "21.00"],
        # 000001.SZ
        ["000001", "2021-09-29", "10.0", "11.0", "9.0", "10.5", "500", "5250",
         "4", "1", "11.55", "9.45", "10.40"],
        ["000001", "2021-09-30", "10.6", "11.0", "10.2", "10.6", "500", "5300",
         "4", "1", "11.55", "9.45", "10.50"],
        ["000001", "2021-10-08", "10.6", "11.0", "10.0", "10.6", "600", "6360",
         "4", "1", "11.66", "9.54", "10.60"],
        # 300750.SZ：完全无复权事件
        ["300750", "2021-09-29", "10.0", "11.0", "9.0", "10.5", "300", "3150",
         "4", "1", "11.55", "9.45", "10.40"],
        # 600001 代码段判沪、Markettype=4 判深 → 以 Markettype 为准
        ["600001", "2021-09-29", "10.0", "11.0", "9.0", "10.0", "100", "1000",
         "4", "1", "11.00", "9.00", "10.00"],
        # B 股（Markettype 2/8）应被剔除
        ["900901", "2021-09-29", "1.0", "1.1", "0.9", "1.0", "100", "100",
         "2", "1", "1.10", "0.90", "1.00"],
        ["200001", "2021-09-29", "1.0", "1.1", "0.9", "1.0", "100", "100",
         "8", "1", "1.10", "0.90", "1.00"],
    ]


def _adjust_rows() -> list[list[str]]:
    return [
        # 600519 首事件晚于 HISTORY_START：基线 = 2.0 / 1.25 = 1.6
        ["2021-10-01", "600519", "0.8", "1.25", "1.6", "2.0"],
        ["2022-01-01", "600519", "0.8", "1.25", "2.0", "2.5"],
        # 000001 首事件早于 HISTORY_START：无需基线
        ["2021-05-14", "000001", "0.992198", "1.007864", "0.781647", "151.523527"],
        ["2021-10-05", "000001", "0.947", "1.0559", "0.78", "160.0"],
    ]


def _cale_rows() -> list[list[str]]:
    return [
        # 1 与 4 市场状态相反，并集为开市；2 市场应被忽略
        ["1", "2021-09-28", "2", "O"],
        ["1", "2021-09-29", "3", "O"],
        ["1", "2021-09-30", "4", "C"],
        ["1", "2021-10-01", "5", "C"],
        ["4", "2021-09-29", "3", "C"],
        ["4", "2021-09-30", "4", "O"],
        ["4", "2021-10-01", "5", "C"],
        ["2", "2021-09-29", "3", "O"],
    ]


def _co_rows() -> list[list[str]]:
    return [
        # 与基础表冲突（list_date 不同）
        ["600519", "贵州茅台", "2001-08-27", "A", ""],
        # 基础表没有的新票
        ["300750", "宁德时代", "2018-06-11", "A", ""],
        ["920001", "纬达光电", "2022-12-27", "A", ""],
        # 退市票：Statco=D 取 Statdt 为 delist_date
        ["600002", "退市示例", "1998-01-22", "D", "2009-12-29"],
        # B 股代码无法归一化，应跳过
        ["900901", "黄山B股", "1996-11-22", "A", ""],
    ]


def _base_info(tmp_path: Path) -> Path:
    path = tmp_path / "instruments.parquet"
    pl.DataFrame(
        [
            {
                "instrument": "600519.SH",
                "name": "贵州茅台",
                "board": "main",
                "list_date": date(2001, 8, 28),
                "delist_date": None,
            },
            {
                "instrument": "000001.SZ",
                "name": "平安银行",
                "board": "main",
                "list_date": date(1991, 4, 3),
                "delist_date": None,
            },
        ],
        schema=INSTRUMENT_INFO,
    ).write_parquet(path)
    return path


@pytest.fixture
def csmar_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "csmar"
    directory.mkdir()
    _write_csv(directory / "TRD_Dalyr.csv", _DALYR_HEADER, _dalyr_rows(), bom=True)
    _write_csv(
        directory / "TRD_AdjustFactor.csv",
        ["TradingDate", "Symbol", "FwardFactor", "BwardFactor",
         "CumulateFwardFactor", "CumulateBwardFactor"],
        _adjust_rows(),
    )
    _write_csv(
        directory / "TRD_Cale.csv",
        ["Markettype", "Clddt", "Daywk", "State"],
        _cale_rows(),
    )
    _write_csv(
        directory / "TRD_Co.csv",
        ["Stkcd", "Stknme", "Listdt", "Statco", "Statdt"],
        _co_rows(),
    )
    return directory


@pytest.fixture
def base_info(tmp_path: Path) -> Path:
    return _base_info(tmp_path)


# ---------------------------------------------------------------------------
# 字段映射 / 过滤
# ---------------------------------------------------------------------------


def test_daily_bars_maps_units_and_drops_zero_volume(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).daily_bars(
        ["600519.SH"], HISTORY_START, date(2021, 10, 8)
    )
    check_daily_bars(out)
    # 零成交的 9/30 被剔除，只剩 9/29 与 10/08
    assert out["date"].to_list() == [date(2021, 9, 29), date(2021, 10, 8)]
    first = out.row(0, named=True)
    assert first["volume"] == 1000.0  # 股
    assert first["amount"] == 10500.0  # 元
    assert first["vwap"] == pytest.approx(10.5)  # amount / volume
    assert first["limit_up"] is None
    assert first["limit_down"] is None


def test_daily_bars_filters_markettype(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).daily_bars(
        ["900901.SZ", "200001.SZ"], HISTORY_START, date(2021, 10, 8)
    )
    assert out.height == 0
    check_daily_bars(out)


def test_daily_bars_exchange_cross_check_uses_markettype(
    csmar_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="quant.data.source.csmar")
    out = CsmarSource(csmar_dir).daily_bars(
        ["600001.SZ"], HISTORY_START, date(2021, 10, 8)
    )
    # 代码段判沪、Markettype 判深：以 Markettype 为准归为 600001.SZ
    assert out["instrument"].to_list() == ["600001.SZ"]
    assert any("交叉校验" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 逐日复权因子
# ---------------------------------------------------------------------------


def test_adjfactor_baseline_and_event_switch(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).daily_bars(
        ["600519.SH"], HISTORY_START, date(2021, 10, 8)
    )
    factors = dict(zip(out["date"].to_list(), out["adjfactor"].to_list()))
    # 首事件 2021-10-01 晚于起点 → 反推基线 2.0 / 1.25
    assert factors[date(2021, 9, 29)] == pytest.approx(1.6)
    # 事件日当天切换为新因子（第二个事件 2022-01-01 不生效）
    assert factors[date(2021, 10, 8)] == pytest.approx(2.0)


def test_adjfactor_forward_fill_between_events(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).daily_bars(
        ["000001.SZ"], HISTORY_START, date(2021, 10, 8)
    )
    factors = dict(zip(out["date"].to_list(), out["adjfactor"].to_list()))
    # 首事件早于起点：窗口内直接用 2021-05-14 的事件因子，不做基线反推
    assert factors[date(2021, 9, 29)] == pytest.approx(151.523527)
    assert factors[date(2021, 9, 30)] == pytest.approx(151.523527)
    # 10-05 事件在 10-08 前向填充
    assert factors[date(2021, 10, 8)] == pytest.approx(160.0)


def test_adjfactor_defaults_to_one_without_events(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).daily_bars(
        ["300750.SZ"], HISTORY_START, date(2021, 10, 8)
    )
    assert out["adjfactor"].to_list() == [1.0]


def test_factor_scale_applied(csmar_dir: Path) -> None:
    source = CsmarSource(csmar_dir, factor_scale={"600519.SH": 2.0})
    out = source.daily_bars(["600519.SH", "300750.SZ"], HISTORY_START, date(2021, 10, 8))
    scaled = out.filter(pl.col("instrument") == "600519.SH")["adjfactor"].to_list()
    unscaled = out.filter(pl.col("instrument") == "300750.SZ")["adjfactor"].to_list()
    assert scaled[0] == pytest.approx(3.2)  # 1.6 × 2
    assert unscaled == [1.0]


def test_max_date_and_limit_reference(csmar_dir: Path) -> None:
    source = CsmarSource(csmar_dir)
    assert source.max_date() == date(2021, 10, 8)
    ref = source.limit_reference()
    assert set(ref.columns) == {
        "date", "instrument", "trdsta", "ref_limit_up", "ref_limit_down",
        "ref_pre_close",
    }
    row = ref.filter(
        (pl.col("instrument") == "600519.SH") & (pl.col("date") == date(2021, 9, 29))
    ).row(0, named=True)
    assert row["trdsta"] == 1
    assert row["ref_limit_up"] == pytest.approx(11.55)
    assert row["ref_pre_close"] == pytest.approx(10.40)


# ---------------------------------------------------------------------------
# 交易日历
# ---------------------------------------------------------------------------


def test_trade_calendar_union_and_history_start(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).trade_calendar(date(2021, 9, 28), date(2021, 10, 1))
    check_schema(out, TRADE_CALENDAR)
    # 9/28 早于 HISTORY_START 被裁掉；两市场状态取并集；10/1 全休市
    assert out.rows() == [
        (date(2021, 9, 29), True),
        (date(2021, 9, 30), True),
        (date(2021, 10, 1), False),
    ]


def test_trade_calendar_clips_range(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).trade_calendar(date(2021, 9, 30), date(2021, 9, 30))
    assert out.rows() == [(date(2021, 9, 30), True)]


# ---------------------------------------------------------------------------
# 证券信息
# ---------------------------------------------------------------------------


def test_instrument_info_merges_base_and_csmar(
    csmar_dir: Path, base_info: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="quant.data.source.csmar")
    out = CsmarSource(csmar_dir, base_info=base_info).instrument_info()
    check_schema(out, INSTRUMENT_INFO)
    got = {row[0]: row for row in out.rows()}
    # 基础表保留且冲突不覆盖
    assert got["600519.SH"][3] == date(2001, 8, 28)
    assert any("冲突" in record.message for record in caplog.records)
    # CSMAR 补入新票
    assert got["300750.SZ"][1] == "宁德时代"
    assert got["920001.BJ"][2] == "bj"
    assert got["600002.SH"][4] == date(2009, 12, 29)
    # 无法归一化的 B 股被跳过
    assert "900901.SZ" not in got


def test_instrument_info_without_base_only_csmar(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).instrument_info()
    check_schema(out, INSTRUMENT_INFO)
    assert out["instrument"].to_list() == [
        "300750.SZ", "600002.SH", "600519.SH", "920001.BJ"
    ]


# ---------------------------------------------------------------------------
# 公司行为 / 协议
# ---------------------------------------------------------------------------


def test_corporate_actions_empty(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).corporate_actions(
        ["600519.SH"], HISTORY_START, date(2021, 10, 8)
    )
    assert out.height == 0
    check_schema(out, CORPORATE_ACTIONS)


def test_daily_bars_satisfies_schema(csmar_dir: Path) -> None:
    out = CsmarSource(csmar_dir).daily_bars(
        ["600519.SH", "000001.SZ"], HISTORY_START, date(2021, 10, 8)
    )
    check_schema(out, DAILY_BARS)
    check_daily_bars(out)


def test_satisfies_protocol(csmar_dir: Path) -> None:
    assert isinstance(CsmarSource(csmar_dir), DataSource)
