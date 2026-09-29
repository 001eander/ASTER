"""统一字段定义：全项目唯一的行情 / 日历 / 公司行为 / 证券信息 schema。

约定：
- 证券代码统一格式 ``600000.SH`` / ``000001.SZ`` / ``920001.BJ``（六位数字 + 交易所后缀）。
- 价格单位元，成交量单位股（不是手），成交额单位元。
- ``adjfactor`` 为后复权因子：``后复权价 = 未复权价 × adjfactor``。
  后复权因子对历史日期不随新的公司行为变化，适合增量更新。
- 停牌约定：交易日当天无该证券的 K 线行即视为停牌，不补空行。
"""
from __future__ import annotations

import re
from datetime import date
from typing import Literal

import polars as pl

Board = Literal["main", "cyb", "kcb", "bj"]

#: 全系统历史起点 = CSMAR 日线建库窗口首日（issue #59）。
#: 该日之前的 akshare 缓存弃用，日期序列型计算（回测、标签、因子）一律从此日起。
HISTORY_START: date = date(2021, 9, 29)

# ---------------------------------------------------------------------------
# 证券代码
# ---------------------------------------------------------------------------

INSTRUMENT_RE = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")


def normalize_instrument(code: str) -> str:
    """把 akshare 风格的六位代码归一化为 ``600000.SH`` 形式。

    已带后缀的输入原样返回（大小写不敏感）。
    """
    code = code.strip().upper()
    if INSTRUMENT_RE.match(code):
        return code
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"无法识别的证券代码: {code!r}")
    return f"{code}.{exchange_of_digits(code)}"


def exchange_of_digits(digits: str) -> Literal["SH", "SZ", "BJ"]:
    """按六位代码段判定交易所。"""
    head3 = digits[:3]
    if head3 in {"600", "601", "603", "605", "688", "689"}:
        return "SH"
    if head3 in {"000", "001", "002", "003", "300", "301", "302"}:
        return "SZ"
    if head3.startswith(("4", "8")) or head3 == "920":
        return "BJ"
    raise ValueError(f"无法判定交易所的代码段: {digits!r}")


def board_of(instrument: str) -> Board:
    """按证券代码判定板块：主板 / 创业板 / 科创板 / 北交所。"""
    instrument = normalize_instrument(instrument)
    digits, exchange = instrument.split(".")
    head3 = digits[:3]
    if exchange == "BJ":
        return "bj"
    if head3 in {"688", "689"}:
        return "kcb"
    if head3 in {"300", "301", "302"}:
        return "cyb"
    return "main"


# ---------------------------------------------------------------------------
# 表 schema
# ---------------------------------------------------------------------------

#: 日线行情表。factor_api 的因子输入即此表的前 10 列；
#: limit_up / limit_down 由涨跌停预计算（issue #5）填充，抓取阶段为 null。
DAILY_BARS = pl.Schema(
    {
        "date": pl.Date,
        "instrument": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "vwap": pl.Float64,
        "volume": pl.Float64,
        "amount": pl.Float64,
        "adjfactor": pl.Float64,
        "limit_up": pl.Float64,
        "limit_down": pl.Float64,
    }
)

#: 交易日历表。is_open 覆盖周末与节假日，便于对齐任意区间。
TRADE_CALENDAR = pl.Schema(
    {
        "date": pl.Date,
        "is_open": pl.Boolean,
    }
)

#: 公司行为表（分红送转）。cash_per_share 为税前每股派息（元），
#: share_per_share 为每股送转股数；同一天同一证券可能同时有两类，合并为一行。
CORPORATE_ACTIONS = pl.Schema(
    {
        "date": pl.Date,
        "instrument": pl.String,
        "cash_per_share": pl.Float64,
        "share_per_share": pl.Float64,
    }
)

#: 指数日线表（issue #67）。``index_code`` 为六位数字、不带交易所后缀，
#: 与证券代码空间不混淆。用于指数增强 / 超额绩效的基准。
INDEX_BARS = pl.Schema(
    {
        "date": pl.Date,
        "index_code": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }
)

#: 落地的基准指数集合：沪深 300 / 中证 500 / 中证 1000 / 中证 2000。
INDEX_CODES: tuple[str, ...] = ("000300", "000905", "000852", "932000")

#: 证券信息表。ST 状态随时间变化，不放在此表，由涨跌停预计算按日期区间处理。
INSTRUMENT_INFO = pl.Schema(
    {
        "instrument": pl.String,
        "name": pl.String,
        "board": pl.String,
        "list_date": pl.Date,
        "delist_date": pl.Date,
    }
)

#: 行业分类表（东财口径，issue #65）。``effective_from`` 为该归属的抓取日：
#: 东财行业分类调整频率低，akshare / 东财只提供当前截面，历史段按当前截面回填，
#: 因此该列是「已知时点」而不是「分类真正生效日」（与成分股 PIT 的严格性不同）。
#: 同一 ``instrument`` 可以有多行，读取方取 ``effective_from`` 最大的一行。
INDUSTRY = pl.Schema(
    {
        "instrument": pl.String,
        "industry_l1": pl.String,
        "industry_l2": pl.String,
        "effective_from": pl.Date,
    }
)


class SchemaError(ValueError):
    """数据表与约定 schema 不符。"""


def check_schema(df: pl.DataFrame, schema: pl.Schema, *, name: str = "table") -> None:
    """校验列名、顺序与 dtype，不符即抛 SchemaError。"""
    actual = df.schema
    expected_names = list(schema.keys())
    if list(actual.keys()) != expected_names:
        raise SchemaError(
            f"{name} 列不符: 期望 {expected_names}, 实际 {list(actual.keys())}"
        )
    mismatched = {
        col: (str(schema[col]), str(actual[col]))
        for col in expected_names
        if actual[col] != schema[col]
    }
    if mismatched:
        raise SchemaError(f"{name} dtype 不符: {mismatched}")


def check_daily_bars(df: pl.DataFrame, *, name: str = "daily_bars") -> None:
    """日线表的额外约定：按 (instrument, date) 排序，同键不重复。"""
    check_schema(df, DAILY_BARS, name=name)
    key = ["instrument", "date"]
    if df.select(key).is_duplicated().any():
        raise SchemaError(f"{name} 存在重复的 (instrument, date)")
    if not df.equals(df.sort(key)):
        raise SchemaError(f"{name} 未按 (instrument, date) 排序")


def check_index_bars(df: pl.DataFrame, *, name: str = "index_bars") -> None:
    """指数日线表的额外约定：按 (index_code, date) 排序，同键不重复。"""
    check_schema(df, INDEX_BARS, name=name)
    key = ["index_code", "date"]
    if df.select(key).is_duplicated().any():
        raise SchemaError(f"{name} 存在重复的 (index_code, date)")
    if not df.equals(df.sort(key)):
        raise SchemaError(f"{name} 未按 (index_code, date) 排序")


def check_industry(df: pl.DataFrame, *, name: str = "industry") -> None:
    """行业表的额外约定：按 (instrument, effective_from) 排序，同键不重复。"""
    check_schema(df, INDUSTRY, name=name)
    key = ["instrument", "effective_from"]
    if df.select(key).is_duplicated().any():
        raise SchemaError(f"{name} 存在重复的 (instrument, effective_from)")
    if not df.equals(df.sort(key)):
        raise SchemaError(f"{name} 未按 (instrument, effective_from) 排序")
