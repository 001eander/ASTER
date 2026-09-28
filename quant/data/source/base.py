"""DataSource 协议：上层只面向本协议编程，换数据源只新增一个实现类。

所有实现必须遵守 ``quant.data.schema`` 的表结构与口径约定
（单位、adjfactor 语义、停牌约定、排序约定）。
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataSource(Protocol):
    """行情与基础数据源的统一接口。"""

    def daily_bars(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        """日线行情，schema 见 ``schema.DAILY_BARS``。

        - 返回未复权价格；``adjfactor`` 为后复权因子。
        - ``vwap = amount / volume``，成交量为 0 的行 vwap 取 null。
        - ``limit_up`` / ``limit_down`` 本层不填（null），由预计算补。
        - 停牌日不产生行。
        - 输出按 (instrument, date) 排序。
        """
        ...

    def trade_calendar(self, start: date, end: date) -> pl.DataFrame:
        """交易日历，schema 见 ``schema.TRADE_CALENDAR``，按 date 排序。"""
        ...

    def corporate_actions(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        """分红送转（除权除息日生效），schema 见 ``schema.CORPORATE_ACTIONS``。"""
        ...

    def instrument_info(self) -> pl.DataFrame:
        """证券基本信息（含退市股），schema 见 ``schema.INSTRUMENT_INFO``。"""
        ...
