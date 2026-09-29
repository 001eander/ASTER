"""factor_api 接口规范：因子输入输出的 schema 与契约。

一个因子是一个 ``.py`` 文件，实现 ``compute(data: pl.DataFrame) -> pl.DataFrame``。
本模块集中定义该契约的常量与文档，后续 issue #18（截断重算检测）、
#19（复杂度度量）、#20（因子评估管线）都以此为公共基础。

输入
    长表面板数据，列固定为 :data:`quant.data.schema.DAILY_BARS` 的前 10 列：
    ``date, instrument, open, high, low, close, vwap, volume, amount, adjfactor``，
    已按 ``(instrument, date)`` 排序。价格未复权，复权请用 ``adjfactor`` 自行处理
    （``后复权价 = 未复权价 × adjfactor``）。

输出
    ``date, instrument, value`` 三列长表，缺失用 null，不做截面标准化。
    行序不作要求，下游用 join 对齐，因此因子无须对输出排序。

三条禁令
    1. 禁止网络与文件 IO：因子只允许基于传入的 DataFrame 计算。
    2. 禁止随机性：同一输入必须得到同一输出，结果可复现。
    3. 禁止使用 cutoff 之后的数据：只能用当日及之前的历史。
"""
from __future__ import annotations

from typing import Callable

import polars as pl

from quant.data.schema import DAILY_BARS

#: 因子输入列（保持 DAILY_BARS 的列序），即 DAILY_BARS 的前 10 列。
FACTOR_INPUT_COLUMNS: tuple[str, ...] = tuple(DAILY_BARS.keys())[:10]

#: 因子输入的完整 schema。
FACTOR_INPUT_SCHEMA: pl.Schema = pl.Schema(
    {name: DAILY_BARS[name] for name in FACTOR_INPUT_COLUMNS}
)

#: 因子输出 schema：``(date, instrument, value)``，value 为 Float64，缺失用 null。
FACTOR_OUTPUT_SCHEMA: pl.Schema = pl.Schema(
    {
        "date": pl.Date,
        "instrument": pl.String,
        "value": pl.Float64,
    }
)

#: 因子输出的键列，用于去重校验。
FACTOR_OUTPUT_KEY_COLUMNS: tuple[str, ...] = ("date", "instrument")

#: 因子契约的函数类型。
FactorCompute = Callable[[pl.DataFrame], pl.DataFrame]
