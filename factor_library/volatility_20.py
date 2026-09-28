"""volatility_20：20 日已实现波动率。

经济假设：短期日收益标准差衡量价格波动水平，高波动常对应情绪不稳与风险溢价。
作为低波动异象的基线因子：波动越高，风险调整后收益往往越差。
使用字段：close, adjfactor。
窗口：20（在 20 个日收益上计算，需 21 个收盘价）。
复权口径：close × adjfactor（后复权），日收益对除权除息不敏感。未年化。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 回看窗口（交易日收益数）。
WINDOW: int = 20

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日日收益标准差（未年化）。

    日收益 ``ret(T) = adj_close(T) / adj_close(T-1) - 1``，输出
    ``(date, instrument, value)``，前 20 个交易日为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"volatility_20 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
        )
        .with_columns(
            (
                pl.col("_adj_close")
                / pl.col("_adj_close").shift(1).over("instrument")
                - 1.0
            ).alias("_ret")
        )
        .with_columns(
            pl.col("_ret")
            .rolling_std(WINDOW, min_samples=WINDOW)
            .over("instrument")
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
