"""price_volume_corr_10：价量时序相关。

经济假设：个股自身复权收盘价与成交量在过去 10 日的时序相关，刻画「放量上涨 /
放量下跌」的配合关系。正相关表示量价同向，常对应趋势确认；负相关对应量价背离。
使用字段：close, adjfactor, volume。
窗口：10。
复权口径：close × adjfactor（后复权），避免除权跳空污染价格序列的相关结构；
成交量不复权。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 回看窗口（交易日）。
WINDOW: int = 10

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "volume", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """价量时序相关：``corr(adj_close, volume)`` 的 10 日滚动值（逐票）。

    输出 ``(date, instrument, value)``，前 9 个交易日为 null；窗口内价格或成交量
    无方差（相关系数无定义）时为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"price_volume_corr_10 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
        )
        .with_columns(
            pl.rolling_corr(
                pl.col("_adj_close"),
                pl.col("volume"),
                window_size=WINDOW,
                min_samples=WINDOW,
            )
            .over("instrument")
            .fill_nan(None)
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
