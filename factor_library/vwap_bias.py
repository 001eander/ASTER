"""vwap_bias：收盘对 VWAP 的偏离。

经济假设：收盘价高于当日成交量加权均价（VWAP）说明尾盘买盘占优，日内资金
净流入；低于 VWAP 则相反。作为最短周期的量价强弱信号。
使用字段：close, vwap。
窗口：无（当日截面）。
复权口径：同一交易日内复权因子在比值中约去，未复权价与复权价结果相同。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "vwap")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """收盘对 VWAP 的偏离：``close / vwap - 1``。

    输出 ``(date, instrument, value)``，无窗口预热期；``vwap`` 非正时为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"vwap_bias 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            pl.when(pl.col("vwap") > 0)
            .then(pl.col("close") / pl.col("vwap") - 1.0)
            .otherwise(None)
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
