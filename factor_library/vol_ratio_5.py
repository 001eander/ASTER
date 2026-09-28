"""vol_ratio_5：短长期均量比。

经济假设：短期（5 日）均量相对长期（60 日）均量放大，说明资金关注度抬升，
常伴随趋势启动或加速。量能类因子无需复权。
使用字段：volume。
窗口：5 与 60。
复权口径：成交量不复权；换手相关口径按股数原值计算。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 短期窗口（交易日）。
SHORT_WINDOW: int = 5

#: 长期窗口（交易日）。
LONG_WINDOW: int = 60

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "volume")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """短长期均量比：``mean(volume, 5) / mean(volume, 60)``。

    输出 ``(date, instrument, value)``，长期窗口未填满（前 59 日）为 null；
    长期均量为 0 或 null 时为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"vol_ratio_5 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            pl.col("volume")
            .rolling_mean(SHORT_WINDOW, min_samples=SHORT_WINDOW)
            .over("instrument")
            .alias("_vol_short"),
            pl.col("volume")
            .rolling_mean(LONG_WINDOW, min_samples=LONG_WINDOW)
            .over("instrument")
            .alias("_vol_long"),
        )
        .with_columns(
            pl.when(pl.col("_vol_long") > 0)
            .then(pl.col("_vol_short") / pl.col("_vol_long"))
            .otherwise(None)
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
