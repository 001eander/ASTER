"""量能：5 日均量相对 20 日均量取反，缩量看多、放量看空。

经济假设
    放量往往伴随情绪过热与短期超买，后续收益回落；缩量说明抛压衰竭、筹码稳定，
    后续相对占优。因此短期相对放量的股票看空，外层取负号后缩量股得分高。
    该机制是纯量能维度，不依赖价格方向，与短期反转因子机制正交。

使用字段
    volume（成交量，单位股）。日内的比值口径与复权因子无关，成交量本身不做复权。

窗口
    ``SHORT_WINDOW = 5``、``LONG_WINDOW = 20``，``WARMUP = 40`` 覆盖长窗铺满所需。

复权口径
    纯量能因子，不涉及价格比较与收益计算，故不使用 adjfactor。

构造
    ``vol_s = mean(volume, 5)``，``vol_l = mean(volume, 20)``，
    ``value = -(vol_s / vol_l)``。两个滚动均值都按 ``.over("instrument")`` 分组、
    基于已按 ``(instrument, date)`` 排序的输入计算，只使用当日及之前的数据。
    长窗未铺满处自然为 null；``vol_l`` 非正处同样输出 null，不填 0。
    输出恰好 ``(date, instrument, value)`` 三列，Float64，不做截面标准化，
    不留中间列。全程向量化，无循环、无 IO、无随机。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 短期均量窗口（交易日）。
SHORT_WINDOW: int = 5

#: 长期均量窗口（交易日），也是窗口不足判 null 的依据。
LONG_WINDOW: int = 20

#: 预热期：长窗前 19 个交易日无值，取 40 留出余量。
WARMUP: int = 40

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "volume")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """短长期均量比取反：``-mean(volume, 5) / mean(volume, 20)``。

    值越大越看多（缩量股得分高）。窗口未铺满或长期均量非正处为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"因子缺少输入列：{missing}")

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
            .then(-(pl.col("_vol_short") / pl.col("_vol_long")))
            .otherwise(None)
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
