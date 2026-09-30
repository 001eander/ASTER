"""量：20 日成交量变异系数取负（量能稳定性做多）。

经济假设
    成交量是筹码换手的直接证据。过去 20 个交易日成交量波动剧烈（变异系数 CV 高）
    的股票，说明筹码在短线资金之间频繁易手、交易博弈激烈，持有者结构松散、抛压
    不可预期，市场对这个「不确定性」索取负的风险补偿；反过来，成交量长期平稳的
    股票筹码锁定良好、参与者结构稳定、潜在抛压可预期，未来一段时间的相对收益更高。
    因此把 20 日成交量变异系数取负作为信号：量能越稳定越看多（因子值越大越看多）。

    与库内量能因子的区别：库内已有的是「量能水平」口径（均量、换手代理、量比），
    本因子刻画的是同一水平之下的**离散度**——均量相同但忽大忽小的股票与涓涓细流
    的股票被区分开，量能水平本身在 CV 里被均值归一化掉了。

使用字段
    volume（以及键列 date、instrument）。不涉及任何价格比较。

窗口
    WINDOW = 20 个交易日滚动。CV = std(volume, 20) / mean(volume, 20)，
    value = -CV。窗口未填满（前 19 行）为 null，保持 null 不填 0；
    20 日均量为 0（长期停牌等）时同样记 null，避免除零产生 inf。

复权口径
    成交量以「股数」计，除权除息不改变当日成交股数的经济含义，无需复权，故只取
    volume 原值。本因子不接触任何价格字段，价格跳空不可能污染信号。

契约
    compute(data) 全程向量化：排序、rolling_mean / rolling_std 配 over("instrument")、
    逐元素四则运算，无循环、无 IO、无随机数；统计量只用滚动窗口，不使用任何全序列
    聚合，也只用当日及之前的数据（无 shift，更无负向 shift）。输出恰好
    (date, instrument, value) 三列，value 为 Float64，行序不作要求，不做截面标准化。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 成交量波动率的滚动窗口（交易日）。
WINDOW: int = 20

#: 预热期（交易日），供截断重算检测避开起步期（窗口 20 日，留出余量）。
WARMUP: int = 40

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "volume")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """计算 20 日成交量变异系数取负，返回 ``(date, instrument, value)`` 长表。

    :param data: 长表面板，至少含 :data:`REQUIRED_COLUMNS`，已按
        ``(instrument, date)`` 排序（函数内部重新排序以防漂移）。
    :return: 三列长表，``value = -std(volume, 20) / mean(volume, 20)``；
        窗口不足或 20 日均量为 0 时为 null，不填 0。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"vol_cv_20_neg 缺少输入列：{missing}")

    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            pl.col("volume")
            .rolling_mean(WINDOW, min_samples=WINDOW)
            .over("instrument")
            .alias("_vol_mean"),
            pl.col("volume")
            .rolling_std(WINDOW, min_samples=WINDOW)
            .over("instrument")
            .alias("_vol_std"),
        )
        .with_columns(
            pl.when(
                (pl.col("_vol_mean") > 0.0) & pl.col("_vol_std").is_finite()
            )
            .then(-(pl.col("_vol_std") / pl.col("_vol_mean")))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("value")
        )
        .select("date", "instrument", "value")
    )
