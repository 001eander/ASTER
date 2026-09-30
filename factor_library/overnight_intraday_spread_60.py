"""overnight_intraday_spread_60：隔夜与日内收益价差的 60 日标准化累计（收益分解）。

经济假设
    个股的日收益可以干净地拆成两段：隔夜分量 ``今开 / 昨收 - 1``（隔夜消息、公告、
    开盘集合竞价的挂单）与日内分量 ``今收 / 今开 - 1``（连续竞价里的价格发现）。两段
    背后是性质不同的资金——隔夜段缺少连续竞价的检验，容易留下开盘情绪的溢价；日内段
    是真金白银换手出来的结果。本因子取两者的**价差**在 60 个交易日上的标准化累计：

        value = [sum60(隔夜_t - 日内_t)] / std60(隔夜_t - 日内_t)

    价差高（反复高开、盘中却回吐）的股票，开盘溢价一次次被日内抛压否定；价差低
    （反复低开、盘中收复）的股票，开盘抛压被日内买盘接走。全市场日线样本
    （2021-09-29 起）给出的结论是：日内分量反转（当日涨得多的后续跑输）、隔夜分量
    弱延续，二者的价差因此同时吃两种效应，方向为「价差越大越看多」。因子值越大越看多。

    为什么是 60 日而不是 20 日
        同口径的 20 日版本（``overnight_intraday_spread_20``）在全样本上 rank_ic≈0.041、
        分层单调、共线性 0.46，四项门槛过其三，唯一卡点是 ICIR≈0.32（日频 RankIC 的
        mean/std，不年化）。瓶颈在分母：20 日的价差估计太短，横截面名次被短期 regime
        噪声反复重排，IC 日间波动大。把聚合窗口拉长到 60 日，估计量更稳、个股排名更
        粘，用同一信号换更低的 IC 波动。除以滚动标准差相当于对该价差的 60 日 t 值再乘
        ``sqrt(60)``：同样 5 个百分点的累计价差，日常波动大的票信息量更低，这里把
        横截面的尺度差异抹平。

    与库内同方向因子的机制区别
        库内 price 系列因子都在同一条价格序列的端点上做文章：``mom_5`` / ``mom_20``
        比收盘对收盘，``reversal_5`` 比开盘对开盘，``ma_bias_*`` 比收盘对均线，
        ``vwap_bias`` 比收盘对当日 VWAP，``volatility_20`` / ``range_pct`` 度量波动与
        振幅。它们用的都是「日与日之间的总收益」或「日内的整体振幅」；本因子做的是
        收益率内部的分解，把同一天拆成隔夜段与日内段的对比，总收益本身被消掉，度量
        的是「开盘时点的定价相对盘中定价的偏离」。库内无任何因子使用隔夜 / 日内分解
        口径。

使用字段：open, close, adjfactor（外加键列 date、instrument）。
窗口：单日分解，WINDOW = 60 个交易日滚动求和与滚动标准差；``shift(1)`` 与滚动窗口
    不足处为 null，保持 null，不填充。
复权口径：后复权。隔夜分量 = (open_t × adjfactor_t) / (close_{t-1} × adjfactor_{t-1}) - 1，
    日内分量 = (close_t × adjfactor_t) / (open_t × adjfactor_t) - 1；除权除息日的
    未复权价跳空被 adjfactor 抵消，不进入因子。

数据质量兜底：A 股涨跌幅限制为 ±10% / ±20% / ±30%，复权后有据可依的单日隔夜跳空或
    日内振幅不会超过 50%；超界的点只可能来自复权因子接缝一类的脏数据，这类行记 null、
    不进入累计与波动，避免脏数据霸占截面两端的名次。横截面去均值只用当日截面，
    属当日信息，无前视。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 隔夜 / 日内价差的累计窗口（交易日）。
WINDOW: int = 60

#: 回看昨收的滞后阶数，只允许正数。
PREV_LAG: int = 1

#: 单日分量的合理上界：隔夜跳空或日内振幅的绝对值超过它即视为脏数据，记 null。
MAX_ABS_LEG: float = 0.5

#: 预热交易日数：1 阶滞后 + 60 日滚动窗口，第 62 行起才有首个有效值，留余量取 70。
WARMUP: int = 70

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "open",
    "close",
    "adjfactor",
)


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """``sum60(隔夜 - 日内) / std60(隔夜 - 日内)``，值越大越看多。

    输入按契约含 10 列行情面板，输出恰好 ``(date, instrument, value)`` 三列，
    ``value`` 为 Float64；窗口不足处为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"overnight_intraday_spread_60 缺少输入列：{missing}")

    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("open") * pl.col("adjfactor")).alias("_adj_open"),
            (pl.col("close") * pl.col("adjfactor")).alias("_adj_close"),
        )
        .with_columns(
            (
                pl.col("_adj_open")
                / pl.col("_adj_close").shift(PREV_LAG).over("instrument")
                - 1.0
            ).alias("_overnight"),
            (pl.col("_adj_close") / pl.col("_adj_open") - 1.0).alias("_intraday"),
        )
        # 脏数据兜底：两条腿任一出界即整行作废（记 null，不进入累计与波动）。
        .with_columns(
            pl.when(
                (pl.col("_overnight").abs() <= MAX_ABS_LEG)
                & (pl.col("_intraday").abs() <= MAX_ABS_LEG)
            )
            .then(pl.col("_overnight") - pl.col("_intraday"))
            .otherwise(None)
            .alias("_spread")
        )
        # 横截面去均值：只看个股相对当日全市场的偏离，当日信息，无前视。
        .with_columns(
            (
                pl.col("_spread") - pl.col("_spread").mean().over("date")
            ).alias("_spread")
        )
        .with_columns(
            pl.col("_spread")
            .rolling_sum(WINDOW, min_samples=WINDOW)
            .over("instrument")
            .alias("_sum"),
            pl.col("_spread")
            .rolling_std(WINDOW, min_samples=WINDOW)
            .over("instrument")
            .alias("_std"),
        )
        .with_columns(
            pl.when(pl.col("_std") > 0)
            .then(pl.col("_sum") / pl.col("_std"))
            .otherwise(None)
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
