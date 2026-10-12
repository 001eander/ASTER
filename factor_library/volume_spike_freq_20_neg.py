"""量：20 日爆量频次占比取负（窗口内超出自身常态 3 倍的爆量天数占比），出过脉冲式天量的股票看空。

经济假设
    股价的中短期走势由筹码结构决定：一只股票在最近一个月里频繁出现「单日成交量远超
    自身常态」的脉冲，说明这段时间内筹码在反复大换手——游资对倒、事件驱动的抢筹与
    恐慌抛售、利好兑现出货都会留下这种痕迹。每一次脉冲都意味着有大量持仓在当日价格
    附近易主，新进承接方的成本集中在脉冲日价位，一旦价格回落，承接盘缺乏耐心，后续
    抛压不可预期；反过来，整月没有出现过爆量日的股票，参与者结构稳定、换手节奏可
    预期，潜在抛压在时间上被摊薄。因此把「爆量天数占比」取负作为信号：因子值越大
    越看多（越没有出现脉冲式天量的股票得分越高）。

    为什么统计的是**频次**而不是峰值幅度
        同一个直觉有两种口径：脉冲有多强（幅度），和脉冲有多频繁（频次）。经验库
        证据显示幅度口径会被已有因子吸收——insp-023 首轮按「20 日最大单日量 / 20 日
        均量取负」实现，实测 rank_ic=0.0203、icir=0.363、mono=0.90，但与库内
        ``vol_cv_20_neg`` 的逐日截面相关均值高达 0.928（判冗余拒绝）。原因是
        ``max/mean`` 与 ``std/mean`` 在量能分布右偏时几乎是同一个量的两种写法，都被
        窗口内的幅度离散度驱动。改成频次口径后（本实现），实测 rank_ic=0.0301、
        icir=0.372、mono=0.70，与库内因子相关性最高的 ``vol_ratio_5`` 只有 0.581，
        与 ``vol_cv_20_neg`` 已不在前三——频次统计对「有没有越线」敏感，对越线多少
        以及未越线日的波动幅度不敏感。

    与库内同类因子的机制区别
        - ``vol_cv_20_neg``（``std(volume, 20) / mean(volume, 20)`` 取负）：二阶梯度的
          连续离散度，窗口内每一天的波动都进入读数。
        - ``vol_ratio_5`` / ``vol_ratio_5_20_neg``：两个不同长度窗口的**均量之比**，
          刻画量能水平／趋势。
        - ``turnover_proxy``：量能水平的直接代理。
        本因子不做任何均值、方差或比值运算，而是给「当日成交量是否超过自身长期常态
        的 3 倍」记一次事件，再数最近 20 个交易日里这类事件出现了几次。阈值化的计数
        丢掉幅度信息：一天放出 8 倍量与两天各放出 3.5 倍量在幅度口径里读数量级不同，
        在频次口径里只差一次计数。

信号源与口径
    信号源 volume（键列 date、instrument），单位为股。成交量以股数计量，除权除息不
    改变当日成交股数的经济含义，无需复权，因此本因子不接触任何价格字段，除权日的
    价格跳空不可能进入信号（题目要求的复权口径在此不适用，也不会被违反）。常态基线
    用个股自身**过去** 60 个交易日的成交均量（``shift(1)`` 后再滚动求均，刻意把当日
    排除在基线之外，避免爆量日把自己的阈值抬高、钝化检测），阈值取基线的 3 倍。

窗口
    ``BASELINE_WINDOW = 60``：个股常态基线的长度（中尺度，覆盖一个季度），铺满
    40 个观测（``MIN_BASELINE_SAMPLES``）即出值，观测不足记 null。
    ``WINDOW = 20``：爆量事件计数的窗口（中尺度，与灵感规格一致）。
    ``SPIKE_MULTIPLE = 3.0``：爆量判定倍数。
    取值 ``value = -count(爆量日, 20) / 20``，落在 ``[-1, 0]``，是离散步长 0.05 的
    占比取负。基线观测不足（每券前 40 行）或其均量非正时记 null，计数窗口不足 20 个
    观测时记 null，都不填 0。预热期 ``WARMUP = 80``（60 日基线 + 20 日计数窗口）。

契约
    全程向量化：``sort`` → ``shift`` / ``rolling_mean`` / ``rolling_sum`` 配
    ``over("instrument")`` → 逐元素比较、计数与取负，无循环、无 IO、无随机数。所有
    统计量都是个股自身的向后滚动窗口，只用当日及之前的数据（``shift`` 参数恒为正，
    无全序列聚合、无排名回填）。输出恰好 ``(date, instrument, value)`` 三列，value 为
    Float64，不做截面标准化，不留中间列。
"""
from __future__ import annotations

import polars as pl

#: 爆量事件计数的滚动窗口（交易日）：最近 20 日里出现过几次爆量。中尺度。
WINDOW: int = 20

#: 个股常态成交量基线的滚动窗口（交易日），基线刻意排除当日。中尺度。
BASELINE_WINDOW: int = 60

#: 基线的最少观测数：窗口 60 日、铺满 40 个观测即出值，把「shift(1) + 基线 +
#: 20 日计数」的整条链压到 60 个交易日内（库内因子的窗口局部性预算，与
#: tests/test_baseline_factors.py 的尾部重算口径一致）；观测不足 40 仍输出 null。
MIN_BASELINE_SAMPLES: int = 40

#: 爆量判定倍数：当日成交量超过自身常态基线的该倍数即记一次爆量事件。
SPIKE_MULTIPLE: float = 3.0

#: 预热交易日数：60 日基线铺满后还需 20 日计数窗口，共 80 日。
WARMUP: int = BASELINE_WINDOW + WINDOW

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "volume")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日爆量频次占比取负，返回 ``(date, instrument, value)`` 三列长表。

    ``value`` 为个股最近 :data:`WINDOW` 个交易日中「成交量超过自身前
    :data:`BASELINE_WINDOW` 日成交均量 :data:`SPIKE_MULTIPLE` 倍」的天数占比取负，
    落在 ``[-1, 0]``，越大越看多（越没有出过脉冲式天量的股票得分越高）。
    基线窗口未铺满、基线均量非正或计数窗口不足处输出 null，不填 0，也不做截面标准化。

    :param data: 长表面板，至少含 :data:`REQUIRED_COLUMNS` 各列；函数内部按
        ``(instrument, date)`` 重新排序，防止行序漂移让 ``over`` 分组窗口错位。
    :return: 恰好 ``(date, instrument, value)`` 三列，``value`` 为 Float64。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"vol_spike_freq_20_neg 缺少输入列：{missing}")

    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])

    baseline = (
        pl.col("volume")
        .shift(1)
        .rolling_mean(BASELINE_WINDOW, min_samples=MIN_BASELINE_SAMPLES)
        .over("instrument")
    )
    spike = (
        pl.when(baseline.is_null() | (baseline <= 0.0))
        .then(None)
        .otherwise((pl.col("volume") > SPIKE_MULTIPLE * baseline).cast(pl.Float64))
    )
    value = -(
        spike.rolling_mean(WINDOW, min_samples=WINDOW).over("instrument")
    )

    return ordered.with_columns(value.cast(pl.Float64).alias("value")).select(
        "date", "instrument", "value"
    )
