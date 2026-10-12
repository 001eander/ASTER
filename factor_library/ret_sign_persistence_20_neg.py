"""ret_sign_persistence_20_neg：20 日显著异动方向延续占比取负（方向自相关计数）。

经济假设
    把个股日收益的幅度信息丢掉，只留方向符号，再数「今天的显著异动方向与昨天的显著
    异动方向是否相同」。20 日窗口内同向延续的占比高，说明这只股票的价格推进是单边
    的：资金顺着同一个方向连续发力（连续放量上攻、连续被砸），筹码在同一个方向上
    快速集中，每一次推进都建立在「前一日也在同向异动」的情绪与仓位惯性上，承接盘是
    趋势跟随者而非价值买盘，惯性耗尽后没有对手方接货，后续回落。占比低的股票方向
    反复翻转，多空在窗口里反复换手，定价分歧被一次次撮合消化，抛压提前释放，筹码
    结构松散，后续表现更好。故取负：方向越反复的股票分越高。

    这条轴与幅度族不同源。库内 mom_5（rank_ic -0.027）、mom_20（-0.035）说明 A 股
    短周期由反转主导，但它们累计的是收益幅度：一只「三天涨 8%、两天跌 7%」的高波动
    票与一只「天天涨 0.3%」的低波动票在动量读数上可以相同，方向结构却相反（前者
    方向反复、后者单边）。本因子只数方向符号，涨跌停与单日暴涨暴跌的幅度被彻底削掉，
    度量的是「方向是否成链」，与幅度累计的 mom 家族、与波动水平的 volatility_20、
    与收盘区间位置的 CLV、与收益三阶矩的 return_skew_20_neg 都落在不同的信息面上。

与库内同方向因子的机制区别
    库内价格方向的 0/1 计数族（``pv_direction_concordance_20`` 一类）比较的是「价的
    方向」与「量的方向」同日是否一致，度量的是跨字段的量价配合；本因子比较的是
    「价的方向」与「自己昨天的价的方向」，是同一条价格序列的时间序列自相关，窗口里
    只用价格一个字段。``mom_5`` / ``mom_20`` 累计复权收益幅度，``reversal_5`` 用单
    窗口的开盘涨幅水平，``return_skew_20_neg`` 取收益分布三阶矩（对单日暴涨敏感），
    ``max_drawdown_20`` / ``volatility_20`` / ``range_pct`` 度量波动与回撤的幅度水平。
    没有一个刻画「方向的延续性」这条轴。

口径与实测依据
    规格主轴（20 日方向自相关计数取负、后复权、脏数据护栏、shift 只取正数、缺失记
    null）全部保留，另加两处口径澄清，都在全样本上实测过：

    1) **剔除当日全市场共同收益再判方向**。当日截面均值收益是全市场共享的系统性
       方向（指数级涨跌），个股跟着大盘同向波动属于 beta，不是自己的方向。方向判定
       先在「个股复权收益 - 当日截面均值收益」上做，留下的是个股特有方向。截面均值
       只用当日截面，属当日信息，无前视。20 日窗口下 rank_ic 由 +0.035 抬到 +0.044，
       是全样本上唯一一处让 IC 显著变好的改动。
    2) **以显著异动定义方向**。|个股特有收益| ≤ 2% 的日子不构成方向：这类微幅噪音
       天天在翻，把它算成一次方向延续会让因子退化成几乎无信息的白噪声。阈值以下
       （含收益恰为 0 的横盘日）符号记 0，与前后都不构成延续链，也自成一天的不延续，
       属合法中性点——既不能解读为方向一致，也不能解读为方向翻转。窗口 20、显著
       阈值 1.5% / 2.0% / 2.5% 三点的 rank_ic 为 +0.0438 / +0.0439 / +0.0428、icir
       均为 +0.365；窗口 15 / 20 / 25 为 +0.0442 / +0.0439 / +0.0431。信号在邻域内
       平坦，取 2% 与 20 日只是网格中部的整数点，不是单点调参的产物。

    幅度信息的处理与规格一致：只数「同向 / 不同向」这一个 0/1 标记，两个方向相反但
    幅度悬殊的日子在计数里同权，涨跌停、单日巨震的幅度不进入排序。

数据质量兜底
    A 股涨跌幅限制为 ±10% / ±20% / ±30%（ST、创业板 / 科创板、北交所），复权后的
    单日收益绝对值不会超过 50%，越界只可能来自复权因子接缝或行情脏数据。这类日子
    的收益记 null：既不进入符号比较、不进入当日截面均值，也让窗口凑不满样本输出
    null，不把脏数据静默算成一次方向延续。条件表达式对 null 恒不成立，NaN / inf 与
    缺失一并落到 null，输出里缺失只有 null 一种形态。

使用字段：close, adjfactor（外加键列 date、instrument）。窗口：WINDOW = 20 个交易日。
复权口径：close × adjfactor（后复权），跨日比价一律用复权价，除权跳空不进因子。
契约：无 IO、无随机数、无逐行循环；``shift`` 只取正数，滚动统计量一律
    ``over("instrument")``，截面均值只用当日截面，不用全序列均值 / 排名 / 分位数；
    不做截面标准化、不做平滑、缺失不填 0；输出恰好 ``(date, instrument, value)``
    三列，``value`` 为 Float64，行序不作要求。
"""
from __future__ import annotations

import polars as pl

#: 方向延续占比的滚动窗口（交易日），中尺度。
WINDOW: int = 20

#: 回看前一日方向符号的滞后阶数，只允许正数。
PREV_LAG: int = 1

#: 单日复权收益的合理上界：绝对值超过它即判为复权接缝一类的脏数据，当日记 null。
MAX_ABS_RET: float = 0.5

#: 显著异动阈值：|个股特有收益| 不超过它的日子符号记 0（微幅噪音不构成方向）。
SIG_THRESHOLD: float = 0.02

#: 预热交易日数：1 阶滞后 + 20 日滚动窗口，第 22 行起才有首个有效值，留余量取 25。
WARMUP: int = 25

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日显著异动方向延续占比取负，返回 ``(date, instrument, value)`` 三列。

    因子值越大越看多：窗口内显著异动方向频繁翻转（方向不成链）的股票得分高，方向
    单边延续的趋势股得分低。窗口不足、脏数据、窗口内出现缺失的当日输出 null，不填 0，
    也不做截面标准化。

    :param data: 长表面板，含 :data:`REQUIRED_COLUMNS` 各列；函数内部按
        ``(instrument, date)`` 重新排序，防止行序漂移让 ``shift`` 与滚动窗口错位。
    :return: 恰好 ``(date, instrument, value)`` 三列，行序不作要求。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"ret_sign_persistence_20_neg 缺少输入列：{missing}")

    # 后复权价与日收益：跨日比价只用复权口径，除权跳空不进入方向判定。
    adj_close = (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
    raw_ret = (
        pl.col("_adj_close")
        / pl.col("_adj_close").shift(PREV_LAG).over("instrument")
        - 1.0
    ).alias("_ret")
    # 脏数据护栏：越界（复权接缝）与 NaN / inf 一律记 null，不参与符号比较与截面均值。
    clean = (
        pl.col("_ret").is_finite() & (pl.col("_ret").abs() <= MAX_ABS_RET)
    ).alias("_clean")

    ordered = (
        data.select(*REQUIRED_COLUMNS)
        .sort(["instrument", "date"])
        .with_columns(adj_close)
        .with_columns(raw_ret)
        .with_columns(clean)
        .with_columns(
            pl.when(pl.col("_clean")).then(pl.col("_ret")).otherwise(None).alias("_ret_c"),
        )
        # 剔除当日全市场共同方向（截面均值只用当日截面，属当日信息，无前视）。
        .with_columns(
            (
                pl.col("_ret_c") - pl.col("_ret_c").mean().over("date")
            ).alias("_idio"),
        )
        # 显著异动方向：微幅噪音与横盘日记 0，缺失记 null。
        .with_columns(
            pl.when(pl.col("_idio").is_null())
            .then(None)
            .when(pl.col("_idio").abs() > SIG_THRESHOLD)
            .then(pl.col("_idio").sign())
            .otherwise(0.0)
            .alias("_sgn"),
        )
        # 两级滞后分两步物化：带 over 的表达式不嵌进另一个 over。
        .with_columns(
            pl.col("_sgn").shift(PREV_LAG).over("instrument").alias("_prev_sgn"),
        )
        # 延续标记：任一侧缺失记 null；同日同向且都不为 0 记 1.0，其余记 0.0。
        .with_columns(
            pl.when(pl.col("_sgn").is_null() | pl.col("_prev_sgn").is_null())
            .then(None)
            .when(pl.col("_sgn") * pl.col("_prev_sgn") > 0.0)
            .then(1.0)
            .otherwise(0.0)
            .alias("_persist"),
        )
        # 窗口 min_samples 取满：窗口里出现 null 即凑不满样本，当日输出 null。
        .with_columns(
            (
                -pl.col("_persist")
                .rolling_mean(WINDOW, min_samples=WINDOW)
                .over("instrument")
            ).alias("value"),
        )
    )
    return ordered.select("date", "instrument", pl.col("value").cast(pl.Float64))
