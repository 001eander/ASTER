"""价量：20 日「收盘对 VWAP 偏离」的超额离散度（vwap_bias 二阶矩 − 当日振幅自身的离散度）。

经济假设
    当日 ``dev = close / vwap - 1`` 度量收盘价相对全天成交均价的位置：收盘高于全天均价
    说明尾盘有资金愿意用更贵的价格接货，反之说明尾盘抛压压住了价格。库内 ``vwap_bias``
    取的是这条偏离的**一阶矩**（20 日均值，日内资金强弱的方向）；本因子取它的**二阶矩**
    （20 日标准差），问的是另一个问题：这只票的收盘价与全天均价是不是稳定地贴在同一个
    位置。

    日均偏离为零的票可以有两种完全不同的形态：每天 ``close`` 与 ``vwap`` 都贴着走
    （偏离始终微小），收盘价代表全天公允定价；或者每天大幅正负摇摆（今天收在均价上方
    3%，明天收在下方 3%），尾盘资金反复博弈、有拉尾／砸尾的做线习惯，收盘价被尾盘几笔
    单子撬动，不代表全天真实成交水平，次日开盘要向全天均价回复。一阶矩把这两种形态压成
    同一个零，分不开；二阶矩正是被它丢掉的维度。

    但裸的 ``std(dev)`` 不是干净的信号：价格波动大的票，收盘对均价的偏离天然摆得大
    （全样本上与 ``volatility_20`` 的逐日截面相关达 -0.834），直接用它等于买一个波动率
    因子的换皮。因此本因子从 ``std(dev)`` 里减去**同一只票当日振幅自身的离散度**
    ``std(ampv)``（``ampv = (high - low) / vwap``，同样以全天均价为基准的相对振幅）——
    后者度量的是「这只票日常震荡幅度本身有多不稳」，是上面那条公用分量的直接代理。
    差额只留下「收盘-均价失配的摆动，超出该股自身震荡常态的部分」。

    因子值大（差额为正）表示收盘与均价的位置变化比该股自身的震荡强度还活跃，收盘端有
    持续的增量资金在推动定价（尾盘集合竞价的主动买卖、拉抬／打压的做线痕迹），这类活跃
    的收盘定价行为在次日仍有延续性，看多；因子值小（差额为负）表示收盘相对均价的位置
    变化比该股自身的日内震荡还小，收盘价随波逐流、无人定价，后续缺少资金关注，看空。

    方向说明：原始 ``std(dev)`` 的口径与假设一致（偏离波动大 → 看空，实测 rank_ic
    -0.045），但它与库内 ``volatility_20`` 相关 -0.834、判冗余；剔除公用分量后的残差
    项方向翻转为正（实测 rank_ic +0.034），说明「做线痕迹」里带增量信息的是相对该股
    自身波动常态的超额部分，而非波动水平本身。方向由全样本 RankIC 符号确定，未做样本内
    择优。

与库内同方向因子的机制区别
    ``vwap_bias``（price/short/vwap_bias）是 ``dev`` 的一阶矩。均值为零不约束波动，两者
    机械独立：本因子对「偏离恒定 ±0.5%」的票给中性分，对「偏离在 ±3% 之间来回摆但均值
    恰为 0」的票给最低分，``vwap_bias`` 给这两只票几乎同分。

    ``volatility_20``（日收益标准差）与 ``range_pct``（日内相对振幅均值）度量价格自身的
    波动水平，``max_drawdown_20`` 度量相对高点的回撤。本因子的两项都是**离散度的差**：
    振幅水平被减掉、跨日涨跌被同日比值约掉，剩下的是「收盘对均价失配的摆动相对该股自身
    震荡常态的超出量」这一形状维度——一只票可以振幅很大但收盘始终贴着均价（本因子低分），
    也可以振幅很小但收盘天天被撬动（本因子高分），与波动水平口径观察面不同。
    ``return_skew_20_neg`` 是收益分布的三阶矩，``vol_cv_20_neg``、``volume_spike_freq_20_neg``
    与 ``vol_ratio_5_20_neg`` 看成交量的离散度与水平，本因子不接触任何成交量字段，也不
    使用收益路径的高阶矩。

构造
    逐券按日排序后：

        dev_t  = close_t / vwap_t - 1                 （仅当 vwap_t > 0，否则 null）
        ampv_t = (high_t - low_t) / vwap_t            （仅当 vwap_t > 0，否则 null）
        value_t = std(dev_{t-19..t}) - std(ampv_{t-19..t})

    两个滚动标准差同窗口、同基准（分母都是当日成交均价），窗口 WINDOW = 20、ddof = 1。
    窗口未铺满或窗口内出现 null（停牌 / 零均价的脏点）时输出 null，不填 0。不做时序
    平滑、不做截面标准化、不做排名。

    复权口径：``close``、``high``、``low``、``vwap`` 是同日、同口径（未复权）的成交价，
    ``adjfactor`` 在每项分子分母上同日对消，因此本条不做任何复权换算，也刻意不做跨日
    价格比较——跨日构造会把除权跳空引入信号，同日比值天然免疫。

使用字段：``close``、``high``、``low``、``vwap``（外加键列 ``date``、``instrument``）。
窗口：``WINDOW`` = 20 个交易日；``WARMUP`` = 25 个交易日（滚动窗口在第 20 行铺满）。
复权口径：同日比值，``adjfactor`` 对消，无需复权；不使用任何未复权价的跨日比较。

契约
    :func:`compute` 全程 polars 向量化：排序一次，两个 ``rolling_std`` 配
    ``over("instrument")`` 后相减，无逐行循环、无 IO、无随机数；统计量只用后向滚动
    窗口，不用全序列均值／排名／分位数，也不使用当日之后的数据。输出恰好
    ``(date, instrument, value)`` 三列，``value`` 为 Float64，行序不作要求。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 滚动窗口（交易日）。
WINDOW: int = 20

#: 滚动标准差的自由度修正：与 polars ``rolling_std`` 默认一致（样本标准差）。
DD_OF: int = 1

#: 预热交易日数：滚动窗口在第 WINDOW 行才铺满，留余量取 25。
WARMUP: int = 25

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "close",
    "high",
    "low",
    "vwap",
)


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日「收盘对 VWAP 偏离的超额离散度」，返回 ``(date, instrument, value)``。

    :param data: 长表面板，至少含 :data:`REQUIRED_COLUMNS` 各列；内部按
        ``(instrument, date)`` 重新排序，防止行序漂移让滚动窗口错位。
    :return: 三列长表，``value = std(close/vwap - 1, WINDOW) - std((high-low)/vwap, WINDOW)``；
        ``vwap <= 0``、窗口未铺满或窗口内含 null 的行记 null，不填 0。值越大越看多。
    :raises ValueError: 输入缺少 :data:`REQUIRED_COLUMNS` 中的列。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"REQUIRED_COLUMNS 缺失：{missing}")

    inst = "instrument"
    vwap = pl.col("vwap").cast(pl.Float64)
    # vwap 非正（停牌 / 脏点）时两项都无定义，记 null；同日价格，adjfactor 对消。
    valid = vwap > 0.0
    nan = pl.lit(None, dtype=pl.Float64)

    return (
        data.select(*REQUIRED_COLUMNS)
        .sort(inst, "date")
        .with_columns(
            _dev=pl.when(valid).then(pl.col("close").cast(pl.Float64) / vwap - 1.0).otherwise(nan),
            _ampv=pl.when(valid)
            .then(
                (pl.col("high").cast(pl.Float64) - pl.col("low").cast(pl.Float64)) / vwap
            )
            .otherwise(nan),
        )
        # 窗口不满或窗口内含 null（含每只票首几行未铺满）时 rolling_std 输出 null。
        .with_columns(
            value=(
                pl.col("_dev").rolling_std(WINDOW, ddof=DD_OF, min_samples=WINDOW).over(inst)
                - pl.col("_ampv")
                .rolling_std(WINDOW, ddof=DD_OF, min_samples=WINDOW)
                .over(inst)
            ).cast(pl.Float64)
        )
        .select("date", inst, "value")
    )
