"""量：成交活跃度轨迹的「顺序结构」腿池（差分自相关 + 路径效率），轨迹杂乱无章的股票做多。

经济假设
    成交量／成交额的时间序列「顺序结构」刻画资金行为的可计划性，是量的边际分布
    （均值、标准差、极值）丢掉的维度。过去 10／20 个交易日里，量的日变化高度自相关
    （今日的放量倾向延续昨日的放量）、且 20 日净变化占日变化绝对值之和的比例很高，
    说明成交活跃度的演变是一条有方向、有节奏的轨迹：多来自指数化调仓、程序化拆单、
    按计划的增持减持等「日常化、可预期」的资金流，没有新信息含量，在 A 股短期反转与
    流动性补偿的定价下对应次日相对弱势。反过来，量的变化无自相关、净变化被来回的
    日间波动淹没（路径效率低），是信息冲击驱动的偶发交易，短期关注度溢价为正。

    两条腿测同一机制的不同侧面：量的日变化一阶自相关（成交股数 10 日、成交额 20 日）
    度量顺序依赖，路径效率 ``|20 日净变化| / Σ|日变化|``（成交股数与成交额各一条）
    度量轨迹形状。四条腿取负后等权平均，因子值越大（轨迹越无序）越看多。

与库内同方向因子的机制区别
    库内 ``vol_cv_20_neg``（volume/medium/volume_dispersion）用滚动标准差／滚动均值
    度量量的离散度，``vol_ratio_5_20_neg`` 度量均量的水平收缩，
    ``volume_spike_freq_20_neg`` 度量超阈天数占比。三者都把当日量与同一窗口内量的
    边际分布作比较，对「量的先后顺序」完全不敏感：同样的 CV = 0.8 既可以是「三天
    爆量三天枯竭交替」，也可以是「平滑爬升后平滑回落」，变异系数分不开，一阶自相关
    与路径效率正是分开这两者的统计量。本因子不看量的绝对水平与分布形状，只看量的
    轨迹如何随时间展开。

构造
    先按 ``(instrument, date)`` 排序，成交股数与成交额各取 ``log(x + 1)`` 后作一阶
    差分。四条腿（全部只用后向窗口与正滞后）：

        ac10 = corr(Δlv_t, Δlv_{t−1})                窗口 10，lv = log(volume + 1)
        ac20 = corr(Δla_t, Δla_{t−1})                窗口 20，la = log(amount + 1)
        pev  = |lv_t − lv_{t−19}| / Σ|Δlv|           窗口 20
        pea  = |la_t − la_{t−19}| / Σ|Δla|           窗口 20

    value = −(ac10 + ac20 + pev + pea) / 4

    polars 没有滚动相关，``rolling_map`` 逐窗口 Python 回调在全市场上会超时，故 Pearson
    相关用 ``rolling_sum`` 拼标准公式：

        corr = (n·Σx·y − Σx·Σy) / sqrt((n·Σx² − (Σx)²)·(n·Σy² − (Σy)²))

    其中 ``y`` 是 ``x`` 的滞后 1 期，故 Σy 与 Σy² 恰好等于 Σx 与 Σx² 的滞后 1 行
    （同一窗口整体平移一期，恒等），直接对已算出的滚动和取 ``shift`` 即可，省掉两次
    滚动求和。``n`` 为窗口长度。任一侧方差项非正（窗口内量能一成不变，如长期停牌；
    或窗口含 null）时相关无定义，记 null；路径效率的分母为 0 时同样记 null。窗口
    未铺满、或任一腿取不到值时整个 value 记 null，不填 0。

    方向：库内 ``vol_cv_20_neg``、``volume_spike_freq_20_neg`` 都是「量的分布平稳做多、
    极端脉冲做空」；本因子在顺序维度上测出的符号相反（轨迹平滑可预期做空、轨迹杂乱
    做多），全样本 RankIC 为正、ICIR 约 0.35，与 ``vol_cv_20_neg`` 的截面相关约 0.39，
    是同一信号源上互补而非重叠的一条腿。

使用字段：volume、amount（外加键列 date、instrument），两者按契约均为 Float64。
    成交股数与成交额不受除权除息影响，无需复权；本因子不接触价格字段，价格跳空
    不可能污染信号。
窗口：自相关短窗 10 日、中窗 20 日；路径效率窗口 20 日；滞后取正数 1 与窗口长度
    减一。
复权口径：不涉及价格比较，volume／amount 原值直接用。

契约
    compute(data) 全程 polars 向量化：``shift`` + ``rolling_sum`` 全部配
    ``over("instrument")``，无逐行循环、无 IO、无随机数；统计量只用后向滚动窗口与
    正滞后，不使用任何全序列聚合，也不使用当日之后的数据。输出恰好
    ``(date, instrument, value)`` 三列，value 为 Float64，行序不作要求。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 自相关短窗（交易日），用于成交股数腿。
WINDOW_SHORT: int = 10

#: 自相关中窗与路径效率窗口（交易日）。
WINDOW: int = 20

#: 滞后阶数，只允许正数。
LAG: int = 1

#: 预热交易日数：最慢的腿（中窗自相关，含差分与滞后）在第 22 行才铺满，留余量取 25。
WARMUP: int = 25

#: 腿数，用于等权平均（两条自相关腿 + 两条路径效率腿）。
N_LEGS: float = 4.0

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "volume", "amount")


# ---------------------------------------------------------------------------
# 滚动统计量
# ---------------------------------------------------------------------------


def _flow_legs(col: str, corr_window: int, path_window: int) -> tuple[pl.Expr, pl.Expr]:
    """``col`` 的对数增量上的两条腿：滞后 1 阶自相关与路径效率。

    :param col: 输入列名（``volume`` 或 ``amount``）。
    :param corr_window: 自相关窗口长度。
    :param path_window: 路径效率窗口长度。
    :return: ``(自相关, 路径效率)`` 两个 Float64 表达式；样本不足或统计量无定义时为
        null（自相关在任一侧方差项非正时，路径效率在分母为 0 时），不填 0。

    自相关用 ``rolling_sum`` 拼标准公式
    ``(n·Σxy − Σx·Σy) / sqrt((n·Σx² − (Σx)²)·(n·Σy² − (Σy)²))``，其中 ``y`` 为
    ``x`` 的滞后 1 期，因此 Σy／Σy² 就是 Σx／Σx² 的滞后一行；路径效率为
    ``|净变化| / Σ|日变化|``，取值落在 [0, 1]。全部只用后向窗口与正滞后。
    """
    level = pl.col(col).log1p()
    diff = level - level.shift(LAG).over("instrument")
    lag = diff.shift(LAG).over("instrument")
    sx = diff.rolling_sum(corr_window).over("instrument")
    sxx = (diff * diff).rolling_sum(corr_window).over("instrument")
    sy = sx.shift(LAG).over("instrument")
    syy = sxx.shift(LAG).over("instrument")
    sxy = (diff * lag).rolling_sum(corr_window).over("instrument")
    den = (corr_window * sxx - sx * sx) * (corr_window * syy - sy * sy)
    corr = pl.when(den > 0.0).then((corr_window * sxy - sx * sy) / den.sqrt())
    path = diff.abs().rolling_sum(path_window).over("instrument")
    net = (level - level.shift(path_window - 1).over("instrument")).abs()
    eff = pl.when(path > 0.0).then(net / path)
    return corr, eff


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """成交活跃度轨迹的顺序结构腿池，返回 ``(date, instrument, value)`` 三列长表。

    :param data: 长表面板，至少含 :data:`REQUIRED_COLUMNS` 各列；内部按
        ``(instrument, date)`` 重新排序，防止行序漂移让 ``shift`` 与滚动窗口错位。
    :return: 三列长表，值为 ``−(ac10 + ac20 + pev + pea) / 4``；窗口不足或任一条腿
        取不到值时该行为 null，不填 0。值越大（量能轨迹越无序）越看多。
    """
    if not set(REQUIRED_COLUMNS).issubset(data.columns):
        raise ValueError(f"REQUIRED_COLUMNS 缺失：{REQUIRED_COLUMNS}")

    ac_v, pe_v = _flow_legs("volume", WINDOW_SHORT, WINDOW)
    ac_a, pe_a = _flow_legs("amount", WINDOW, WINDOW)
    return (
        data.select(*REQUIRED_COLUMNS)
        .sort("instrument", "date")
        .with_columns(value=-(ac_v + ac_a + pe_v + pe_a) / N_LEGS)
        .select("date", "instrument", "value")
    )
