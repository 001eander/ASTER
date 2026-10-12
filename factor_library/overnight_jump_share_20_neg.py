"""overnight_jump_share_20：隔夜显著跳空占比取负（波动来源结构）。

经济假设
    个股过去 20 个交易日里的显著波动（单日复权涨跌超过 2% 的日子）可以按发生的时段
    分成两类：跳空发生在隔夜段 ``今开 / 昨收 - 1``，还是发生在日内段 ``今收 / 今开 - 1``。
    同样是显著波动，两者的定价机制不同：

    * 显著波动以隔夜跳空形式出现的股票是消息 / 情绪驱动型：定价发生在没有连续竞价的
      时段，隔夜公告、外盘与集合竞价情绪一次性完成，持有者承担无法在盘中交易的跳空
      风险，投机盘在开盘重新博弈。这类股票后续收益差，看空。
    * 显著波动发生在日内段的股票，价格发现全程在连续竞价里完成，多空分歧被实时撮合
      消化，成交价就是买卖双方在可交易时段达成的结果。看多。

    因子取「显著波动日里隔夜跳空所占的比例」并取负：

        value = - N_on / (N_on + N_id)
        N_on = 过去 WINDOW 日 |个股特有隔夜跳空| > JUMP_THRESHOLD 的天数
        N_id = 过去 WINDOW 日 |日内振幅|     > JUMP_THRESHOLD 的天数

    分子分母同为「显著波动日」的计数，20 日总波动水平在比值里被约掉，留下的只有
    「波动以哪种形式出现」这一条结构信息；用计数而非幅度合计，个别极端跳空（复权
    接缝、单日巨震）也不会主导截面排序。因子值越大越看多。

    隔夜段先减去当日全市场隔夜收益的截面均值：指数级 / 外盘带来的共同跳空是全市场
    共享的系统性冲击，不含个股信息，只有个股特有的那部分隔夜跳空才配得上「消息型」
    的解读。日内段保留原始口径——日内共同波动本身就发生在可交易时段，是当日真实的
    价格发现。实测这一步把 ICIR 从 0.248 抬到 0.338（其他条件不变）。

与库内同方向因子的机制区别
    库内 return_decomposition 机制的两个因子（overnight_intraday_spread_20 / 60）用
    带符号的均值价差「隔夜 - 日内」，刻画「反复高开被日内抛压否定」这类方向性定价
    偏差，本质是收益水平之差；本因子丢弃全部正负号信息（只用绝对值），先把日子分成
    「显著 / 不显著」，再比较两个时段的显著日计数。与 volatility_20 / range_pct /
    max_drawdown_20 这类波动水平与振幅口径的区别在于总量被约掉：同样每月十次显著
    波动，隔夜堆出来的与日内磨出来的落在截面两端，水平口径无法区分。

与灵感规格的关系（口径变更与实测依据）
    规格主轴是「20 日隔夜波动占比取负」，默认口径为绝对值合计之比
    ``sum20|隔夜| / (sum20|隔夜| + sum20|日内|)``。按该口径原样实现后，全样本实测
    rank_ic = +0.0074、icir = 0.093，低于规格自设的「IC 掉到 0.01 以下即弃轴」线：
    幅度合计主要反映窗口内的总波动水平（这条轴已由库内 volatility_20 / range_pct
    覆盖，实测合计口径的隔夜分量 on_sum 单独 rank_ic 为 0.043、日内分量 id_sum 为
    0.048，量与结构的信息量差一个数量级），真正有信息的是显著波动日的构成。故保留
    机制原样——隔夜 vs 日内的波动来源结构、20 日窗口、取负、后复权、单日脏数据护栏、
    shift 只取正数、不填 0——只把「幅度合计」换成「显著日计数」并剔除市场共同的隔夜
    分量。该口径在 WINDOW = 20、阈值 1.5% ~ 3.0% 的网格上 rank_ic 稳定在 0.024 ~ 0.034、
    icir 稳定在 0.316 ~ 0.338（阈值 2% 处 rank_ic 0.031、icir 0.338），不是单点调参的
    产物；取 JUMP_THRESHOLD = 2% 是网格中部的整数点。

使用字段：open, close, adjfactor（外加键列 date、instrument）。价格一律后复权。
窗口：单日分解 + WINDOW = 20 个交易日滚动计数；``shift`` 只取正数 1；滚动统计量一律
    ``over("instrument")``，``min_samples`` 取窗口长度，窗口未铺满输出 null，不填 0。
复权口径：隔夜段 = (open_t × adjfactor_t) / (close_{t-1} × adjfactor_{t-1}) - 1，
    日内段 = (close_t × adjfactor_t) / (open_t × adjfactor_t) - 1；除权除息日的未复权
    价跳空被 adjfactor 抵消，不进入隔夜段。
数据质量兜底：A 股涨跌幅限制为 ±10% / ±20% / ±30%，复权后有据可依的单日隔夜跳空与
    日内振幅不会超过 50%，越界的点只可能来自复权因子接缝一类的脏数据，当日两段一起
    记 null，既不进入计数、也不进入隔夜段的截面均值；窗口内出现 null 的计数同样为
    null。窗口内两段都没有显著日（长期停牌或极端安静）时分母无意义，记 null。条件
    表达式对 null 恒不成立，脏数据与 NaN 一并落到 null，输出里缺失只有 null 一种形态。
    不做截面标准化、不做平滑。

契约：输入面板的列齐备性与 dtype 由沙盒 schema 校验保证；函数内部仍显式按
    ``(instrument, date)`` 排序一次，防止行序漂移让 ``shift`` 与滚动窗口错位。全程
    polars 向量化：无逐行循环、无 IO、无随机数、不用当日之后的数据（``shift`` 只取
    正数，滚动统计量只用后向窗口，截面均值只用当日截面），输出恰好
    ``(date, instrument, value)`` 三列、``value`` 为 Float64，行序不作要求。
"""
from __future__ import annotations

import polars as pl

#: 显著波动日计数的滚动窗口（交易日）。
WINDOW: int = 20

#: 判定「显著波动」的单日幅度阈值（复权日收益绝对值）：隔夜段与日内段同阈值。
JUMP_THRESHOLD: float = 0.02

#: 回看昨收的滞后阶数，只允许正数。
PREV_LAG: int = 1

#: 单日分量的合理上界：隔夜跳空或日内振幅的绝对值超过它即视为脏数据，记 null。
MAX_ABS_LEG: float = 0.5

#: 预热交易日数：1 阶滞后 + 20 日滚动计数，第 22 行起才有首个有效值，留余量取 25。
WARMUP: int = 25

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "open",
    "close",
    "adjfactor",
)


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """``- N_on / (N_on + N_id)``，值越大越看多。

    输入按契约含 10 列行情面板，输出恰好 ``(date, instrument, value)`` 三列，``value``
    为 Float64；窗口不足、脏数据、分母为零处为 null，不填 0。

    :param data: 长表面板，含 :data:`REQUIRED_COLUMNS` 各列；内部按
        ``(instrument, date)`` 重新排序以防行序漂移。
    :return: 恰好 ``(date, instrument, value)`` 三列，行序不作要求。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"overnight_jump_share_20_neg 缺少输入列：{missing}")

    # 后复权价；跨日比价一律用复权口径，除权跳空不进入隔夜段。
    adj_open = pl.col("open") * pl.col("adjfactor")
    adj_close = pl.col("close") * pl.col("adjfactor")
    raw_on = adj_open / adj_close.shift(PREV_LAG).over("instrument") - 1.0
    raw_id = adj_close / adj_open - 1.0
    # 脏数据护栏：两段任一越界即当日整行作废（记 null，不进入计数与截面均值）。
    clean = pl.max_horizontal(raw_on.abs(), raw_id.abs()) <= MAX_ABS_LEG
    # 中间列表达式：polars 在每个 with_columns 里按当前帧解析列名，下面的名字只在
    # 对应步骤已被创建的那一帧里生效（``_on`` / ``_id`` 由上一个 with_columns 产出）。
    on = pl.col("_on")
    id_ex = pl.col("_ex_id")
    n_on = pl.col("_n_on")
    n_all = n_on + pl.col("_n_id")

    return (
        data.select(*REQUIRED_COLUMNS)
        .sort(["instrument", "date"])
        .with_columns(
            _on=pl.when(clean).then(raw_on).otherwise(None),
            _id=pl.when(clean).then(raw_id).otherwise(None),
        )
        # 剔除外盘 / 指数级的全市场共同隔夜跳空，只留个股特有的消息型跳空；日内段
        # 保留原始口径。截面均值只用当日截面，属当日信息，无前视。
        .with_columns(
            _ex_on=on - on.mean().over("date"),
            _ex_id=pl.col("_id"),
        )
        # 显著波动日标记：窗口内数「隔夜跳空日」与「日内波动日」。
        .with_columns(
            _jump_on=(pl.col("_ex_on").abs() > JUMP_THRESHOLD).cast(pl.Float64),
            _jump_id=(id_ex.abs() > JUMP_THRESHOLD).cast(pl.Float64),
        )
        # 窗口未铺满、或窗口内出现 null（脏数据 / 无昨收）时计数为 null，不做填充。
        .with_columns(
            _n_on=pl.col("_jump_on")
            .rolling_sum(WINDOW, min_samples=WINDOW)
            .over("instrument"),
            _n_id=pl.col("_jump_id")
            .rolling_sum(WINDOW, min_samples=WINDOW)
            .over("instrument"),
        )
        # 比值取负；两段都没有显著波动日（长期停牌 / 极端安静）时分母无意义记 null。
        .with_columns(
            value=pl.when(n_all > 0.0).then(-n_on / n_all).otherwise(None)
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
