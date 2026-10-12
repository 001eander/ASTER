"""价量三腿加权 rank 合成（0014）沿时间尺度轴改一处：量能 CV 腿窗口 20 → 40 日。

经济假设
    本方向的瓶颈已从 IC 均值转到 ICIR。0014（量 CV 0.6、价差与下影各 0.2）把权重轴推到
    第二档后，rank_ic 由 0.0451 小降到 0.0397、icir 只由 0.47 抬到 0.48，权重轴的边际
    收益在递减；而量 CV 单腿的 icir 约 0.43，靠继续加权也够不到 0.50 的门。故本轮不动
    权重，改走另一条尚未试过的轴：量 CV 腿自身的估计时间尺度。

    量能变异系数（CV = 滚动 std / 滚动 mean）度量的是「筹码换手节奏的离散度」。均量相同
    但忽大忽小的票，筹码在短线资金间频繁易手、持有者结构松散；量能平稳的票筹码锁定良好、
    潜在抛压可预期。这条腿的定价偏差来自筹码结构这个慢变量，信息本来就以月为尺度释放，
    因此 20 日窗口对它偏短：窗口里混入的少数异常放量日（消息、大宗、指数调仓）会把腿部
    排名在日间大幅抖动，这部分抖动是噪声、不是信号。把 CV 窗口由 20 日拉长到 40 日，
    单日异常量的权重被摊薄、腿部排名的日间自相关抬升，腿的 IC 流标准差下降；合成 IC 的
    方差由三腿 IC 方差按权重平方叠加，量 CV 腿占六成，其方差下行直接压低合成方差，
    icir 因此抬升，而 IC 均值只小幅让渡。

    三条腿的公式、符号、权重、护栏一律照搬 0014，唯一改动是 :data:`VOL_CV_WINDOW`
    由 20 改为 40、:data:`WARMUP` 由 30 提到 45：

    * 腿 A（收益分解，20 日）：日收益拆成隔夜段 ``今开/昨收 - 1`` 与日内段
      ``今收/今开 - 1``，取价差 ``隔夜 - 日内`` 的 20 日累计 / 20 日标准差。价差高的票
      反复高开、盘中回吐；价差低的票反复低开、盘中收复。窗口不动。
    * 腿 B（量能稳定性，40 日）：成交量变异系数取负，本轮唯一改动。窗口拉长压的是估计
      噪声，不是把信号做时序平滑——改的是信号本身的估计窗口，与 0010（对合成后的
      value 再做 5 日均值）在机制上不同：平滑会把三腿共有的当日信息一起钝化，拉长单腿
      窗口只降该腿的估计方差、不改另两腿的时点响应。
    * 腿 C（K 线形状，10 日）：下影长度合计 / 振幅合计。长下影是盘中下探被买回的痕迹，
      按振幅加权让真正走出区间的日子主导份额。窗口不动。

    诊断上另有三条替代路径已被证伪——时序平滑（0010）、继续加腿（0011）、规模/流动性
    中性化（0012）都没能抬 ICIR，故本轮只动量 CV 腿的窗口这一处。因子值越大越看多。

构造
    三条腿量纲互不相同（A 是标准化累计、B 是比率、C 是占比），直接加权相加会让 A 独大，
    故先逐日截面取 rank 再按权重加权，权重与 0014 完全一致：

        value = 0.2 × rank(腿A) + 0.6 × rank(腿B) + 0.2 × rank(腿C)

    rank 只用当日截面（当日信息，无前视）；任一腿窗口不足或记 null 的当日，value 沿
    空值传播为 null，不填 0。不做时序平滑、不做截面中性化、不做二次标准化。权重之和为
    1，value 与三个 rank 同量级，恒为 Float64。

使用字段：open, high, low, close, volume, adjfactor（外加键列 date、instrument）。
窗口：腿 A 20 个交易日、腿 B 40 个交易日、腿 C 10 个交易日；``shift`` 只取正数 1；
    滚动统计量一律 ``over("instrument")``，窗口未铺满时输出 null（polars 的
    ``min_samples`` 默认等于窗口长度，此处依赖该默认，不显式传参）。
复权口径：后复权。隔夜段 = (open_t × adjfactor_t) / (close_{t-1} × adjfactor_{t-1}) - 1，
    日内段 = (close_t × adjfactor_t) / (open_t × adjfactor_t) - 1；腿 C 的影线与振幅
    按当日 adjfactor 缩放后再跨日合计，除权跳空不进入因子。腿 B 只用量（股数），
    除权不改变成交股数的经济含义，无需复权。

数据质量兜底：A 股涨跌幅限制为 ±10% / ±20% / ±30%，复权后单日隔夜跳空或日内振幅
    不会超过 50%，超界的点只可能来自复权接缝一类的脏数据，当日价差记 null、不进入
    累计与波动。腿 A 的分母（价差的标准差）由去均值后的构造保证为正，腿 B 的分母
    （40 日均量）恒正，故都不另加兜底；腿 C 的分母加 1e-9，避免连续一字板把振幅合计压到
    0 时出现 0/0。条件表达式对 null 恒判假，输出缺失只有 null 一种形态。

与库内同方向因子的区别：库内 13 个 pool 因子覆盖动量 / 反转 / 均线偏离 / 波动 / 振幅
    水平 / 量比 / 均量 / 换手 / 量价相关 / VWAP 偏离，都是「单一口径的时序统计」。本因子
    把三条机制互异的腿在截面 rank 空间按 IC 稳定性加权集成（量 CV 腿占六成），信号来自
    腿部互补与权重倾斜，任何单一口径的窗口或符号变动都复现不了这个结构；本轮再改量 CV
    腿的估计窗口，进一步远离库内任何单一量能口径。

契约：输入面板的列齐备性由沙盒 schema 校验保证；行序按契约已为 ``(instrument, date)``
    递增，函数内部仍显式排序一次，防止行序漂移让 ``shift`` 与滚动窗口错位。无 IO、无
    随机数、无当日之后的数据。中间表达式写成张量化的短链，是为把 AST 节点数压在沙盒
    阈值 500 以内，不改变任何计算语义。
"""
from __future__ import annotations

import polars as pl
from polars import col as C, max_horizontal as hmax, min_horizontal as hmin, when as W

#: 腿 A：隔夜 / 日内价差的累计与标准化窗口（交易日）。
SPREAD_WINDOW: int = 20

#: 腿 B：成交量变异系数的滚动窗口（交易日）。本轮由 20 拉长到 40，压估计噪声抬 ICIR。
VOL_CV_WINDOW: int = 40

#: 腿 C：下影承接力的合计窗口（交易日）。
SHADOW_WINDOW: int = 10

#: 腿 A 单日分量的合理上界，越界视为脏数据记 null。
MAX_ABS_LEG: float = 0.5

#: 腿 A（隔夜/日内价差 t 值）在合成中的权重：IC 方差大，压到两成。
W_SPREAD: float = 0.2

#: 腿 B（量能 CV 取负）在合成中的权重：IC 流最稳，第二档倾斜给六成。
W_VOLCV: float = 0.6

#: 腿 C（下影承接力）在合成中的权重：IC 方差大，压到两成。
W_SHADOW: float = 0.2

#: 预热交易日数：最长链路为 1 阶滞后 + 40 日滚动窗口，留余量取 45。
WARMUP: int = 45

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjfactor",
)


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """三腿加权截面 rank 合成（量 CV 腿窗口 40 日），返回 ``(date, instrument, value)``。

    ``value`` 为 Float64；任一腿窗口不足或记 null 的当日输出 null，不填 0。值越大越看多。

    :param data: 长表面板，含 :data:`REQUIRED_COLUMNS` 各列；内部按
        ``(instrument, date)`` 重新排序以防行序漂移。
    :return: 恰好 ``(date, instrument, value)`` 三列，行序不作要求。
    """
    # 跨日比价一律后复权（未复权价 × adjfactor），除权跳空不进入因子。
    sc = C("adjfactor")
    adj_close = C("close") * sc
    prev = adj_close.shift(1).over("instrument")

    # 腿 A 的隔夜段与日内段；价差超界只可能来自复权接缝一类的脏数据，当日记 null。
    overnight = C("open") * sc / prev - 1.0
    intraday = adj_close / (C("open") * sc) - 1.0
    raw_spread = W(hmax(overnight.abs(), intraday.abs()) <= MAX_ABS_LEG).then(
        overnight - intraday
    )

    # 腿 C 的日度分子 / 分母：复权口径的影线长度与振幅。
    shadow = sc * (hmin(C("open"), C("close")) - C("low"))
    day_range = sc * (C("high") - C("low"))

    # 三条腿的滚动中间量：窗口未铺满时为 null，不做任何填充。
    sp = C("_spread")
    frame = (
        data.sort("instrument", "date")
        # 腿 A 的当日截面去均值：只取个股相对当日全市场的偏离，当日信息、无前视。
        # 带 ``over`` 的窗口表达式不能嵌进另一个 ``over``，故先落成列再做下一步。
        .with_columns(_spread=raw_spread)
        .with_columns(_spread=sp - sp.mean().over("date"))
        .with_columns(
            _leg_spread=sp.rolling_sum(SPREAD_WINDOW).over("instrument")
            / sp.rolling_std(SPREAD_WINDOW).over("instrument"),
            # 量 CV 取负：量能平稳（筹码锁定）的票记正分；窗口 40 日摊薄单日异常量。
            _leg_volcv=-C("volume").rolling_std(VOL_CV_WINDOW).over("instrument")
            / C("volume").rolling_mean(VOL_CV_WINDOW).over("instrument"),
            # 下影合计 / 振幅合计；分母加 1e-9 防连续一字板压出 0/0。
            _leg_shadow=shadow.rolling_sum(SHADOW_WINDOW).over("instrument")
            / (day_range.rolling_sum(SHADOW_WINDOW).over("instrument") + 1e-9),
        )
    )

    # 逐日截面 rank 拉平量纲后按 IC 稳定性加权相加：量能 CV 腿六成、另两腿各两成。
    # 任一腿的 null 沿加法自然传播，当日 value 记 null，不填 0。
    value = (
        W_SPREAD * C("_leg_spread").rank().over("date")
        + W_VOLCV * C("_leg_volcv").rank().over("date")
        + W_SHADOW * C("_leg_shadow").rank().over("date")
    )
    return frame.with_columns(value=value).select("date", "instrument", "value")
