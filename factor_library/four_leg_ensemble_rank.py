"""四腿等权 rank 合成：收益分解 + 量能二阶矩 + K 线形状 + Amihud 非流动性。

经济假设
    本方向的系统性瓶颈是 ICIR。9 条已评分记录（含 0008 三腿合成 rank_ic=0.0451、
    icir=0.44）的 IC 均值都已够用，卡在日间抖动上。0008 证明「腿间集成」走对了路：
    三条腿单独 icir 为 0.43 / 0.32 / 0.28，等权合成后 0.44，高于任一单腿；0010 在
    合成值上再做 5 日时序平滑，rank_ic 与 icir 双降（0.0451→0.0415、0.44→0.40），把
    「往时间轴方向压噪声」这条路证伪。0009（20 日 Amihud 非流动性水平）rank_ic 0.0212
    过线、单腿 icir 仅 0.14，却与库内 pool 因子逐日截面相关均值最大只有 0.05——低
    ICIR 正说明它的噪声构成与前三腿不同源，是集成需要的正交成分。

    本因子因此沿「加腿」方向走：在 0008 的三条腿之外并入 Amihud 非流动性腿，四条腿
    各刻画一种互不相通的定价偏差：

    * 腿 A（收益分解，20 日）：日收益拆成隔夜段 ``今开/昨收 - 1`` 与日内段
      ``今收/今开 - 1``，取价差 ``隔夜 - 日内`` 在 20 日窗口上的累计 / 标准差。价差高
      （反复高开、盘中回吐）的票，开盘溢价一次次被日内抛压否定；价差低（反复低开、
      盘中收复）的票，开盘抛压被日内买盘接走。
    * 腿 B（量能二阶矩，20 日）：成交量变异系数取负。均量相同但忽大忽小的票筹码在短线
      资金间频繁易手、持有者结构松散；量能平稳的票筹码锁定良好、潜在抛压可预期。CV
      把量能水平本身约掉，只留离散度。
    * 腿 C（K 线形状，10 日）：下影长度合计 / 振幅合计。长下影是盘中下探被买回的痕迹，
      窗口内合计刻画护盘是否持续；按振幅加权让真正走出区间的日子主导份额。
    * 腿 D（流动性水平，20 日）：Amihud 非流动性 ``|日收益| / 成交额`` 的 20 日均值。
      同样吃下一元成交额，价格被推得越远的票越缺乏流动性，持有需要更高的预期收益作为
      补偿，故比值越大越看多。它只看「一元成交额在盘口上砸出的价格幅度」，不看成交额
      相对自身历史的放大，与腿 B 的量能离散度、与库内两个变化率口径的流动性因子都不
      是同一维度。

    四条腿面对的资金、信息与时间尺度互不重合（收益分解 / 量能二阶矩 / 日内价格路径
    形状 / 单位成交额的价格冲击），IC 的噪声成分近似独立：等权 rank 合成后信号成分
    叠加、噪声成分对消，IC 均值基本保留而 IC 标准差随腿数下降，ICIR 抬升。因子值越大
    越看多。

与库内同方向因子的区别
    库内 13 个 pool 因子（mom_5/20、ma_bias_10/20、volatility_20、max_drawdown_20、
    range_pct、reversal_5、price_volume_corr_10、turnover_proxy、vol_ratio_5、
    vol_ratio_5_20_neg、vwap_bias）都是「单一口径的时序统计量」，一个因子一条腿。本
    因子不在任一单口径上微调窗口或符号，而是把四条机制互异的腿在截面 rank 空间等权
    集成，信号来自腿间互补与噪声对消，任何单条腿独自都拿不到同样的日间稳定性。

构造
    三条长窗口腿共用 :data:`LONG_WINDOW`（20 日：腿 A 的累计与标准化、腿 B 的变异系数、
    腿 D 的均值），腿 C 用 :data:`SHORT_WINDOW`（10 日）。流程是「排序 → 四条腿 → 逐日
    截面 ``rank().over("date")`` → 等权平均」：

        value = ( rank(腿A) + rank(腿B) + rank(腿C) + rank(腿D) ) / 4

    rank 只用当日截面信息，对 null 产出 null；四腿中任一腿当日缺值即输出 null，不填 0、
    不做截面标准化、也不做任何时序平滑（0010 已证伪）。腿的窗口与公式对齐 0008 / 0009，
    便于把本轮效果单独归因到「新增一条正交腿」这一处改动。

使用字段：open, high, low, close, volume, amount, adjfactor（外加键列 date、
    instrument）。
窗口：腿 A / 腿 B / 腿 D 各 20 个交易日（:data:`LONG_WINDOW`），腿 C 10 个交易日
    （:data:`SHORT_WINDOW`）；``shift`` 只取正数 1；滚动统计量一律 ``over("instrument")``，
    窗口未铺满时输出 null（polars 的 ``min_samples`` 默认等于窗口长度，腿内依赖该默认，
    不显式传参）。
复权口径：后复权。隔夜段 = (open_t × adjfactor_t) / (close_{t-1} × adjfactor_{t-1}) - 1，
    日内段 = (close_t × adjfactor_t) / (open_t × adjfactor_t) - 1；腿 C 的影线与振幅按
    当日 adjfactor 缩放后再跨日合计，除权跳空不进入因子；腿 D 的日收益同样用
    ``close × adjfactor`` 计算，成交额为当日原值（元），不需要复权。腿 B 只用量（股数），
    除权不改变成交股数的经济含义，无需复权。

数据质量兜底：A 股涨跌幅限制为 ±10% / ±20% / ±30%，复权后单日隔夜跳空或日内振幅不会
    超过 50%，超界的点只可能来自复权接缝一类的脏数据，当日价差记 null、不进入累计与
    波动。腿 D 在停牌日（``amount ≤ 0``，分母无意义）与脏收益（``|ret| > MAX_ABS_LEG``）
    上记 null，当日不参与均值；条件表达式对 NaN 恒判假，脏价产生的 NaN 也一并落到
    null，故输出里的缺失只有 null 一种形态。窗口内有效样本不足时 value 为 null，不填 0。

契约：输入面板的列齐备性与 dtype 由沙盒 schema 校验保证，行序按契约已为
    ``(instrument, date)`` 递增，函数内部仍显式排序一次，防止行序漂移让 ``shift`` 与
    滚动窗口错位。全程 polars 向量化：无逐行循环、无 IO、无随机数、无当日之后的数据
    （shift 只取正数，统计量只用后向滚动窗口），输出恰好 (date, instrument, value)
    三列、value 为 Float64。模块内用 ``from polars import col as C, ...`` 一类的短别名
    引用高频 API，纯粹为把 AST 节点数压在沙盒阈值 500 以内，不改变任何计算语义。
"""
from __future__ import annotations

import polars as pl
from polars import col as C, max_horizontal as hi_max, min_horizontal as hi_min
from polars import sum_horizontal as sh, when as W

#: 三条长窗口腿（A 收益分解 / B 量能变异系数 / D Amihud 非流动性）共用的滚动窗口。
LONG_WINDOW: int = 20

#: 腿 C（K 线形状）的合计窗口。
SHORT_WINDOW: int = 10

#: 单日复权收益 / 价差的合理上界，越界视为复权接缝一类的脏数据记 null。
MAX_ABS_LEG: float = 0.5

#: 预热交易日数：最长链路为 1 阶滞后 + 20 日滚动窗口，留余量取 30。
WARMUP: int = 30

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adjfactor",
)


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """四腿等权截面 rank 合成，返回 ``(date, instrument, value)`` 三列长表。

    ``value`` 为 Float64；任一腿窗口不足或记 null 的当日输出 null，不填 0。值越大越
    看多。

    :param data: 长表面板，含 :data:`REQUIRED_COLUMNS` 各列；内部按
        ``(instrument, date)`` 重新排序以防行序漂移。
    :return: 恰好 ``(date, instrument, value)`` 三列，行序不作要求。
    """
    # 列表达式取短名，后面反复引用；跨日比价一律后复权（价 × adjfactor）。
    op, cl, sc = C("open"), C("close"), C("adjfactor")
    vol, amt = C("volume"), C("amount")
    ac = cl * sc
    oa = op * sc
    prev_ac = ac.shift(1).over("instrument")
    raw_ret = ac / prev_ac - 1.0

    # 腿 A：隔夜段与日内段，价差超界视为复权接缝一类的脏数据。
    overnight = oa / prev_ac - 1.0
    intraday = ac / oa - 1.0
    # 不满足条件（含条件为 null）的行在 when/then 下自动为 null，不填 0。
    raw_spread = W(hi_max(overnight.abs(), intraday.abs()) <= MAX_ABS_LEG).then(
        overnight - intraday
    )

    # 腿 C 的日度分子 / 分母：复权口径的影线长度与振幅。
    shadow = sc * (hi_min(op, cl) - C("low"))
    range_sum = (sc * (C("high") - C("low"))).rolling_sum(SHORT_WINDOW).over("instrument")

    # 各腿的滚动中间量：窗口未铺满时为 null，不做任何填充。
    sp = C("_spread")
    spread_std = sp.rolling_std(LONG_WINDOW).over("instrument")
    vol_mean = vol.rolling_mean(LONG_WINDOW).over("instrument")

    return (
        data.sort("instrument", "date")
        .with_columns(_spread=raw_spread)
        # 腿 A 的当日截面去均值：只取个股相对当日全市场的偏离，当日信息、无前视。
        # 去均值分两步按列做：把窗口表达式嵌进另一个 ``over`` 在 polars 下会静默失真。
        .with_columns(_spread=sp - sp.mean().over("date"))
        # 四条腿先各自落成一列。腿内带 ``over("instrument")`` 的表达式不能直接嵌进外层的
        # ``over("date")``：外层按日期分组后每组只剩一只票，组内的 shift / rolling 全部
        # 落空并静默返回 null（本地实测过）。物化成列后，rank 只对列取当日截面。
        .with_columns(
            _leg_spread=W(spread_std > 0.0).then(
                sp.rolling_sum(LONG_WINDOW).over("instrument") / spread_std
            ),
            _leg_volcv=W(vol_mean > 0.0).then(
                -vol.rolling_std(LONG_WINDOW).over("instrument") / vol_mean
            ),
            _leg_shadow=W(range_sum > 0.0).then(
                shadow.rolling_sum(SHORT_WINDOW).over("instrument") / range_sum
            ),
            # 腿 D：单位成交额造成的价格冲击；停牌（amount ≤ 0）与脏收益记 null。
            _leg_illiq=W((amt > 0.0) & (raw_ret.abs() <= MAX_ABS_LEG))
            .then(raw_ret.abs() / amt)
            .rolling_mean(LONG_WINDOW)
            .over("instrument"),
        )
        # 逐日截面 rank 拉平量纲后等权平均。腿名是静态字面量，这里只拼接表达式、不遍历
        # 数据行；ignore_nulls=False 让任一腿缺值的当日 value 记 null，不填 0。
        .select(
            "date",
            "instrument",
            value=sh(
                [
                    C(leg).rank().over("date")
                    for leg in (
                        "_leg_spread",
                        "_leg_volcv",
                        "_leg_shadow",
                        "_leg_illiq",
                    )
                ],
                ignore_nulls=False,
            )
            / 4.0,  # 四条腿等权平均
        )
    )
