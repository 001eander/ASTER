"""价量：新四腿等权 rank 合成（路径效率取负 + 爆量频次取负 + 收益偏度取负 + 尾部厚度取负）。

经济假设
    本方向的瓶颈一直在 ICIR：单口径时序统计量的 IC 均值够用，卡在日间抖动（库内 9 条
    已评分记录 ICIR 全在 0.14~0.44，最高的是量能 CV 取负的 0.43）。0011 证明「腿间集成」
    走对了路——三条 icir 0.43 / 0.32 / 0.28 的腿等权 rank 合成后 icir 0.44、rank_ic
    0.0451，高于任一单腿，且 0010（合成值再加 5 日时序平滑）与 0012（合成值再截面规模
    中性化）双双把 IC/ICIR 打下去，说明增益只能来自「加进互不相通的正交信息」，不能
    来自对已有信号做时间或截面上的加工。0013 / 0014 进一步证明腿的权重除等权外无增益。

    本因子把同一套集成术施加于一套全新腿：四条腿各刻画一种互不相通的定价偏差，每条腿
    都是「把每日行为压成一个与量纲无关的标量、再在 20 日窗口上取滚动统计」，量纲统一、
    方向一致（值越大越看多），彼此机制不重叠。

    腿 A（路径效率取负，20 日）：Kaufman 效率比的反面——20 日净位移占同期路径总长
    ``Σ|日收益|`` 的比例。比例高（单边趋势、少回头）说明这段涨幅是被连续同向推动的，
    分歧没有通过换手充分释放、筹码集中在趋势途中，透支后续空间；比例低（来回拉锯）
    说明多空反复交换筹码、分歧已被消化，后续上行动力不易枯竭。
    腿 B（爆量频次取负，60 日基线 / 20 日计数）：最近 20 日里「成交量超过自身前 60
    日常态均量 3 倍」的天数占比。窗口内出现过脉冲式天量的票，当日价格附近有大量筹码
    易手，新承接方缺乏耐心、抛压不可预期；整月没出过爆量日的票换手节奏平稳、浮筹锁定。
    腿 C（收益路径偏度取负，20 日）：20 日日收益的三阶矩。涨幅靠少数暴涨日堆出来
    （右偏、彩票型）的票含情绪与博彩溢价，暴涨日吸引跟风盘后缺少承接；涨势平滑推进或
    伴随单日急跌后修复（左偏 / 对称）的票定价不过热。
    腿 D（尾部厚度取负，20 日）：20 日日收益 95 分位与标准差的比值。同样是 2% 的波动，
    由频繁中小幅日收益堆出来（尾部薄）与由偶发单日暴涨撑起（尾部厚）在定价上不等价：
    后者是博彩偏好资金追逐的形态，当期价格被推到透支水平后回吐。

    四条腿面对的信息、资金与统计矩不同（路径的一阶几何 / 量能的事件频次 / 收益的三阶矩 /
    收益的尾部形状），IC 流的噪声成分近似独立：等权 rank 合成后信号成分叠加、噪声对消，
    IC 均值基本保留而 IC 日间标准差下降，ICIR 抬升。因子值越大越看多。

与灵感规格的偏离及其证据（本地全样本实测，口径与沙盒一致）
    灵感给的配方是「ER 取负 + 爆量峰值比取负 + 量价同步率取负 + 量能 CV 取负」。逐条实测
    后发现两条腿与库内 pool 因子是同一个量，合成后直接撞冗余线：

    * 腿 D（量能 CV 取负）与库内 ``vol_cv_20_neg`` 的逐日截面相关为 **1.000**（同一
      统计量），含它的四腿合成与 ``vol_cv_20_neg`` 的相关 0.767、与 ``three_leg_ic_
      weighted_cv40`` 0.565，超 0.7 判冗余；四腿换成「峰值比」版本（``max20 / mean20``）
      更差：峰值比与 CV 的相关 0.928，「ER + 峰值比 + 同步率 + CV」四腿与
      ``vol_cv_20_neg`` 的相关 0.767。
    * 腿 C（量价方向同步率取负）与库内 ``pv_direction_concordance_20_neg`` 的相关为
      **1.000**，含它的四腿合成与该库因子 0.646~0.667——虽然勉强过 0.7 线，但共线性
      折扣已把分数压到 0.003 量级。
    * 腿 A（ER 取负）实测 rank_ic = 0.0076、ICIR = 0.075，单腿几乎没有信息，与库内任一
      因子的最大相关只有 0.205、与其余三条腿的相关都不超过 0.11——是「零信号但完全
      正交」的腿。
    * 腿 B 的峰值比口径同理（与 ``vol_cv_20_neg`` 相关 0.928），事件频次口径
      （``1{volume_t > 3 × 常态均量}`` 的 20 日占比取负）实测 rank_ic = 0.0302、
      ICIR = 0.373、与库内最大相关 0.581，是同一机制里唯一低相关的写法。

    因此本轮保留灵感的两条腿（A 路径效率、B 爆量——B 用事件频次口径，峰值比口径因
    0.928 的冗余相关不可用），把与库内 pool 因子完全重合的 C、D 换成两条同样属于
    「收益路径形状」、但库内尚无的口径：收益路径偏度（腿 C，实测单腿 0.0240 / 0.308，
    与库内最大相关 0.314）与收益尾部厚度（腿 D，0.0245 / 0.241，与库内最大相关 0.514）。
    这四条腿两两之间的逐日截面相关全部 ≤ 0.35，且与库内 19 个 pool 因子的最大相关
    0.502 ——低于七条已知冗余配方（0.739~0.928）。偏离只改「用哪条腿」，集成术本身
    （逐日截面 rank、等权、无平滑、无中性化、无加权）与灵感规格完全一致。

与库内同方向因子的机制区别
    库内 19 个 pool 因子里，单口径时序统计量（mom / ma_bias / volatility / range /
    reversal / vwap_bias / vol_ratio / turnover_proxy / price_volume_corr）都是「一条腿」
    的因子；两个 ensemble（``four_leg_ensemble_rank``、``three_leg_ic_weighted_cv40``）
    与本因子同属集成族，但腿的构成完全不重叠：它们的腿是「隔夜-日内价差 t 值 / 量能
    CV / 下影承接力 / Amihud 非流动性」，本因子是「路径效率 / 爆量频次 / 收益偏度 /
    收益尾部厚度」，四条腿无一条重合（实测本因子与 ``four_leg_ensemble_rank`` 的相关
    0.464、与 ``three_leg_ic_weighted_cv40`` 0.40 量级）。``pv_direction_concordance_20_neg``
    与 ``vol_cv_20_neg`` 正是被替换掉的两条腿的载体，本因子不含这两条腿。

信号源、窗口与复权口径
    信号源 close、volume、adjfactor（外加键列 date、instrument）。跨日比价一律后复权
    ``adj_close = close × adjfactor``，除权日的价格跳空不进入收益；成交股数不受除权
    影响、除权不改变「换了多少股」的经济含义，故 volume 不做复权。窗口：腿 A、C、D 用
    20 个交易日（:data:`WINDOW`），腿 B 的常态基线用 60 个交易日（:data:`BASELINE_WINDOW`）、
    事件计数 20 日；``shift`` 全部取正数，统计量全是 ``over("instrument")`` 的向后滚动
    窗口，窗口未铺满处输出 null。

构造
    1) 排序 ``sort(["instrument", "date"])``；``ac = close × adjfactor``；
       ``ret = ac / ac.shift(1).over("instrument") - 1``，``|ret| > 0.5`` 视为复权接缝
       一类的脏数据记 null（A 股涨跌停 ±10%/20%/30%，正常日频收益不会超过 50%）。
    2) 四条腿各自物化成一列（带 ``over("instrument")`` 的表达式必须先落地成列，再进
       ``over("date")``；嵌套的 ``over`` 在 polars 下会静默失真——0011 实测过）：
       A ``-(|20 日净位移| / Σ|ret|)``，分母非正（长期停牌）记 null；
       B ``-mean20(1{volume > 3 × mean60(volume).shift(1)})``，基线缺失或非正记 null；
       C ``-rolling_skew(ret, 20)``，窗口内收益恒定（二阶矩为 0）时 polars 返回 NaN，
         用 ``is_finite`` 挡住，输出里只有 null 一种缺失形态；
       D ``-rolling_quantile(ret, 0.95, 20) / rolling_std(ret, 20)``，标准差非正记 null。
    3) 四条腿各自逐日截面 ``rank().over("date")`` 拉平量纲后等权平均：
       ``value = sum_horizontal(rankA, rankB, rankC, rankD, ignore_nulls=False) / 4``，
       任一腿当日缺值即输出 null。不做截面标准化、不做时序平滑（0010 证伪）、不做规模
       中性化（0012 证伪）、不改权重（0013 / 0014 证伪），也不留中间列。
    4) ``WARMUP = 90``：最长链路是腿 B 的 1 阶滞后 + 60 日基线（最少 40 个观测出值）
       + 20 日计数窗口，约 60 个交易日后才有首个有效值，留余量供截断重算检测避开起步期。

契约
    输出恰好 ``(date, instrument, value)`` 三列、``value`` 为 Float64，行序不作要求；
    窗口不足与退化情形一律 null，不填 0。全程 polars 向量化：无逐行循环、无 IO、无随机数、
    不使用任何全序列聚合（无排名回填、无分位数回填），只用当日及之前的数据。
"""
from __future__ import annotations

import polars as pl
from polars import col as C, sum_horizontal as sh, when as W

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 四条腿共用的滚动窗口（交易日）：腿 A / C / D 的统计窗口、腿 B 的事件计数窗口。
WINDOW: int = 20

#: 腿 B 的常态成交量基线窗口（交易日）：个股自身过去 60 个交易日的成交均量。
BASELINE_WINDOW: int = 60

#: 腿 B 基线的最少观测数：窗口 60 日、铺满 40 个观测即出值，把「shift(1) + 基线 +
#: 20 日计数」的整条链压到 60 个交易日内（库内因子的窗口局部性预算，与
#: tests/test_baseline_factors.py 的尾部重算口径一致）；观测不足 40 仍输出 null。
MIN_BASELINE_SAMPLES: int = 40

#: 腿 B 的爆量判定倍数：当日成交量超过基线的该倍数即记一次事件。
SPIKE_MULTIPLE: float = 3.0

#: 腿 D 的上分位点：20 日窗口的上尾高度取 95 分位（线性插值）。
TAIL_QUANTILE: float = 0.95

#: 单日复权收益的合理上界：越界视为复权接缝一类的脏数据，当日记 null、不参与统计。
MAX_ABS_RET: float = 0.5

#: 预热交易日数：腿 B 链路为 1 阶滞后 + 60 日基线（最少 40 个观测）+ 20 日计数，
#: 约 60 日后才有值。
WARMUP: int = 90

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "volume", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """四腿等权截面 rank 合成，返回 ``(date, instrument, value)`` 三列长表。

    ``value`` 为 Float64，越大越看多（路径越拉锯、越没有爆量、收益越左偏、尾部越薄得分
    越高）。任一腿窗口未铺满或记 null 的当日输出 null，不填 0，也不做截面标准化。

    :param data: 长表面板，含 :data:`REQUIRED_COLUMNS` 各列；函数内部按
        ``(instrument, date)`` 重新排序，防止行序漂移让 ``over`` 分组窗口错位。
    :return: 恰好 ``(date, instrument, value)`` 三列，行序不作要求。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"pv_four_leg_ensemble_rank 缺少输入列：{missing}")

    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    sc = C("adjfactor")
    ac = C("close") * sc  # 后复权价：除权跳空不进入收益
    vol = C("volume")
    ret = ac / ac.shift(1).over("instrument") - 1.0
    # 脏收益记 null：条件为 null（序列首行）时 when/then 也落 null，不留 0。
    clean = W(ret.abs() <= MAX_ABS_RET).then(ret)

    # 腿 A：20 日净位移占路径总长的比例取负。
    path = clean.abs().rolling_sum(WINDOW).over("instrument")
    net_move = (ac / ac.shift(WINDOW).over("instrument") - 1.0).abs()

    # 腿 B：常态均量刻意不含当日（shift(1) 后再滚动求均），避免爆量日抬高自己的基线。
    base = (
        vol.shift(1)
        .rolling_mean(BASELINE_WINDOW, min_samples=MIN_BASELINE_SAMPLES)
        .over("instrument")
    )
    spike = (
        W(base.is_null() | (base <= 0.0))
        .then(None)
        .otherwise((vol > SPIKE_MULTIPLE * base).cast(pl.Float64))
    )

    # 四条腿先各自落成一列：带 over("instrument") 的表达式不能直接嵌进 over("date")，
    # 外层按日期分组后组内只剩一只票，组内的 shift / rolling 会静默落空（0011 实测）。
    # 偏度与尾部的原料（rolling_skew / rolling_std / rolling_quantile）先各算一次落列：
    # 急求值模式下 polars 不做公共子表达式消除，同一滚动表达式写两遍会算两遍。
    legs = ordered.with_columns(
        _leg_path=W(path > 0.0).then(-(net_move / path)),
        _leg_spike=-spike.rolling_mean(WINDOW, min_samples=WINDOW).over("instrument"),
        _skew=clean.rolling_skew(WINDOW, min_samples=WINDOW).over("instrument"),
        _sd=clean.rolling_std(WINDOW).over("instrument"),
        _q95=clean.rolling_quantile(
            quantile=TAIL_QUANTILE, interpolation="linear", window_size=WINDOW
        ).over("instrument"),
    )
    # 护栏：窗口内收益恒定（长期停牌）时二阶矩为 0，rolling_skew 返回 NaN、比值退化成
    # inf，两者都记 null；输出里只有 null 一种缺失形态，不填 0、不产生 inf。
    legs = legs.with_columns(
        _leg_skew=W(C("_skew").is_finite()).then(-C("_skew")),
        _leg_tail=W(C("_sd") > 0.0).then(-(C("_q95") / C("_sd"))),
    )

    # 逐日截面 rank 拉平量纲后等权平均；ignore_nulls=False 让任一腿缺值的当日记 null。
    return legs.select(
        "date",
        "instrument",
        value=(
            sh(
                [C(leg).rank().over("date") for leg in ("_leg_path", "_leg_spike", "_leg_skew", "_leg_tail")],
                ignore_nulls=False,
            )
            / 4.0
        ).cast(pl.Float64),
    )
