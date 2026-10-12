"""20 日量价方向同步率取负：涨跌与量能增减同向天数占比高的股票看空。

经济假设
    把每日的「价格往哪走」与「量能往哪走」各压成方向符号，再看两者是否同号，得到一条
    与量纲、幅度都无关的行为序列：放量上涨 / 缩量下跌记同向，缩量上涨 / 放量下跌记
    背离。20 日窗口内同向天数占比高，说明这只票的价格位移是由增量资金抢筹或存量资金
    撤离推动的——买盘在涨势里追加、卖盘在跌势里斩仓，价格运动依赖量能推动，情绪交易
    主导、定价容易过热，后续回落概率高；背离天数占比高（缩量上涨、放量下跌）说明价格
    位移不以量能扩张为前提，浮筹锁定、洗盘充分，后续更稳。因此对 20 日同步率取负：
    背离占比越高得分越高，同向占比越高得分越低。

    方向上的旁证：库内 ``price_volume_corr_10`` 的 rank_ic 为 -0.038（量价正相关看空）、
    ``vol_ratio_5`` 为 -0.036（放量看空），都指向「靠量能推动的上涨后续偏弱」，本因子
    与该方向一致。

与库内同方向因子的机制区别
    ``price_volume_corr_10`` 取的是复权收盘价与成交量的**水平值时序相关**，量纲与幅度
    直接进入皮尔逊相关：同样的方向一致性，若放量日集中在窗口两端，相关系数会与小步放量
    完全不同。本因子先把每日收益与量能变化压成符号再做 0/1 计数，对幅度完全不敏感，
    是同一族信息在**秩口径**下的版本：只有「方向是否一致」的频次结构留存，单日爆量的
    幅度不再放大权重。``vol_cv_20_neg``（量能二阶矩）、``vol_ratio_5_20_neg``（均量比）
    只打量能的水平与波动，不含价格方向，与本因子的信号源不同。``turnover_proxy`` 看成交
    额的绝对放大，本因子看的是价格方向与量能方向的**配对关系**，量能整体放大而价格方向
    无规律时该因子不给出信号。

信号源与口径
    信号源 close、adjfactor、volume（键列 date、instrument）。收益倍数用后复权口径
    ``adj_close = close × adjfactor``，除权日的价格跳空不进入 ret；成交股数不受除权
    影响，除权不改变「成交了多少股」的经济含义，故 volume 不做复权。窗口 20 个交易日
    （中尺度），只用个股自身历史的滚动窗口，``shift`` 一律取正数。

构造
    1) ``ret = adj_close / adj_close.shift(1).over("instrument") - 1``；
       ``dvol = volume - volume.shift(1).over("instrument")``。
       单日 ``|ret| > 0.5`` 的样本记 null（长期停牌复牌、极端跳空不是正常日频行为，
       按灵感规格剔除），不参与计数。
    2) 逐日同步标记 ``sync``：ret、dvol 任一为 null，或 ret 与 dvol 有一侧恰为 0
       （方向缺失），一律记 null（停牌 / 零变动日不计，避免污染占比的分子分母）；
       否则 ``(ret > 0 且 dvol > 0) 或 (ret < 0 且 dvol < 0)`` 记 1.0，异号记 0.0。
       注意这是**等权计数**，不按 |ret|、|dvol| 加权。
    3) ``value = -sync.rolling_mean(20, min_samples=20).over("instrument")``。
       polars 的滚动窗口按非 null 计数，``min_samples=20`` 要求窗口内 20 个交易日
       全部有有效标记，值才落地：此时 value 恰为「有效日同向比例」的相反数，取值落在
       [-1, 0]；窗口里出现停牌等无效日则顺延到下一个满 20 个有效日为止，输出 null，
       不填 0。
    4) 起步期（每只证券的前若干行）输出 null；``WARMUP = 25`` 告知评估方预热期
       （ret 需 1 日 shift，再加 20 日窗口）。
       先 ``sort(["instrument", "date"])`` 防止行序漂移让 ``over`` 分组窗口错位。
       排序后只做**一次投影**：``select("date", "instrument", value)`` 直接把表达式算成
       输出三列，中间态（后复权价、ret、dvol、有效标记、sync）不落 DataFrame，输出恰好
       ``(date, instrument, value)`` 三列，value 为 Float64，不做截面标准化。

上一轮沙盒被杀（``Killed``）的排查结论
    日志停在库因子相关性循环中途、没有 Python 栈，是容器被 OOM killer 杀掉的形态。
    本机（同一台机，Docker VM ``MemTotal`` = 7.7 GB）实测量到：
    - 对照组：把 ``compute`` 换成 ``value = close`` 的平凡因子，同一条评估管线峰值仍是
      **5.86 GB**，这是管线自身的底——行情面板常驻 2.9 GB ＋ 库内 18 个 pool 因子取值
      2.5 GB ＋ 相关性逐因子 join 的瞬时量。分阶段看：读盘后 2.9 GB、截断重算后
      3.1 GB、指标阶段 3.5 GB、库因子取完 5.4 GB、相关性循环峰值 5.7–6.1 GB。
    - 本因子整条管线峰值 6.02 GB，即因子侧（含 6 次 ``compute`` 调用）的边际开销约
      0.16 GB，占底数的 3%。
    run 配置是 ``maxSandboxes = 3``：单次评估 6 GB 能塞进 7.7 GB 的 VM，任意两次重叠就
    把 12 GB 压上去、必然触发 OOM，谁被杀取决于调度时序。这属于评估管线与沙盒配额，
    因子侧改不动。能做的是把自身开销压到最小、并缩短占内存的时间：只搬 5 个输入列、
    重活只有一条 ``rolling_mean`` 套 ``over("instrument")``、排序后一次投影成三列、
    不落任何中间列、不复制整表。这条改法不改变因子口径，逐点与改前的双 DataFrame 写法
    比对完全一致（含 null 形态）。根治手段是并发配额（``maxSandboxes`` 调到 1）或给
    Docker VM 加内存，不在因子文件能触及的范围内。

契约：无 IO、无随机数、无当日之后的数据；全程向量化，无逐行循环。
"""
from __future__ import annotations

import polars as pl

#: 同步率统计窗口（交易日）。中尺度。
SYNC_WINDOW: int = 20

#: 窗口内有效标记的最少个数：等于窗口长度，要求 20 个交易日全部有有效标记。
MIN_VALID_DAYS: int = 20

#: 单日收益绝对值上限：超过此值的样本（复牌跳空等）记为 null，不参与计数。
MAX_ABS_RETURN: float = 0.5

#: 预热交易日数：1 日 shift + 20 日窗口，再留余量，取 25。
WARMUP: int = 25

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "close",
    "adjfactor",
    "volume",
)


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日量价方向同步率取负，返回 ``(date, instrument, value)`` 三列。

    ``value`` 为 Float64，落在 ``[-1, 0]``，越大越看多（量价背离天数占比越高得分越高）。
    窗口不满 20 个有效交易日、或单日收益越过 :data:`MAX_ABS_RETURN` 的当日输出 null，
    不填 0，也不做截面标准化。

    :param data: 长表面板，含 :data:`REQUIRED_COLUMNS` 各列；函数内部按
        ``(instrument, date)`` 重新排序，防止行序漂移让 ``over`` 分组窗口错位。
    :return: 恰好 ``(date, instrument, value)`` 三列，行序不作要求。
    """
    # 只取依赖的 5 列再排序：不把 10 列整表复制一遍，排序的也是窄表。
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    # 后复权价：除权日的跳空不进入收益。
    adj_close = pl.col("close") * pl.col("adjfactor")
    # shift(1) 只取前一交易日，回看，窗口内不出现当日之后的数据。
    ret = adj_close / adj_close.shift(1).over("instrument") - 1.0
    dvol = pl.col("volume") - pl.col("volume").shift(1).over("instrument")
    # 有效日：两侧都有历史、收益在正常日频幅度内、且收益与量能变化都不为 0（方向存在）。
    valid = (
        ret.is_not_null()
        & dvol.is_not_null()
        & (ret.abs() <= MAX_ABS_RETURN)
        & (ret != 0.0)
        & (dvol != 0.0)
    )
    # 方向一致记 1.0，背离记 0.0，等权、不按幅度加权；无效日记 null 不参与统计。
    same_direction = ((ret > 0.0) & (dvol > 0.0)) | ((ret < 0.0) & (dvol < 0.0))
    sync = pl.when(valid).then(same_direction.cast(pl.Float64))
    # 取负：同向占比越高（追涨杀跌盘主导）记分越低。
    value = -sync.rolling_mean(SYNC_WINDOW, min_samples=MIN_VALID_DAYS).over(
        "instrument"
    )
    # 单次投影直接产出三列：中间态不落 DataFrame，输出无多余列。
    return ordered.select("date", "instrument", value.alias("value"))
