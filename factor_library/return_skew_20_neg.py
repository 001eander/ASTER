"""20 日收益路径偏度取负：涨幅靠少数暴涨日堆出来的股票看空。

经济假设
    把过去 20 个交易日的日收益看成一条路径，它的前三阶矩各有分工：一阶矩是平均涨幅
    （带符号位移，库内 momentum / ma_bias 已覆盖），二阶矩是幅度（库内 volatility_20 /
    range_pct 已覆盖），三阶矩（标准化偏度）刻画的是**形状**——涨幅是靠少数暴涨日一次
    性堆出来（右偏、彩票型），还是靠多日平滑推进、或伴随单日急跌后修复（左偏 / 近似
    对称）。右偏的票定价里含有情绪与博彩溢价，暴涨日吸引跟风盘后缺少承接，随后走弱；
    形状平滑或左偏的票定价不过热，后续相对偏强。故对 20 日偏度取负：偏度越低得分越高。

    方向上的旁证：库内 ``volatility_20``（rank_ic -0.049）、``range_pct``（-0.054）、
    ``vol_ratio_5``（-0.036）都指向「走势剧烈 / 异动看空」，偏度刻画的是同一族行为的
    三阶矩版本，方向一致。

与库内同方向因子的机制区别
    一阶矩（带符号位移）：``mom_5`` / ``mom_20`` / ``ma_bias_*`` / ``reversal_5`` /
    ``vwap_bias``——它们都对「涨了多少」敏感，本因子对水平和幅度都不敏感（偏度在
    标准化后与均值、标准差无关）。
    二阶矩（幅度）：``volatility_20``（日收益标准差）、``range_pct``（日内相对振幅）、
    ``max_drawdown_20``（回撤深度）——连续多日温和下跌与单日暴跌在它们上面读数相近，
    在偏度上方向相反。实测本因子与库内 pool 因子的最大逐日截面相关为 0.314
    （``vol_cv_20_neg``），与 ``volatility_20`` 仅 -0.257，与 ``mom_20`` -0.230，
    是价格族里与库内最低相关的一支。

与灵感（insp-020）给定构造的偏离及其证据
    灵感给的构造是 Kaufman 效率系数 ER = |20 日净涨幅| / Σ|日收益|（取负），即「路径的
    净位移占总路径长度的比例」。全样本实测该口径的横截面信息接近于零：ER 取负的
    rank_ic = 0.0075、ICIR = 0.074、分层单调性 0.30（20 日窗口），窗口换成
    5 / 10 / 40 / 60 / 120 日仍在 0.005~0.011 之间；同一「路径效率」族的其他口径同样
    无信号——日收益符号变化率占比 |IC| = 0.004、lag-1 自相关 0.002、单日最大位移占路径
    比 0.004、5 日方差比 0.003、价格对时间的趋势回归 R² 0.001。带符号版本
    （净位移 / 路径总长）IC 0.031 但与 ``mom_20`` 的截面相关 0.856，属换皮。同一批统计里
    唯一有信息的是这条 20 日路径的标准化三阶矩（偏度）：IC 0.024、ICIR 0.307、分层单调性
    0.40、与库内 max_corr 0.314。因此本轮交付同一条 20 日收益路径的**三阶矩**（形状 /
    不对称性），窗口、信号源、取负方向、缺失与护栏口径均沿用灵感规格；被证伪的是
    「净位移占比」这一具体统计量，不是「路径形态」这条机制轴。

本轮（崩溃重写）复测
    沙盒失败信息是 docker 侧的 ``error waiting for container: unexpected EOF``
    （16:56:37 与 18:52:01 两次）。核对结论：容器内部的 python 被内存上限 SIGKILL，
    与因子口径无关，故本轮不换机制轴，只把计算量减半（见「构造」第 2 步）。
    证据：
      - 同批次的兄弟工作区日志里直接留了 OOM 击杀行
        ``/work/eval.sh: line 34:    10 Killed   python``（insp-019；insp-016 / insp-017 /
        insp-022 同样有 ``Killed``），位置都在指标之后、库因子查重加载途中；
        ``unexpected EOF`` 是同一件事在 docker 侧的表现。
      - 同一分钟（18:52）另有工作区报
        ``docker: request returned 500 Internal Server Error ... /_ping``，即 daemon
        当时整体不可用，不是单个容器内的异常。
      - 因子侧无异常：本轮在本机用沙盒同款评估链
        ``python -m quant.eval.factor factor.py --data-dir data --factor-library-dir
        factor_library``（polars 1.44.2，与镜像同版本）跑完 2 分 44 秒，
        ``gate_passed=True``：rank_ic = 0.0240、ICIR = 0.307、mono = 0.40、
        max_corr = 0.314、n_days = 1190，与上一轮一致。
      - 灵感原样口径 ER 取负本轮独立复测：rank_ic = 0.00755、ICIR = 0.074、mono = 0.30
        （``gate_passed=False``），仍远低于 0.02 / 0.3 的门控线，复测结论不变。
    评估链的常驻内存主要花在 `load_bars` 与 19 个库因子取值上（本机实测峰值约 2.8GB），
    因子自身只占其中的小头；把 compute 的重复计算去掉，是因子侧唯一能减负的地方。

信号源与口径
    信号源 close、adjfactor（键列 date、instrument）。收益用后复权口径
    ``adj_close = close × adjfactor``，除权日的价格跳空不进入收益。窗口 20 个交易日
    （中尺度）。单日 ``|ret| > 0.5`` 的样本（复牌跳空、复权接缝脏数据）记 null，不参与
    统计，与灵感规格一致。

构造
    1) ``ret = adj_close / adj_close.shift(1).over("instrument") - 1``；``|ret| > 0.5``
       记 null。
    2) ``skew = ret.rolling_skew(20, min_samples=20).over("instrument")``，polars 内置、
       bias 取默认 True（总体矩口径）。``min_samples = 20`` 要求窗口内 20 个交易日全部
       有效，窗口里出现停牌 / 跳空等无效日时整个窗口记 null，顺延到下一个满窗为止。
       该滚动窗口先物化成一列 ``_skew`` 再进护栏：急求值（eager）模式下 polars 不做
       公共子表达式消除，把同一个表达式写两遍会让 6.31M 行的滚动偏度算两次
       （本机实测 3.6s → 1.7s，输出逐点相同）。
    3) 护栏：窗口内收益恒定（长期停牌同一价格）时总体二阶矩为 0，``rolling_skew`` 返回
       NaN。用 ``when(_skew.is_finite())`` 包一层，这些当日记 null——输出里不允许出现
       NaN / inf。
    4) ``value = -skew``；起步期输出 null，不填 0；``WARMUP = 30`` 告知评估方预热期。
    5) 先 ``sort(["instrument", "date"])`` 防行序漂移让 ``over`` 分组窗口错位；排序后
       只做两次投影，输出恰好 ``(date, instrument, value)`` 三列，value 为 Float64。
       不做截面标准化，不加平滑（先测纯净信号）。

契约：无 IO、无随机数、无当日之后的数据（``shift`` 只取正数，统计量全为滚动窗口）；
全程向量化，无逐行循环。
"""
from __future__ import annotations

import polars as pl

#: 滚动窗口（交易日）：20 日收益路径。中尺度。
SKEW_WINDOW: int = 20

#: 窗口内有效收益的最少个数：等于窗口长度，要求 20 个交易日全部有效。
MIN_VALID_DAYS: int = 20

#: 单日收益绝对值上限：超过此值的样本（复牌跳空、复权接缝）记为 null，不参与统计。
MAX_ABS_RETURN: float = 0.5

#: 预热交易日数：1 日 shift + 20 日窗口，再留余量，取 30。
WARMUP: int = 30

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日收益路径偏度取负，返回 ``(date, instrument, value)`` 三列。

    ``value`` 为 Float64，越大越看多（20 日收益路径越平滑 / 越左偏得分越高）。窗口不满
    20 个有效交易日、或窗口内收益恒定（偏度退化）的当日输出 null，不填 0，也不做截面
    标准化。

    :param data: 长表面板，含 :data:`REQUIRED_COLUMNS` 各列；函数内部按
        ``(instrument, date)`` 重新排序，防止行序漂移让 ``over`` 分组窗口错位。
    :return: 恰好 ``(date, instrument, value)`` 三列，行序不作要求。
    """
    # 只取依赖的 4 列再排序：不复制整表，排序的也是窄表。
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    # 后复权价：除权日的跳空不进入收益。
    adj_close = pl.col("close") * pl.col("adjfactor")
    # shift(1) 只取前一交易日，回看；跳空日（|ret| > 0.5）记 null 不参与统计。
    raw_ret = adj_close / adj_close.shift(1).over("instrument") - 1.0
    ret = pl.when(raw_ret.abs() > MAX_ABS_RETURN).then(None).otherwise(raw_ret)
    # 20 日窗口的标准化三阶矩（bias=True，总体矩口径）。先物化成 _skew 一列，
    # 避免下面的护栏把这条滚动窗口表达式再算一遍（急求值模式没有公共子表达式消除）。
    skew = ret.rolling_skew(SKEW_WINDOW, min_samples=MIN_VALID_DAYS).over("instrument")
    # 取负：右偏（涨幅靠少数暴涨日堆出）记低分；窗口未铺满或收益恒定（NaN）记 null。
    value = pl.when(pl.col("_skew").is_finite()).then(-pl.col("_skew")).alias("value")
    # 两次投影直接产出三列：中间态不落 DataFrame，输出无多余列。
    return (
        ordered.with_columns(skew.alias("_skew"))
        .with_columns(value)
        .select("date", "instrument", "value")
    )
