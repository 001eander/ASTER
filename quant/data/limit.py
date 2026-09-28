"""涨跌停上下限预计算。

量化口径
--------
``limit_up`` / ``limit_down`` 是「当日允许成交的价格边界」，按**前一交易日收盘价**
计算，与当日行情无关（无前视）：::

    limit_up   = round_half_up(prev_close × (1 + ratio), 0.01)
    limit_down = round_half_up(prev_close × (1 - ratio), 0.01)

``prev_close`` 是同一证券按 ``date`` 排序后的 ``close.shift(1)``；证券首个交易日无
前收，两列保持 null。

交易所规则表（2026-09 现行）
----------------------------
================  =========  ======================  ============================
板块              标识       非 ST                    ST（含 ``*ST``）
================  =========  ======================  ============================
主板              ``main``   10%                     5%
创业板            ``cyb``    20%（2020-08-24 起）    改革后 20%；改革前 5%
科创板            ``kcb``    20%                     20%
北交所            ``bj``     30%                     30%
================  =========  ======================  ============================

创业板改革日 2020-08-24（``CYB_REFORM_DATE``）：之前创业板涨跌幅 10%、ST 5%，
之后统一放宽到 20%，改革后的创业板 ST 同样是 20%。

舍入口径
--------
交易所按「四舍五入到分」公布涨跌停价，Python 内置 ``round`` 用的是银行家舍入
（``round(2.5) == 2``），不适用。本模块用 :mod:`decimal` 的 ``ROUND_HALF_UP``，
且乘法全程用 :class:`decimal.Decimal`，避免二进制浮点误差。经典用例：::

    prev_close=5.255, ratio=10% -> limit_up=5.78
    prev_close=5.245, ratio=10% -> limit_up=5.77

ST 状态的来源与局限（重要）
---------------------------
ST 状态随时间变化，且没有权威的免费日期化接口，本模块采用两段式判定：

1. 优先用曾用名历史：若能拿到带**变更日期**的曾用名记录，就把名称含 ``ST`` 的连续
   区间合并成 ST 区间（:func:`st_intervals_from_name_history`）。
2. 退化方案：``akshare 1.18.97`` 的 ``stock_info_change_name`` 实测只返回名称序列、
   **不含变更日期**（见下），无法定位时间区间。此时只按**当前名称**判定——当前名称
   含 ``ST`` 则该证券从上市日到最新都按 ST 处理，否则全程按非 ST 处理。

实测结论（2026-09-28，akshare 1.18.97）
---------------------------------------
``ak.stock_info_change_name(symbol="600519")`` 返回两列 ``index`` / ``name``，
逐行是「贵州茅台」的历史简称序列，**没有**变更日期列；000503 返回 7 个曾用名，
其中含 ``ST海虹``，同样无日期。因此当前版本走退化方案。

退化方案的偏差方向
------------------
- 当前名称**不含** ST：历史上曾经 ST 的日子会被当成非 ST 处理，涨跌幅带比真实更宽
  （主板按 10% 而非 5%），回测会**高估**这些日子的可成交性，偏激进。
- 当前名称**含** ST：把非 ST 时期也按 ST 处理，涨跌幅带比真实更窄（5% 而非 10%），
  回测会**低估**可成交性，偏保守。

因此本模块填出的涨跌停价在历史 ST 段附近不是逐日精确值，回测中触及涨跌停的边界
样本需要意识到这一不确定性。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import akshare as ak
import polars as pl

from quant.data.schema import (
    DAILY_BARS,
    board_of,
    check_daily_bars,
    check_schema,
    normalize_instrument,
)
from quant.data.source.akshare import _call, _from_pandas
from quant.data.source.base import DataSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置：规则表常量
# ---------------------------------------------------------------------------

#: 主板非 ST 涨跌幅比例。
MAIN_LIMIT_RATIO: float = 0.10
#: 主板 ST 涨跌幅比例。
MAIN_ST_LIMIT_RATIO: float = 0.05
#: 创业板放开后的涨跌幅比例（含改革后的 ST）。
CYB_LIMIT_RATIO: float = 0.20
#: 创业板注册制改革生效日：2020-08-24 起涨跌幅由 10% 放宽到 20%。
CYB_REFORM_DATE: date = date(2020, 8, 24)
#: 科创板涨跌幅比例（恒 20%，含 ST）。
KCB_LIMIT_RATIO: float = 0.20
#: 北交所涨跌幅比例（恒 30%，含 ST）。
BJ_LIMIT_RATIO: float = 0.30

#: 无上市日时的历史起点兜底（上交所开市日）。
EARLIEST_TRADE_DATE: date = date(1990, 12, 19)
#: 开区间 end_date 的判定哨兵（远未来，仅用于区间比较，不写回结果）。
END_SENTINEL: date = date(9999, 12, 31)
#: 价格最小变动单位。
CENT: Decimal = Decimal("0.01")
#: 曾用名记录中出现 ST 的判定关键字。
ST_MARKER: str = "ST"

#: ST 区间表结构。``end_date`` 为 null 表示至今。
ST_INTERVALS = pl.Schema(
    {
        "instrument": pl.String,
        "start_date": pl.Date,
        "end_date": pl.Date,
    }
)

#: 曾用名变更记录表结构（带日期时）。
NAME_HISTORY = pl.Schema(
    {
        "date": pl.Date,
        "name": pl.String,
    }
)

# ---------------------------------------------------------------------------
# 规则函数
# ---------------------------------------------------------------------------


def limit_ratio(board: str, day: date, is_st: bool) -> float:
    """返回指定板块 / 日期 / ST 状态下的涨跌幅比例。

    与 :func:`compute_limits` 向量化实现共用同一张规则表，测试保证二者一致。
    """
    if board == "kcb":
        return KCB_LIMIT_RATIO
    if board == "bj":
        return BJ_LIMIT_RATIO
    if board == "cyb" and day >= CYB_REFORM_DATE:
        return CYB_LIMIT_RATIO
    return MAIN_ST_LIMIT_RATIO if is_st else MAIN_LIMIT_RATIO


def is_st_name(name: str | None) -> bool:
    """证券简称含 ``ST``（含 ``*ST``）即视为 ST。"""
    if not name:
        return False
    return ST_MARKER in name.upper()


# ---------------------------------------------------------------------------
# 舍入
# ---------------------------------------------------------------------------


def _round_half_up(value: Decimal) -> float:
    """四舍五入到分（交易所口径）。"""
    return float(value.quantize(CENT, rounding=ROUND_HALF_UP))


def _limit_price(
    prev_close: float | None, ratio: float | None, direction: int
) -> float | None:
    """按前收与比例算一侧涨跌停价，全程 Decimal + ROUND_HALF_UP。"""
    if prev_close is None or ratio is None:
        return None
    factor = Decimal(1) + Decimal(direction) * Decimal(str(ratio))
    return _round_half_up(Decimal(str(prev_close)) * factor)


def _limit_up(prev_close: float | None, ratio: float | None) -> float | None:
    return _limit_price(prev_close, ratio, 1)


def _limit_down(prev_close: float | None, ratio: float | None) -> float | None:
    return _limit_price(prev_close, ratio, -1)


def _limit_up_row(row: dict[str, Any]) -> float | None:
    """:meth:`polars.Expr.map_elements` 的 struct 入口，保持类型标注完整。"""
    return _limit_up(row["prev_close"], row["ratio"])


def _limit_down_row(row: dict[str, Any]) -> float | None:
    return _limit_down(row["prev_close"], row["ratio"])


# ---------------------------------------------------------------------------
# 曾用名历史
# ---------------------------------------------------------------------------


def _empty_name_history() -> pl.DataFrame:
    return pl.DataFrame(schema=NAME_HISTORY)


def _empty_intervals() -> pl.DataFrame:
    return pl.DataFrame(schema=ST_INTERVALS)


def _normalize_name_history(raw: pl.DataFrame) -> pl.DataFrame:
    """把 akshare 的曾用名返回归一化成 ``date`` / ``name`` 两列。

    akshare 1.18.97 只给 ``index`` / ``name``、不含日期，此时返回空表，调用方据此
    走退化方案。若未来接口补上日期列（``date`` / ``变更日期`` / ``日期``），这里会
    自动识别。
    """
    if raw.height == 0:
        return _empty_name_history()
    lower = {str(col).lower(): col for col in raw.columns}
    name_col = lower.get("name") or lower.get("名称") or lower.get("证券简称")
    date_col = (
        lower.get("date") or lower.get("变更日期") or lower.get("日期") or lower.get("change_date")
    )
    if name_col is None or date_col is None:
        return _empty_name_history()
    return (
        raw.select(
            pl.col(date_col).cast(pl.Utf8).str.to_date(strict=False).alias("date"),
            pl.col(name_col).cast(pl.String).alias("name"),
        )
        .drop_nulls()
        .sort("date")
    )


def _fetch_name_history(instrument: str) -> pl.DataFrame:
    """拉取单只证券的曾用名变更记录（``date`` / ``name``）。

    经 :func:`_call` 做有限重试。当前 akshare 1.18.97 无日期信息，恒返回空表，
    :func:`build_st_intervals` 只在探测到日期后才会逐票调用本函数。
    """
    digits = normalize_instrument(instrument).split(".")[0]
    raw = _call(ak.stock_info_change_name, symbol=digits)
    return _normalize_name_history(_from_pandas(raw))


def st_intervals_from_name_history(
    history: pl.DataFrame, instrument: str
) -> pl.DataFrame:
    """把带日期的曾用名变更记录转成 ST 区间表。

    约定每条记录自其 ``date`` 起生效，到下一记录的前一天为止；最后一条记录延伸到
    至今（``end_date`` 为 null）。名称含 ``ST`` 的**连续**记录合并为一个区间，这样
    相邻区间不会重叠；若期间摘帽又再次戴帽，则拆成两个区间。
    """
    if history.height == 0:
        return _empty_intervals()
    records = (
        history.select(
            pl.col("date").cast(pl.Date),
            pl.col("name").cast(pl.String),
        )
        .drop_nulls("date")
        .sort("date")
        .unique(subset=["date"], keep="last")
        .rows()
    )
    intervals: list[dict[str, Any]] = []
    open_start: date | None = None
    open_end: date | None = None
    for index, (record_date, name) in enumerate(records):
        next_date = records[index + 1][0] if index + 1 < len(records) else None
        tenure_end = next_date - timedelta(days=1) if next_date is not None else None
        if is_st_name(name):
            if open_start is None:
                open_start = record_date
            open_end = tenure_end
        elif open_start is not None:
            intervals.append(
                {"instrument": instrument, "start_date": open_start, "end_date": open_end}
            )
            open_start = None
            open_end = None
    if open_start is not None:
        intervals.append(
            {"instrument": instrument, "start_date": open_start, "end_date": open_end}
        )
    if not intervals:
        return _empty_intervals()
    return pl.DataFrame(intervals, schema=ST_INTERVALS).sort(["instrument", "start_date"])


def build_st_intervals(source: DataSource, instruments: list[str]) -> pl.DataFrame:
    """构建 ST 日期区间表。

    先探测带日期的曾用名接口是否可用：可用则逐票从名称历史推区间；不可用（akshare
    1.18.97 即如此）则退化为按当前简称判定，当前含 ``ST`` 的证券从上市日覆盖到至今。
    单票出错记 warning 后继续，不影响其余证券。偏差方向见模块 docstring。
    """
    wanted = [normalize_instrument(instrument) for instrument in instruments]
    if not wanted:
        return _empty_intervals()

    info = source.instrument_info()
    info_sub = info.filter(pl.col("instrument").is_in(wanted))
    current = {
        row["instrument"]: row for row in info_sub.iter_rows(named=True)
    }

    dated_supported = False
    try:
        probe = _fetch_name_history(wanted[0])
        dated_supported = probe.height > 0
    except Exception as exc:  # noqa: BLE001 - 探测失败即退回退化方案
        logger.warning("曾用名接口探测失败（%s），改用当前简称判定 ST", exc)

    frames: list[pl.DataFrame] = []
    if dated_supported:
        for instrument in wanted:
            try:
                history = _fetch_name_history(instrument)
            except Exception as exc:  # noqa: BLE001 - 单票失败不影响整批
                logger.warning("抓取 %s 曾用名历史失败：%s", instrument, exc)
                continue
            frame = (
                st_intervals_from_name_history(history, instrument)
                if history.height
                else _fallback_intervals(instrument, current)
            )
            if frame.height:
                frames.append(frame)
    else:
        for instrument in wanted:
            frame = _fallback_intervals(instrument, current)
            if frame.height:
                frames.append(frame)

    if not frames:
        return _empty_intervals()
    out = pl.concat(frames).sort(["instrument", "start_date"]).cast(ST_INTERVALS)
    check_schema(out, ST_INTERVALS, name="st_intervals")
    return out


def _fallback_intervals(
    instrument: str, current: dict[str, dict[str, Any]]
) -> pl.DataFrame:
    """退化方案：当前简称含 ST 时，从上市日（缺失则兜底）覆盖到至今。"""
    meta = current.get(instrument)
    if meta is None:
        logger.warning("证券信息中缺少 %s，按非 ST 处理", instrument)
        return _empty_intervals()
    if not is_st_name(meta.get("name")):
        return _empty_intervals()
    start = meta.get("list_date") or EARLIEST_TRADE_DATE
    return pl.DataFrame(
        {"instrument": [instrument], "start_date": [start], "end_date": [None]},
        schema=ST_INTERVALS,
    )


# ---------------------------------------------------------------------------
# 预计算
# ---------------------------------------------------------------------------


def compute_limits(
    bars: pl.DataFrame,
    instruments: pl.DataFrame,
    st_intervals: pl.DataFrame,
) -> pl.DataFrame:
    """填充 ``DAILY_BARS`` 的 ``limit_up`` / ``limit_down`` 两列。

    - ``prev_close`` 为同证券前一交易日收盘，证券首行 null，对应两列保持 null。
    - 板块取自 ``instruments.board``，缺失时按证券代码用 ``board_of`` 兜底。
    - ``is_st`` 由 ``(instrument, date)`` 是否落在 ``st_intervals`` 判定，区间为闭区间，
      ``end_date`` 为 null 表示至今。
    - 比例判定用 ``pl.when`` 向量化，价格舍入用 Decimal（``map_elements``），不做逐行
      Python 循环处理行情。
    """
    check_schema(bars, DAILY_BARS, name="bars")
    if bars.height == 0:
        return bars.cast(DAILY_BARS)

    out = bars.sort(["instrument", "date"])

    # 板块：证券信息优先，代码推断兜底（board_of 是纯函数，不引入前视）。
    code_board = (
        out.select("instrument")
        .unique()
        .with_columns(
            pl.col("instrument")
            .map_elements(board_of, return_dtype=pl.String)
            .alias("_code_board")
        )
    )
    info_board = instruments.select(
        pl.col("instrument"), pl.col("board").alias("_info_board")
    )
    out = out.join(code_board, on="instrument", how="left").join(
        info_board, on="instrument", how="left"
    )
    board = pl.coalesce(pl.col("_info_board"), pl.col("_code_board"))

    # ST 标记：(instrument, date) 落在任一 ST 区间内。
    if st_intervals.height:
        intervals = st_intervals.with_columns(
            pl.col("end_date").fill_null(END_SENTINEL).alias("_end_eff")
        )
        st_flag = (
            out.select("instrument", "date")
            .join(
                intervals.select("instrument", "start_date", "_end_eff"),
                on="instrument",
                how="left",
            )
            .with_columns(
                (
                    pl.col("start_date").is_not_null()
                    & (pl.col("date") >= pl.col("start_date"))
                    & (pl.col("date") <= pl.col("_end_eff"))
                ).alias("_is_st")
            )
            .group_by(["instrument", "date"])
            .agg(pl.col("_is_st").any().alias("_is_st"))
        )
        out = out.join(st_flag, on=["instrument", "date"], how="left")
    else:
        out = out.with_columns(pl.lit(False).alias("_is_st"))
    is_st = pl.col("_is_st").fill_null(False)
    out = out.with_columns(is_st.alias("_is_st"))

    out = out.with_columns(
        pl.col("close")
        .shift(1)
        .over("instrument", order_by="date")
        .alias("prev_close")
    )

    pre_reform_ratio = (
        pl.when(is_st).then(MAIN_ST_LIMIT_RATIO).otherwise(MAIN_LIMIT_RATIO)
    )
    cyb_ratio = (
        pl.when(pl.col("date") >= CYB_REFORM_DATE)
        .then(CYB_LIMIT_RATIO)
        .otherwise(pre_reform_ratio)
    )
    ratio = (
        pl.when(board == "kcb")
        .then(KCB_LIMIT_RATIO)
        .when(board == "bj")
        .then(BJ_LIMIT_RATIO)
        .when(board == "cyb")
        .then(cyb_ratio)
        .otherwise(pre_reform_ratio)
        .alias("ratio")
    )
    out = out.with_columns(ratio)

    out = out.with_columns(
        pl.struct(["prev_close", "ratio"])
        .map_elements(_limit_up_row, return_dtype=pl.Float64)
        .alias("limit_up"),
        pl.struct(["prev_close", "ratio"])
        .map_elements(_limit_down_row, return_dtype=pl.Float64)
        .alias("limit_down"),
    )

    out = out.select(list(DAILY_BARS.keys())).sort(["instrument", "date"]).cast(DAILY_BARS)
    check_daily_bars(out)
    return out


__all__ = [
    "BJ_LIMIT_RATIO",
    "CYB_LIMIT_RATIO",
    "CYB_REFORM_DATE",
    "KCB_LIMIT_RATIO",
    "MAIN_LIMIT_RATIO",
    "MAIN_ST_LIMIT_RATIO",
    "NAME_HISTORY",
    "ST_INTERVALS",
    "build_st_intervals",
    "compute_limits",
    "is_st_name",
    "limit_ratio",
    "st_intervals_from_name_history",
]
