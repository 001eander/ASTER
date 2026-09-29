"""因子评价指标内核：IC / RankIC / ICIR / 分层 / 换手。

本模块是 IC、RankIC、ICIR、分层收益这些指标的**唯一实现处**（AGENTS.md 量化纪律第 2 条），
其他模块一律引用这里，不要另行实现。

输入约定
--------
所有函数的输入是长表 ``(date, instrument, factor, label)``，列名固定：

- ``date``：截面日期（``pl.Date``）。
- ``instrument``：证券代码（``pl.String``，如 ``600000.SH``）。
- ``factor``：因子值（``pl.Float64``），可含 null。
- ``label``：未来收益标签（``pl.Float64``），可含 null。标签口径由 ``quant/labels/`` 定义，
  本模块只消费，不产生。

方向约定
--------
因子值越大越看多。因此 IC 为正表示因子与未来收益同向。
分层编号：``layer 0`` 是因子值最低的一层，``layer n_layers - 1`` 是最高的一层。

null 与样本量
-------------
逐日截面内，因子或标签为 null 的证券一律剔除；当日有效证券数低于 ``MIN_CROSS_SECTION_COUNT``
（默认 30）时 IC 记 null 并写 warning。分层不设该阈值，但要求当日有效证券数不少于层数。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 输入长表固定列名。
DATE_COL: str = "date"
INSTRUMENT_COL: str = "instrument"
FACTOR_COL: str = "factor"
LABEL_COL: str = "label"

#: 基准绩效函数的输入 / 输出列名（issue #67）。
NAV_COL: str = "nav"
CLOSE_COL: str = "close"
BENCHMARK_NAV_COL: str = "benchmark_nav"
EXCESS_NAV_COL: str = "excess_nav"
PORTFOLIO_RET_COL: str = "portfolio_ret"
BENCHMARK_RET_COL: str = "benchmark_ret"
EXCESS_RET_COL: str = "excess_ret"

#: 跟踪误差低于此阈值视为 0（浮点噪声），信息比率记 None。
TRACKING_ERROR_MIN: float = 1e-12

#: 逐日 IC 的最少有效证券数，低于此值当日 IC 记 null。
MIN_CROSS_SECTION_COUNT: int = 30

#: 分层默认层数。
DEFAULT_LAYERS: int = 5

#: ICIR 年化用的年交易日数（仅在 ``annualize=True`` 时使用）。
TRADING_DAYS_PER_YEAR: int = 252

Method = Literal["spearman", "pearson"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _require_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> None:
    """缺列时抛 ``ValueError``，消息里列出缺失列名。"""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"输入缺少必需列：{missing}；实际列：{df.columns}")


def _require_non_negative_int(value: int, name: str, *, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，实际为 {value}")


# ---------------------------------------------------------------------------
# IC / RankIC
# ---------------------------------------------------------------------------


def ic_series(
    df: pl.DataFrame,
    *,
    method: Method = "spearman",
    min_count: int = MIN_CROSS_SECTION_COUNT,
) -> pl.DataFrame:
    """逐日截面 IC 序列，返回 ``(date, ic)``。

    - ``method="spearman"``（默认）先对当日因子值与标签各自取秩（``rank``，并列取平均），
      再算 Pearson 相关，即 RankIC。``method="pearson"`` 直接算因子值与标签的 Pearson 相关。
    - 当日因子或标签为 null 的证券剔除后，有效证券数小于 ``min_count`` 时该日 ``ic`` 为 null，
      并写一条 warning。输入里出现但当日无任何有效证券的日期同样保留为 null。
    - 当日常数因子（或常数标签）方差为 0，相关无定义，该日 ``ic`` 为 null，不额外报 warning。
    - 返回的 ``date`` 为输入中出现过的所有日期，升序；``ic`` 为 ``Float64``。
    """
    _require_columns(df, (DATE_COL, FACTOR_COL, LABEL_COL))
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method 只支持 'spearman' / 'pearson'，实际为 {method!r}")
    _require_non_negative_int(min_count, "min_count", minimum=1)

    valid = df.select(DATE_COL, FACTOR_COL, LABEL_COL).drop_nulls(
        [FACTOR_COL, LABEL_COL]
    )
    if method == "spearman":
        valid = valid.with_columns(
            pl.col(FACTOR_COL).rank().over(DATE_COL).alias("_factor"),
            pl.col(LABEL_COL).rank().over(DATE_COL).alias("_label"),
        )
    else:
        valid = valid.with_columns(
            pl.col(FACTOR_COL).alias("_factor"),
            pl.col(LABEL_COL).alias("_label"),
        )

    aggregated = valid.group_by(DATE_COL).agg(
        pl.len().alias("_count"),
        pl.corr("_factor", "_label").alias("_ic"),
    )

    out = (
        df.select(DATE_COL)
        .unique()
        .sort(DATE_COL)
        .join(aggregated, on=DATE_COL, how="left")
        .with_columns(pl.col("_count").fill_null(0))
    )

    low = out.filter(pl.col("_count") < min_count)
    if low.height:
        logger.warning(
            "ic_series: %d 个交易日的有效证券数少于 %d，IC 记 null：%s%s",
            low.height,
            min_count,
            low.select(DATE_COL).head(3).to_series().to_list(),
            "..." if low.height > 3 else "",
        )

    return out.with_columns(
        pl.when(pl.col("_count") >= min_count)
        .then(pl.col("_ic").fill_nan(None))
        .otherwise(None)
        .alias("ic")
    ).select(DATE_COL, "ic")


@dataclass(frozen=True)
class ICSummary:
    """IC 序列的汇总统计。

    字段口径：``n_days`` 为 ``ic`` 非 null 的天数；``mean`` / ``std`` 在这批非 null 值上计算，
    ``std`` 用样本标准差（``ddof=1``）；``icir = mean / std``；``ic_win_rate`` 为 ``ic > 0`` 的占比。

    ICIR 默认**不年化**（日频 IC 的 mean/std 已是日频口径）。``annualize=True`` 时乘以
    ``sqrt(252)``。``std`` 为 0 或有效天数不足 2 时，``icir`` 为 None。
    """

    mean: float | None
    std: float | None
    icir: float | None
    ic_win_rate: float | None
    n_days: int
    annualized: bool = False


def summarize_ic(ic: pl.DataFrame, *, annualize: bool = False) -> ICSummary:
    """把 :func:`ic_series` 的输出汇总成 :class:`ICSummary`。

    ``ic`` 列全为 null（或输入为空）时，除 ``n_days=0`` 外各统计量为 None。
    """
    _require_columns(ic, (DATE_COL, "ic"))
    raw = ic.get_column("ic")
    series = raw.filter(raw.is_finite())
    n_days = series.len()
    if n_days == 0:
        return ICSummary(None, None, None, None, 0, annualize)

    mean = float(series.mean())
    std = float(series.std()) if n_days >= 2 else None

    icir: float | None = None
    if std is not None and std != 0.0:
        icir = mean / std
        if annualize:
            icir *= math.sqrt(TRADING_DAYS_PER_YEAR)

    ic_win_rate = float((series > 0).sum()) / n_days
    return ICSummary(mean, std, icir, ic_win_rate, n_days, annualize)


# ---------------------------------------------------------------------------
# 分层
# ---------------------------------------------------------------------------


def layered_returns(
    df: pl.DataFrame,
    *,
    n_layers: int = DEFAULT_LAYERS,
) -> pl.DataFrame:
    """逐日按因子值分 ``n_layers`` 层，返回 ``(date, layer, ret, count)``。

    - 每层 ``ret`` 是当日该层证券标签的等权平均（label 为 null 的证券已剔除）。
    - 层编号：``layer 0`` = 因子值最低层，``layer n_layers - 1`` = 因子值最高层。
    - 分层按当日因子值升序排名后等分（``rank(method="ordinal")`` 保证并列有确定归属，
      并列时按输入行序），因子值相同的证券可能被分到相邻层。
    - 要求每个交易日有效证券数不少于 ``n_layers``，否则抛 ``ValueError``（不静默降层，
      避免不同日期的层号含义漂移）。有效证券数充足时每层至少一只证券。
    - 返回按 ``(date, layer)`` 升序。
    """
    _require_columns(df, (DATE_COL, INSTRUMENT_COL, FACTOR_COL, LABEL_COL))
    _require_non_negative_int(n_layers, "n_layers", minimum=2)

    valid = df.select(DATE_COL, INSTRUMENT_COL, FACTOR_COL, LABEL_COL).drop_nulls(
        [FACTOR_COL, LABEL_COL]
    )
    if valid.height == 0:
        raise ValueError("输入没有任何有效的 (factor, label) 行，无法分层")

    counts = valid.group_by(DATE_COL).agg(pl.len().alias("_n"))
    too_few = counts.filter(pl.col("_n") < n_layers)
    if too_few.height:
        examples = too_few.select(DATE_COL).head(3).to_series().to_list()
        raise ValueError(
            f"{too_few.height} 个交易日的有效证券数少于层数 {n_layers}，"
            f"无法分层（示例日期：{examples}）"
        )

    valid = valid.with_columns(
        pl.col(FACTOR_COL)
        .rank(method="ordinal")
        .over(DATE_COL)
        .cast(pl.Int64)
        .alias("_rank"),
        pl.len().over(DATE_COL).cast(pl.Int64).alias("_n"),
    ).with_columns(
        (((pl.col("_rank") - 1) * n_layers) // pl.col("_n")).alias("layer")
    )

    return (
        valid.group_by(DATE_COL, "layer")
        .agg(
            pl.col(LABEL_COL).mean().alias("ret"),
            pl.len().cast(pl.Int64).alias("count"),
        )
        .sort(DATE_COL, "layer")
        .select(DATE_COL, "layer", "ret", "count")
    )


def layer_monotonicity(layered: pl.DataFrame) -> float | None:
    """各层平均收益对层号的 Spearman 相关。

    先对每层求全样本 ``ret`` 均值（跨日等权，不按 count 加权），再算该均值与 ``layer`` 的
    Spearman 相关。``1.0`` 表示层号越高收益越高，完美单调；``-1.0`` 表示完全反向。
    层数不足 2 或无方差时返回 None。
    """
    _require_columns(layered, (DATE_COL, "layer", "ret"))
    per_layer = (
        layered.group_by("layer")
        .agg(pl.col("ret").mean().alias("_ret"))
        .drop_nulls("_ret")
        .sort("layer")
    )
    if per_layer.height < 2:
        return None

    result = per_layer.select(
        pl.corr(
            pl.col("layer").rank(),
            pl.col("_ret").rank(),
        ).alias("_corr")
    ).item()
    if result is None or not math.isfinite(result):
        return None
    return float(result)


# ---------------------------------------------------------------------------
# 换手
# ---------------------------------------------------------------------------


def turnover(
    df: pl.DataFrame,
    *,
    top_n: int,
    ascending: bool = False,
) -> pl.DataFrame:
    """因子 Top-N 组合的逐日换手率序列，返回 ``(date, turnover)``。

    口径：每日按因子值取 Top-``top_n`` 只证券构成入选集合 :math:`S_t`（``ascending=False``
    取因子值最大的 N 只，``False`` 为默认方向；``ascending=True`` 取最小的 N 只），
    换手率定义为单边替换比例

    .. math::

        turnover_t = |S_t \\setminus S_{t-1}| / top_n

    即当日新买入的证券数占组合规模的比例（等权组合的买卖各为这一比例）。首个交易日没有
    前一日集合，``turnover`` 为 null。因子为 null 的证券不参与入选。

    当某日有效证券数少于 ``top_n`` 时，入选集合会小于 ``top_n``，分母仍取 ``top_n``，
    换手率可能低于实际调仓幅度。
    """
    _require_columns(df, (DATE_COL, INSTRUMENT_COL, FACTOR_COL))
    _require_non_negative_int(top_n, "top_n", minimum=1)

    valid = (
        df.select(DATE_COL, INSTRUMENT_COL, FACTOR_COL)
        .drop_nulls(FACTOR_COL)
        .with_columns(
            pl.col(FACTOR_COL)
            .rank(method="ordinal", descending=not ascending)
            .over(DATE_COL)
            .alias("_rank")
        )
        .filter(pl.col("_rank") <= top_n)
    )

    daily = (
        valid.group_by(DATE_COL)
        .agg(pl.col(INSTRUMENT_COL).sort().alias("_members"))
        .sort(DATE_COL)
        .with_columns(pl.col("_members").shift(1).alias("_prev_members"))
    )

    changed = (
        pl.col("_members").list.set_difference(pl.col("_prev_members")).list.len()
    )
    return daily.with_columns(
        pl.when(pl.col("_prev_members").is_null())
        .then(None)
        .otherwise(changed.cast(pl.Float64) / top_n)
        .alias("turnover")
    ).select(DATE_COL, "turnover")


# ---------------------------------------------------------------------------
# 基准绩效（组合 vs 指数，issue #67）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSummary:
    """组合相对基准指数的超额绩效汇总。

    口径
    ----
    - 输入按日期 **inner join**，只保留组合 nav 与基准收盘都为正的交易日。
    - ``benchmark_nav`` / ``excess_nav`` 均以期初（首个共同交易日）归一为 1.0；
      ``excess_nav = 组合归一净值 / 基准归一净值``。
    - ``benchmark_annualized`` 由基准归一净值几何年化：
      ``benchmark_nav[-1] ** (TRADING_DAYS_PER_YEAR / n_days) - 1``。
    - ``excess_annualized`` 取日超额收益均值的算术年化
      （``mean(excess_ret) * TRADING_DAYS_PER_YEAR``），与跟踪误差同量纲。
    - ``tracking_error = std(excess_ret, ddof=1) * sqrt(TRADING_DAYS_PER_YEAR)``，
      有效超额收益少于 2 个时为 None。
    - ``information_ratio = excess_annualized / tracking_error``，跟踪误差为 0 或
      样本不足时为 None。

    有效超额收益少于 2 个时除 ``n_days`` / 总值外各统计量为 None；无共同交易日时
    全部为 None 且 ``n_days=0``。
    """

    n_days: int
    benchmark_total_return: float | None
    benchmark_annualized: float | None
    excess_total_return: float | None
    excess_annualized: float | None
    tracking_error: float | None
    information_ratio: float | None


def benchmark_nav_series(
    portfolio_nav: pl.DataFrame, benchmark_close: pl.DataFrame
) -> pl.DataFrame:
    """对齐组合 nav 与基准收盘，返回基准 / 超额净值序列。

    输入 ``portfolio_nav`` 含 ``(date, nav)``，``benchmark_close`` 含 ``(date, close)``。
    输出列固定为 ``(date, portfolio_nav, benchmark_nav, excess_nav, portfolio_ret,
    benchmark_ret, excess_ret)``：``portfolio_nav`` / ``benchmark_nav`` 以期初归一为
    1.0，``excess_nav`` 为两者之比，收益率列由相邻净值 / 收盘相除得到（首个交易日
    为 null）。两个输入都按 ``date`` 去重 ``keep="last"`` 后 inner join，只有双方
    都有正值的交易日保留。
    """
    _require_columns(portfolio_nav, (DATE_COL, NAV_COL))
    _require_columns(benchmark_close, (DATE_COL, CLOSE_COL))

    aligned = (
        portfolio_nav.select(DATE_COL, NAV_COL)
        .unique(subset=[DATE_COL], keep="last")
        .join(
            benchmark_close.select(DATE_COL, CLOSE_COL).unique(
                subset=[DATE_COL], keep="last"
            ),
            on=DATE_COL,
            how="inner",
        )
        .filter((pl.col(NAV_COL) > 0) & (pl.col(CLOSE_COL) > 0))
        .sort(DATE_COL)
    )
    if aligned.height == 0:
        return pl.DataFrame(
            schema={
                DATE_COL: pl.Date,
                "portfolio_nav": pl.Float64,
                BENCHMARK_NAV_COL: pl.Float64,
                EXCESS_NAV_COL: pl.Float64,
                PORTFOLIO_RET_COL: pl.Float64,
                BENCHMARK_RET_COL: pl.Float64,
                EXCESS_RET_COL: pl.Float64,
            }
        )

    nav0 = float(aligned[NAV_COL][0])
    close0 = float(aligned[CLOSE_COL][0])
    return (
        aligned.with_columns(
            (pl.col(NAV_COL) / nav0).alias("portfolio_nav"),
            (pl.col(CLOSE_COL) / close0).alias(BENCHMARK_NAV_COL),
        )
        .with_columns(
            (pl.col("portfolio_nav") / pl.col(BENCHMARK_NAV_COL)).alias(
                EXCESS_NAV_COL
            ),
            (pl.col(NAV_COL) / pl.col(NAV_COL).shift(1) - 1.0).alias(
                PORTFOLIO_RET_COL
            ),
            (pl.col(CLOSE_COL) / pl.col(CLOSE_COL).shift(1) - 1.0).alias(
                BENCHMARK_RET_COL
            ),
        )
        .with_columns(
            (pl.col(PORTFOLIO_RET_COL) - pl.col(BENCHMARK_RET_COL)).alias(
                EXCESS_RET_COL
            )
        )
        .select(
            DATE_COL,
            "portfolio_nav",
            BENCHMARK_NAV_COL,
            EXCESS_NAV_COL,
            PORTFOLIO_RET_COL,
            BENCHMARK_RET_COL,
            EXCESS_RET_COL,
        )
    )


def benchmark_performance(
    portfolio_nav: pl.DataFrame, benchmark_close: pl.DataFrame
) -> BenchmarkSummary:
    """按 :func:`benchmark_nav_series` 的对齐口径汇总基准与超额绩效。

    跟踪误差与信息比率的定义见 :class:`BenchmarkSummary`。
    """
    series = benchmark_nav_series(portfolio_nav, benchmark_close)
    n_days = series.height
    if n_days == 0:
        return BenchmarkSummary(0, None, None, None, None, None, None)

    benchmark_final = float(series[BENCHMARK_NAV_COL][-1])
    excess_final = float(series[EXCESS_NAV_COL][-1])
    benchmark_total = benchmark_final - 1.0
    benchmark_annualized = benchmark_final ** (
        TRADING_DAYS_PER_YEAR / n_days
    ) - 1.0

    raw = series.get_column(EXCESS_RET_COL)
    excess = raw.filter(raw.is_finite())
    n_excess = excess.len()
    if n_excess >= 2:
        mean = float(excess.mean())
        std = float(excess.std())
        excess_annualized: float | None = mean * TRADING_DAYS_PER_YEAR
        tracking_error: float | None = std * math.sqrt(TRADING_DAYS_PER_YEAR)
        information_ratio: float | None = (
            excess_annualized / tracking_error
            if tracking_error > TRACKING_ERROR_MIN
            else None
        )
    else:
        excess_annualized = None
        tracking_error = None
        information_ratio = None

    return BenchmarkSummary(
        n_days=n_days,
        benchmark_total_return=benchmark_total,
        benchmark_annualized=benchmark_annualized,
        excess_total_return=excess_final - 1.0,
        excess_annualized=excess_annualized,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
    )


__all__ = [
    "BENCHMARK_NAV_COL",
    "BENCHMARK_RET_COL",
    "BenchmarkSummary",
    "CLOSE_COL",
    "DATE_COL",
    "DEFAULT_LAYERS",
    "EXCESS_NAV_COL",
    "EXCESS_RET_COL",
    "FACTOR_COL",
    "ICSummary",
    "INSTRUMENT_COL",
    "LABEL_COL",
    "MIN_CROSS_SECTION_COUNT",
    "NAV_COL",
    "PORTFOLIO_RET_COL",
    "TRADING_DAYS_PER_YEAR",
    "benchmark_nav_series",
    "benchmark_performance",
    "ic_series",
    "layer_monotonicity",
    "layered_returns",
    "summarize_ic",
    "turnover",
]
