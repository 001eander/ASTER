"""持仓风险分析四表：指数分布 / 市值分布 / 风格暴露 / 主动行业暴露（issue #69）。

范式参照 ``reference-analyze-weights.py``（bydata 数据源，pandas 原型），本模块用
polars 重写，数据源全部换成仓库本地缓存（``index_members`` / ``industry`` / ``bars``）。

四表口径
--------
1. **指数分布**：逐日持仓权重落在 沪深300 / 中证500 / 中证1000 / 中证2000 /
   ``3800以外`` 五桶。归属按 :data:`quant.data.schema.INDEX_CODES` 顺序，同票多指数
   取排名靠前者；不属于任一指数（含无成分数据）归 ``3800以外``。
2. **市值分布**：逐日持仓权重落在 大 / 中 / 小 / 微盘四桶，来源为
   :func:`quant.portfolio.style.equivalent_market_value` 的等效市值（``close ×
   adjfactor × share_const``）。缺等效市值归 ``未知``。
3. **风格暴露**：六风格因子的逐日持仓加权平均。某票缺该因子值时**剔除该票权重并
   在剩余权重上归一**（与参考脚本一致）；当日全部缺失时记 ``None``。
4. **主动行业暴露**：组合东财一级行业权重 − 基准行业权重，逐日。基准权重由调用方
   提供（如 ``index_weights`` 的 PIT 权重），缺基准参数时不出此表。

输入约定
--------
持仓权重序列为长表 ``(date, instrument, weight)``（``weight`` 为小数，和约 1）。
辅助表统一带 ``date`` 列，与持仓按 ``(date, instrument)`` 对齐；行业表也可不带
``date``（静态截面，按 ``instrument`` 广播）。所有 ``date`` 一律归一到 ``pl.Date``。

等效市值与分档阈值
------------------
仓库无历史股本数据，:func:`equivalent_market_value` 的等效市值在 ``share_const = 1.0``
时只是后复权价。为沿用 A 股惯例的市值阈值（500 亿 / 100 亿 / 30 亿），本模块用
:data:`MV_SHARE_CONST`（默认 10 亿股常数股本）把等效市值放大到近似市值量纲，
:data:`MV_LARGE_MIN` / :data:`MV_MID_MIN` / :data:`MV_SMALL_MIN` 即按元计。历史股本到达后
把 ``MV_SHARE_CONST`` 换成逐票股本即可，阈值不变。

无前视
------
全部辅助表只应按 cutoff 及以前的数据构造；本模块不做时间过滤，由调用方保证。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from quant.data.index_members import INDEX_NAMES
from quant.data.schema import INDEX_CODES
from quant.portfolio.enhanced import EnhancedOptimizer, compute_exposures
from quant.portfolio.style import STYLE_FACTOR_NAMES

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

DATE_COL: str = "date"
INSTRUMENT_COL: str = "instrument"
WEIGHT_COL: str = "weight"

#: 五桶中最末的「其余」桶名。
INDEX_BUCKET_OTHER: str = "3800以外"
#: 指数分布列顺序：四个指数 + 其余。
INDEX_BUCKETS: tuple[str, ...] = (
    *(INDEX_NAMES[code] for code in INDEX_CODES),
    INDEX_BUCKET_OTHER,
)

#: 指数归属优先级（0 最高）。
INDEX_RANK: dict[str, int] = {code: i for i, code in enumerate(INDEX_CODES)}

#: 市值四桶 + 缺失桶，输出列顺序。
MV_BUCKET_LARGE: str = "大盘股"
MV_BUCKET_MID: str = "中盘股"
MV_BUCKET_SMALL: str = "小盘股"
MV_BUCKET_MICRO: str = "微盘股"
MV_BUCKET_UNKNOWN: str = "未知"
MV_BUCKETS: tuple[str, ...] = (
    MV_BUCKET_LARGE,
    MV_BUCKET_MID,
    MV_BUCKET_SMALL,
    MV_BUCKET_MICRO,
    MV_BUCKET_UNKNOWN,
)

#: 等效市值 → 近似市值 的常数股本（10 亿股），仅用于把分档阈值换算到价格量纲。
#: 见模块文档「等效市值与分档阈值」。
MV_SHARE_CONST: float = 1.0e9
#: 市值分档阈值（元，近似市值口径）：> 大盘 / > 中盘 / >= 小盘，其余为微盘。
MV_LARGE_MIN: float = 5.0e10
MV_MID_MIN: float = 1.0e10
MV_SMALL_MIN: float = 3.0e9

#: 行业缺失占位符。
UNKNOWN_INDUSTRY: str = "未知"

#: 自洽性检查的数值松弛：求解器容差 + ``_finalize`` 约零归一带来的微小偏移。
#: 判定超带的条件为 ``deviation > tol × (1 + REL_SLACK) + ABS_SLACK``。
CONSISTENCY_REL_SLACK: float = 0.05
CONSISTENCY_ABS_SLACK: float = 1.0e-4


class RiskReportError(ValueError):
    """持仓风险分析的输入或计算不满足约定。"""


class ConsistencyError(AssertionError):
    """指增组合的实测暴露超出优化器设定带。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskReport:
    """四表产出；未提供对应辅助表时该表为 ``None``。

    Attributes
    ----------
    index_distribution:
        ``date`` + :data:`INDEX_BUCKETS`，逐日各桶权重和。
    market_value_distribution:
        ``date`` + :data:`MV_BUCKETS`，逐日各市值段权重和。
    style_exposure:
        ``date`` + 六风格因子，逐日持仓加权暴露。
    active_industry:
        ``date`` + 行业列，组合行业权重 − 基准行业权重；无基准时为 ``None``。
    """

    index_distribution: pl.DataFrame | None = None
    market_value_distribution: pl.DataFrame | None = None
    style_exposure: pl.DataFrame | None = None
    active_industry: pl.DataFrame | None = None

    def tables(self) -> dict[str, pl.DataFrame]:
        """按固定顺序返回非空表 ``{表名: DataFrame}``。"""
        out: dict[str, pl.DataFrame] = {}
        if self.index_distribution is not None:
            out["指数分布"] = self.index_distribution
        if self.market_value_distribution is not None:
            out["市值分布"] = self.market_value_distribution
        if self.style_exposure is not None:
            out["风格暴露"] = self.style_exposure
        if self.active_industry is not None:
            out["主动行业暴露"] = self.active_industry
        return out


# ---------------------------------------------------------------------------
# 输入归一
# ---------------------------------------------------------------------------


def _with_normalized_date(frame: pl.DataFrame) -> pl.DataFrame:
    """把 ``date`` 列归一到 ``pl.Date``，容忍常见字符串格式。"""
    dtype = frame.schema[DATE_COL]
    if dtype == pl.Date:
        expr = pl.col(DATE_COL)
    elif dtype == pl.Datetime:
        expr = pl.col(DATE_COL).dt.date()
    else:
        expr = (
            pl.col(DATE_COL)
            .cast(pl.String)
            .str.replace_all(r"[-/.]", "")
            .str.to_date("%Y%m%d", strict=False)
        )
    return frame.with_columns(expr.alias(DATE_COL))


def normalize_holdings(holdings: pl.DataFrame) -> pl.DataFrame:
    """校验并归一持仓权重长表，输出 ``(date, instrument, weight)`` 按日期排序。"""
    missing = [
        c for c in (DATE_COL, INSTRUMENT_COL, WEIGHT_COL) if c not in holdings.columns
    ]
    if missing:
        raise RiskReportError(f"持仓权重缺少列: {missing}")
    out = _with_normalized_date(holdings).with_columns(
        pl.col(INSTRUMENT_COL).cast(pl.String),
        pl.col(WEIGHT_COL).cast(pl.Float64),
    ).select(DATE_COL, INSTRUMENT_COL, WEIGHT_COL)
    if out[DATE_COL].null_count():
        raise RiskReportError("持仓权重的 date 存在无法解析的值")
    return out.sort([DATE_COL, INSTRUMENT_COL])


def _weights_frame(frame: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """校验 ``(date, instrument, value)`` 辅助表并归一日期。"""
    missing = [c for c in (DATE_COL, INSTRUMENT_COL, value_col) if c not in frame.columns]
    if missing:
        raise RiskReportError(f"辅助表缺少列: {missing}")
    return (
        _with_normalized_date(frame)
        .with_columns(pl.col(INSTRUMENT_COL).cast(pl.String))
        .select(DATE_COL, INSTRUMENT_COL, value_col)
        .sort([DATE_COL, INSTRUMENT_COL])
    )


# ---------------------------------------------------------------------------
# 表 1：指数分布
# ---------------------------------------------------------------------------


def compute_index_distribution(
    holdings: pl.DataFrame, members: pl.DataFrame
) -> pl.DataFrame:
    """逐日持仓权重在五个指数桶上的分布。

    ``members`` 为 :data:`quant.data.schema.INDEX_MEMBERS` 形态
    ``(date, instrument, index_code)``；同票多指数按 :data:`INDEX_RANK` 取排名靠前者。
    """
    hold = normalize_holdings(holdings)
    if members.height == 0:
        tagged = hold.with_columns(pl.lit(INDEX_BUCKET_OTHER).alias("_bucket"))
    else:
        missing = [
            c for c in (DATE_COL, INSTRUMENT_COL, "index_code") if c not in members.columns
        ]
        if missing:
            raise RiskReportError(f"成分表缺少列: {missing}")
        tagged = _members_buckets(members).join(
            hold, on=[DATE_COL, INSTRUMENT_COL], how="right"
        )
        tagged = tagged.with_columns(pl.col("_bucket").fill_null(INDEX_BUCKET_OTHER))

    return _pivot_weights(tagged, INDEX_BUCKETS)


def _members_buckets(members: pl.DataFrame) -> pl.DataFrame:
    """把成分表归约为每 ``(date, instrument)`` 一个指数桶名。"""
    return (
        _with_normalized_date(members)
        .with_columns(
            pl.col(INSTRUMENT_COL).cast(pl.String),
            pl.col("index_code").cast(pl.String),
        )
        .with_columns(
            pl.col("index_code")
            .replace_strict(INDEX_RANK, default=len(INDEX_RANK))
            .alias("_rank"),
            pl.col("index_code")
            .replace_strict(INDEX_NAMES, default=INDEX_BUCKET_OTHER)
            .alias("_bucket"),
        )
        .sort("_rank")
        .unique(subset=[DATE_COL, INSTRUMENT_COL], keep="first", maintain_order=True)
        .select(DATE_COL, INSTRUMENT_COL, "_bucket")
    )


# ---------------------------------------------------------------------------
# 表 2：市值分布
# ---------------------------------------------------------------------------


def compute_market_value_distribution(
    holdings: pl.DataFrame, equiv_mv: pl.DataFrame
) -> pl.DataFrame:
    """逐日持仓权重在大 / 中 / 小 / 微盘四桶上的分布。

    ``equiv_mv`` 为 ``(date, instrument, equiv_mv)``，由
    :func:`quant.portfolio.style.equivalent_market_value` 生成（见模块文档的分档口径）。
    """
    hold = normalize_holdings(holdings)
    mv = _weights_frame(equiv_mv, "equiv_mv")
    tagged = hold.join(mv, on=[DATE_COL, INSTRUMENT_COL], how="left").with_columns(
        _mv_bucket_expr().alias("_bucket")
    )
    return _pivot_weights(tagged, MV_BUCKETS)


def _mv_bucket_expr() -> pl.Expr:
    """等效市值 → 市值桶（``None`` 归 :data:`MV_BUCKET_UNKNOWN`）。"""
    return (
        pl.when(pl.col("equiv_mv").is_null())
        .then(pl.lit(MV_BUCKET_UNKNOWN))
        .when(pl.col("equiv_mv") > MV_LARGE_MIN)
        .then(pl.lit(MV_BUCKET_LARGE))
        .when(pl.col("equiv_mv") > MV_MID_MIN)
        .then(pl.lit(MV_BUCKET_MID))
        .when(pl.col("equiv_mv") >= MV_SMALL_MIN)
        .then(pl.lit(MV_BUCKET_SMALL))
        .otherwise(pl.lit(MV_BUCKET_MICRO))
    )


# ---------------------------------------------------------------------------
# 表 3：风格暴露
# ---------------------------------------------------------------------------


def compute_style_exposure(
    holdings: pl.DataFrame,
    style_factors: pl.DataFrame,
    *,
    factor_names: Sequence[str] = STYLE_FACTOR_NAMES,
) -> pl.DataFrame:
    """六风格因子的逐日持仓加权暴露；缺因子值的票剔除权重后归一。"""
    hold = normalize_holdings(holdings)
    missing = [c for c in (DATE_COL, INSTRUMENT_COL) if c not in style_factors.columns]
    if missing:
        raise RiskReportError(f"风格因子表缺少列: {missing}")
    present = [name for name in factor_names if name in style_factors.columns]
    if not present:
        raise RiskReportError(f"风格因子表缺少全部因子列: {list(factor_names)}")

    factors = _with_normalized_date(style_factors).with_columns(
        pl.col(INSTRUMENT_COL).cast(pl.String)
    ).select(DATE_COL, INSTRUMENT_COL, *present)

    out = hold.select(DATE_COL).unique()
    for name in present:
        joined = hold.join(
            factors.select(DATE_COL, INSTRUMENT_COL, name),
            on=[DATE_COL, INSTRUMENT_COL],
            how="left",
        )
        daily = (
            joined.filter(pl.col(name).is_not_null())
            .group_by(DATE_COL)
            .agg(
                (pl.col(WEIGHT_COL) * pl.col(name)).sum().alias("_num"),
                pl.col(WEIGHT_COL).sum().alias("_den"),
            )
            .with_columns(
                pl.when(pl.col("_den") > 0)
                .then(pl.col("_num") / pl.col("_den"))
                .otherwise(None)
                .alias(name)
            )
            .select(DATE_COL, name)
        )
        out = out.join(daily, on=DATE_COL, how="left")

    for name in factor_names:
        if name not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    return out.select(DATE_COL, *factor_names).sort(DATE_COL)


# ---------------------------------------------------------------------------
# 表 4：主动行业暴露
# ---------------------------------------------------------------------------


def compute_active_industry(
    holdings: pl.DataFrame,
    bench_weights: pl.DataFrame,
    industry: pl.DataFrame,
    *,
    bench_index: str | None = None,
) -> pl.DataFrame:
    """逐日组合行业权重 − 基准行业权重（东财一级行业）。

    ``bench_weights`` 若含 ``index_code`` 列可按 ``bench_index`` 过滤；``industry`` 为
    ``(instrument(, date), industry_l1)``，无 ``date`` 时按证券静态广播。
    """
    hold = normalize_holdings(holdings)
    bench = _benchmark_weights(bench_weights, bench_index)
    if bench.height == 0:
        raise RiskReportError("基准权重为空，无法计算主动行业暴露")

    port = _industry_weights(hold, industry, WEIGHT_COL).rename({WEIGHT_COL: "port_w"})
    bench_ind = (
        _industry_weights(bench, industry, WEIGHT_COL)
        .rename({WEIGHT_COL: "_bench_raw"})
        .with_columns(
            (pl.col("_bench_raw") / pl.col("_bench_raw").sum().over(DATE_COL)).alias(
                "bench_w"
            )
        )
        .select(DATE_COL, "_industry", "bench_w")
    )

    industries = (
        pl.concat(
            [port.select("_industry"), bench_ind.select("_industry")],
            how="vertical",
        )
        .unique()
        .sort("_industry")["_industry"]
        .to_list()
    )
    dates = hold.select(DATE_COL).unique()
    grid = dates.join(pl.DataFrame({"_industry": industries}), how="cross")
    active = (
        grid.join(port, on=[DATE_COL, "_industry"], how="left")
        .join(bench_ind, on=[DATE_COL, "_industry"], how="left")
        .with_columns(
            (pl.col("port_w").fill_null(0.0) - pl.col("bench_w").fill_null(0.0)).alias(
                "_active"
            )
        )
    )
    wide = active.pivot(
        on="_industry", index=DATE_COL, values="_active", aggregate_function="sum"
    )
    for name in industries:
        if name not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(name))
    return wide.select(DATE_COL, *industries).sort(DATE_COL).fill_null(0.0)


def _benchmark_weights(frame: pl.DataFrame, index_code: str | None) -> pl.DataFrame:
    """从基准表（可含 ``index_code``）取 ``(date, instrument, weight)``。"""
    if index_code is not None:
        if "index_code" not in frame.columns:
            raise RiskReportError("指定 bench_index 但基准表缺少 index_code 列")
        frame = frame.filter(pl.col("index_code").cast(pl.String) == index_code)
    return _weights_frame(frame, WEIGHT_COL)


def _industry_weights(
    frame: pl.DataFrame, industry: pl.DataFrame, value_col: str
) -> pl.DataFrame:
    """按行业汇总某一权重表的权重，输出 ``(date, _industry, value_col)``。"""
    ind_col = "industry_l1" if "industry_l1" in industry.columns else "industry"
    if ind_col not in industry.columns:
        raise RiskReportError("行业表缺少 industry_l1 / industry 列")
    ind = industry.with_columns(
        pl.col(INSTRUMENT_COL).cast(pl.String),
        pl.col(ind_col).cast(pl.String),
    )
    if DATE_COL in industry.columns:
        ind = _with_normalized_date(ind).select(DATE_COL, INSTRUMENT_COL, ind_col)
        joined = frame.join(ind, on=[DATE_COL, INSTRUMENT_COL], how="left")
    else:
        ind = ind.select(INSTRUMENT_COL, ind_col).unique(
            subset=[INSTRUMENT_COL], keep="last"
        )
        joined = frame.join(ind, on=INSTRUMENT_COL, how="left")
    return (
        joined.with_columns(
            pl.col(ind_col).fill_null(UNKNOWN_INDUSTRY).alias("_industry")
        )
        .group_by([DATE_COL, "_industry"])
        .agg(pl.col(value_col).sum().alias(value_col))
    )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def build_risk_report(
    holdings: pl.DataFrame,
    *,
    members: pl.DataFrame | None = None,
    equiv_mv: pl.DataFrame | None = None,
    style_factors: pl.DataFrame | None = None,
    bench_weights: pl.DataFrame | None = None,
    industry: pl.DataFrame | None = None,
    bench_index: str | None = None,
) -> RiskReport:
    """按提供的辅助表构造四表；缺对应输入的表记 ``None``。

    - ``members`` → 指数分布；
    - ``equiv_mv`` → 市值分布；
    - ``style_factors`` → 风格暴露；
    - ``bench_weights`` + ``industry`` → 主动行业暴露。
    """
    index_distribution = (
        compute_index_distribution(holdings, members) if members is not None else None
    )
    market_value_distribution = (
        compute_market_value_distribution(holdings, equiv_mv) if equiv_mv is not None else None
    )
    style_exposure = (
        compute_style_exposure(holdings, style_factors) if style_factors is not None else None
    )
    active_industry = (
        compute_active_industry(holdings, bench_weights, industry, bench_index=bench_index)
        if bench_weights is not None and industry is not None
        else None
    )
    return RiskReport(
        index_distribution=index_distribution,
        market_value_distribution=market_value_distribution,
        style_exposure=style_exposure,
        active_industry=active_industry,
    )


# ---------------------------------------------------------------------------
# 自洽性检查
# ---------------------------------------------------------------------------


def check_enhanced_consistency(
    holdings: pl.DataFrame,
    *,
    bench_weights: pl.DataFrame,
    industry: pl.DataFrame,
    style: pl.DataFrame | None = None,
    equiv_mv: pl.DataFrame | None = None,
    optimizer: EnhancedOptimizer | None = None,
    industry_tol: float | None = None,
    style_tol: float | None = None,
    market_value_tol: float | None = None,
    bench_index: str | None = None,
) -> pl.DataFrame:
    """断言指增组合的实测行业 / 市值 / 风格偏离落在优化器设定带内。

    逐日调用 :func:`quant.portfolio.enhanced.compute_exposures`（与优化器同一口径）：
    行业偏离为 ``|组合 − 基准| / 基准``，风格与市值为以基准同口径 std 为单位的 z 值。
    任一偏离超带即抛 :class:`ConsistencyError`；全部通过返回逐日检查表。

    ``industry_tol`` / ``style_tol`` / ``market_value_tol`` 缺省取 ``optimizer`` 的对应
    参数；``optimizer`` 也为空时要求显式给出 ``industry_tol``。``style`` / ``equiv_mv``
    为空则不检查对应维度。
    """
    if optimizer is not None:
        if industry_tol is None:
            industry_tol = optimizer.industry_exposure
        if style_tol is None:
            style_tol = optimizer.style_exposure
        if market_value_tol is None:
            market_value_tol = optimizer.market_value_exposure
    if industry_tol is None:
        raise RiskReportError("未提供 optimizer 时必须显式给出 industry_tol")

    hold = normalize_holdings(holdings)
    bench = _benchmark_weights(bench_weights, bench_index)
    if bench.height == 0:
        raise RiskReportError("基准权重为空，无法做自洽性检查")

    hold_by_date = _partition(hold, WEIGHT_COL)
    bench_by_date = _partition_bench(bench)
    industry_by_date, industry_static = _industry_maps(industry)
    style_by_date = _style_maps(style) if style is not None else None
    mv_by_date = _mv_maps(equiv_mv) if equiv_mv is not None else None

    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    for day in sorted(hold_by_date):
        port = hold_by_date[day]
        bench_map = bench_by_date.get(day)
        if not bench_map:
            continue
        ind_map = industry_by_date.get(day, industry_static)
        style_map = style_by_date.get(day, {}) if style_by_date is not None else {}
        mv_map = mv_by_date.get(day, {}) if mv_by_date is not None else {}

        exposures = compute_exposures(
            port,
            bench_weights=bench_map,
            industry=ind_map,
            float_mv=mv_map,
            style=style_map,
        )

        before = len(violations)
        ind_ratio = _max_industry_ratio(
            exposures["industry"], violations, day, industry_tol
        )
        style_std = (
            _max_style_std(exposures["style"], violations, day, style_tol)
            if style_by_date is not None
            else None
        )
        mv_std = None
        if mv_by_date is not None and market_value_tol is not None:
            mv_std = abs(float(exposures["market_value"]["std"]))
            if _out_of_band(mv_std, market_value_tol):
                violations.append(
                    f"{day.isoformat()} 市值偏离 {mv_std:.6f} 超出 {market_value_tol:.6f}"
                )

        rows.append(
            {
                DATE_COL: day,
                "industry_max_ratio": ind_ratio,
                "style_max_std": style_std,
                "market_value_std": mv_std,
                "violations": len(violations) - before,
            }
        )

    if violations:
        preview = "；".join(violations[:5])
        more = f"（其余 {len(violations) - 5} 条略）" if len(violations) > 5 else ""
        raise ConsistencyError(f"指增组合暴露超出设定带：{preview}{more}")

    schema = {
        DATE_COL: pl.Date,
        "industry_max_ratio": pl.Float64,
        "style_max_std": pl.Float64,
        "market_value_std": pl.Float64,
        "violations": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema).sort(DATE_COL)


def _out_of_band(deviation: float, tol: float) -> bool:
    """偏离是否超带：允许求解器容差与约零归一的数值松弛。"""
    return deviation > tol * (1.0 + CONSISTENCY_REL_SLACK) + CONSISTENCY_ABS_SLACK


def _max_industry_ratio(
    industry_exp: Mapping[str, Mapping[str, float]],
    violations: list[str],
    day: date,
    tol: float,
) -> float:
    """行业偏离最大值，并记录超带项。

    基准权重非零的行业按相对偏离比较；基准为零的行业组合权重必须约为 0
    （与优化器 ``port ≤ 0`` 的约束一致），否则记绝对偏离。
    """
    worst = 0.0
    for code, values in industry_exp.items():
        bench = float(values["bench"])
        port = float(values["portfolio"])
        if bench > 0.0:
            ratio = abs(port - bench) / bench
            worst = max(worst, ratio)
            if _out_of_band(ratio, tol):
                violations.append(
                    f"{day.isoformat()} 行业 {code} 偏离 {ratio:.6f} 超出 {tol:.6f}"
                )
        elif port > CONSISTENCY_ABS_SLACK:
            worst = max(worst, port)
            violations.append(
                f"{day.isoformat()} 行业 {code} 基准权重为 0 但组合持有 {port:.6f}"
            )
    return worst


def _max_style_std(
    style_exp: Mapping[str, Mapping[str, float]],
    violations: list[str],
    day: date,
    tol: float | None,
) -> float:
    """风格偏离（std 倍数）最大值，并记录超带项。"""
    if tol is None:
        return 0.0
    worst = 0.0
    for factor, values in style_exp.items():
        std = abs(float(values["std"]))
        worst = max(worst, std)
        if _out_of_band(std, tol):
            violations.append(
                f"{day.isoformat()} 风格 {factor} 偏离 {std:.6f} 超出 {tol:.6f}"
            )
    return worst


# ---------------------------------------------------------------------------
# 分区辅助
# ---------------------------------------------------------------------------


def _partition(frame: pl.DataFrame, value_col: str) -> dict[date, dict[str, float]]:
    """``{date: {instrument: value}}``。"""
    out: dict[date, dict[str, float]] = {}
    for key, group in frame.group_by(DATE_COL, maintain_order=True):
        day = key[0] if isinstance(key, tuple) else key
        out[day] = {
            str(row[INSTRUMENT_COL]): float(row[value_col])
            for row in group.iter_rows(named=True)
        }
    return out


def _partition_bench(frame: pl.DataFrame) -> dict[date, dict[str, float]]:
    return {
        day: {inst: w for inst, w in values.items() if w > 0.0}
        for day, values in _partition(frame, WEIGHT_COL).items()
    }


def _industry_maps(
    industry: pl.DataFrame,
) -> tuple[dict[date, dict[str, str]], dict[str, str]]:
    """行业表 → ``(逐日映射, 静态映射)``；有 ``date`` 时静态映射为空。"""
    ind_col = "industry_l1" if "industry_l1" in industry.columns else "industry"
    if ind_col not in industry.columns:
        raise RiskReportError("行业表缺少 industry_l1 / industry 列")
    if DATE_COL in industry.columns:
        frame = _with_normalized_date(industry).with_columns(
            pl.col(INSTRUMENT_COL).cast(pl.String),
            pl.col(ind_col).cast(pl.String),
        ).select(DATE_COL, INSTRUMENT_COL, ind_col)
        by_date: dict[date, dict[str, str]] = {}
        for key, group in frame.group_by(DATE_COL, maintain_order=True):
            day = key[0] if isinstance(key, tuple) else key
            by_date[day] = {
                str(row[INSTRUMENT_COL]): str(row[ind_col])
                for row in group.iter_rows(named=True)
            }
        return by_date, {}
    static = {
        str(row[INSTRUMENT_COL]): str(row[ind_col]) for row in industry.iter_rows(named=True)
    }
    return {}, static


def _style_maps(style: pl.DataFrame) -> dict[date, dict[str, dict[str, float]]]:
    """风格因子表 → ``{date: {instrument: {factor: value}}}``。"""
    names = [name for name in STYLE_FACTOR_NAMES if name in style.columns]
    frame = _with_normalized_date(style).with_columns(
        pl.col(INSTRUMENT_COL).cast(pl.String)
    )
    out: dict[date, dict[str, dict[str, float]]] = {}
    for key, group in frame.group_by(DATE_COL, maintain_order=True):
        day = key[0] if isinstance(key, tuple) else key
        mapping: dict[str, dict[str, float]] = {}
        for row in group.iter_rows(named=True):
            item: dict[str, float] = {}
            for name in names:
                value = row.get(name)
                if value is not None:
                    item[name] = float(value)
            mapping[str(row[INSTRUMENT_COL])] = item
        out[day] = mapping
    return out


def _mv_maps(equiv_mv: pl.DataFrame) -> dict[date, dict[str, float]]:
    return _partition(_weights_frame(equiv_mv, "equiv_mv"), "equiv_mv")


# ---------------------------------------------------------------------------
# 透视 / 渲染 / 落盘
# ---------------------------------------------------------------------------


def _pivot_weights(tagged: pl.DataFrame, buckets: Sequence[str]) -> pl.DataFrame:
    """``(date, instrument, weight, _bucket)`` → ``date × buckets`` 宽表。"""
    wide = tagged.group_by([DATE_COL, "_bucket"]).agg(
        pl.col(WEIGHT_COL).sum().alias(WEIGHT_COL)
    )
    if wide.height == 0:
        return pl.DataFrame(
            {
                DATE_COL: pl.Series([], dtype=pl.Date),
                **{b: pl.Series([], dtype=pl.Float64) for b in buckets},
            }
        )
    wide = wide.pivot(
        on="_bucket", index=DATE_COL, values=WEIGHT_COL, aggregate_function="sum"
    )
    for bucket in buckets:
        if bucket not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(bucket))
    return wide.select(DATE_COL, *buckets).sort(DATE_COL).fill_null(0.0)


def frame_to_markdown(df: pl.DataFrame, *, digits: int = 4) -> str:
    """把宽表渲染为 markdown 表；日期列输出 ISO 字符串。"""
    if df.height == 0:
        return "（空）"
    headers = ["日期" if col == DATE_COL else str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in df.iter_rows():
        cells: list[str] = []
        for col, value in zip(df.columns, row):
            if col == DATE_COL:
                cells.append(_fmt_date(value))
            elif value is None:
                cells.append("-")
            elif isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_date(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def render_markdown(
    report: RiskReport,
    *,
    title: str = "持仓风险分析（issue #69）",
    notes: Sequence[str] = (),
) -> str:
    """把四表渲染为一份 markdown 文档。"""
    lines = [f"# {title}", ""]
    if notes:
        lines.append("## 说明")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    for name, df in report.tables().items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(frame_to_markdown(df))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_xlsx(path: Path, sheets: Mapping[str, pl.DataFrame]) -> bool:
    """写出 xlsx；无 xlsxwriter / openpyxl 时返回 ``False``（不改依赖）。"""
    try:
        import xlsxwriter  # noqa: PLC0415
    except ImportError:
        return _write_xlsx_openpyxl(path, sheets)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path))
    header_fmt = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd"})
    num_fmt = workbook.add_format({"num_format": "0.0000"})
    try:
        for sheet_name, df in sheets.items():
            ws = workbook.add_worksheet(sheet_name)
            for col_idx, name in enumerate(df.columns):
                ws.write(0, col_idx, "日期" if name == DATE_COL else str(name), header_fmt)
            for row_idx, row in enumerate(df.iter_rows(), start=1):
                for col_idx, (name, value) in enumerate(zip(df.columns, row)):
                    if name == DATE_COL:
                        ws.write_datetime(row_idx, col_idx, _to_datetime(value), date_fmt)
                    elif value is None:
                        ws.write_blank(row_idx, col_idx, None, num_fmt)
                    else:
                        ws.write_number(row_idx, col_idx, float(value), num_fmt)
            ws.freeze_panes(1, 1)
            ws.set_column(0, 0, 12)
    finally:
        workbook.close()
    return True


def _write_xlsx_openpyxl(path: Path, sheets: Mapping[str, pl.DataFrame]) -> bool:
    try:
        from openpyxl import Workbook  # noqa: PLC0415
    except ImportError:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    for sheet_name, df in sheets.items():
        ws = workbook.create_sheet(title=str(sheet_name)[:31])
        ws.append(["日期" if c == DATE_COL else str(c) for c in df.columns])
        for row in df.iter_rows():
            ws.append(
                [_to_datetime(v) if c == DATE_COL else v for c, v in zip(df.columns, row)]
            )
    workbook.save(str(path))
    return True


def _to_datetime(value: Any) -> Any:
    from datetime import datetime  # noqa: PLC0415

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return value


__all__ = [
    "CONSISTENCY_ABS_SLACK",
    "CONSISTENCY_REL_SLACK",
    "ConsistencyError",
    "DATE_COL",
    "INDEX_BUCKETS",
    "INDEX_BUCKET_OTHER",
    "INDEX_RANK",
    "INSTRUMENT_COL",
    "MV_BUCKETS",
    "MV_BUCKET_UNKNOWN",
    "MV_LARGE_MIN",
    "MV_MID_MIN",
    "MV_SHARE_CONST",
    "MV_SMALL_MIN",
    "RiskReport",
    "RiskReportError",
    "UNKNOWN_INDUSTRY",
    "WEIGHT_COL",
    "build_risk_report",
    "check_enhanced_consistency",
    "compute_active_industry",
    "compute_index_distribution",
    "compute_market_value_distribution",
    "compute_style_exposure",
    "frame_to_markdown",
    "normalize_holdings",
    "render_markdown",
    "write_xlsx",
]
