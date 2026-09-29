"""指数增强优化输入组装（issue #71 抽出，供每日跑批与端到端回测共用）。

组装四类按 cutoff 截取的输入，口径与 :mod:`quant.portfolio.enhanced` 的约束族一一对应：

1. **基准权重**：:func:`quant.data.index_members.index_weights_on` 取 ``<= ref_date``
   最近的 PIT 日频权重（原始口径，和约 1）。
2. **行业**：:func:`quant.data.cache.load_industry` 取 ``effective_from <= ref_date``
   的最新一行（东财一级）。历史快照缺失时退化到表内**最新一份**并把 ``fallback``
   置 True——没有历史快照时这是唯一可用的行业截面，属于数据可得性限制，
   调用方须显式记账（见 PR 说明），不得静默使用。
3. **流通市值**：优先用 ``data/float_mv.parquet`` 的真实流通市值（千元，
   :func:`quant.data.float_mv.read_float_mv`）；该表缺失或覆盖为空时退化到
   :func:`quant.portfolio.style.equivalent_market_value`（后复权价 × 常数股本）。
   市值约束只用到截面相对量，两种口径都可用，但口径不同会影响组合的加权市值锚。
4. **六风格**：:func:`quant.portfolio.style.compute_style_factors` 全窗口算一次，
   按 cutoff 取每票最近一行。

另提供 :func:`returns_matrix`：候选证券的日收益矩阵（列顺序与候选清单严格一致），
协方差估计的输入。

无前视
------
所有截取只用 ``date <= ref_date`` 的行；``load_industry`` 的退化分支是唯一的例外，
它读的是「当前」行业截面，由调用方决定是否接受并在报告中说明。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import polars as pl

from quant.data.cache import load_industry
from quant.data.float_mv import read_float_mv
from quant.data.index_members import index_weights_on
from quant.portfolio.style import (
    STYLE_FACTOR_NAMES,
    compute_style_factors,
    equivalent_market_value,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DATE_COL: str = "date"
INSTRUMENT_COL: str = "instrument"

#: 收益矩阵默认回溯交易日数（协方差窗口）。
DEFAULT_RETURNS_WINDOW: int = 250

#: 流通市值列的规范名（真实流通市值与等效市值共用）。
FLOAT_MV_COL: str = "float_mv"


# ---------------------------------------------------------------------------
# 基准权重
# ---------------------------------------------------------------------------


def benchmark_weights(
    data_dir: str | Path, index_code: str, ref_date: date
) -> dict[str, float]:
    """基准在 ``ref_date`` 或之前最近的 PIT 权重 ``{instrument: weight}``。

    表内没有该指数、或没有不晚于 ``ref_date`` 的记录时返回空字典。
    """
    frame = index_weights_on(Path(data_dir), index_code, ref_date)
    return {
        str(row[INSTRUMENT_COL]): float(row["weight"])
        for row in frame.iter_rows(named=True)
    }


# ---------------------------------------------------------------------------
# cutoff 截取
# ---------------------------------------------------------------------------


def latest_rows(
    frame: pl.DataFrame,
    ref_date: date,
    columns: Sequence[str],
    *,
    date_col: str = DATE_COL,
    instrument_col: str = INSTRUMENT_COL,
) -> pl.DataFrame:
    """每只证券取 ``<= ref_date`` 最近一行，只保留 ``instrument`` + ``columns``。

    ``frame`` 需含 ``date`` / ``instrument``；空表或区间内没有行时返回零行表
    （schema 为 ``instrument`` + ``columns``）。
    """
    schema: dict[str, pl.DataType] = {instrument_col: pl.String}
    for name in columns:
        schema[name] = frame.schema[name] if name in frame.schema else pl.Float64
    if frame.height == 0:
        return pl.DataFrame(schema=schema)
    window = frame.filter(pl.col(date_col) <= ref_date).sort(
        [instrument_col, date_col]
    )
    if window.height == 0:
        return pl.DataFrame(schema=schema)
    return window.group_by(instrument_col).agg(
        [pl.col(name).last() for name in columns]
    )


def float_mv_map(frame: pl.DataFrame, ref_date: date) -> dict[str, float]:
    """每只证券 ``<= ref_date`` 最近一个流通市值（:data:`FLOAT_MV_COL` 列）。"""
    latest = latest_rows(frame, ref_date, [FLOAT_MV_COL])
    return {
        str(row[INSTRUMENT_COL]): float(row[FLOAT_MV_COL])
        for row in latest.iter_rows(named=True)
        if row[FLOAT_MV_COL] is not None
    }


def style_map(
    frame: pl.DataFrame, ref_date: date
) -> dict[str, dict[str, float]]:
    """每只证券 ``<= ref_date`` 最近一日的六风格暴露。"""
    latest = latest_rows(frame, ref_date, list(STYLE_FACTOR_NAMES))
    out: dict[str, dict[str, float]] = {}
    for row in latest.iter_rows(named=True):
        values = {
            name: float(row[name])
            for name in STYLE_FACTOR_NAMES
            if row[name] is not None
        }
        if values:
            out[str(row[INSTRUMENT_COL])] = values
    return out


def industry_table(
    data_dir: str | Path, ref_date: date, instruments: Sequence[str]
) -> tuple[pl.DataFrame, bool]:
    """行业归属表 + 是否退化为「最新一份快照」（见模块文档）。

    返回 ``(frame, fallback)``；``frame`` 列固定为
    ``instrument / industry_l1 / industry_l2 / effective_from``。
    """
    frame = load_industry(Path(data_dir), as_of=ref_date, instruments=list(instruments))
    if frame.height:
        return frame, False
    # 没有不晚于 ref_date 的快照：退化到表内最新一份（可能为空表）。
    return load_industry(Path(data_dir), instruments=list(instruments)), True


def industry_map(frame: pl.DataFrame) -> dict[str, str]:
    """行业表 → ``{instrument: industry_l1}``。"""
    if frame.height == 0:
        return {}
    return {
        str(row[INSTRUMENT_COL]): str(row["industry_l1"])
        for row in frame.iter_rows(named=True)
    }


# ---------------------------------------------------------------------------
# 流通市值来源
# ---------------------------------------------------------------------------


def float_mv_frame(data_dir: str | Path, bars: pl.DataFrame) -> pl.DataFrame:
    """``(date, instrument, float_mv)`` 全窗口表：真实流通市值优先，缺失退化到等效市值。"""
    frame = read_float_mv(Path(data_dir))
    if frame.height:
        return frame.select(DATE_COL, INSTRUMENT_COL, FLOAT_MV_COL)
    return equivalent_market_value(bars).rename({"equiv_mv": FLOAT_MV_COL})


def style_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """六风格因子全窗口表（``compute_style_factors`` 的别名，便于调用方统一入口）。"""
    return compute_style_factors(bars)


# ---------------------------------------------------------------------------
# 收益矩阵
# ---------------------------------------------------------------------------


def returns_matrix(
    bars: pl.DataFrame,
    instruments: Sequence[str],
    ref_date: date,
    *,
    window: int = DEFAULT_RETURNS_WINDOW,
) -> pl.DataFrame:
    """候选证券的日收益矩阵（列顺序与 ``instruments`` 一致）。

    收益用后复权价 ``close × adjfactor`` 计算；缺失 / 非有限值填 0，候选证券在窗口内
    完全没有行情时整列填 0，保证优化器拿到的矩阵始终有限。取 ``ref_date`` 及以前最近
    ``window`` 个交易日。
    """
    instruments = list(instruments)
    selected = bars.filter(
        (pl.col(DATE_COL) <= ref_date)
        & pl.col(INSTRUMENT_COL).is_in(instruments)
    ).sort([INSTRUMENT_COL, DATE_COL])
    selected = selected.with_columns(
        (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
    ).with_columns(
        pl.when(
            (pl.col("_adj_close").shift(1).over(INSTRUMENT_COL) > 0)
            & pl.col("_adj_close").is_finite()
        )
        .then(
            pl.col("_adj_close")
            / pl.col("_adj_close").shift(1).over(INSTRUMENT_COL)
            - 1.0
        )
        .otherwise(None)
        .alias("_ret")
    )
    pivot = selected.pivot(on=INSTRUMENT_COL, index=DATE_COL, values="_ret")
    for instrument in instruments:
        if instrument not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0.0).alias(instrument))
    cleaned = pivot.select(instruments).with_columns(
        [
            pl.when(pl.col(instrument).is_finite())
            .then(pl.col(instrument))
            .otherwise(0.0)
            .alias(instrument)
            for instrument in instruments
        ]
    )
    return cleaned.tail(window)


def subset_weights(
    weights: Mapping[str, float], instruments: Sequence[str]
) -> dict[str, float]:
    """把权重裁剪到候选集内，只保留正权重。"""
    allowed = set(instruments)
    return {
        instrument: float(weight)
        for instrument, weight in weights.items()
        if instrument in allowed and float(weight) > 0.0
    }


__all__ = [
    "DATE_COL",
    "DEFAULT_RETURNS_WINDOW",
    "FLOAT_MV_COL",
    "INSTRUMENT_COL",
    "benchmark_weights",
    "float_mv_frame",
    "float_mv_map",
    "industry_map",
    "industry_table",
    "latest_rows",
    "returns_matrix",
    "style_frame",
    "style_map",
    "subset_weights",
]
