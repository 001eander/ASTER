"""Barra 风格因子本地计算（issue #68）。

指数增强优化器需要六类风格暴露约束：``beta`` / ``momentum`` / ``nlsize`` /
``reverse`` / ``sigma`` / ``turnover``。仓库没有外购的风格因子库，本模块从自有的
后复权行情面板（``close × adjfactor``）与等效市值推导这六个因子，逐日截面标准化
（z-score）后供 :mod:`quant.portfolio.enhanced` 使用。

口径
----
- **收益**：``ret = pct_change(px)``，``px = close × adjfactor``（后复权），停牌日无行
  时收益为缺失，不参与滚动窗口。
- **市场收益**：当日全部证券等权平均 ``ret``，用于 ``beta``。
- **beta**：``cov(ret_i, mkt) / var(mkt)``，窗口 :data:`BETA_WINDOW`。
- **momentum**：``px[t−21] / px[t−252] − 1``，跳过最近一个月（Barra 口径）。
- **reverse**：``−(px[t] / px[t−21] − 1)``，即最近一个月收益取负。
- **sigma**：``ret`` 的滚动标准差，窗口 :data:`SIGMA_WINDOW`。
- **turnover**：``amount / px`` 的滚动均值（成交额 / 等效市值，缺历史股本时的代理），
  窗口 :data:`TURNOVER_WINDOW`。
- **nlsize**：``(log(px))³``，``px`` 即等效市值（见下）。
- **标准化**：每个因子按 ``date`` 做截面 z-score（``ddof=1``），缺失值填 0（中性）。
  窗口预热不足（新股、停牌）的原始值为缺失，标准化后同样归 0。

等效市值口径
------------
无历史股本数据，用 ``px = close × adjfactor × share_const`` 作为等效市值，``share_const``
为常数（默认 1.0）。常数因子在截面标准化里被约掉，不影响六因子的相对比较；
``turnover`` 的分子 ``amount`` 与分母 ``px`` 同乘常数后仍为比例量，同样不受影响。
历史股本数据到达后可替换 ``share_const`` 之外的市值来源，接口不变。

输出
----
``compute_style_factors`` 返回宽表，列为 ``date, instrument`` + 六个因子名，
按 ``(instrument, date)`` 排序。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 六风格因子名（顺序固定，供约束族逐因子遍历）。
STYLE_FACTOR_NAMES: tuple[str, ...] = (
    "beta",
    "momentum",
    "nlsize",
    "reverse",
    "sigma",
    "turnover",
)

#: beta 回归窗口（交易日）。
BETA_WINDOW: int = 60
#: momentum 总窗口与跳过窗口（跳过最近一个月）。
MOMENTUM_WINDOW: int = 252
MOMENTUM_SKIP: int = 21
#: reverse 窗口（最近一个月）。
REVERSE_WINDOW: int = 21
#: sigma 窗口。
SIGMA_WINDOW: int = 60
#: turnover 代理窗口。
TURNOVER_WINDOW: int = 21

#: 滚动窗口最小有效观测比例（预热不足则为缺失）。
MIN_PERIODS_RATIO: float = 2.0 / 3.0

#: 等效市值常数股本；常数在截面标准化与比例量中约掉，取 1.0 即可。
SHARE_CONST_DEFAULT: float = 1.0

#: 截面 z-score 的自由度。
ZSCORE_DDOF: int = 1

#: 计算所需的行情列。
REQUIRED_BAR_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "close",
    "adjfactor",
    "amount",
)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def compute_style_factors(
    bars: pl.DataFrame,
    *,
    share_const: float = SHARE_CONST_DEFAULT,
) -> pl.DataFrame:
    """从后复权行情面板计算六风格因子，输出逐日截面标准化的宽表。

    Parameters
    ----------
    bars:
        日行情，至少含 ``date / instrument / close / adjfactor / amount``；重复
        ``(date, instrument)`` 取最后一行。
    share_const:
        等效市值常数股本，见模块文档。

    Returns
    -------
    ``pl.DataFrame``，列 ``date, instrument`` + :data:`STYLE_FACTOR_NAMES`，
    按 ``(instrument, date)`` 排序。
    """
    missing = [c for c in REQUIRED_BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"行情面板缺少列: {missing}")

    df = (
        bars.select(list(REQUIRED_BAR_COLUMNS))
        .unique(subset=["date", "instrument"], keep="last")
        .sort(["instrument", "date"])
    )

    min_beta = max(2, int(BETA_WINDOW * MIN_PERIODS_RATIO))
    min_sigma = max(2, int(SIGMA_WINDOW * MIN_PERIODS_RATIO))
    min_turnover = max(1, int(TURNOVER_WINDOW * MIN_PERIODS_RATIO))

    df = df.with_columns(
        (pl.col("close") * pl.col("adjfactor") * share_const).alias("_px")
    )
    df = df.with_columns(
        pl.col("_px").pct_change().over("instrument", order_by="date").alias("_ret")
    )
    df = df.with_columns(
        pl.col("_ret").mean().over("date").alias("_mkt")
    )

    # beta = cov(ret_i, mkt) / var(mkt) = corr × std_i / std_mkt
    corr_expr = pl.rolling_corr(
        pl.col("_ret"),
        pl.col("_mkt"),
        window_size=BETA_WINDOW,
        min_samples=min_beta,
    ).over("instrument", order_by="date")
    std_i_expr = (
        pl.col("_ret")
        .rolling_std(window_size=BETA_WINDOW, min_samples=min_beta)
        .over("instrument", order_by="date")
    )
    std_mkt_expr = (
        pl.col("_mkt")
        .rolling_std(window_size=BETA_WINDOW, min_samples=min_beta)
        .over("instrument", order_by="date")
    )

    df = df.with_columns(
        [
            (corr_expr * std_i_expr / std_mkt_expr).alias("beta"),
            (
                pl.col("_px").shift(MOMENTUM_SKIP).over("instrument", order_by="date")
                / pl.col("_px").shift(MOMENTUM_WINDOW).over("instrument", order_by="date")
                - 1.0
            ).alias("momentum"),
            (
                -(
                    pl.col("_px")
                    / pl.col("_px").shift(REVERSE_WINDOW).over("instrument", order_by="date")
                    - 1.0
                )
            ).alias("reverse"),
            pl.col("_ret")
            .rolling_std(window_size=SIGMA_WINDOW, min_samples=min_sigma)
            .over("instrument", order_by="date")
            .alias("sigma"),
            (pl.col("amount") / pl.col("_px"))
            .rolling_mean(window_size=TURNOVER_WINDOW, min_samples=min_turnover)
            .over("instrument", order_by="date")
            .alias("turnover"),
            (pl.col("_px").log() ** 3).alias("nlsize"),
        ]
    )

    for name in STYLE_FACTOR_NAMES:
        df = df.with_columns(_zscore(df, name))

    out = df.select(
        ["date", "instrument", *STYLE_FACTOR_NAMES]
    ).sort(["instrument", "date"])
    return out


def equivalent_market_value(
    bars: pl.DataFrame,
    *,
    share_const: float = SHARE_CONST_DEFAULT,
) -> pl.DataFrame:
    """等效市值 ``close × adjfactor × share_const``，列为 ``date, instrument, equiv_mv``。

    无历史股本数据时的本地口径，常数因子在截面比较中约掉（见模块文档）。指数增强
    优化器的 ``float_mv`` 可由本函数在 cutoff 日截取得到。
    """
    missing = [c for c in ("date", "instrument", "close", "adjfactor") if c not in bars.columns]
    if missing:
        raise ValueError(f"行情面板缺少列: {missing}")
    return (
        bars.select("date", "instrument", "close", "adjfactor")
        .unique(subset=["date", "instrument"], keep="last")
        .with_columns(
            (pl.col("close") * pl.col("adjfactor") * share_const).alias("equiv_mv")
        )
        .select("date", "instrument", "equiv_mv")
        .sort(["instrument", "date"])
    )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _zscore(df: pl.DataFrame, name: str) -> pl.Expr:
    """逐日截面 z-score；标准差为 0 或缺失时该日因子归 0。"""
    std = pl.col(name).std(ddof=ZSCORE_DDOF).over("date")
    safe_std = pl.when(std > 0).then(std).otherwise(None)
    z = (pl.col(name) - pl.col(name).mean().over("date")) / safe_std
    return (
        pl.when(z.is_finite())
        .then(z)
        .otherwise(0.0)
        .alias(name)
    )


__all__ = [
    "BETA_WINDOW",
    "MIN_PERIODS_RATIO",
    "MOMENTUM_SKIP",
    "MOMENTUM_WINDOW",
    "REVERSE_WINDOW",
    "SHARE_CONST_DEFAULT",
    "SIGMA_WINDOW",
    "STYLE_FACTOR_NAMES",
    "TURNOVER_WINDOW",
    "compute_style_factors",
    "equivalent_market_value",
]
