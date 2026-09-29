"""``quant.portfolio.style`` 单元测试。

覆盖：六因子输出形状与截面标准化、beta 对已知共动、momentum/reverse 方向、
nlsize 单调、预热不足归零、等效市值辅助函数、输入校验。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quant.portfolio.style import (
    STYLE_FACTOR_NAMES,
    compute_style_factors,
    equivalent_market_value,
)


def _dates(days: int) -> list[str]:
    start = date(2024, 1, 1)
    return [(start + timedelta(days=t)).isoformat() for t in range(days)]


def _frame(series: dict[str, np.ndarray], days: int) -> pl.DataFrame:
    """把 {instrument: 价格序列} 展成行情面板。"""
    rows: list[dict[str, object]] = []
    dates = _dates(days)
    for inst, prices in series.items():
        for t in range(days):
            rows.append(
                {
                    "date": dates[t],
                    "instrument": inst,
                    "close": float(prices[t]),
                    "adjfactor": 1.0,
                    "amount": 1.0e7,
                }
            )
    return pl.DataFrame(rows)


def _random_walk(n: int, days: int, rng: np.random.Generator, drift: float = 0.0) -> np.ndarray:
    ret = drift + rng.standard_normal(days) * 0.01
    level = np.empty(days)
    px = 10.0
    for t in range(days):
        px *= 1.0 + ret[t]
        level[t] = px
    return level


def test_output_shape_and_no_nulls() -> None:
    rng = np.random.default_rng(0)
    bars = _frame({f"60000{i}.SH": _random_walk(0, 300, rng) for i in range(3)}, 300)
    out = compute_style_factors(bars)
    assert out.columns == ["date", "instrument", *STYLE_FACTOR_NAMES]
    assert out.height == 3 * 300
    for factor in STYLE_FACTOR_NAMES:
        assert out[factor].null_count() == 0
        assert out[factor].is_finite().all()


def test_cross_sectional_standardization() -> None:
    rng = np.random.default_rng(3)
    bars = _frame({f"60000{i}.SH": _random_walk(0, 300, rng) for i in range(8)}, 300)
    out = compute_style_factors(bars)
    last = out.filter(pl.col("date") == out["date"].max())
    for factor in STYLE_FACTOR_NAMES:
        assert abs(float(last[factor].mean())) < 1e-9
        assert abs(float(last[factor].std(ddof=1)) - 1.0) < 1e-6


def test_momentum_and_reverse_directions() -> None:
    """持续上涨票的 momentum 高于持续下跌票，reverse 低于后者。"""
    days = 300
    rising = np.exp(np.linspace(0.0, 0.5, days))
    falling = np.exp(np.linspace(0.0, -0.5, days))
    bars = _frame({"600000.SH": rising, "600001.SH": falling}, days)
    out = compute_style_factors(bars)
    last = out.filter(pl.col("date") == out["date"].max())
    values = {inst: row for inst, row in zip(last["instrument"].to_list(), last.iter_rows(named=True))}
    assert values["600000.SH"]["momentum"] > values["600001.SH"]["momentum"]
    assert values["600000.SH"]["reverse"] < values["600001.SH"]["reverse"]


def test_beta_tracks_market_loading() -> None:
    """高市场暴露的票 beta 排名应高于低暴露的票。"""
    rng = np.random.default_rng(7)
    days = 300
    market = rng.standard_normal(days) * 0.01
    series = {}
    for j, loading in enumerate([0.2, 2.0]):
        ret = 0.0005 + loading * market + rng.standard_normal(days) * 0.002
        series[f"60000{j}.SH"] = np.cumprod(1.0 + ret) * 10.0
    out = compute_style_factors(_frame(series, days))
    last = out.filter(pl.col("date") == out["date"].max())
    beta_map = dict(zip(last["instrument"].to_list(), last["beta"].to_list()))
    assert beta_map["600001.SH"] > beta_map["600000.SH"]


def test_nlsize_monotone_in_price() -> None:
    days = 300
    bars = _frame(
        {"600000.SH": np.full(days, 5.0), "600001.SH": np.full(days, 50.0)}, days
    )
    out = compute_style_factors(bars)
    last = out.filter(pl.col("date") == out["date"].max())
    nlsize = dict(zip(last["instrument"].to_list(), last["nlsize"].to_list()))
    assert nlsize["600001.SH"] > nlsize["600000.SH"]


def test_warmup_filled_neutral() -> None:
    """窗口预热不足的早期日期，历史类因子归 0（中性），不产生 NaN。"""
    rng = np.random.default_rng(9)
    bars = _frame({f"60000{i}.SH": _random_walk(0, 300, rng) for i in range(2)}, 300)
    out = compute_style_factors(bars)
    first = out.filter(pl.col("date") == out["date"].min())
    # nlsize 只需当期价格，首日即有定义；其余因子需要历史窗口，首日归 0。
    for factor in STYLE_FACTOR_NAMES:
        if factor == "nlsize":
            continue
        assert float(first[factor][0]) == 0.0


def test_equivalent_market_value() -> None:
    rng = np.random.default_rng(1)
    bars = _frame({"600000.SH": _random_walk(0, 10, rng)}, 10)
    out = equivalent_market_value(bars, share_const=2.0)
    assert out.columns == ["date", "instrument", "equiv_mv"]
    joined = out.join(bars, on=["date", "instrument"])
    expected = joined["close"] * joined["adjfactor"] * 2.0
    assert np.allclose(joined["equiv_mv"].to_numpy(), expected.to_numpy())


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="缺少列"):
        compute_style_factors(pl.DataFrame({"date": [], "instrument": []}))
    with pytest.raises(ValueError, match="缺少列"):
        equivalent_market_value(pl.DataFrame({"date": [], "instrument": []}))
