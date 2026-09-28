"""``quant.eval.metrics`` 单元测试。

全部用合成数据，不触网。IC 用例用解析解核对，分层用例构造严格线性关系核对单调性，
换手用例手算集合变化，边界用例覆盖样本不足 / 全 null / 层数超过证券数。
"""
from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from datetime import date, timedelta

import polars as pl
import pytest

from quant.eval.metrics import (
    ICSummary,
    ic_series,
    layer_monotonicity,
    layered_returns,
    summarize_ic,
    turnover,
)

DAY = date(2024, 1, 2)


def _daily(
    day: date,
    factors: Sequence[float | None],
    labels: Sequence[float | None],
    *,
    instruments: Sequence[str] | None = None,
) -> pl.DataFrame:
    """构造单日长表；证券代码缺省 ``600000.SH`` 起递增。"""
    n = len(factors)
    if instruments is None:
        instruments = [f"{600000 + i:06d}.SH" for i in range(n)]
    return pl.DataFrame(
        {
            "date": [day] * n,
            "instrument": list(instruments),
            "factor": list(factors),
            "label": list(labels),
        }
    )


# ---------------------------------------------------------------------------
# IC / RankIC
# ---------------------------------------------------------------------------


def test_ic_series_spearman_vs_pearson() -> None:
    """RankIC 对单调非线性关系为 1，Pearson 小于 1。"""
    df = _daily(DAY, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 100.0])

    rank_ic = ic_series(df, min_count=4)
    assert rank_ic.columns == ["date", "ic"]
    assert rank_ic["date"].to_list() == [DAY]
    assert rank_ic["ic"][0] == pytest.approx(1.0)

    pearson_ic = ic_series(df, method="pearson", min_count=4)
    # 手算：cov=149, var_x=5, var_y=7205 → r = 149 / sqrt(5*7205)
    assert pearson_ic["ic"][0] == pytest.approx(0.7850264209630101, rel=1e-9)


def test_ic_series_pearson_equals_rank_on_linear() -> None:
    """严格线性关系下 pearson 与 spearman 一致。"""
    df = _daily(DAY, [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    rank_ic = ic_series(df, min_count=4)["ic"][0]
    pearson_ic = ic_series(df, method="pearson", min_count=4)["ic"][0]
    assert rank_ic == pytest.approx(1.0)
    assert pearson_ic == pytest.approx(1.0)


def test_ic_series_noise_near_zero() -> None:
    """独立噪声的日均 IC 应接近 0。"""
    rng = random.Random(20240101)
    frames: list[pl.DataFrame] = []
    for offset in range(120):
        day = DAY + timedelta(days=offset)
        factors = [rng.gauss(0.0, 1.0) for _ in range(50)]
        labels = [rng.gauss(0.0, 1.0) for _ in range(50)]
        frames.append(_daily(day, factors, labels))

    ic = ic_series(pl.concat(frames))
    summary = summarize_ic(ic)
    assert summary.n_days == 120
    assert summary.mean is not None
    assert abs(summary.mean) < 0.05


def test_ic_series_constant_factor_is_null() -> None:
    """常因子方差为 0，相关无定义，记 null 而非 NaN。"""
    df = _daily(DAY, [1.0] * 35, [float(i) for i in range(35)])
    ic = ic_series(df)
    assert ic["ic"].to_list() == [None]


def test_ic_series_insufficient_count_warns(caplog: pytest.LogCaptureFixture) -> None:
    """有效证券数不足 min_count 时 IC 记 null 并写 warning。"""
    df = _daily(DAY, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with caplog.at_level(logging.WARNING, logger="quant.eval.metrics"):
        ic = ic_series(df)
    assert ic["ic"].to_list() == [None]
    assert any("少于" in record.getMessage() for record in caplog.records)


def test_ic_series_all_null_keeps_date() -> None:
    """因子与标签全 null 时日期仍保留，IC 为 null。"""
    df = _daily(DAY, [None, None], [1.0, 2.0])
    ic = ic_series(df)
    assert ic["date"].to_list() == [DAY]
    assert ic["ic"].to_list() == [None]


def test_ic_series_single_instrument_is_null() -> None:
    """只有 1 只证券无法算相关，IC 为 null。"""
    df = _daily(DAY, [1.0], [0.5])
    ic = ic_series(df)
    assert ic["ic"].to_list() == [None]


def test_ic_series_invalid_method() -> None:
    df = _daily(DAY, [1.0], [1.0])
    with pytest.raises(ValueError, match="method"):
        ic_series(df, method="kendall")  # type: ignore[arg-type]


def test_ic_series_missing_column() -> None:
    df = pl.DataFrame({"date": [DAY], "factor": [1.0]})
    with pytest.raises(ValueError, match="缺少必需列"):
        ic_series(df)


# ---------------------------------------------------------------------------
# ICIR / 胜率
# ---------------------------------------------------------------------------


def test_summarize_ic_values() -> None:
    ic = pl.DataFrame(
        {
            "date": [date(2024, 1, d) for d in range(1, 5)],
            "ic": [0.1, 0.2, -0.1, 0.3],
        }
    )
    expected_std = math.sqrt(0.0875 / 3)
    summary = summarize_ic(ic)
    assert isinstance(summary, ICSummary)
    assert summary.n_days == 4
    assert summary.mean == pytest.approx(0.125)
    assert summary.std == pytest.approx(expected_std)
    assert summary.icir == pytest.approx(0.125 / expected_std)
    assert summary.ic_win_rate == pytest.approx(0.75)
    assert summary.annualized is False


def test_summarize_ic_annualized() -> None:
    ic = pl.DataFrame(
        {
            "date": [date(2024, 1, d) for d in range(1, 5)],
            "ic": [0.1, 0.2, -0.1, 0.3],
        }
    )
    expected_std = math.sqrt(0.0875 / 3)
    summary = summarize_ic(ic, annualize=True)
    assert summary.icir == pytest.approx(0.125 / expected_std * math.sqrt(252))
    assert summary.annualized is True


def test_summarize_ic_ignores_nulls() -> None:
    ic = pl.DataFrame(
        {
            "date": [date(2024, 1, d) for d in range(1, 4)],
            "ic": [0.1, None, -0.1],
        }
    )
    summary = summarize_ic(ic)
    assert summary.n_days == 2
    assert summary.mean == pytest.approx(0.0)
    assert summary.std == pytest.approx(math.sqrt(0.02))
    assert summary.ic_win_rate == pytest.approx(0.5)


def test_summarize_ic_all_null() -> None:
    ic = pl.DataFrame({"date": [DAY], "ic": [None]})
    summary = summarize_ic(ic)
    assert summary.n_days == 0
    assert summary.mean is None
    assert summary.std is None
    assert summary.icir is None
    assert summary.ic_win_rate is None


def test_summarize_ic_single_day_std_none() -> None:
    ic = pl.DataFrame({"date": [DAY], "ic": [0.2]})
    summary = summarize_ic(ic)
    assert summary.n_days == 1
    assert summary.std is None
    assert summary.icir is None


# ---------------------------------------------------------------------------
# 分层
# ---------------------------------------------------------------------------


def test_layered_returns_exact_and_monotone() -> None:
    """因子与收益同向，layer 0 最低、收益严格递增。"""
    df = _daily(DAY, [float(x) for x in range(1, 11)], [float(x) for x in range(1, 11)])
    layered = layered_returns(df, n_layers=5)

    assert layered.columns == ["date", "layer", "ret", "count"]
    assert layered["layer"].to_list() == [0, 1, 2, 3, 4]
    assert layered["ret"].to_list() == pytest.approx([1.5, 3.5, 5.5, 7.5, 9.5])
    assert layered["count"].to_list() == [2, 2, 2, 2, 2]
    assert layer_monotonicity(layered) == pytest.approx(1.0)


def test_layer_monotonicity_reversed() -> None:
    """因子与收益反向，单调性为 -1。"""
    df = _daily(
        DAY,
        [float(x) for x in range(1, 11)],
        [float(11 - x) for x in range(1, 11)],
    )
    layered = layered_returns(df, n_layers=5)
    assert layer_monotonicity(layered) == pytest.approx(-1.0)


def test_layered_returns_multi_day() -> None:
    """多日时每日独立分层。"""
    day2 = DAY + timedelta(days=1)
    df = pl.concat(
        [
            _daily(DAY, [float(x) for x in range(1, 11)], [float(x) for x in range(1, 11)]),
            _daily(
                day2,
                [float(x) for x in range(1, 11)],
                [float(2 * x) for x in range(1, 11)],
            ),
        ]
    )
    layered = layered_returns(df, n_layers=5)
    assert set(layered["date"].to_list()) == {DAY, day2}
    assert layered.filter(pl.col("date") == day2)["ret"].to_list() == pytest.approx(
        [3.0, 7.0, 11.0, 15.0, 19.0]
    )
    # 两日均同向，层均收益仍严格递增
    assert layer_monotonicity(layered) == pytest.approx(1.0)


def test_layered_returns_too_few_securities() -> None:
    """层数超过当日证券数时报错，不静默降层。"""
    df = _daily(DAY, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="少于层数"):
        layered_returns(df, n_layers=5)


def test_layered_returns_invalid_n_layers() -> None:
    df = _daily(DAY, [1.0, 2.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="n_layers"):
        layered_returns(df, n_layers=1)


def test_layered_returns_empty_raises() -> None:
    df = _daily(DAY, [None, None], [0.1, 0.2])
    with pytest.raises(ValueError, match="没有任何有效"):
        layered_returns(df, n_layers=2)


def test_layer_monotonicity_single_layer() -> None:
    layered = pl.DataFrame({"date": [DAY], "layer": [0], "ret": [0.1], "count": [40]})
    assert layer_monotonicity(layered) is None


def test_layer_monotonicity_flat_is_none() -> None:
    layered = pl.DataFrame(
        {"date": [DAY] * 2, "layer": [0, 1], "ret": [0.1, 0.1], "count": [20, 20]}
    )
    assert layer_monotonicity(layered) is None


# ---------------------------------------------------------------------------
# 换手
# ---------------------------------------------------------------------------


def _two_day(factors_day1: Sequence[float], factors_day2: Sequence[float]) -> pl.DataFrame:
    instruments = ["600000.SH", "600001.SH", "600002.SH"]
    return pl.DataFrame(
        {
            "date": [DAY] * 3 + [DAY + timedelta(days=1)] * 3,
            "instrument": instruments * 2,
            "factor": list(factors_day1) + list(factors_day2),
        }
    )


def test_turnover_two_day_top2() -> None:
    """手算：day1 最高两层 {A,B}，day2 {B,C}，替换 1 只 → 0.5。"""
    df = _two_day([3.0, 2.0, 1.0], [1.0, 2.0, 3.0])
    out = turnover(df, top_n=2)
    assert out.columns == ["date", "turnover"]
    assert out["turnover"].to_list() == [None, pytest.approx(0.5)]


def test_turnover_unchanged_is_zero() -> None:
    df = _two_day([3.0, 2.0, 1.0], [3.0, 2.0, 1.0])
    out = turnover(df, top_n=2)
    assert out["turnover"].to_list() == [None, pytest.approx(0.0)]


def test_turnover_ascending_selects_lowest() -> None:
    """ascending=True 时选因子最低层，与默认方向结果不同。"""
    df = _two_day([1.0, 2.0, 3.0], [2.0, 1.0, 3.0])
    out = turnover(df, top_n=1, ascending=True)
    # day1 最低 A，day2 最低 B → 替换 1 只 → 1.0
    assert out["turnover"].to_list() == [None, pytest.approx(1.0)]

    out_desc = turnover(df, top_n=1)
    # 默认选最高：两日都是 C → 0.0
    assert out_desc["turnover"].to_list() == [None, pytest.approx(0.0)]


def test_turnover_null_factor_excluded() -> None:
    df = pl.DataFrame(
        {
            "date": [DAY] * 2,
            "instrument": ["600000.SH", "600001.SH"],
            "factor": [1.0, None],
        }
    )
    out = turnover(df, top_n=2)
    assert out.height == 1
    assert out["turnover"].to_list() == [None]


def test_turnover_invalid_top_n() -> None:
    df = _two_day([3.0, 2.0, 1.0], [3.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="top_n"):
        turnover(df, top_n=0)
