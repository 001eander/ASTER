"""``quant.eval.risk_report`` 单元测试（issue #69）。

覆盖四张表的**手工算例**（合成持仓 + 合成成分 / 市值 / 风格 / 行业数据）、
自洽性检查的带内通过 / 带外报错、空基准参数行为与渲染 / 落盘辅助。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant.eval.risk_report import (
    INDEX_BUCKETS,
    MV_BUCKETS,
    ConsistencyError,
    RiskReportError,
    build_risk_report,
    check_enhanced_consistency,
    compute_active_industry,
    compute_index_distribution,
    compute_market_value_distribution,
    compute_style_exposure,
    frame_to_markdown,
    normalize_holdings,
    render_markdown,
    write_xlsx,
)
from quant.portfolio.enhanced import EnhancedOptimizer
from quant.portfolio.style import STYLE_FACTOR_NAMES

D1 = date(2026, 1, 5)
D2 = date(2026, 1, 6)


def _holdings(rows: list[tuple[date, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            "instrument": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
        },
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )


# ---------------------------------------------------------------------------
# 表 1：指数分布
# ---------------------------------------------------------------------------


def test_index_distribution_manual() -> None:
    """600000 同时属 300/500 取 300；300001 无成分归 3800以外。"""
    holdings = _holdings(
        [
            (D1, "600000.SH", 0.5),
            (D1, "000001.SZ", 0.3),
            (D1, "300001.SZ", 0.2),
        ]
    )
    members = pl.DataFrame(
        {
            "date": [D1, D1, D1, D1, D1],
            "instrument": [
                "600000.SH",
                "600000.SH",
                "000001.SZ",
                "300002.SZ",
                "600000.SH",
            ],
            "index_code": ["000905", "000300", "000905", "932000", "000510"],
        },
        schema={"date": pl.Date, "instrument": pl.String, "index_code": pl.String},
    )
    out = compute_index_distribution(holdings, members)
    assert out.columns == ["date", *INDEX_BUCKETS]
    assert out["date"].to_list() == [D1]
    row = out.row(0, named=True)
    assert row["沪深300"] == pytest.approx(0.5)
    assert row["中证500"] == pytest.approx(0.3)
    assert row["中证2000"] == pytest.approx(0.0)
    assert row["3800以外"] == pytest.approx(0.2)


def test_index_distribution_multiple_dates_and_empty_members() -> None:
    holdings = _holdings([(D1, "600000.SH", 1.0), (D2, "600000.SH", 1.0)])
    out = compute_index_distribution(holdings, pl.DataFrame(schema={
        "date": pl.Date, "instrument": pl.String, "index_code": pl.String,
    }))
    assert out.height == 2
    assert out["3800以外"].to_list() == [1.0, 1.0]
    for bucket in INDEX_BUCKETS[:-1]:
        assert out[bucket].to_list() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# 表 2：市值分布
# ---------------------------------------------------------------------------


def test_market_value_distribution_manual() -> None:
    """四条分界：> 500 亿 / > 100 亿 / >= 30 亿 / 其余，缺失归未知。"""
    holdings = _holdings(
        [
            (D1, "600000.SH", 0.3),
            (D1, "000001.SZ", 0.3),
            (D1, "300001.SZ", 0.2),
            (D1, "300002.SZ", 0.1),
            (D1, "300003.SZ", 0.1),
        ]
    )
    equiv_mv = pl.DataFrame(
        {
            "date": [D1, D1, D1, D1],
            "instrument": ["600000.SH", "000001.SZ", "300001.SZ", "300002.SZ"],
            "equiv_mv": [6e10, 1.5e10, 3e9, 2e9],
        },
        schema={"date": pl.Date, "instrument": pl.String, "equiv_mv": pl.Float64},
    )
    out = compute_market_value_distribution(holdings, equiv_mv)
    assert out.columns == ["date", *MV_BUCKETS]
    row = out.row(0, named=True)
    assert row["大盘股"] == pytest.approx(0.3)
    assert row["中盘股"] == pytest.approx(0.3)
    assert row["小盘股"] == pytest.approx(0.2)
    assert row["微盘股"] == pytest.approx(0.1)
    assert row["未知"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 表 3：风格暴露
# ---------------------------------------------------------------------------


def test_style_exposure_drops_missing_and_renormalizes() -> None:
    """缺因子值的票剔除其权重：(0.5×1 + 0.3×2) / 0.8 = 1.375。"""
    holdings = _holdings(
        [
            (D1, "600000.SH", 0.5),
            (D1, "000001.SZ", 0.3),
            (D1, "300001.SZ", 0.2),
        ]
    )
    style = pl.DataFrame(
        {
            "date": [D1, D1, D1],
            "instrument": ["600000.SH", "000001.SZ", "300001.SZ"],
            "beta": [1.0, 2.0, None],
        },
        schema={"date": pl.Date, "instrument": pl.String, "beta": pl.Float64},
    )
    out = compute_style_exposure(holdings, style, factor_names=["beta"])
    assert out.columns == ["date", "beta"]
    assert out["beta"][0] == pytest.approx(1.375)


def test_style_exposure_all_missing_is_null() -> None:
    holdings = _holdings([(D1, "600000.SH", 1.0)])
    style = pl.DataFrame(
        {
            "date": [D1],
            "instrument": ["600000.SH"],
            "beta": [None],
        },
        schema={"date": pl.Date, "instrument": pl.String, "beta": pl.Float64},
    )
    out = compute_style_exposure(holdings, style, factor_names=["beta"])
    assert out["beta"][0] is None


# ---------------------------------------------------------------------------
# 表 4：主动行业暴露
# ---------------------------------------------------------------------------


def test_active_industry_manual() -> None:
    """组合银行 0.8 / 电子 0.2，基准银行 0.5 / 电子 0.5 → 各偏 ±0.3。"""
    holdings = _holdings(
        [
            (D1, "600000.SH", 0.8),
            (D1, "300001.SZ", 0.2),
        ]
    )
    bench = pl.DataFrame(
        {
            "date": [D1, D1],
            "instrument": ["600000.SH", "300001.SZ"],
            "weight": [0.5, 0.5],
        },
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )
    industry = pl.DataFrame(
        {
            "instrument": ["600000.SH", "300001.SZ"],
            "industry_l1": ["银行", "电子"],
        },
        schema={"instrument": pl.String, "industry_l1": pl.String},
    )
    out = compute_active_industry(holdings, bench, industry)
    assert set(out.columns) == {"date", "银行", "电子"}
    row = out.row(0, named=True)
    assert row["银行"] == pytest.approx(0.3)
    assert row["电子"] == pytest.approx(-0.3)


def test_active_industry_uses_index_code_filter() -> None:
    """基准表含多指数时按 bench_index 取对应权重。"""
    holdings = _holdings([(D1, "600000.SH", 1.0)])
    bench = pl.DataFrame(
        {
            "date": [D1, D1],
            "instrument": ["600000.SH", "600000.SH"],
            "index_code": ["000300", "000905"],
            "weight": [1.0, 1.0],
        },
        schema={
            "date": pl.Date,
            "instrument": pl.String,
            "index_code": pl.String,
            "weight": pl.Float64,
        },
    )
    industry = pl.DataFrame(
        {"instrument": ["600000.SH"], "industry_l1": ["银行"]},
        schema={"instrument": pl.String, "industry_l1": pl.String},
    )
    out = compute_active_industry(holdings, bench, industry, bench_index="000905")
    assert out["银行"][0] == pytest.approx(0.0)
    with pytest.raises(RiskReportError, match="index_code"):
        compute_active_industry(holdings, bench.drop("index_code"), industry, bench_index="000905")


# ---------------------------------------------------------------------------
# 汇总与空基准行为
# ---------------------------------------------------------------------------


def test_build_risk_report_without_benchmark_has_no_active_industry() -> None:
    holdings = _holdings([(D1, "600000.SH", 1.0)])
    members = pl.DataFrame(
        {"date": [D1], "instrument": ["600000.SH"], "index_code": ["000300"]},
        schema={"date": pl.Date, "instrument": pl.String, "index_code": pl.String},
    )
    report = build_risk_report(holdings, members=members)
    assert report.index_distribution is not None
    assert report.market_value_distribution is None
    assert report.style_exposure is None
    assert report.active_industry is None
    assert list(report.tables()) == ["指数分布"]


def test_build_risk_report_with_benchmark_adds_industry() -> None:
    holdings = _holdings([(D1, "600000.SH", 1.0)])
    bench = pl.DataFrame(
        {"date": [D1], "instrument": ["600000.SH"], "weight": [1.0]},
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )
    industry = pl.DataFrame(
        {"instrument": ["600000.SH"], "industry_l1": ["银行"]},
        schema={"instrument": pl.String, "industry_l1": pl.String},
    )
    report = build_risk_report(holdings, bench_weights=bench, industry=industry)
    assert list(report.tables()) == ["主动行业暴露"]
    assert report.active_industry is not None
    # 给了基准但没给行业表：仍不出主动行业暴露表。
    assert build_risk_report(holdings, bench_weights=bench).active_industry is None


# ---------------------------------------------------------------------------
# 自洽性检查
# ---------------------------------------------------------------------------


def _consistency_fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """两票同行业、beta 1.0 / 3.0；基准 0.5 / 0.5。"""
    holdings = _holdings([(D1, "600000.SH", 0.5), (D1, "000001.SZ", 0.5)])
    bench = pl.DataFrame(
        {
            "date": [D1, D1],
            "instrument": ["600000.SH", "000001.SZ"],
            "weight": [0.5, 0.5],
        },
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )
    industry = pl.DataFrame(
        {"instrument": ["600000.SH", "000001.SZ"], "industry_l1": ["银行", "银行"]},
        schema={"instrument": pl.String, "industry_l1": pl.String},
    )
    style = pl.DataFrame(
        {
            "date": [D1, D1],
            "instrument": ["600000.SH", "000001.SZ"],
            "beta": [1.0, 3.0],
        },
        schema={"date": pl.Date, "instrument": pl.String, "beta": pl.Float64},
    )
    return holdings, bench, industry, style


def test_consistency_passes_within_band() -> None:
    holdings, bench, industry, style = _consistency_fixture()
    checks = check_enhanced_consistency(
        holdings,
        bench_weights=bench,
        industry=industry,
        style=style,
        optimizer=EnhancedOptimizer(industry_exposure=0.01, style_exposure=0.5),
    )
    assert checks.height == 1
    assert checks["industry_max_ratio"][0] == pytest.approx(0.0)
    assert checks["style_max_std"][0] == pytest.approx(0.0)
    assert checks["violations"][0] == 0


def test_consistency_raises_outside_band_industry() -> None:
    """组合银行 0.9 / 电子 0.1，基准各 0.5 → 行业偏离 0.8 超带。"""
    holdings = _holdings([(D1, "600000.SH", 0.9), (D1, "300001.SZ", 0.1)])
    bench = pl.DataFrame(
        {
            "date": [D1, D1],
            "instrument": ["600000.SH", "300001.SZ"],
            "weight": [0.5, 0.5],
        },
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )
    industry = pl.DataFrame(
        {"instrument": ["600000.SH", "300001.SZ"], "industry_l1": ["银行", "电子"]},
        schema={"instrument": pl.String, "industry_l1": pl.String},
    )
    with pytest.raises(ConsistencyError, match="行业"):
        check_enhanced_consistency(
            holdings, bench_weights=bench, industry=industry, industry_tol=0.005
        )


def test_consistency_raises_outside_band_style() -> None:
    holdings, bench, industry, style = _consistency_fixture()
    shifted = _holdings([(D1, "600000.SH", 0.1), (D1, "000001.SZ", 0.9)])
    with pytest.raises(ConsistencyError, match="风格"):
        check_enhanced_consistency(
            shifted,
            bench_weights=bench,
            industry=industry,
            style=style,
            optimizer=EnhancedOptimizer(industry_exposure=1.0, style_exposure=0.005),
        )


def test_consistency_flags_industry_absent_from_benchmark() -> None:
    """基准没有电子行业，组合持有电子 0.2 → 视为超带（优化器约束 port ≤ 0）。"""
    holdings = _holdings([(D1, "600000.SH", 0.8), (D1, "300001.SZ", 0.2)])
    bench = pl.DataFrame(
        {"date": [D1], "instrument": ["600000.SH"], "weight": [1.0]},
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )
    industry = pl.DataFrame(
        {"instrument": ["600000.SH", "300001.SZ"], "industry_l1": ["银行", "电子"]},
        schema={"instrument": pl.String, "industry_l1": pl.String},
    )
    with pytest.raises(ConsistencyError, match="电子"):
        check_enhanced_consistency(
            holdings, bench_weights=bench, industry=industry, industry_tol=0.005
        )


def test_consistency_requires_tolerance() -> None:
    holdings, bench, industry, _ = _consistency_fixture()
    with pytest.raises(RiskReportError, match="industry_tol"):
        check_enhanced_consistency(holdings, bench_weights=bench, industry=industry)


def test_consistency_passes_for_optimizer_output() -> None:
    """指增优化器的解必然落在其自身设定带内（自洽性检查与优化器同口径）。"""
    import numpy as np

    rng = np.random.default_rng(11)
    n, n_members = 12, 8
    instruments = [f"{600000 + i:06d}.SH" for i in range(n)]
    bench_raw = np.zeros(n)
    bench_raw[:n_members] = rng.dirichlet(np.ones(n_members))
    bench = {inst: float(bench_raw[i]) for i, inst in enumerate(instruments)}
    industry = {inst: f"I{int(rng.integers(0, 3))}" for inst in instruments}
    float_mv = {inst: float(np.exp(rng.normal(24.0, 1.0))) for inst in instruments}
    style = {
        inst: {f: float(rng.standard_normal()) for f in STYLE_FACTOR_NAMES}
        for inst in instruments
    }
    alpha = {inst: float(rng.standard_normal()) for inst in instruments}
    returns = rng.standard_normal((80, n)) * 0.01

    optimizer = EnhancedOptimizer()
    result = optimizer.optimize_day(
        date=D1,
        instruments=instruments,
        alpha=alpha,
        bench_weights=bench,
        industry=industry,
        float_mv=float_mv,
        style=style,
        covariance=returns,
    )
    assert result.weights

    day = D1
    holdings = _holdings([(day, inst, w) for inst, w in result.weights.items()])
    bench_frame = pl.DataFrame(
        {
            "date": [day] * len(bench),
            "instrument": list(bench),
            "weight": list(bench.values()),
        },
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )
    industry_frame = pl.DataFrame(
        {
            "instrument": list(industry),
            "industry_l1": list(industry.values()),
        },
        schema={"instrument": pl.String, "industry_l1": pl.String},
    )
    style_frame = pl.DataFrame(
        {
            "date": [day] * n,
            "instrument": instruments,
            **{f: [style[inst][f] for inst in instruments] for f in STYLE_FACTOR_NAMES},
        }
    )
    mv_frame = pl.DataFrame(
        {
            "date": [day] * n,
            "instrument": instruments,
            "equiv_mv": [float_mv[inst] for inst in instruments],
        },
        schema={"date": pl.Date, "instrument": pl.String, "equiv_mv": pl.Float64},
    )

    checks = check_enhanced_consistency(
        holdings,
        bench_weights=bench_frame,
        industry=industry_frame,
        style=style_frame,
        equiv_mv=mv_frame,
        industry_tol=result.thresholds["industry"],
        style_tol=result.thresholds["style"],
        market_value_tol=result.thresholds["market_value"],
    )
    assert checks.height == 1
    assert checks["violations"][0] == 0


# ---------------------------------------------------------------------------
# 输入校验与渲染
# ---------------------------------------------------------------------------


def test_normalize_holdings_rejects_missing_columns() -> None:
    with pytest.raises(RiskReportError, match="缺少列"):
        normalize_holdings(pl.DataFrame({"date": [], "instrument": []}))


def test_normalize_holdings_parses_string_dates() -> None:
    frame = pl.DataFrame(
        {
            "date": ["2026-01-05", "20260106"],
            "instrument": ["600000.SH", "600000.SH"],
            "weight": [0.5, 0.5],
        }
    )
    out = normalize_holdings(frame)
    assert out["date"].to_list() == [D1, D2]


def test_frame_to_markdown_and_render() -> None:
    holdings = _holdings([(D1, "600000.SH", 1.0)])
    members = pl.DataFrame(
        {"date": [D1], "instrument": ["600000.SH"], "index_code": ["000300"]},
        schema={"date": pl.Date, "instrument": pl.String, "index_code": pl.String},
    )
    report = build_risk_report(holdings, members=members)
    text = render_markdown(report, notes=["合成用例"])
    assert "## 指数分布" in text
    assert "2026-01-05" in text
    assert "合成用例" in text
    assert frame_to_markdown(pl.DataFrame({"date": [], "x": []})) == "（空）"


def test_write_xlsx_roundtrip_or_graceful(tmp_path: Path) -> None:
    holdings = _holdings([(D1, "600000.SH", 1.0)])
    members = pl.DataFrame(
        {"date": [D1], "instrument": ["600000.SH"], "index_code": ["000300"]},
        schema={"date": pl.Date, "instrument": pl.String, "index_code": pl.String},
    )
    report = build_risk_report(holdings, members=members)
    path = tmp_path / "risk.xlsx"
    wrote = write_xlsx(path, report.tables())
    if wrote:
        assert path.exists() and path.stat().st_size > 0
    else:  # pragma: no cover - 无 xlsxwriter / openpyxl 时跳过
        assert not path.exists()
