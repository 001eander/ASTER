"""站点资产生成脚本的验收：指标口径、幂等、回填与缺输入退出码。

测试全部用 ``tmp_path`` 造的迷你 parquet 与 report.md，不依赖仓库里的 ``runs/``。
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from scripts.build_site_assets import (
    MissingInputError,
    backfill_metrics,
    build_metrics,
    format_metric,
    main,
    render_nav_svg,
)

TRADING_DAYS = 252


def _write_run(
    run_dir: Path,
    *,
    nav: list[float],
    portfolio_nav: list[float],
    benchmark_nav: list[float],
    config_rows: dict[str, str],
    perf_rows: dict[str, str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    dates = [date(2025, 1, 2) + timedelta(days=index) for index in range(len(nav))]
    pl.DataFrame(
        {
            "date": dates,
            "cash": [value * 0.1 for value in nav],
            "market_value": [value * 0.9 for value in nav],
            "nav": nav,
            "traded_amount": [0.0] * len(nav),
            "turnover": [0.5] * len(nav),
        }
    ).write_parquet(run_dir / "nav.parquet")

    rets = [None] + [
        portfolio_nav[index] / portfolio_nav[index - 1] - 1
        for index in range(1, len(portfolio_nav))
    ]
    bench_rets = [None] + [
        benchmark_nav[index] / benchmark_nav[index - 1] - 1
        for index in range(1, len(benchmark_nav))
    ]
    excess = [
        None if rets[index] is None else rets[index] - bench_rets[index]
        for index in range(len(rets))
    ]
    pl.DataFrame(
        {
            "date": dates,
            "portfolio_nav": portfolio_nav,
            "benchmark_nav": benchmark_nav,
            "excess_nav": [
                p / b for p, b in zip(portfolio_nav, benchmark_nav)
            ],
            "portfolio_ret": rets,
            "benchmark_ret": bench_rets,
            "excess_ret": excess,
        }
    ).write_parquet(run_dir / "benchmark.parquet")

    lines = ["# 端到端回测报告（issue #16）", "", "## 配置", "", "| 项 | 值 |", "| --- | --- |"]
    for key, value in config_rows.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 绩效", "", "| 指标 | 值 |", "| --- | --- |"])
    for key, value in perf_rows.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


NAV = [100_000.0, 110_000.0, 105_000.0, 120_000.0]
PORTFOLIO_NAV = [1.0, 1.1, 1.05, 1.2]
BENCHMARK_NAV = [1.0, 1.02, 1.01, 1.03]
CONFIG_REPORT = {
    "start": "2025-01-02",
    "end": "2025-01-05",
    "initial_cash": "100000.00",
    "strategy": "stock_selection",
    "rebalance_freq": "W",
    "benchmark": "000300",
}
PERF_REPORT = {
    "总收益": "20.0000%",
    "成交笔数": "7",
    "拒单笔数": "2",
    "费用占初始资金": "3.0000%",
}


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    registry = tmp_path / "factor_library" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "factors": [
                    {"lineage": {"op": "seed"}},
                    {"lineage": {"op": "seed"}},
                    {"lineage": {"op": "mutation"}},
                    {"lineage": {"op": "crossover"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_run(
        tmp_path / "runs" / "flagship",
        nav=NAV,
        portfolio_nav=PORTFOLIO_NAV,
        benchmark_nav=BENCHMARK_NAV,
        config_rows=CONFIG_REPORT,
        perf_rows=PERF_REPORT,
    )
    zz_config = dict(CONFIG_REPORT)
    zz_config.update({"benchmark": "000852", "rebalance_freq": "M", "initial_cash": "1000000.00"})
    zz_perf = dict(PERF_REPORT)
    zz_perf.pop("费用占初始资金")
    _write_run(
        tmp_path / "runs" / "zz1000",
        nav=[value * 10 for value in NAV],
        portfolio_nav=PORTFOLIO_NAV,
        benchmark_nav=BENCHMARK_NAV,
        config_rows=zz_config,
        perf_rows=zz_perf,
    )
    return tmp_path


def _build(workspace: Path, out_name: str = "assets", corr_png: Path | None = None) -> dict:
    return build_metrics(
        registry_path=workspace / "factor_library" / "registry.json",
        flagship_run=workspace / "runs" / "flagship",
        zz1000_run=workspace / "runs" / "zz1000",
        corr_png=corr_png,
        out_dir=workspace / out_name,
    )


def test_factor_counts_follow_registry(workspace: Path) -> None:
    metrics = _build(workspace)

    assert metrics["factors"] == {
        "total": 4,
        "seed": 2,
        "mutation": 1,
        "crossover": 1,
    }


def test_metrics_follow_backtest_caliber(workspace: Path) -> None:
    metrics = _build(workspace)
    sample = metrics["samples"]["flagship"]

    returns = [NAV[index] / NAV[index - 1] - 1 for index in range(1, len(NAV))]
    expected_sharpe = statistics.mean(returns) / statistics.stdev(returns) * TRADING_DAYS**0.5
    expected_drawdown = min(
        NAV[index] / max(NAV[: index + 1]) - 1 for index in range(len(NAV))
    )

    assert sample["total_return"] == pytest.approx(0.2)
    assert sample["max_drawdown"] == pytest.approx(expected_drawdown, abs=1e-6)
    assert sample["sharpe"] == pytest.approx(expected_sharpe, abs=1e-6)
    assert sample["turnover"] == pytest.approx(0.5 * len(NAV), abs=1e-6)
    assert sample["excess_return"] == pytest.approx(
        PORTFOLIO_NAV[-1] / BENCHMARK_NAV[-1] - 1, abs=1e-6
    )


def test_turnover_is_rounded_to_two_decimals(workspace: Path) -> None:
    metrics = _build(workspace)
    flagship = metrics["samples"]["flagship"]
    zz1000 = metrics["samples"]["zz1000"]

    assert flagship["turnover"] == pytest.approx(2.0)
    assert zz1000["turnover"] == pytest.approx(2.0)
    assert format_metric("flagship.turnover", 38.7148) == "38.71"
    assert format_metric("zz1000.turnover", 3.8832) == "3.88"


def test_report_fields_are_parsed_not_invented(workspace: Path) -> None:
    metrics = _build(workspace)
    flagship = metrics["samples"]["flagship"]
    zz1000 = metrics["samples"]["zz1000"]

    assert flagship["trades"] == 7
    assert flagship["rejects"] == 2
    assert flagship["cost_ratio"] == pytest.approx(0.03)
    assert "cost_ratio" not in zz1000
    assert flagship["benchmark"] == "000300"
    assert flagship["rebalance"] == "W"
    assert flagship["initial_cash"] == pytest.approx(100_000)
    assert zz1000["benchmark"] == "000852"
    assert zz1000["rebalance"] == "M"
    assert zz1000["initial_cash"] == pytest.approx(1_000_000)
    assert flagship["start"] == "2025-01-02"
    assert flagship["end"] == "2025-01-05"


def test_metrics_json_is_utf8_lf_with_sorted_keys(workspace: Path) -> None:
    _build(workspace)
    raw = (workspace / "assets" / "metrics.json").read_bytes()

    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    assert text == json.dumps(json.loads(text), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_missing_input_returns_nonzero_and_keeps_existing_outputs(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = workspace / "assets"
    out_dir.mkdir()
    sentinel = out_dir / "metrics.json"
    sentinel.write_text("keep me\n", encoding="utf-8")

    code = main(
        [
            "--factor-library-dir",
            str(workspace / "factor_library"),
            "--flagship-run",
            str(workspace / "runs" / "absent"),
            "--zz1000-run",
            str(workspace / "runs" / "zz1000"),
            "--out-dir",
            str(out_dir),
            "--index",
            str(workspace / "index.html"),
        ]
    )

    assert code != 0
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    captured = capsys.readouterr()
    assert "absent" in captured.out + captured.err


def test_build_metrics_reports_every_missing_path(workspace: Path) -> None:
    absent = workspace / "runs" / "absent"
    with pytest.raises(MissingInputError) as excinfo:
        build_metrics(
            registry_path=workspace / "factor_library" / "registry.json",
            flagship_run=absent,
            zz1000_run=workspace / "runs" / "zz1000",
            out_dir=workspace / "assets",
        )

    message = str(excinfo.value)
    for name in ("nav.parquet", "benchmark.parquet", "report.md"):
        assert name in message
    assert "absent" in message


OUTPUT_NAMES = ("metrics.json", "nav-flagship.svg", "nav-zz1000.svg", "factor-corr.png")


def test_repeated_builds_are_byte_identical(workspace: Path) -> None:
    corr = workspace / "source-corr.png"
    corr.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    _build(workspace, "first", corr_png=corr)
    _build(workspace, "second", corr_png=corr)

    for name in OUTPUT_NAMES:
        first = (workspace / "first" / name).read_bytes()
        second = (workspace / "second" / name).read_bytes()
        assert first == second, f"{name} 两次生成字节不一致"

    before = {name: (workspace / "first" / name).read_bytes() for name in OUTPUT_NAMES}
    _build(workspace, "first", corr_png=corr)
    after = {name: (workspace / "first" / name).read_bytes() for name in OUTPUT_NAMES}
    assert before == after, "同一目录重复生成应覆盖为相同字节"


def test_corr_png_is_copied_and_missing_source_keeps_existing_file(workspace: Path) -> None:
    corr = workspace / "source-corr.png"
    corr.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    _build(workspace, corr_png=corr)
    copied = workspace / "assets" / "factor-corr.png"
    assert copied.read_bytes() == corr.read_bytes()

    _build(workspace, corr_png=workspace / "absent.png")
    assert copied.read_bytes() == corr.read_bytes(), "缺失源文件时不应删除既有热图"


def test_nav_svg_is_deterministic_and_self_contained() -> None:
    first = render_nav_svg(PORTFOLIO_NAV, BENCHMARK_NAV)
    second = render_nav_svg(PORTFOLIO_NAV, BENCHMARK_NAV)

    assert first == second
    assert 'viewBox="0 0 720 260"' in first
    assert "<metadata" not in first
    assert "<title" not in first
    assert "<!--" not in first
    assert "<image" not in first
    assert first.count("<polyline") == 2, "净值图只画组合与基准两条线"
    assert len(re.findall(r"-?\d+\.\d{2},-?\d+\.\d{2}", first)) >= 2 * len(PORTFOLIO_NAV)
    assert first.rstrip().endswith("</svg>")


def test_index_backfill_only_rewrites_metric_text(workspace: Path) -> None:
    metrics = _build(workspace)
    index = workspace / "index.html"
    index.write_text(
        "<section id=\"evidence\">\n"
        "<h2><span data-metric=\"factors.total\">0</span> 个因子工件</h2>\n"
        "<details><summary>查看</summary>\n"
        "<p data-metric=\"flagship.total_return\">旧值</p>\n"
        "<p data-metric=\"zz1000.note\">旧值</p>\n"
        "</details></section>\n",
        encoding="utf-8",
    )
    tags_before = re.findall(r"</?[a-zA-Z][^>]*>", index.read_text(encoding="utf-8"))

    assert backfill_metrics(index, metrics) is True
    updated = index.read_text(encoding="utf-8")
    assert backfill_metrics(index, metrics) is False, "第二次回填应无变化"

    assert f'<span data-metric="factors.total">{metrics["factors"]["total"]}</span>' in updated
    assert f'data-metric="flagship.total_return">{format_metric("flagship.total_return", metrics["samples"]["flagship"]["total_return"])}<' in updated
    assert 'data-metric="zz1000.note">多次实验中的一个配置，不是旗舰策略<' in updated
    assert re.findall(r"</?[a-zA-Z][^>]*>", updated) == tags_before, "回填不应改动标签结构"


def test_backfill_rejects_metric_keys_missing_from_metrics(workspace: Path) -> None:
    metrics = _build(workspace)
    index = workspace / "index.html"
    index.write_text('<p data-metric="flagship.unknown">x</p>\n', encoding="utf-8")

    with pytest.raises(KeyError):
        backfill_metrics(index, metrics)


def test_main_generates_assets_and_backfills_index(workspace: Path) -> None:
    index = workspace / "site" / "index.html"
    index.parent.mkdir()
    index.write_text(
        '<p data-metric="flagship.sharpe">0.00</p>\n', encoding="utf-8"
    )
    argv = [
        "--factor-library-dir",
        str(workspace / "factor_library"),
        "--flagship-run",
        str(workspace / "runs" / "flagship"),
        "--zz1000-run",
        str(workspace / "runs" / "zz1000"),
        "--corr-png",
        str(workspace / "absent.png"),
        "--out-dir",
        str(workspace / "assets"),
        "--index",
        str(index),
    ]

    assert main(argv) == 0
    first = {name: (workspace / "assets" / name).read_bytes() for name in OUTPUT_NAMES[:-1]}
    assert main(argv) == 0
    second = {name: (workspace / "assets" / name).read_bytes() for name in OUTPUT_NAMES[:-1]}
    assert first == second
    assert "0.00" not in index.read_text(encoding="utf-8")

