"""从因子注册表与两个回测 run 生成主页证据区资产（issue #157）。

只写 ``site/assets/`` 与 ``site/index.html`` 里 ``data-metric`` 元素的文本：
``metrics.json``、两张净值 SVG、复制因子相关性热图。数字口径与
``scripts/backtest_e2e.py:compute_metrics`` 一致，成交 / 拒单 / 费用从 ``report.md`` 解析。

运行：``uv run python scripts/build_site_assets.py``
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

TRADING_DAYS_PER_YEAR: int = 252

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "factor_library" / "registry.json"
DEFAULT_FLAGSHIP_RUN = REPO_ROOT / "runs" / "flagship"
DEFAULT_ZZ1000_RUN = REPO_ROOT / "runs" / "e2e-zz1000-enhanced-1m-M"
DEFAULT_CORR_PNG = REPO_ROOT / "runs" / "reports" / "factor_corr_heatmap.png"
DEFAULT_OUT_DIR = REPO_ROOT / "site" / "assets"
DEFAULT_INDEX = REPO_ROOT / "site" / "index.html"

DISCLAIMER = "历史回测，非实盘，不构成投资建议"
ZZ1000_NOTE = "多次实验中的一个配置，不是旗舰策略"

RUN_FILES = ("nav.parquet", "benchmark.parquet", "report.md")

REBALANCE_LABELS = {"W": "周频", "M": "月频"}

SVG_VIEWBOX = (720, 260)
SVG_MARGIN = {"left": 44.0, "right": 12.0, "top": 14.0, "bottom": 26.0}
SVG_COLOR_GRID = "#e2dbcd"
SVG_COLOR_BASE = "#a99f8f"
SVG_COLOR_PORTFOLIO = "#7a5bb5"
SVG_COLOR_BENCHMARK = "#8a8175"

METRIC_PATTERN = re.compile(
    r'(<(?P<tag>[a-zA-Z][a-zA-Z0-9]*)(?P<attrs>[^>]*?\sdata-metric="(?P<key>[^"]+)"[^>]*)>)'
    r"(?P<body>.*?)"
    r"(</(?P=tag)>)",
    re.DOTALL,
)


class MissingInputError(RuntimeError):
    """必需输入缺失；列出全部缺失路径，调用方负责退出码。"""


# ---------------------------------------------------------------------------
# 读取与解析
# ---------------------------------------------------------------------------


def count_factors(registry_path: Path) -> dict[str, int]:
    """按 ``lineage.op`` 统计因子工件数量；注册表只读。"""
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    factors = payload.get("factors")
    if not isinstance(factors, list):
        raise ValueError(f"{registry_path} 缺少 factors 数组")
    counts = Counter(
        str(factor.get("lineage", {}).get("op", "")) for factor in factors
    )
    return {
        "total": len(factors),
        "seed": counts["seed"],
        "mutation": counts["mutation"],
        "crossover": counts["crossover"],
    }


def _parse_report(path: Path) -> dict[str, dict[str, str]]:
    """把 report.md 的二级标题表解析成 ``{节名: {行名: 值}}``。"""
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = sections.setdefault(stripped[3:].strip(), {})
            continue
        if current is None or not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 2:
            continue
        name, value = cells
        if not name or set(name) == {"-"} or set(value) == {"-"}:
            continue
        current[name] = value
    return sections


def _require(rows: Mapping[str, str], key: str, source: Path) -> str:
    value = rows.get(key)
    if value is None or value in {"", "-"}:
        raise ValueError(f"{source} 缺少字段 {key}")
    return value


def _parse_percent(text: str) -> float:
    cleaned = text.strip()
    if not cleaned.endswith("%"):
        raise ValueError(f"不是百分比：{text!r}")
    return float(cleaned[:-1]) / 100.0


def _read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort("date")


def _series_metrics(nav: pl.DataFrame) -> tuple[float, float, float, float]:
    """复算总收益、最大回撤、夏普与总换手，口径同 compute_metrics。"""
    if nav.height < 2:
        raise ValueError("nav.parquet 至少需要两个交易日")
    navs = nav["nav"]
    initial = float(navs[0])
    final = float(navs[-1])
    if initial <= 0.0 or final <= 0.0:
        raise ValueError("nav.parquet 的净值必须为正")
    total_return = final / initial - 1.0

    daily = nav.select(pl.col("nav").pct_change().alias("ret"))["ret"].drop_nulls()
    std = float(daily.std(ddof=1)) if daily.len() >= 2 else 0.0
    sharpe = (
        float(daily.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if std > 0.0
        else 0.0
    )
    max_drawdown = float((navs / navs.cum_max() - 1.0).min())
    return total_return, max_drawdown, sharpe, float(nav["turnover"].sum())


def _build_sample(
    *,
    sample_id: str,
    label: str,
    run_dir: Path,
    note: str | None = None,
) -> dict[str, Any]:
    nav_path = run_dir / "nav.parquet"
    benchmark_path = run_dir / "benchmark.parquet"
    report_path = run_dir / "report.md"

    nav = _read_parquet(nav_path)
    benchmark = _read_parquet(benchmark_path)
    if nav.height != benchmark.height:
        raise ValueError(
            f"{run_dir} 的 nav 与 benchmark 行数不一致：{nav.height} / {benchmark.height}"
        )
    if nav["date"].to_list() != benchmark["date"].to_list():
        raise ValueError(f"{run_dir} 的 nav 与 benchmark 交易日不一致")

    report = _parse_report(report_path)
    config = report.get("配置", {})
    records = report.get("绩效", {})

    initial_cash = float(_require(config, "initial_cash", report_path))
    if not math.isclose(initial_cash, float(nav["nav"][0]), rel_tol=1e-9):
        raise ValueError(
            f"{report_path} 的 initial_cash 与 nav.parquet 首日净值不一致："
            f"{initial_cash} / {float(nav['nav'][0])}"
        )

    total_return, max_drawdown, sharpe, _ = _series_metrics(nav)
    portfolio_last = float(benchmark["portfolio_nav"][-1])
    benchmark_last = float(benchmark["benchmark_nav"][-1])
    if benchmark_last <= 0.0:
        raise ValueError(f"{benchmark_path} 的基准净值必须为正")
    excess_return = portfolio_last / benchmark_last - 1.0

    dates = nav["date"].to_list()
    sample: dict[str, Any] = {
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "benchmark": _require(config, "benchmark", report_path),
        "rebalance": _require(config, "rebalance_freq", report_path),
        "strategy": _require(config, "strategy", report_path),
        "initial_cash": round(initial_cash, 2),
        "total_return": round(total_return, 6),
        "excess_return": round(excess_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe": round(sharpe, 6),
        "nav_svg": f"nav-{sample_id}.svg",
        "label": label,
        "disclaimer": DISCLAIMER,
    }
    if "成交笔数" in records:
        sample["trades"] = int(records["成交笔数"])
    if "拒单笔数" in records:
        sample["rejects"] = int(records["拒单笔数"])
    if "费用占初始资金" in records:
        sample["cost_ratio"] = round(_parse_percent(records["费用占初始资金"]), 6)
    if note is not None:
        sample["note"] = note
    return sample


# ---------------------------------------------------------------------------
# 净值 SVG
# ---------------------------------------------------------------------------


def _polyline(
    values: Sequence[float],
    y_min: float,
    y_max: float,
    width: float,
    height: float,
) -> str:
    left = SVG_MARGIN["left"]
    top = SVG_MARGIN["top"]
    span = y_max - y_min
    last = len(values) - 1
    points: list[str] = []
    for index, value in enumerate(values):
        x = left + (width * index / last if last > 0 else 0.0)
        y = top + height * (1.0 - (value - y_min) / span)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def render_nav_svg(portfolio: Sequence[float], benchmark: Sequence[float]) -> str:
    """纯字符串拼接的确定性净值图：组合与基准两条折线。"""
    if len(portfolio) < 2 or len(portfolio) != len(benchmark):
        raise ValueError("净值序列长度必须一致且至少两个点")

    view_width, view_height = SVG_VIEWBOX
    plot_width = view_width - SVG_MARGIN["left"] - SVG_MARGIN["right"]
    plot_height = view_height - SVG_MARGIN["top"] - SVG_MARGIN["bottom"]
    low = min(min(portfolio), min(benchmark))
    high = max(max(portfolio), max(benchmark))
    spread = high - low
    if spread <= 0.0:
        spread = 0.01
    y_min = low - spread * 0.06
    y_max = high + spread * 0.06

    grid: list[str] = []
    for step in range(5):
        y = SVG_MARGIN["top"] + plot_height * step / 4.0
        grid.append(
            f'<line x1="{SVG_MARGIN["left"]:.2f}" y1="{y:.2f}" '
            f'x2="{view_width - SVG_MARGIN["right"]:.2f}" y2="{y:.2f}" '
            f'stroke="{SVG_COLOR_GRID}" stroke-width="1"/>'
        )

    base_y = SVG_MARGIN["top"] + plot_height * (1.0 - (1.0 - y_min) / (y_max - y_min))
    base_y = min(max(base_y, SVG_MARGIN["top"]), SVG_MARGIN["top"] + plot_height)

    portfolio_points = _polyline(portfolio, y_min, y_max, plot_width, plot_height)
    benchmark_points = _polyline(benchmark, y_min, y_max, plot_width, plot_height)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_width} {view_height}" '
        f'width="{view_width}" height="{view_height}">',
        *grid,
        f'<line x1="{SVG_MARGIN["left"]:.2f}" y1="{base_y:.2f}" '
        f'x2="{view_width - SVG_MARGIN["right"]:.2f}" y2="{base_y:.2f}" '
        f'stroke="{SVG_COLOR_BASE}" stroke-width="1" stroke-dasharray="4 4"/>',
        f'<polyline fill="none" stroke="{SVG_COLOR_BENCHMARK}" stroke-width="1.2" '
        f'points="{benchmark_points}"/>',
        f'<polyline fill="none" stroke="{SVG_COLOR_PORTFOLIO}" stroke-width="1.8" '
        f'points="{portfolio_points}"/>',
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 生成与回填
# ---------------------------------------------------------------------------


def _missing_inputs(
    registry_path: Path, flagship_run: Path, zz1000_run: Path
) -> list[Path]:
    missing: list[Path] = []
    if not registry_path.is_file():
        missing.append(registry_path)
    for run_dir in (flagship_run, zz1000_run):
        for name in RUN_FILES:
            path = run_dir / name
            if not path.is_file():
                missing.append(path)
    return missing


def _write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def build_metrics(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    flagship_run: Path = DEFAULT_FLAGSHIP_RUN,
    zz1000_run: Path = DEFAULT_ZZ1000_RUN,
    corr_png: Path | None = DEFAULT_CORR_PNG,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """生成 metrics.json、两张净值 SVG，并在热图存在时复制；返回 metrics。"""
    registry_path = Path(registry_path)
    flagship_run = Path(flagship_run)
    zz1000_run = Path(zz1000_run)
    out_dir = Path(out_dir)

    missing = _missing_inputs(registry_path, flagship_run, zz1000_run)
    if missing:
        raise MissingInputError("\n".join(str(path) for path in missing))

    flagship_nav = _read_parquet(flagship_run / "nav.parquet")
    zz1000_nav = _read_parquet(zz1000_run / "nav.parquet")
    flagship_bench = _read_parquet(flagship_run / "benchmark.parquet")
    zz1000_bench = _read_parquet(zz1000_run / "benchmark.parquet")

    metrics: dict[str, Any] = {
        "factors": count_factors(registry_path),
        "samples": {
            "flagship": _build_sample(
                sample_id="flagship",
                label="旗舰 10 万样本",
                run_dir=flagship_run,
            ),
            "zz1000": _build_sample(
                sample_id="zz1000",
                label="中证 1000 指数增强月度样本",
                run_dir=zz1000_run,
                note=ZZ1000_NOTE,
            ),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        out_dir / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text(
        out_dir / "nav-flagship.svg",
        render_nav_svg(
            flagship_bench["portfolio_nav"].to_list(),
            flagship_bench["benchmark_nav"].to_list(),
        ),
    )
    _write_text(
        out_dir / "nav-zz1000.svg",
        render_nav_svg(
            zz1000_bench["portfolio_nav"].to_list(),
            zz1000_bench["benchmark_nav"].to_list(),
        ),
    )

    if corr_png is not None and Path(corr_png).is_file():
        shutil.copyfile(Path(corr_png), out_dir / "factor-corr.png")
    else:
        print(
            f"警告：缺少因子相关性热图 {corr_png}，保留现有 factor-corr.png",
            file=sys.stderr,
        )

    return metrics


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """把嵌套 metrics 摊平成 ``data-metric`` 使用的点分键。"""
    flat: dict[str, Any] = {}
    for key, value in metrics["factors"].items():
        flat[f"factors.{key}"] = value
    for sample_id, sample in metrics["samples"].items():
        for key, value in sample.items():
            flat[f"{sample_id}.{key}"] = value
    return flat


def format_metric(key: str, value: Any) -> str:
    """把 metrics 值格式化成页面展示文本；HTML 与测试共用这一处。"""
    if key.endswith(".initial_cash"):
        return f"{float(value) / 10_000:g} 万"
    if key.endswith(".rebalance"):
        return REBALANCE_LABELS.get(str(value), str(value))
    if key.endswith((".total_return", ".excess_return", ".max_drawdown")):
        return f"{float(value):.2%}"
    if key.endswith(".cost_ratio"):
        return f"{float(value):.1%}"
    if key.endswith(".sharpe"):
        return f"{float(value):.2f}"
    if key.endswith((".trades", ".rejects")) or key.startswith("factors."):
        return str(int(value))
    return str(value)


def _read_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_preserving_newlines(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def backfill_metrics(index_path: Path, metrics: Mapping[str, Any]) -> bool:
    """回填 index.html 的 data-metric 文本，只改文本；返回是否有变化。"""
    index_path = Path(index_path)
    source = _read_preserving_newlines(index_path)
    flat = flatten_metrics(metrics)

    seen = {match.group("key") for match in METRIC_PATTERN.finditer(source)}
    unknown = sorted(key for key in seen if key not in flat)
    if unknown:
        raise KeyError(f"页面存在 metrics 未提供的 data-metric：{', '.join(unknown)}")

    def replace(match: re.Match[str]) -> str:
        key = match.group("key")
        value = escape(format_metric(key, flat[key]), quote=False)
        return f"{match.group(1)}{value}</{match.group('tag')}>"

    updated = METRIC_PATTERN.sub(replace, source)
    if updated == source:
        return False
    _write_preserving_newlines(index_path, updated)
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成主页证据区资产：metrics.json、净值 SVG、热图副本，并回填 data-metric。"
    )
    parser.add_argument(
        "--factor-library-dir",
        type=Path,
        default=DEFAULT_REGISTRY.parent,
        help="因子库目录（含 registry.json，只读）",
    )
    parser.add_argument(
        "--flagship-run", type=Path, default=DEFAULT_FLAGSHIP_RUN, help="旗舰回测目录"
    )
    parser.add_argument(
        "--zz1000-run", type=Path, default=DEFAULT_ZZ1000_RUN, help="中证 1000 指增回测目录"
    )
    parser.add_argument(
        "--corr-png", type=Path, default=DEFAULT_CORR_PNG, help="因子相关性热图源文件"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="站点资源输出目录"
    )
    parser.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX, help="需要回填 data-metric 的主页"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        metrics = build_metrics(
            registry_path=args.factor_library_dir / "registry.json",
            flagship_run=args.flagship_run,
            zz1000_run=args.zz1000_run,
            corr_png=args.corr_png,
            out_dir=args.out_dir,
        )
        changed = backfill_metrics(args.index, metrics)
    except MissingInputError as exc:
        print("错误：缺少必需输入文件：", file=sys.stderr)
        for path in str(exc).splitlines():
            print(f"  {path}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(f"metrics：{args.out_dir / 'metrics.json'}")
    print(f"净值图：{args.out_dir / 'nav-flagship.svg'}、{args.out_dir / 'nav-zz1000.svg'}")
    print(f"主页回填：{args.index}（{'已更新' if changed else '无变化'}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
