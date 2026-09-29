"""持仓风险分析四表 CLI（issue #69）。

读入持仓权重长表 ``(date, instrument, weight)``，输出指数分布 / 市值分布 / 风格暴露 /
主动行业暴露四表（markdown 必出，xlsx 视 xlsxwriter / openpyxl 是否可用）。数据源全部
为本地缓存：``index_members.parquet`` / ``index_weights.parquet`` / ``industry.parquet``
/ ``bars/``。

用法::

    # 直接给持仓权重 parquet
    uv run python scripts/risk_report.py \
        --weights runs/e2e/holdings.parquet --data-dir data \
        --benchmark 000905 --out runs/e2e/risk_report.md

    # 读 e2e 产出目录里的 holdings.parquet（backtest_e2e 落盘）
    uv run python scripts/risk_report.py \
        --from-e2e runs/e2e --data-dir data --benchmark 000905

持仓权重的构造由调用方负责（e2e 回测持仓或 run_daily 调仓单）；本脚本只消费。
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from quant.data.cache import load_bars, load_calendar, load_industry  # noqa: E402
from quant.data.index_members import read_index_members, read_index_weights  # noqa: E402
from quant.eval.risk_report import (  # noqa: E402
    MV_SHARE_CONST,
    ConsistencyError,
    RiskReport,
    build_risk_report,
    check_enhanced_consistency,
    render_markdown,
    write_xlsx,
)
from quant.portfolio.style import (  # noqa: E402
    MOMENTUM_WINDOW,
    compute_style_factors,
    equivalent_market_value,
)

#: 风格因子预热余量（交易日）。
STYLE_WARMUP_MARGIN: int = 80

#: 持仓权重列名别名 → 内部规范名。
COLUMN_ALIASES: dict[str, str] = {
    "date": "date",
    "trade_date": "date",
    "DATE": "date",
    "instrument": "instrument",
    "ticker": "instrument",
    "TICKER": "instrument",
    "weight": "weight",
    "WEIGHT": "weight",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="持仓风险分析四表（issue #69）")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="持仓权重 parquet/csv 路径")
    source.add_argument(
        "--from-e2e", type=Path, help="e2e 产出目录，读取其 holdings.parquet"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="缓存目录，默认 data/")
    parser.add_argument(
        "--benchmark", default=None, help="基准指数代码（六位，如 000905），给了才出主动行业暴露"
    )
    parser.add_argument("--out", type=Path, default=None, help="markdown 输出路径")
    parser.add_argument("--xlsx", type=Path, default=None, help="xlsx 输出路径（默认与 md 同名）")
    parser.add_argument(
        "--share-const",
        type=float,
        default=MV_SHARE_CONST,
        help=f"等效市值常数股本，默认 {MV_SHARE_CONST:.0e}",
    )
    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="给基准时附带自洽性检查（需 --industry-tol / --style-tol）",
    )
    parser.add_argument("--industry-tol", type=float, default=None, help="行业偏离带（相对基准）")
    parser.add_argument("--style-tol", type=float, default=None, help="风格偏离带（std 倍数）")
    parser.add_argument("--market-value-tol", type=float, default=None, help="市值偏离带（std 倍数）")
    return parser.parse_args(argv)


def load_holdings_weights(path: Path) -> pl.DataFrame:
    """读持仓权重，支持 parquet / csv，容忍常见列名别名。"""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path, try_parse_dates=True)
    rename = {
        src: COLUMN_ALIASES[src]
        for src in frame.columns
        if src in COLUMN_ALIASES and COLUMN_ALIASES[src] != src
    }
    if rename:
        frame = frame.rename(rename)
    return frame


def _style_window_start(data_dir: Path, ref_date: date) -> date:
    """取 ``ref_date`` 往前 ``MOMENTUM_WINDOW + STYLE_WARMUP_MARGIN`` 个开市日。"""
    calendar = load_calendar(data_dir, end=ref_date)
    days = (
        calendar.filter(pl.col("is_open") & (pl.col("date") <= ref_date))
        .sort("date")["date"]
        .to_list()
    )
    if not days:
        return ref_date
    index = max(0, len(days) - (MOMENTUM_WINDOW + STYLE_WARMUP_MARGIN))
    return days[index]


def run(
    *,
    holdings: pl.DataFrame,
    data_dir: Path,
    benchmark: str | None,
    share_const: float,
    check_consistency: bool,
    industry_tol: float | None,
    style_tol: float | None,
    market_value_tol: float | None,
) -> tuple[str, RiskReport]:
    """构造四表并渲染 markdown，返回 ``(markdown, RiskReport)``。"""
    if holdings.height == 0:
        raise ValueError("持仓权重为空")
    hold = holdings.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("instrument").cast(pl.String),
        pl.col("weight").cast(pl.Float64),
    ).select("date", "instrument", "weight")
    dates = sorted(set(hold["date"].to_list()))
    instruments = hold["instrument"].unique().to_list()

    window_start = _style_window_start(data_dir, dates[0])
    bars = load_bars(data_dir, instruments=instruments, start=window_start, end=dates[-1])
    if bars.height == 0:
        raise ValueError(f"行情窗口 [{window_start}, {dates[-1]}] 内没有数据")

    date_set = pl.Series("date", dates, dtype=pl.Date).implode()
    equiv_mv = equivalent_market_value(bars, share_const=share_const).filter(
        pl.col("date").is_in(date_set)
    )
    style_factors = compute_style_factors(bars).filter(pl.col("date").is_in(date_set))

    members = read_index_members(data_dir)
    industry = load_industry(data_dir)

    bench_weights = None
    if benchmark is not None:
        bench_weights = read_index_weights(data_dir)

    report = build_risk_report(
        hold,
        members=members,
        equiv_mv=equiv_mv,
        style_factors=style_factors,
        bench_weights=bench_weights,
        industry=industry if bench_weights is not None else None,
        bench_index=benchmark,
    )

    notes: list[str] = [
        f"持仓 {hold.height} 行，{len(dates)} 个交易日 {dates[0]}–{dates[-1]}，"
        f"{len(instruments)} 只证券。",
        f"指数桶归属用 index_members 当日 PIT（覆盖 {members['date'].min() if members.height else '无'} 起）。",
        f"等效市值 = close × adjfactor × {share_const:.0e}（近似市值口径）。",
    ]
    if benchmark is None:
        notes.append("未给基准，主动行业暴露表未产出。")
    if check_consistency and benchmark is not None and bench_weights is not None:
        try:
            checks = check_enhanced_consistency(
                hold,
                bench_weights=bench_weights,
                industry=industry,
                style=style_factors,
                equiv_mv=equiv_mv,
                industry_tol=industry_tol,
                style_tol=style_tol,
                market_value_tol=market_value_tol,
                bench_index=benchmark,
            )
            notes.append(
                f"自洽性检查通过：{checks.height} 个交易日，"
                f"行业最大偏离 {checks['industry_max_ratio'].max():.4f}，"
                f"风格最大偏离 {checks['style_max_std'].max():.4f}。"
            )
        except ConsistencyError as exc:
            notes.append(f"自洽性检查未通过：{exc}")

    return render_markdown(report, notes=notes), report


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = parse_args(argv)
    if args.from_e2e is not None:
        weights_path = args.from_e2e / "holdings.parquet"
        out = args.out if args.out is not None else args.from_e2e / "risk_report.md"
    else:
        weights_path = args.weights
        out = args.out if args.out is not None else weights_path.with_name("risk_report.md")

    holdings = load_holdings_weights(weights_path)
    text, report = run(
        holdings=holdings,
        data_dir=args.data_dir,
        benchmark=args.benchmark,
        share_const=args.share_const,
        check_consistency=args.check_consistency,
        industry_tol=args.industry_tol,
        style_tol=args.style_tol,
        market_value_tol=args.market_value_tol,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"已写入 {out}")

    xlsx_path = args.xlsx if args.xlsx is not None else out.with_suffix(".xlsx")
    if write_xlsx(xlsx_path, report.tables()):
        print(f"已写入 {xlsx_path}")
    else:
        print("未安装 xlsxwriter / openpyxl，跳过 xlsx（可选产出）")

    for name, df in report.tables().items():
        print(f"\n## {name}（{df.height} 行）")
        print(df.tail(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
