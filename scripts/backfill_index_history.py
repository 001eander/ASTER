"""CSMAR 取样变更表历史回填 CLI（issue #64 历史段）。

以官方最新成分名单为基准，用 CSMAR 指数取样变更表逆放重建 2021-09 起的调样生效日
成分快照，并用「等效市值加权」近似调样日权重锚；随后重建日频成分 / 权重表。

流程::

    1. 解析 CSMAR 取样变更表（过滤股票类与目标指数，去重归一化代码）；
    2. 以 ``data/index_member_snapshots.parquet`` 里各指数最新一份为基准逆放重建，
       正放回推做自洽校验（不变式不通过即中止）；
    3. 合并快照落盘（同 (指数, 生效日) 整份覆盖）；
    4. 用 ``close × adjfactor`` 构造每个调样生效日的等效权重锚，与既有官方锚合并
       （官方锚优先，整份覆盖同日权重）；
    5. :func:`quant.data.index_members.build_daily_tables` 重建日频表；
    6. 在官方锚日用同一等效口径重算并与官方权重对比，打印偏差中位数 / 90 分位 / 最大值。

用法::

    uv run python scripts/backfill_index_history.py \\
        --csmar-dir "C:\\Windows\\Temp\\opencode\\csmar-index"
    uv run python scripts/backfill_index_history.py --csmar-dir <dir> --dry-run

``--csmar-dir`` 可重复传入（CSMAR 分年度导出包各一个目录）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data import cache  # noqa: E402
from quant.data.index_history import (  # noqa: E402
    build_equivalent_anchors,
    check_reconstruction_invariant,
    compare_equivalent_to_official,
    csmar_change_files,
    read_csmar_changes,
    reconstruct_snapshots,
)
from quant.data.index_members import (  # noqa: E402
    INDEX_CODES,
    MEMBER_SNAPSHOTS_FILE,
    WEIGHT_ANCHORS_FILE,
    merge_member_snapshots,
    merge_weight_anchors,
    read_anchor_weights,
    read_snapshots,
    build_daily_tables,
)
from quant.data.schema import HISTORY_START  # noqa: E402

logger = logging.getLogger("backfill_index_history")

#: 偏差披露的分位数。
P90: float = 0.9


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSMAR 变更表历史回填指数成分与权重")
    parser.add_argument("--data-dir", default="data", help="缓存目录，默认 data/")
    parser.add_argument(
        "--csmar-dir",
        action="append",
        required=True,
        metavar="DIR",
        help="CSMAR 取样变更表导出目录（可重复传入）",
    )
    parser.add_argument("--end", type=_parse_date, default=None, help="结束日期，默认最新")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    return parser


def _summarize_snapshots(snapshots: pl.DataFrame) -> None:
    print("\n# 重建后成分快照覆盖")
    summary = (
        snapshots.group_by("index_code")
        .agg(
            pl.col("snapshot_date").min().alias("min_date"),
            pl.col("snapshot_date").max().alias("max_date"),
            pl.col("snapshot_date").n_unique().alias("n_snapshots"),
            pl.len().alias("n_rows"),
        )
        .sort("index_code")
    )
    for row in summary.iter_rows(named=True):
        print(
            f"  {row['index_code']}: {row['n_snapshots']} 份快照 / "
            f"{row['n_rows']} 行，{row['min_date']} ~ {row['max_date']}"
        )


def _summarize_anchors(anchors: pl.DataFrame) -> None:
    print("\n# 权重锚")
    summary = (
        anchors.group_by("index_code")
        .agg(
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            pl.col("date").n_unique().alias("n_dates"),
            pl.len().alias("n_rows"),
        )
        .sort("index_code")
    )
    for row in summary.iter_rows(named=True):
        print(
            f"  {row['index_code']}: {row['n_dates']} 个锚日 / {row['n_rows']} 行，"
            f"{row['min_date']} ~ {row['max_date']}"
        )


def _summarize_deviations(comparison: pl.DataFrame) -> None:
    if comparison.height == 0:
        print("\n# 等效口径偏差披露：无官方锚可比")
        return
    print("\n# 等效市值口径 vs 官方锚 权重绝对偏差披露")
    stats = (
        comparison.group_by("index_code")
        .agg(
            pl.col("abs_deviation").median().alias("median"),
            pl.col("abs_deviation").quantile(P90).alias("p90"),
            pl.col("abs_deviation").max().alias("max"),
            pl.len().alias("n"),
        )
        .sort("index_code")
    )
    for row in stats.iter_rows(named=True):
        print(
            f"  {row['index_code']}: 中位 {row['median']:.2e} / "
            f"90 分位 {row['p90']:.2e} / 最大 {row['max']:.2e}（{row['n']} 票）"
        )
    overall = comparison["abs_deviation"]
    print(
        f"  全指数合计：中位 {overall.median():.2e} / "
        f"90 分位 {overall.quantile(P90):.2e} / 最大 {overall.max():.2e}"
    )


def _summarize_deviations_by_date(comparison: pl.DataFrame) -> None:
    if comparison.height == 0:
        return
    stats = (
        comparison.group_by(["index_code", "anchor_date"])
        .agg(
            pl.col("abs_deviation").median().alias("median"),
            pl.len().alias("n"),
        )
        .sort(["index_code", "anchor_date"])
    )
    print("\n# 分锚日偏差")
    for row in stats.iter_rows(named=True):
        print(
            f"  {row['index_code']}@{row['anchor_date']}: 中位 {row['median']:.2e}"
            f"（{row['n']} 票）"
        )


def _verify_daily_tables(data_dir: Path) -> None:
    members = pl.read_parquet(data_dir / "index_members.parquet")
    weights = pl.read_parquet(data_dir / "index_weights.parquet")
    sums = weights.group_by(["index_code", "date"]).agg(pl.col("weight").sum().alias("s"))
    bad = sums.filter((pl.col("s") < 0.99) | (pl.col("s") > 1.01))
    print("\n# 日频表验收")
    print(
        f"  成分表 {members.height} 行，权重表 {weights.height} 行；"
        f"权重和越界 (指数, 日) {bad.height} 个"
    )
    if weights.height:
        print(
            f"  权重和范围 {sums['s'].min():.6f} ~ {sums['s'].max():.6f}"
        )


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    files: list[Path] = []
    for directory in args.csmar_dir:
        files.extend(csmar_change_files(directory))
    if not files:
        logger.error("未在 %s 下找到 CSMAR 取样变更表（模式 IDX_Chgsmp*.csv）", args.csmar_dir)
        return 2

    changes = read_csmar_changes(files)
    print(f"# 变更流水 {changes.height} 行，来自 {len(files)} 个分片")
    summary = (
        changes.group_by("index_code")
        .agg(
            pl.col("effective_date").min().alias("min_date"),
            pl.col("effective_date").max().alias("max_date"),
            pl.len().alias("n"),
        )
        .sort("index_code")
    )
    for row in summary.iter_rows(named=True):
        print(
            f"  {row['index_code']}: {row['n']} 条变更，"
            f"{row['min_date']} ~ {row['max_date']}"
        )

    snapshots = read_snapshots(data_dir)
    if snapshots.height == 0:
        logger.error("缺少官方成分基准快照 %s，先跑 build_index_members.py", data_dir / MEMBER_SNAPSHOTS_FILE)
        return 2

    if args.dry_run:
        print("\n# dry-run：解析与重建计划如下，不落盘")
        print(f"- 基准快照 {snapshots.height} 行")
        print(f"- 目标指数 {list(INDEX_CODES)}")
        print("- 逆放重建 → 合并快照 → 等效权重锚 → 重建日频表")
        return 0

    reconstructed = reconstruct_snapshots(changes, snapshots, coverage_start=HISTORY_START)
    check_reconstruction_invariant(changes, reconstructed)
    print("\n# 逆放重建自洽校验通过（正放回推逐日集合一致）")

    merged_snapshots = merge_member_snapshots(snapshots, reconstructed)
    _summarize_snapshots(merged_snapshots)
    cache._atomic_write_parquet(data_dir / MEMBER_SNAPSHOTS_FILE, merged_snapshots)
    print(f"已写入 {data_dir / MEMBER_SNAPSHOTS_FILE}")

    calendar = cache.load_calendar(data_dir, end=args.end)
    open_days = [
        day
        for day in calendar.filter(pl.col("is_open"))["date"].to_list()
        if day >= HISTORY_START
    ]
    instruments = merged_snapshots["instrument"].unique().to_list()
    bars = cache.load_bars(
        data_dir, instruments=instruments, start=HISTORY_START, end=args.end
    )
    approx, dropped = build_equivalent_anchors(merged_snapshots, bars, open_days)
    if dropped.height:
        print(f"\n# 等效权重锚剔除无当日行情票 {dropped.height} 条")
        for row in (
            dropped.group_by("index_code", "date")
            .agg(pl.len().alias("n"))
            .sort("index_code", "date")
            .iter_rows(named=True)
        ):
            print(f"  {row['index_code']}@{row['date']}: {row['n']} 票")

    official = read_anchor_weights(data_dir)
    anchors = merge_weight_anchors(approx, official)
    _summarize_anchors(anchors)
    cache._atomic_write_parquet(data_dir / WEIGHT_ANCHORS_FILE, anchors)
    print(f"已写入 {data_dir / WEIGHT_ANCHORS_FILE}")

    counts = build_daily_tables(data_dir, end=args.end)
    print(
        f"\n# 日频表：成分 {counts['members']} 行，权重 {counts['weights']} 行，"
        f"对拍明细 {counts['drift_check']} 行"
    )

    comparison = compare_equivalent_to_official(official, bars)
    _summarize_deviations(comparison)
    _summarize_deviations_by_date(comparison)
    _verify_daily_tables(data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
