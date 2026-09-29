"""全量抓取 A 股数据到本地 Parquet 缓存的命令行入口。

用法::

    # 全市场首抓（默认自 DEFAULT_START（全系统历史起点）至今，含公司行为）
    ASTER_NO_PROXY=1 uv run python scripts/fetch_data.py

    # 调试：只抓两只票
    ASTER_NO_PROXY=1 uv run python scripts/fetch_data.py \
        --instruments 600519,300750 --start 2026-09-01

网络说明：本机系统代理可能损坏，``requests`` 会读 Windows 注册表里的代理配置。
设 ``ASTER_NO_PROXY=1`` 时脚本在导入任何网络库之前把 ``NO_PROXY`` 置为 ``*``，
绕过系统代理直连。

调度（定时增量更新）由 issue #4 负责，本脚本只做一次全量抓取。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

# 约定：ASTER_NO_PROXY=1 时绕过系统代理。必须在 import akshare（连带 requests）之前。
if os.environ.get("ASTER_NO_PROXY") == "1":
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.cache import (  # noqa: E402
    DEFAULT_START,
    FetchReport,
    fetch_full,
)
from quant.data.source.akshare import AkshareSource  # noqa: E402

#: 进度打印间隔（每处理这么多只报一次）。
PROGRESS_EVERY: int = 100


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="全量抓取 A 股数据到 Parquet 缓存（支持断点续传）"
    )
    parser.add_argument("--data-dir", default="data", help="缓存目录，默认 data/")
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=DEFAULT_START,
        help=f"起始日期，默认 {DEFAULT_START.isoformat()}",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=date.today(),
        help="结束日期，默认今天",
    )
    parser.add_argument(
        "--instruments",
        default=None,
        help="逗号分隔的证券代码，调试用；不给则抓全市场",
    )
    ca_group = parser.add_mutually_exclusive_group()
    ca_group.add_argument(
        "--ca", dest="ca", action="store_true", default=True, help="抓公司行为（默认）"
    )
    ca_group.add_argument(
        "--no-ca", dest="ca", action="store_false", help="跳过公司行为"
    )
    return parser


def _progress_printer(started: float):
    def _print(done: int, total: int, instrument: str) -> None:
        if done % PROGRESS_EVERY != 0 and done != total:
            return
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"[{done}/{total}] {instrument} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )

    return _print


def _print_report(report: FetchReport) -> None:
    print("\n# 抓取完成")
    print(
        f"日线：ok={report.ok} failed={report.failed} "
        f"empty={report.empty} skipped={report.skipped}"
    )
    print(
        f"公司行为：ok={report.ca_ok} failed={report.ca_failed} "
        f"empty={report.ca_empty}"
    )
    if report.failures:
        sample = list(report.failures.items())[:10]
        print(f"日线失败样例（共 {len(report.failures)}）：{sample}")
    if report.ca_failures:
        print(f"公司行为失败（共 {len(report.ca_failures)}）")
    print(f"耗时：{report.elapsed_seconds:.1f}s")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    instruments = (
        [item.strip() for item in args.instruments.split(",") if item.strip()]
        if args.instruments
        else None
    )

    started = time.monotonic()
    source = AkshareSource()
    print(
        f"# fetch_full data_dir={args.data_dir} "
        f"range={args.start}~{args.end} ca={args.ca} "
        f"instruments={'全市场' if instruments is None else len(instruments)}",
        flush=True,
    )
    report = fetch_full(
        source,
        Path(args.data_dir),
        start=args.start,
        end=args.end,
        instruments=instruments,
        ca=args.ca,
        progress=_progress_printer(started),
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
