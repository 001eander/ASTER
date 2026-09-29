"""指数成分与权重抓取 CLI（issue #64）。

抓取中证官方最新成分名单与月末权重，增量落地到本地缓存并重建日频 PIT 表。
幂等：成分名单未变时不重复落盘，权重锚按 (指数, 日期) 覆盖写入。

用法::

    ASTER_NO_PROXY=1 uv run python scripts/build_index_members.py
    ASTER_NO_PROXY=1 uv run python scripts/build_index_members.py --end 2026-09-28
    uv run python scripts/build_index_members.py --dry-run

``ASTER_NO_PROXY=1`` 约定与 ``scripts/daily_update.py`` 一致：在导入任何网络库
之前把 ``NO_PROXY`` 置为 ``*``，绕过损坏的系统代理。

注意：中证官网只提供**最新一期**成分名单与月末权重，历史月度权重无公开归档。
本脚本负责把最新锚增量累加并向前漂移；2021-09 起的历史回填需要外部离线导出
（CSMAR 指数成分/权重表）后写入 ``index_member_snapshots.parquet`` 与
``index_weight_anchors.parquet`` 再调用 :func:`build_daily_tables`。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

# 约定：ASTER_NO_PROXY=1 时绕过系统代理。必须在 import akshare 之前。
if os.environ.get("ASTER_NO_PROXY") == "1":
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.index_members import (  # noqa: E402
    INDEX_CODES,
    run_index_update,
)


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取中证指数成分与权重并重建日频表")
    parser.add_argument("--data-dir", default="data", help="缓存目录，默认 data/")
    parser.add_argument("--end", type=_parse_date, default=None, help="结束日期，默认最新")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.dry_run:
        print(f"# dry-run data_dir={data_dir} end={args.end or '最新'}")
        print(f"- 抓取指数：{list(INDEX_CODES)}")
        print("- 内存增量更新成分快照与权重锚，重建 index_members/index_weights")
        return 0

    from quant.data.source.csindex import CsindexSource  # 延迟导入，dry-run 不碰网络

    started = time.monotonic()
    print(f"# build_index_members data_dir={data_dir}", flush=True)
    counts = run_index_update(CsindexSource(), data_dir, end=args.end)
    print("\n# 完成")
    print(
        f"指数成分快照 {counts.get('snapshots', 0)} 行 / 权重锚 "
        f"{counts.get('anchors', 0)} 行"
    )
    print(
        f"日频成分表 {counts.get('members', 0)} 行，日频权重表 "
        f"{counts.get('weights', 0)} 行，对拍明细 {counts.get('drift_check', 0)} 行"
    )
    print(
        f"本次更新指数 {counts.get('updated', 0)}/{counts.get('checked', 0)}，"
        f"耗时 {time.monotonic() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
