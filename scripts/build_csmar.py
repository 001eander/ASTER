"""用 CSMAR 离线导出包重建本地 Parquet 缓存的命令行入口（issue #59）。

用法::

    # dry-run：只打印计划，不动任何文件
    uv run python scripts/build_csmar.py --csmar-dir C:/Windows/Temp/opencode/csmar

    # 正式重建
    uv run python scripts/build_csmar.py --csmar-dir C:/Windows/Temp/opencode/csmar --yes

流程（``--yes`` 时执行）：

1. 删除 ``data_dir/bars/`` 下全部年份 parquet（CSMAR 全量重建，旧起点数据一并清除）。
2. 清空 ``_manifest.json`` 的 ``"bars"`` 条目，保留 ``"ca"``（公司行为账本继续有效）。
3. 落盘 ``instruments.parquet``（以既有 akshare 表为基础，``TRD_Co`` 补充与校验）。
4. 落盘 ``calendar.parquet``（``HISTORY_START`` 至最大日线日）。
5. 全量日线一次合并写盘。
6. 按落盘结果重建 ``"bars"`` 账本条目。
7. 导出 ``csmar_limit_reference.parquet`` 供 issue #5 交叉验证涨跌停。
8. 打印票数、行数、日期范围、分年行数与耗时。

不走 ``cache.fetch_full`` 逐票路径：离线全量已在内存，逐票 flush 会让年份文件反复
重写放大 IO；离线建库崩溃后重跑成本极低，无需断点续传。``corporate_actions.parquet``
保持既有 akshare 缓存不动。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 项目内脚本复用 cache 的私有落盘/账本函数：它们承载 BARS_DIR、INSTRUMENTS_FILE 等
# 磁盘布局契约，直接调用可避免在本脚本里复刻一份。日常业务代码不应依赖这些私有名。
from quant.data.cache import (  # noqa: E402
    BARS_DIR,
    CALENDAR_FILE,
    INSTRUMENTS_FILE,
    _atomic_write_parquet,
    _load_manifest,
    _merge_daily_bars,
    _save_manifest,
    load_bars,
)
from quant.data.schema import HISTORY_START  # noqa: E402
from quant.data.source.csmar import CsmarSource  # noqa: E402

#: 涨跌停参考中间产物文件名（不属缓存布局契约，仅供 issue #5 使用）。
LIMIT_REFERENCE_FILE: str = "csmar_limit_reference.parquet"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 CSMAR 离线包全量重建 Parquet 缓存（先删后建，幂等）"
    )
    parser.add_argument(
        "--csmar-dir", required=True, help="CSMAR 解压目录（含 TRD_Dalyr*.csv）"
    )
    parser.add_argument("--data-dir", default="data", help="缓存目录，默认 data/")
    parser.add_argument(
        "--factor-scale",
        default=None,
        help="可选 JSON 路径，{instrument: 系数}，透传给 CsmarSource 归一化复权基期",
    )
    parser.add_argument(
        "--yes", action="store_true", help="确认执行；不给时只打印计划（dry-run）"
    )
    return parser


def _load_factor_scale(path: str | None) -> dict[str, float] | None:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--factor-scale 需为 {instrument: 系数} 形式的 JSON 对象")
    return {str(key): float(value) for key, value in raw.items()}


def _existing_bar_files(data_dir: Path) -> list[Path]:
    bars_dir = data_dir / BARS_DIR
    if not bars_dir.exists():
        return []
    return sorted(bars_dir.glob("*.parquet"))


def _print_plan(args: argparse.Namespace, factor_scale: dict[str, float] | None) -> None:
    data_dir = Path(args.data_dir)
    existing = _existing_bar_files(data_dir)
    print("# dry-run（未加 --yes，不修改任何文件）")
    print(f"csmar-dir={Path(args.csmar_dir)}")
    print(f"data-dir={data_dir}  history_start={HISTORY_START}")
    print(f"factor-scale={'无' if not factor_scale else f'{len(factor_scale)} 只'}")
    print(f"将删除 bars 年份文件 {len(existing)} 个：{[p.name for p in existing]}")
    print("将重建：instruments.parquet、calendar.parquet、bars/*.parquet、_manifest.json")
    print(f"将导出：{LIMIT_REFERENCE_FILE}")
    print("确认后请加 --yes 执行。")


def _clear_bars(data_dir: Path) -> int:
    removed = 0
    for path in _existing_bar_files(data_dir):
        path.unlink()
        removed += 1
    return removed


def _rebuild_manifest(data_dir: Path, loaded: pl.DataFrame) -> int:
    """按落盘 bars 重建 ``"bars"`` 账本，保留既有 ``"ca"`` 条目。"""
    manifest = _load_manifest(data_dir)
    entries: dict[str, dict[str, object]] = {}
    if loaded.height:
        summary = loaded.group_by("instrument").agg(
            pl.col("date").max().alias("last_date"),
            pl.len().alias("rows"),
        )
        for row in summary.iter_rows(named=True):
            entries[str(row["instrument"])] = {
                "last_date": row["last_date"].isoformat(),
                "rows": int(row["rows"]),
                "status": "ok",
                "error": None,
            }
    manifest["bars"] = entries
    _save_manifest(data_dir, manifest)
    return len(entries)


def _print_report(
    loaded: pl.DataFrame, limit_reference: pl.DataFrame, elapsed: float
) -> None:
    print("\n# 建库完成")
    print(f"票数（ok）：{loaded['instrument'].n_unique()}")
    print(f"总行数：{loaded.height}（涨跌停参考 {limit_reference.height} 行）")
    if loaded.height:
        print(f"日期范围：{loaded['date'].min()} ~ {loaded['date'].max()}")
        per_year = (
            loaded.group_by(pl.col("date").dt.year().alias("year"))
            .agg(pl.len().alias("rows"))
            .sort("year")
        )
        print("分年行数：")
        for row in per_year.iter_rows(named=True):
            print(f"  {row['year']}: {row['rows']}")
    else:
        print("日期范围：无数据")
    print(f"耗时：{elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    factor_scale = _load_factor_scale(args.factor_scale)

    if not args.yes:
        _print_plan(args, factor_scale)
        return 0

    started = time.monotonic()
    data_dir = Path(args.data_dir)
    csmar_dir = Path(args.csmar_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    removed = _clear_bars(data_dir)
    print(f"已删除 bars 年份文件 {removed} 个", flush=True)

    # 先清空 bars 账本、保留 ca；后续按落盘结果重建。
    manifest = _load_manifest(data_dir)
    manifest["bars"] = {}
    _save_manifest(data_dir, manifest)

    base_info = data_dir / INSTRUMENTS_FILE
    source = CsmarSource(
        csmar_dir,
        base_info=base_info if base_info.exists() else None,
        factor_scale=factor_scale,
    )
    print("重建 instruments.parquet ...", flush=True)
    info = source.instrument_info()
    _atomic_write_parquet(base_info, info)
    print(f"证券信息 {info.height} 只", flush=True)

    end = source.max_date()
    if end is None:
        print("CSMAR 日线为空，终止建库", file=sys.stderr)
        return 1

    print(f"重建 calendar.parquet（{HISTORY_START} ~ {end}）...", flush=True)
    calendar = source.trade_calendar(HISTORY_START, end)
    _atomic_write_parquet(data_dir / CALENDAR_FILE, calendar)
    print(f"日历 {calendar.height} 天，其中开市 {calendar['is_open'].sum()} 天", flush=True)

    print("合并全量日线 ...", flush=True)
    bars = source.daily_bars(info["instrument"].to_list(), HISTORY_START, end)
    _merge_daily_bars(data_dir, bars)
    print(f"日线 {bars.height} 行，落盘后校验 ...", flush=True)

    loaded = load_bars(data_dir)
    entries = _rebuild_manifest(data_dir, loaded)
    print(f"账本 bars 条目 {entries} 个", flush=True)

    print(f"导出 {LIMIT_REFERENCE_FILE} ...", flush=True)
    limit_reference = source.limit_reference()
    _atomic_write_parquet(data_dir / LIMIT_REFERENCE_FILE, limit_reference)

    _print_report(loaded, limit_reference, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
