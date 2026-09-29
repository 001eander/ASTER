"""每日增量更新 CLI：行情 + 公司行为 + 涨跌停一体化跑批。

T 日收盘后跑一次，把本地缓存推进到最新交易日：

1. :func:`quant.data.cache.update_daily` 增量抓日线，并全量刷新
   ``calendar.parquet`` / ``instruments.parquet``；
2. :func:`quant.data.cache.update_corporate_actions` 覆盖重抓近
   ``CA_LOOKBACK_DAYS`` 天窗口并合并进 ``corporate_actions.parquet``
   （公司行为会修订历史，故窗口内整段重抓，merge 按 ``(date, instrument)`` 去重）；
3. 对受影响的年份文件重算涨跌停：把该年文件连同**前一年**一起 ``load_bars``
   再 :func:`quant.data.limit.compute_limits`，写回时只写受影响年份，保证跨年首行
   的 ``prev_close`` 有前一年最后一条收盘价可依；
4. 打印各阶段耗时、ok/failed/empty 计数、失败清单与最新数据日期。

指数行情（issue #67）在日线增量之后单独刷新 4 只基准指数（``INDEX_CODES``）到
``effective_end``，与个股行情相互独立，失败只记账不中止其余阶段。

非交易日 ``end`` 也能正常处理：提示「最近开市日为 X」，把数据推进到该日即可。

用法::

    ASTER_NO_PROXY=1 uv run python scripts/daily_update.py
    ASTER_NO_PROXY=1 uv run python scripts/daily_update.py --end 2026-09-25
    uv run python scripts/daily_update.py --dry-run

``ASTER_NO_PROXY=1`` 约定与 ``scripts/fetch_data.py`` 一致：在导入任何网络库之前
把 ``NO_PROXY`` 置为 ``*``，绕过损坏的系统代理。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

# 约定：ASTER_NO_PROXY=1 时绕过系统代理。必须在 import akshare（连带 requests）之前。
if os.environ.get("ASTER_NO_PROXY") == "1":
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.cache import (  # noqa: E402
    CALENDAR_FILE,
    MANIFEST_FILE,
    FetchReport,
    IndexBarsReport,
    ProgressCallback,
    load_bars,
    load_instruments,
    update_corporate_actions,
    update_daily,
    update_index_bars,
)
from quant.data.limit import (  # noqa: E402
    ST_INTERVALS,
    build_st_intervals,
    compute_limits,
)
from quant.data.schema import (  # noqa: E402
    DAILY_BARS,
    INDEX_CODES,
    check_daily_bars,
)
from quant.data.source.base import DataSource  # noqa: E402

logger = logging.getLogger("daily_update")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 默认数据目录。
DEFAULT_DATA_DIR: str = "data"
#: 年线文件所在子目录（与 cache 层契约一致）。
BARS_DIRNAME: str = "bars"
#: ST 区间缓存文件名。
ST_INTERVALS_FILENAME: str = "st_intervals.parquet"
#: 公司行为每次覆盖重抓的回溯窗口（天）。公司行为会修订历史，故整段重抓。
CA_LOOKBACK_DAYS: int = 90
#: 进度打印间隔（每处理这么多只报一次）。
PROGRESS_EVERY: int = 100


@dataclass
class DailyUpdateResult:
    """一次每日更新的结果汇总。"""

    data_dir: Path
    requested_end: date
    effective_end: date
    bars: FetchReport
    ca: FetchReport
    index: IndexBarsReport = field(default_factory=IndexBarsReport)
    limit_years: dict[int, int] = field(default_factory=dict)
    latest_data_date: date | None = None
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 缓存状态查询
# ---------------------------------------------------------------------------


def manifest_max_bars_date(data_dir: Path) -> date | None:
    """从 ``_manifest.json`` 读取全市场日线的最新 ``last_date``，无数据返回 ``None``。"""
    path = Path(data_dir) / MANIFEST_FILE
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    entries = raw.get("bars")
    if not isinstance(entries, dict):
        return None
    dates: list[date] = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        value = entry.get("last_date")
        if not value:
            continue
        try:
            dates.append(date.fromisoformat(str(value)))
        except ValueError:
            continue
    return max(dates) if dates else None


def latest_open_date(data_dir: Path, end: date) -> date | None:
    """从本地 ``calendar.parquet`` 找 ``<= end`` 的最近开市日，缺失返回 ``None``。"""
    data_dir = Path(data_dir)
    if not (data_dir / CALENDAR_FILE).exists():
        return None
    from quant.data.cache import load_calendar  # 局部导入，避免与顶层导入顺序耦合

    calendar = load_calendar(data_dir, end=end)
    opened = calendar.filter(pl.col("is_open"))
    if opened.height == 0:
        return None
    return opened["date"].max()


def affected_years(data_dir: Path, before: date | None, end: date) -> list[int]:
    """返回 ``(before, end]`` 覆盖、且本地已存在年份文件的年份列表。

    ``before`` 为更新前全市场最新数据日（无数据则 ``None``）。只保留文件已存在的
    年份，避免为尚未落盘的空年份建文件。``before >= end`` 时返回空列表（无新增）。
    """
    if before is None or before >= end:
        return []
    start = before + timedelta(days=1)
    bars_dir = Path(data_dir) / BARS_DIRNAME
    years = range(start.year, end.year + 1)
    return [year for year in years if (bars_dir / f"{year:04d}.parquet").exists()]


# ---------------------------------------------------------------------------
# 涨跌停重算
# ---------------------------------------------------------------------------


def load_or_build_st(data_dir: Path, source: DataSource) -> pl.DataFrame:
    """读取 ST 区间缓存 ``st_intervals.parquet``，不存在则构建并写回。"""
    path = Path(data_dir) / ST_INTERVALS_FILENAME
    if path.exists():
        return pl.read_parquet(path).cast(ST_INTERVALS)
    instruments = load_instruments(Path(data_dir))
    out = build_st_intervals(source, instruments["instrument"].to_list())
    out.write_parquet(path)
    logger.info("构建 ST 区间（%d 行）并缓存到 %s", out.height, path)
    return out


def recompute_limits(data_dir: Path, source: DataSource, years: list[int]) -> dict[int, int]:
    """对指定年份重算涨跌停并写回，返回 ``{年份: 行数}``。

    每个年份连同前一年一起 ``load_bars``，让该年首个交易日的 ``prev_close`` 拿到
    前一年最后一条收盘价；写回时只保留该年行，不动前一年文件。
    """
    data_dir = Path(data_dir)
    counts: dict[int, int] = {}
    if not years:
        return counts
    instruments = load_instruments(data_dir)
    st = load_or_build_st(data_dir, source)
    bars_dir = data_dir / BARS_DIRNAME

    for year in sorted(set(years)):
        path = bars_dir / f"{year:04d}.parquet"
        if not path.exists():
            continue
        bars = load_bars(
            data_dir, start=date(year - 1, 1, 1), end=date(year, 12, 31)
        )
        out = compute_limits(bars, instruments, st)
        frame = (
            out.filter(pl.col("date").dt.year() == year)
            .sort(["instrument", "date"])
            .cast(DAILY_BARS)
        )
        check_daily_bars(frame, name=f"bars/{year}")
        frame.write_parquet(path)
        counts[year] = frame.height
        logger.info("写回 %s（%d 行）", path, frame.height)
    return counts


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_daily_update(
    source: DataSource,
    data_dir: Path,
    *,
    end: date,
    ca_lookback_days: int = CA_LOOKBACK_DAYS,
    progress: ProgressCallback | None = None,
) -> DailyUpdateResult:
    """执行一次每日增量更新，返回 :class:`DailyUpdateResult`。"""
    data_dir = Path(data_dir)
    timings: dict[str, float] = {}
    total_started = time.monotonic()

    # 更新前的最新数据日，用于圈定受影响的年份文件。
    before = manifest_max_bars_date(data_dir)

    started = time.monotonic()
    bars_report = update_daily(source, data_dir, end=end, progress=progress)
    timings["bars"] = time.monotonic() - started

    # update_daily 已刷新日历，据此定位 <= end 的最近开市日。
    effective_end = latest_open_date(data_dir, end) or end

    # 基准指数日线增量（issue #67）：4 只指数刷新到 effective_end。
    started = time.monotonic()
    index_report = update_index_bars(source, data_dir, end=effective_end)
    timings["index"] = time.monotonic() - started

    started = time.monotonic()
    ca_report = update_corporate_actions(
        source,
        data_dir,
        start=effective_end - timedelta(days=ca_lookback_days),
        end=effective_end,
    )
    timings["ca"] = time.monotonic() - started

    started = time.monotonic()
    years = affected_years(data_dir, before, effective_end)
    limit_counts = recompute_limits(data_dir, source, years)
    timings["limits"] = time.monotonic() - started

    timings["total"] = time.monotonic() - total_started
    return DailyUpdateResult(
        data_dir=data_dir,
        requested_end=end,
        effective_end=effective_end,
        bars=bars_report,
        ca=ca_report,
        index=index_report,
        limit_years=limit_counts,
        latest_data_date=manifest_max_bars_date(data_dir),
        timings=timings,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="每日增量更新（行情 / 公司行为 / 涨跌停）")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=date.today(),
        help="结束日期，默认今天",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    return parser


def _progress_printer(started: float) -> ProgressCallback:
    def _print(done: int, total: int, instrument: str) -> None:
        if done % PROGRESS_EVERY != 0 and done != total:
            return
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"[{done}/{total}] {instrument} elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )

    return _print


def _print_dry_run(data_dir: Path, end: date) -> None:
    before = manifest_max_bars_date(data_dir)
    latest_open = latest_open_date(data_dir, end)
    effective = latest_open or end
    if latest_open is not None:
        years = affected_years(data_dir, before, effective)
    else:
        years = []
    print(f"# dry-run data_dir={data_dir} end={end.isoformat()}")
    if latest_open is not None and latest_open != end:
        print(f"提示：{end.isoformat()} 非开市日，最近开市日为 {latest_open.isoformat()}")
    print(f"- 增量抓日线 + 刷新 calendar/instruments（推进到 {effective.isoformat()}）")
    print(f"- 基准指数日线：{list(INDEX_CODES)} 刷新到 {effective.isoformat()}")
    ca_start = effective - timedelta(days=CA_LOOKBACK_DAYS)
    print(f"- 公司行为：重抓 [{ca_start.isoformat()}, {effective.isoformat()}] 并合并")
    print(f"- 涨跌停：重算年份 {years}")
    print(f"- 当前缓存最新数据日：{before.isoformat() if before else '无'}")


def _print_summary(result: DailyUpdateResult) -> None:
    bars = result.bars
    ca = result.ca
    timings = result.timings
    print("\n# 每日更新完成")
    print(f"数据目录：{result.data_dir}")
    if result.effective_end != result.requested_end:
        print(
            f"提示：{result.requested_end.isoformat()} 非开市日，"
            f"数据推进到最近开市日 {result.effective_end.isoformat()}"
        )
    print(
        f"最新数据日期："
        f"{result.latest_data_date.isoformat() if result.latest_data_date else '无'}"
    )
    print(
        f"阶段耗时：日线 {timings.get('bars', 0.0):.1f}s，"
        f"指数 {timings.get('index', 0.0):.1f}s，"
        f"公司行为 {timings.get('ca', 0.0):.1f}s，"
        f"涨跌停 {timings.get('limits', 0.0):.1f}s，"
        f"合计 {timings.get('total', 0.0):.1f}s"
    )
    print(
        f"日线：ok={bars.ok} failed={bars.failed} "
        f"empty={bars.empty} skipped={bars.skipped}"
    )
    print(
        f"公司行为：ok={ca.ca_ok} failed={ca.ca_failed} empty={ca.ca_empty}"
    )
    index = result.index
    print(
        f"基准指数：ok={index.ok} failed={index.failed} "
        f"empty={index.empty} skipped={index.skipped} 表内 {index.rows} 行"
    )
    if index.failures:
        print(f"指数失败清单：{index.failures}")
    print(f"涨跌停重算：{result.limit_years}")
    if bars.failures:
        sample = list(bars.failures.items())[:10]
        print(f"日线失败清单（共 {len(bars.failures)}，前 10）：{sample}")
    if ca.ca_failures:
        sample = list(ca.ca_failures.items())[:10]
        print(f"公司行为失败清单（共 {len(ca.ca_failures)}，前 10）：{sample}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.dry_run:
        _print_dry_run(data_dir, args.end)
        return 0

    from quant.data.source.akshare import AkshareSource  # 延迟导入，dry-run 不碰网络

    started = time.monotonic()
    source = AkshareSource()
    print(
        f"# daily_update data_dir={args.data_dir} end={args.end.isoformat()}",
        flush=True,
    )
    result = run_daily_update(
        source,
        data_dir,
        end=args.end,
        progress=_progress_printer(started),
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
