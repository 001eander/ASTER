"""Parquet 缓存层：全量首抓、断点续传与增量更新。

磁盘布局（契约固定，issue #5 也按此读写）::

    data/
      calendar.parquet           # TRADE_CALENDAR，全量刷新
      instruments.parquet        # INSTRUMENT_INFO，全量刷新
      corporate_actions.parquet  # CORPORATE_ACTIONS，全量刷新
      bars/YYYY.parquet          # 每自然年一个文件，schema.DAILY_BARS，按 (instrument, date) 排序
      _manifest.json             # 抓取账本

``_manifest.json`` 结构::

    {
      "bars": {"600000.SH": {"last_date": "2026-09-26", "rows": 1234,
                             "status": "ok", "error": null}, ...},
      "ca":   {"600000.SH": {"status": "ok", "rows": 3, "error": null}, ...},
      "updated_at": "2026-09-28T20:00:00"
    }

设计取舍
--------
- 每只票抓完立即合并写盘并落一次账本，进程被杀后重跑只补缺口。代价是
  ``bars/YYYY.parquet`` 每年一个整体文件，逐票合并会反复读写该文件；全市场
  首抓的写放大明显，换来的是任意时刻中断都可续传。
- 断点状态以账本为准：``last_date`` 是已落盘数据在该票上的最大日期，它不是
  ``end`` 时只补 ``[last_date + 1, end]``，从头重抓的只有账本里无记录的票。
- 公司行为接口逐票抓取，本模块按 ``CA_BATCH_SIZE`` 分批调用数据源以摊薄账本
  写入，批次内成功整批记账；数据源对单票失败的 log+skip 无法区分「无分红」与
  「抓取失败」，故两者统一记为 ``empty``（均为终态，续传时跳过）。
- ``DataSource`` 对单票失败是 log+skip，缓存层自己按票记账：调用抛出异常的票
  记为 ``failed``，返回空表的票记为 ``empty``。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    check_daily_bars,
    check_schema,
    normalize_instrument,
)
from quant.data.source.base import DataSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: ``fetch_full`` / ``update_daily`` 的默认起始日（CLI 默认值同此）。
DEFAULT_START: date = date(2015, 1, 1)

CALENDAR_FILE: str = "calendar.parquet"
INSTRUMENTS_FILE: str = "instruments.parquet"
CORPORATE_ACTIONS_FILE: str = "corporate_actions.parquet"
BARS_DIR: str = "bars"
MANIFEST_FILE: str = "_manifest.json"

#: 公司行为按票分批抓取的批大小，用于控制账本写入频率。
CA_BATCH_SIZE: int = 200
#: ``update_daily`` 中无 ``last_date`` 的失败票重试窗口（天）。
FAILED_RETRY_LOOKBACK_DAYS: int = 30

#: 账本里 bars 条目的状态取值。
STATUS_OK: str = "ok"
STATUS_FAILED: str = "failed"
STATUS_EMPTY: str = "empty"

#: 进度回调：(已处理只数, 总只数, 当前证券代码) -> None。
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class FetchReport:
    """一次抓取的结果统计，失败只记账不抛异常。

    ``ok`` / ``failed`` / ``empty`` / ``skipped`` 与 ``failures`` 描述日线抓取；
    ``ca_*`` 字段单独描述公司行为抓取，避免与日线口径互相污染。
    """

    ok: int = 0
    failed: int = 0
    empty: int = 0
    skipped: int = 0
    #: 日线失败清单：证券代码 -> 错误信息。
    failures: dict[str, str] = field(default_factory=dict)
    ca_ok: int = 0
    ca_failed: int = 0
    ca_empty: int = 0
    #: 公司行为失败清单：证券代码 -> 错误信息。
    ca_failures: dict[str, str] = field(default_factory=dict)
    #: 抓取耗时（秒）。
    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        """日线处理过的票数（含跳过）。"""
        return self.ok + self.failed + self.empty + self.skipped


# ---------------------------------------------------------------------------
# 路径与账本
# ---------------------------------------------------------------------------


def _bars_path(data_dir: Path, year: int) -> Path:
    return data_dir / BARS_DIR / f"{year:04d}.parquet"


def _ensure_data_dir(data_dir: Path) -> None:
    (data_dir / BARS_DIR).mkdir(parents=True, exist_ok=True)


def _empty(schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _atomic_write_parquet(path: Path, df: pl.DataFrame) -> None:
    """先写临时文件再替换，避免中途被杀留下半截 parquet。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _load_manifest(data_dir: Path) -> dict[str, object]:
    path = data_dir / MANIFEST_FILE
    if not path.exists():
        return {"bars": {}, "ca": {}, "updated_at": None}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # 账本损坏时按空账本重来
        logger.warning("_manifest.json 读取失败，按空账本处理：%s", exc)
        return {"bars": {}, "ca": {}, "updated_at": None}
    if not isinstance(raw, dict):
        return {"bars": {}, "ca": {}, "updated_at": None}
    raw.setdefault("bars", {})
    raw.setdefault("ca", {})
    raw.setdefault("updated_at", None)
    return raw


def _save_manifest(data_dir: Path, manifest: dict[str, object]) -> None:
    manifest["updated_at"] = dt.datetime.now().replace(microsecond=0).isoformat()
    path = data_dir / MANIFEST_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)


def _parse_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _dump_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _bar_entries(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = manifest.get("bars")
    if not isinstance(entries, dict):
        entries = {}
        manifest["bars"] = entries
    return entries  # type: ignore[return-value]


def _ca_entries(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = manifest.get("ca")
    if not isinstance(entries, dict):
        entries = {}
        manifest["ca"] = entries
    return entries  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 表归一化与合并
# ---------------------------------------------------------------------------


def _normalize_bars(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return _empty(DAILY_BARS)
    return (
        df.select(list(DAILY_BARS.keys()))
        .cast(DAILY_BARS)
        .sort(["instrument", "date"])
    )


def _normalize_actions(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return _empty(CORPORATE_ACTIONS)
    return (
        df.select(list(CORPORATE_ACTIONS.keys()))
        .cast(CORPORATE_ACTIONS)
        .sort(["instrument", "date"])
    )


def _merge_daily_bars(data_dir: Path, new_bars: pl.DataFrame) -> None:
    """把新行情并入受影响的年份文件：concat → unique(keep=last) → sort。"""
    if new_bars.height == 0:
        return
    new_bars = _normalize_bars(new_bars)
    years = sorted(set(new_bars["date"].dt.year().to_list()))
    for year in years:
        path = _bars_path(data_dir, int(year))
        chunk = new_bars.filter(pl.col("date").dt.year() == year)
        if path.exists():
            chunk = pl.concat([pl.read_parquet(path), chunk], how="vertical_relaxed")
        chunk = (
            chunk.select(list(DAILY_BARS.keys()))
            .cast(DAILY_BARS)
            .unique(subset=["instrument", "date"], keep="last")
            .sort(["instrument", "date"])
        )
        check_daily_bars(chunk, name=f"bars/{year}")
        _atomic_write_parquet(path, chunk)


def _merge_corporate_actions(data_dir: Path, new_actions: pl.DataFrame) -> None:
    if new_actions.height == 0:
        return
    new_actions = _normalize_actions(new_actions)
    path = data_dir / CORPORATE_ACTIONS_FILE
    if path.exists():
        new_actions = pl.concat(
            [pl.read_parquet(path), new_actions], how="vertical_relaxed"
        )
    out = (
        new_actions.select(list(CORPORATE_ACTIONS.keys()))
        .cast(CORPORATE_ACTIONS)
        .unique(subset=["date", "instrument"], keep="last")
        .sort(["instrument", "date"])
    )
    check_schema(out, CORPORATE_ACTIONS, name="corporate_actions")
    _atomic_write_parquet(path, out)


# ---------------------------------------------------------------------------
# 抓取计划
# ---------------------------------------------------------------------------


def _bar_plan(
    entry: dict[str, object] | None,
    start: date,
    end: date,
    no_last_date_start: date | None,
) -> tuple[date, date] | None:
    """返回该票需要补抓的闭区间，``None`` 表示可跳过。

    - 账本无记录：从 ``no_last_date_start``（缺省为 ``start``）全抓。
    - 有 ``last_date``：从 ``last_date + 1`` 抓到 ``end``。
    - status 为 ``ok`` 且 ``last_date >= end``：跳过。
    """
    if entry is None:
        fetch_start = no_last_date_start or start
    else:
        last = _parse_date(entry.get("last_date"))
        if entry.get("status") == STATUS_OK and last is not None and last >= end:
            return None
        fetch_start = last + timedelta(days=1) if last is not None else (
            no_last_date_start or start
        )
    if fetch_start > end:
        return None
    return fetch_start, end


def _emit(progress: ProgressCallback | None, done: int, total: int, instrument: str) -> None:
    if progress is not None:
        progress(done, total, instrument)


# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------


def _fetch_bars(
    source: DataSource,
    data_dir: Path,
    manifest: dict[str, object],
    instruments: list[str],
    start: date,
    end: date,
    report: FetchReport,
    progress: ProgressCallback | None,
    no_last_date_start: date | None = None,
) -> None:
    entries = _bar_entries(manifest)
    total = len(instruments)
    for done, instrument in enumerate(instruments, start=1):
        plan = _bar_plan(entries.get(instrument), start, end, no_last_date_start)
        if plan is None:
            report.skipped += 1
            _emit(progress, done, total, instrument)
            continue
        fetch_start, fetch_end = plan
        try:
            frame = _normalize_bars(
                source.daily_bars([instrument], fetch_start, fetch_end)
            )
        except Exception as exc:  # noqa: BLE001 - 单票失败不影响整批
            previous = entries.get(instrument, {})
            entries[instrument] = {
                "last_date": previous.get("last_date"),
                "rows": int(previous.get("rows") or 0),
                "status": STATUS_FAILED,
                "error": str(exc)[:500],
            }
            report.failed += 1
            report.failures[instrument] = str(exc)
            logger.warning("抓取 %s 日线失败：%s", instrument, exc)
        else:
            if frame.height:
                _merge_daily_bars(data_dir, frame)
                new_rows = frame.select(["instrument", "date"]).unique().height
                previous = entries.get(instrument, {})
                entries[instrument] = {
                    "last_date": _dump_date(frame["date"].max()),
                    "rows": int(previous.get("rows") or 0) + new_rows,
                    "status": STATUS_OK,
                    "error": None,
                }
                report.ok += 1
            else:
                previous = entries.get(instrument, {})
                entries[instrument] = {
                    "last_date": previous.get("last_date"),
                    "rows": int(previous.get("rows") or 0),
                    "status": STATUS_EMPTY,
                    "error": None,
                }
                report.empty += 1
        # 逐票落账：进程被杀后重跑能续。
        _save_manifest(data_dir, manifest)
        _emit(progress, done, total, instrument)


def _fetch_corporate_actions(
    source: DataSource,
    data_dir: Path,
    manifest: dict[str, object],
    instruments: list[str],
    start: date,
    end: date,
    report: FetchReport,
) -> None:
    entries = _ca_entries(manifest)
    # ok / empty 均为终态（公司行为是全历史一次抓取），只补 failed。
    pending = [
        instrument
        for instrument in instruments
        if entries.get(instrument, {}).get("status") not in (STATUS_OK, STATUS_EMPTY)
    ]
    for offset in range(0, len(pending), CA_BATCH_SIZE):
        batch = pending[offset : offset + CA_BATCH_SIZE]
        try:
            frame = _normalize_actions(source.corporate_actions(batch, start, end))
            if frame.height:
                frame = frame.filter(
                    (pl.col("date") >= start) & (pl.col("date") <= end)
                )
        except Exception as exc:  # noqa: BLE001 - 单批失败记账继续
            message = str(exc)[:500]
            for instrument in batch:
                entries[instrument] = {
                    "status": STATUS_FAILED,
                    "rows": 0,
                    "error": message,
                }
                report.ca_failed += 1
                report.ca_failures[instrument] = str(exc)
            logger.warning("抓取公司行为批次失败（%d 只）：%s", len(batch), exc)
            _save_manifest(data_dir, manifest)
            continue
        if frame.height:
            _merge_corporate_actions(data_dir, frame)
        present = set(frame["instrument"].to_list()) if frame.height else set()
        for instrument in batch:
            if instrument in present:
                rows = int(frame.filter(pl.col("instrument") == instrument).height)
                entries[instrument] = {"status": STATUS_OK, "rows": rows, "error": None}
                report.ca_ok += 1
            else:
                entries[instrument] = {"status": STATUS_EMPTY, "rows": 0, "error": None}
                report.ca_empty += 1
        _save_manifest(data_dir, manifest)


def fetch_full(
    source: DataSource,
    data_dir: Path,
    *,
    start: date,
    end: date,
    instruments: list[str] | None = None,
    ca: bool = True,
    progress: ProgressCallback | None = None,
) -> FetchReport:
    """全量首抓：拉基础表 + 全市场日线，支持断点续传。

    ``instruments`` 为 ``None`` 时用 ``source.instrument_info()`` 取全市场代码
    （含退市股），并落盘 ``instruments.parquet``；给定列表（调试用）时只按列表抓，
    若 ``instruments.parquet`` 尚不存在则补抓一次证券信息以补全缓存布局。

    失败只记账不抛异常，返回值见 :class:`FetchReport`。
    """
    started = time.monotonic()
    data_dir = Path(data_dir)
    _ensure_data_dir(data_dir)
    manifest = _load_manifest(data_dir)

    if instruments is None:
        info = source.instrument_info()
        check_schema(info, INSTRUMENT_INFO, name="instrument_info")
        _atomic_write_parquet(data_dir / INSTRUMENTS_FILE, info)
        instruments = info["instrument"].to_list()
    else:
        instruments = [normalize_instrument(item) for item in instruments]
        if not (data_dir / INSTRUMENTS_FILE).exists():
            info = source.instrument_info()
            check_schema(info, INSTRUMENT_INFO, name="instrument_info")
            _atomic_write_parquet(data_dir / INSTRUMENTS_FILE, info)

    calendar = source.trade_calendar(start, end)
    check_schema(calendar, TRADE_CALENDAR, name="trade_calendar")
    _atomic_write_parquet(data_dir / CALENDAR_FILE, calendar)

    report = FetchReport()
    _fetch_bars(source, data_dir, manifest, instruments, start, end, report, progress)
    if ca:
        _fetch_corporate_actions(
            source, data_dir, manifest, instruments, start, end, report
        )
        if not (data_dir / CORPORATE_ACTIONS_FILE).exists():
            # 无任何公司行为时也落一个空表，保证缓存布局契约完整。
            _atomic_write_parquet(
                data_dir / CORPORATE_ACTIONS_FILE, _empty(CORPORATE_ACTIONS)
            )

    _save_manifest(data_dir, manifest)
    report.elapsed_seconds = time.monotonic() - started
    return report


def update_daily(
    source: DataSource,
    data_dir: Path,
    *,
    end: date | None = None,
    progress: ProgressCallback | None = None,
) -> FetchReport:
    """增量更新：只补 ``status == ok`` 的票的缺口，失败票重试一次。

    ``end`` 默认今天。``calendar.parquet`` 与 ``instruments.parquet`` 每次全量
    刷新；公司行为不在本期增量范围内（由 ``fetch_full`` 负责，issue #4 再调度）。
    """
    started = time.monotonic()
    data_dir = Path(data_dir)
    _ensure_data_dir(data_dir)
    end = end or date.today()
    manifest = _load_manifest(data_dir)

    info = source.instrument_info()
    check_schema(info, INSTRUMENT_INFO, name="instrument_info")
    _atomic_write_parquet(data_dir / INSTRUMENTS_FILE, info)

    calendar = source.trade_calendar(DEFAULT_START, end)
    check_schema(calendar, TRADE_CALENDAR, name="trade_calendar")
    _atomic_write_parquet(data_dir / CALENDAR_FILE, calendar)

    entries = _bar_entries(manifest)
    instruments = [
        instrument
        for instrument, entry in entries.items()
        if entry.get("status") in (STATUS_OK, STATUS_FAILED)
    ]

    report = FetchReport()
    _fetch_bars(
        source,
        data_dir,
        manifest,
        instruments,
        DEFAULT_START,
        end,
        report,
        progress,
        no_last_date_start=end - timedelta(days=FAILED_RETRY_LOOKBACK_DAYS),
    )
    _save_manifest(data_dir, manifest)
    report.elapsed_seconds = time.monotonic() - started
    return report


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


def _iter_bar_files(
    data_dir: Path, start: date | None, end: date | None
) -> Iterator[Path]:
    bars_dir = data_dir / BARS_DIR
    if not bars_dir.exists():
        return
    for path in sorted(bars_dir.glob("*.parquet")):
        try:
            year = int(path.stem)
        except ValueError:
            continue
        if start is not None and year < start.year:
            continue
        if end is not None and year > end.year:
            continue
        yield path


def load_bars(
    data_dir: Path,
    *,
    instruments: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """读取日线，按年份文件裁剪读取范围，输出过 ``check_daily_bars``。"""
    data_dir = Path(data_dir)
    frames = [pl.read_parquet(path) for path in _iter_bar_files(data_dir, start, end)]
    if not frames:
        return _empty(DAILY_BARS)
    df = pl.concat(frames, how="vertical_relaxed")
    if instruments is not None:
        wanted = [normalize_instrument(item) for item in instruments]
        df = df.filter(pl.col("instrument").is_in(wanted))
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    out = (
        df.select(list(DAILY_BARS.keys()))
        .cast(DAILY_BARS)
        .sort(["instrument", "date"])
    )
    check_daily_bars(out)
    return out


def load_calendar(
    data_dir: Path, *, start: date | None = None, end: date | None = None
) -> pl.DataFrame:
    """读取交易日历，输出过 ``check_schema``。"""
    df = pl.read_parquet(Path(data_dir) / CALENDAR_FILE)
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    out = df.select(list(TRADE_CALENDAR.keys())).cast(TRADE_CALENDAR).sort("date")
    check_schema(out, TRADE_CALENDAR, name="trade_calendar")
    return out


def load_instruments(data_dir: Path) -> pl.DataFrame:
    """读取证券信息，输出过 ``check_schema``。"""
    df = pl.read_parquet(Path(data_dir) / INSTRUMENTS_FILE)
    out = df.select(list(INSTRUMENT_INFO.keys())).cast(INSTRUMENT_INFO).sort("instrument")
    check_schema(out, INSTRUMENT_INFO, name="instrument_info")
    return out


def load_corporate_actions(
    data_dir: Path,
    *,
    instruments: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """读取公司行为，输出过 ``check_schema``。"""
    df = pl.read_parquet(Path(data_dir) / CORPORATE_ACTIONS_FILE)
    if instruments is not None:
        wanted = [normalize_instrument(item) for item in instruments]
        df = df.filter(pl.col("instrument").is_in(wanted))
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    out = (
        df.select(list(CORPORATE_ACTIONS.keys()))
        .cast(CORPORATE_ACTIONS)
        .sort(["instrument", "date"])
    )
    check_schema(out, CORPORATE_ACTIONS, name="corporate_actions")
    return out


__all__ = [
    "DEFAULT_START",
    "FetchReport",
    "ProgressCallback",
    "fetch_full",
    "load_bars",
    "load_calendar",
    "load_corporate_actions",
    "load_instruments",
    "update_daily",
]
