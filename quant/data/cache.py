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
- 每只票抓完立即记账，日线数据按 ``BARS_FLUSH_EVERY`` 只批量合并写盘，
  避免逐票重写年份文件的写放大。落盘顺序为先 parquet 后账本：崩溃后账本
  落后于 parquet，重跑补抓时 ``unique(keep="last")`` 保证幂等。
- 断点状态以账本为准：``last_date`` 是已确认落盘数据在该票上的最大日期。它对
  ``status == "ok"`` 的票是增量抓取的锚点日，抓取区间为 ``[last_date, end]``
  （含该日，用于重定复权基期）；失败或无 ``last_date`` 的票仍从
  ``no_last_date_start`` 起全抓。
- 公司行为接口逐票抓取，本模块按 ``CA_BATCH_SIZE`` 分批调用数据源以摊薄账本
  写入，批次内成功整批记账；数据源对单票失败的 log+skip 无法区分「无分红」与
  「抓取失败」，故两者统一记为 ``empty``（均为终态，续传时跳过）。
- ``DataSource`` 对单票失败是 log+skip，缓存层自己按票记账：调用抛出异常的票
  记为 ``failed``，返回空表的票记为 ``empty``。
- 增量复权因子重定基（issue #59）：CSMAR 建库把历史 bars 落成 CSMAR 基期因子，
  其后由 akshare 每日增量接续。两源原始行情一致、事件乘数一致（新浪源 197 个事件
  相对差中位 3.2e-5、最大 6.3e-4），但因子绝对水平每股有恒定比例差（历史事件乘数
  口径差几十年复利，000001 CSMAR/akshare≈1.248，600519≈1.0），直接落盘会让接缝日
  每股 ``adjfactor`` 跳变一个固定倍率，破坏跨接缝的后复权收益。因此增量时多抓一个
  锚点日 ``last_date``，用 ``scale = 缓存last_date因子 / akshare同日期因子``
  把增量段整体重定基，误差只剩事件乘数口径差（~3e-5）。锚点日行不落盘，避免覆盖
  缓存基期因子。``scale`` 落在 ``[0.2, 5.0]`` 之外或锚点缺失时告警并不缩放。
- 锚定机制的幂等性：同日重跑因 ``last_date >= end`` 短路跳过；跨日重跑时锚点取
  缓存最新行，``scale`` 稳定，重复段由 ``unique(keep="last")`` 覆盖为同值。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import polars as pl

from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    HISTORY_START,
    INDEX_BARS,
    INDEX_CODES,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    check_daily_bars,
    check_index_bars,
    check_schema,
    normalize_instrument,
)
from quant.data.source.base import DataSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: ``fetch_full`` / ``update_daily`` 的默认起始日（CLI 默认值同此）。
#: 全系统历史起点 = CSMAR 建库窗口首日 2021-09-29（issue #59）。
DEFAULT_START: date = HISTORY_START

CALENDAR_FILE: str = "calendar.parquet"
INSTRUMENTS_FILE: str = "instruments.parquet"
CORPORATE_ACTIONS_FILE: str = "corporate_actions.parquet"
BARS_DIR: str = "bars"
#: 基准指数日线单文件（issue #67），schema 见 ``schema.INDEX_BARS``。
INDEX_BARS_FILE: str = "index_bars.parquet"
MANIFEST_FILE: str = "_manifest.json"

#: 公司行为按票分批抓取的批大小，用于控制账本写入频率。
CA_BATCH_SIZE: int = 200
#: ``update_daily`` 中无 ``last_date`` 的失败票重试窗口（天）。
FAILED_RETRY_LOOKBACK_DAYS: int = 30
#: 日线合并写盘的批大小：累积若干只票再一次合并进年份文件，避免逐票整文件
#: 重写带来的写放大。崩溃后账本落后于 parquet，重抓时 unique 去重保证幂等。
BARS_FLUSH_EVERY: int = 100

#: 增量复权因子重定基的合理缩放区间；超出视为异常，告警并不缩放。
REBASE_SCALE_MIN: float = 0.2
REBASE_SCALE_MAX: float = 5.0

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


@dataclass
class IndexBarsReport:
    """一次指数日线增量的结果统计（issue #67），失败只记账不抛异常。"""

    ok: int = 0
    failed: int = 0
    empty: int = 0
    skipped: int = 0
    #: 本次新增（去重后计入表内的）行数。
    rows: int = 0
    #: 指数代码 -> 错误信息。
    failures: dict[str, str] = field(default_factory=dict)
    #: 指数代码 -> 更新后的最新日期（ISO 字符串）。
    last_dates: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return self.ok + self.failed + self.empty + self.skipped


class _IndexSource(Protocol):
    """指数日线来源。``AkshareSource`` 结构性满足，测试可注入假实现。"""

    def index_bars(
        self, index_codes: list[str], start: date, end: date
    ) -> pl.DataFrame:
        ...


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


def _normalize_index_bars(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return _empty(INDEX_BARS)
    return (
        df.select(list(INDEX_BARS.keys()))
        .cast(INDEX_BARS)
        .sort(["index_code", "date"])
    )


def _index_bars_path(data_dir: Path) -> Path:
    return data_dir / INDEX_BARS_FILE


def _merge_index_bars(data_dir: Path, new_bars: pl.DataFrame) -> int:
    """把新指数行情并入单文件：concat → unique(keep=last) → sort，返回表内总行数。

    ``keep="last"`` 让新抓到的行覆盖旧行，同日重复跑幂等。
    """
    if new_bars.height == 0:
        return 0
    new_bars = _normalize_index_bars(new_bars)
    path = _index_bars_path(data_dir)
    if path.exists():
        new_bars = pl.concat(
            [pl.read_parquet(path), new_bars], how="vertical_relaxed"
        )
    out = (
        new_bars.select(list(INDEX_BARS.keys()))
        .cast(INDEX_BARS)
        .unique(subset=["index_code", "date"], keep="last")
        .sort(["index_code", "date"])
    )
    check_index_bars(out)
    _atomic_write_parquet(path, out)
    return out.height


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
    include_anchor: bool = False,
) -> tuple[date, date] | None:
    """返回该票需要补抓的闭区间，``None`` 表示可跳过。

    - 账本无记录：从 ``no_last_date_start``（缺省为 ``start``）全抓。
    - ``status == "ok"``、有 ``last_date`` 且 ``include_anchor``：从
      ``last_date`` 抓到 ``end``，多抓的锚点日用于增量复权因子重定基（不落盘）。
    - 其余有 ``last_date`` 的情况：从 ``last_date + 1`` 抓到 ``end``。
    - ``last_date >= end``：跳过。
    """
    if entry is None:
        fetch_start = no_last_date_start or start
    else:
        last = _parse_date(entry.get("last_date"))
        is_ok = entry.get("status") == STATUS_OK
        if is_ok and last is not None and last >= end:
            return None
        if last is None:
            fetch_start = no_last_date_start or start
        elif is_ok and include_anchor:
            fetch_start = last  # 含锚点日
        else:
            fetch_start = last + timedelta(days=1)
    if fetch_start > end:
        return None
    return fetch_start, end


def _is_positive_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _load_anchors(
    data_dir: Path, entries: dict[str, dict[str, object]]
) -> dict[str, tuple[date, float]]:
    """构建增量重定基锚点表：``instrument -> (last_date, 缓存adjfactor)``。

    整库只读一遍：先收集 ``status == "ok"`` 且 ``last_date`` 非空的
    ``(instrument, last_date)``，按 ``last_date`` 年份分组，只读涉及年份的 bars
    文件（``last_date`` 通常落在最近一两个年份文件）。没有候选锚点时直接返回空表，
    不读任何文件。
    """
    by_year: dict[int, list[tuple[str, date]]] = {}
    for instrument, entry in entries.items():
        if entry.get("status") != STATUS_OK:
            continue
        last = _parse_date(entry.get("last_date"))
        if last is None:
            continue
        by_year.setdefault(last.year, []).append((instrument, last))
    if not by_year:
        return {}

    anchors: dict[str, tuple[date, float]] = {}
    for year, pairs in by_year.items():
        path = _bars_path(data_dir, year)
        if not path.exists():
            continue
        wanted = sorted({instrument for instrument, _ in pairs})
        frame = pl.read_parquet(
            path, columns=["instrument", "date", "adjfactor"]
        ).filter(pl.col("instrument").is_in(wanted))
        lookup = {
            (row["instrument"], row["date"]): row["adjfactor"]
            for row in frame.iter_rows(named=True)
        }
        for instrument, last in pairs:
            factor = lookup.get((instrument, last))
            if factor is not None:
                anchors[instrument] = (last, float(factor))
    return anchors


def _rebase_on_anchor(
    instrument: str, frame: pl.DataFrame, anchor: tuple[date, float]
) -> pl.DataFrame:
    """把增量段 ``adjfactor`` 重定基到缓存基期，并去掉锚点日行。

    ``scale = 缓存锚点因子 / 数据源锚点因子``；锚点行缺失、因子非正有限值或
    ``scale`` 超出 ``[REBASE_SCALE_MIN, REBASE_SCALE_MAX]`` 时告警并不缩放。
    无论是否缩放，锚点日及其之前的行都不落盘，避免覆盖缓存基期因子。
    """
    last_date, cached_factor = anchor
    anchor_rows = frame.filter(pl.col("date") == last_date)
    scale = 1.0
    if anchor_rows.height == 0:
        logger.warning(
            "增量重定基缺锚点行：%s 在 %s 无数据（返回日期范围 %s ~ %s），原样落盘",
            instrument,
            last_date,
            frame["date"].min(),
            frame["date"].max(),
        )
    else:
        source_factor = anchor_rows["adjfactor"][0]
        if _is_positive_finite(source_factor) and _is_positive_finite(cached_factor):
            candidate = cached_factor / source_factor
            if REBASE_SCALE_MIN <= candidate <= REBASE_SCALE_MAX:
                scale = candidate
            else:
                logger.warning(
                    "增量重定基 scale=%.4g 超出 [%.2g, %.2g]：%s 锚点 %s "
                    "数据源因子=%s 缓存因子=%s，不缩放",
                    candidate,
                    REBASE_SCALE_MIN,
                    REBASE_SCALE_MAX,
                    instrument,
                    last_date,
                    source_factor,
                    cached_factor,
                )
        else:
            logger.warning(
                "增量重定基因子非正有限值：%s 锚点 %s 数据源因子=%s 缓存因子=%s，不缩放",
                instrument,
                last_date,
                source_factor,
                cached_factor,
            )
    frame = frame.filter(pl.col("date") > last_date)
    if scale != 1.0 and frame.height:
        frame = frame.with_columns(
            (pl.col("adjfactor") * scale).alias("adjfactor")
        )
    return frame


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
    anchors: dict[str, tuple[date, float]] | None = None,
) -> None:
    entries = _bar_entries(manifest)
    total = len(instruments)
    buffer: list[pl.DataFrame] = []
    pending: list[str] = []

    def flush() -> None:
        """合并缓冲区进年份文件，然后落账（顺序保证崩溃后可幂等重抓）。"""
        if buffer:
            _merge_daily_bars(data_dir, pl.concat(buffer, how="vertical_relaxed"))
            buffer.clear()
        if pending:
            _save_manifest(data_dir, manifest)
            pending.clear()

    for done, instrument in enumerate(instruments, start=1):
        try:
            anchor = anchors.get(instrument) if anchors else None
            plan = _bar_plan(
                entries.get(instrument),
                start,
                end,
                no_last_date_start,
                include_anchor=anchor is not None,
            )
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
                pending.append(instrument)
            else:
                if anchor is not None:
                    frame = _rebase_on_anchor(instrument, frame, anchor)
                if frame.height:
                    buffer.append(frame)
                    new_rows = frame.select(["instrument", "date"]).unique().height
                    previous = entries.get(instrument, {})
                    entries[instrument] = {
                        "last_date": _dump_date(frame["date"].max()),
                        "rows": int(previous.get("rows") or 0) + new_rows,
                        "status": STATUS_OK,
                        "error": None,
                    }
                    report.ok += 1
                elif anchor is not None:
                    # 只返回锚点行或尚无新数据：视为已是最新，保持原 last_date。
                    previous = entries.get(instrument, {})
                    entries[instrument] = {
                        "last_date": previous.get("last_date"),
                        "rows": int(previous.get("rows") or 0),
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
                pending.append(instrument)
            if len(pending) >= BARS_FLUSH_EVERY:
                flush()
            _emit(progress, done, total, instrument)
        except BaseException:
            # 中断/异常退出前尽量落盘，保住已完成的进度。
            try:
                flush()
            except Exception:  # noqa: BLE001 - 清理失败不掩盖原异常
                logger.exception("中断清理落盘失败")
            raise
    flush()


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
    # 全量首抓同一来源内部基期自洽，无需锚定（anchors 默认 None）。
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

    增量抓取会多带一个锚点日（缓存 ``last_date``）并对新行 ``adjfactor`` 重定基到
    缓存基期，跨数据源接缝不跳变，详见模块 docstring 的「增量复权因子重定基」。
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
    # 增量段复权因子重定基到缓存基期（issue #59），避免跨源接缝处每股因子跳变。
    anchors = _load_anchors(data_dir, entries)

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
        anchors=anchors,
    )
    _save_manifest(data_dir, manifest)
    report.elapsed_seconds = time.monotonic() - started
    return report


def update_corporate_actions(
    source: DataSource,
    data_dir: Path,
    *,
    start: date,
    end: date,
    instruments: list[str] | None = None,
) -> FetchReport:
    """公司行为增量刷新：重抓 ``[start, end]`` 窗口并与缓存合并。

    公司行为会修订历史（补录、更正除权除息日），因此每次对窗口内全市场覆盖重抓，
    而不是像日线那样只补缺口；``_merge_corporate_actions`` 按 ``(date, instrument)``
    去重，同窗口重复跑幂等。``instruments`` 为 ``None`` 时优先用
    ``instruments.parquet`` 的代码，缺失则向数据源取并落盘。

    逐批抓取，单批失败只记账不抛异常，统计写在 ``FetchReport`` 的 ``ca_*`` 字段。
    """
    started = time.monotonic()
    data_dir = Path(data_dir)
    _ensure_data_dir(data_dir)
    report = FetchReport()

    if instruments is None:
        if (data_dir / INSTRUMENTS_FILE).exists():
            instruments = load_instruments(data_dir)["instrument"].to_list()
        else:
            info = source.instrument_info()
            check_schema(info, INSTRUMENT_INFO, name="instrument_info")
            _atomic_write_parquet(data_dir / INSTRUMENTS_FILE, info)
            instruments = info["instrument"].to_list()

    for offset in range(0, len(instruments), CA_BATCH_SIZE):
        batch = instruments[offset : offset + CA_BATCH_SIZE]
        try:
            frame = _normalize_actions(source.corporate_actions(batch, start, end))
            if frame.height:
                frame = frame.filter(
                    (pl.col("date") >= start) & (pl.col("date") <= end)
                )
        except Exception as exc:  # noqa: BLE001 - 单批失败记账继续
            message = str(exc)[:500]
            for instrument in batch:
                report.ca_failed += 1
                report.ca_failures[instrument] = str(exc)
            logger.warning("抓取公司行为批次失败（%d 只）：%s", len(batch), message)
            continue
        if frame.height:
            _merge_corporate_actions(data_dir, frame)
        present = set(frame["instrument"].to_list()) if frame.height else set()
        for instrument in batch:
            if instrument in present:
                report.ca_ok += 1
            else:
                report.ca_empty += 1

    if not (data_dir / CORPORATE_ACTIONS_FILE).exists():
        # 无任何公司行为时也落一个空表，保证缓存布局契约完整。
        _atomic_write_parquet(data_dir / CORPORATE_ACTIONS_FILE, _empty(CORPORATE_ACTIONS))

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


# ---------------------------------------------------------------------------
# 指数日线（issue #67）
# ---------------------------------------------------------------------------


def _index_last_dates(data_dir: Path) -> dict[str, date]:
    """读 ``index_bars.parquet`` 里每只指数的最新日期，缺失或读不动返回空表。"""
    path = _index_bars_path(data_dir)
    if not path.exists():
        return {}
    try:
        df = pl.read_parquet(path, columns=["index_code", "date"])
    except Exception as exc:  # noqa: BLE001 - 损坏文件按全量重抓处理
        logger.warning("index_bars.parquet 读取失败，按全量重抓：%s", exc)
        return {}
    return {
        str(row["index_code"]): row["date"]
        for row in df.group_by("index_code")
        .agg(pl.col("date").max())
        .iter_rows(named=True)
    }


def update_index_bars(
    source: _IndexSource,
    data_dir: Path,
    *,
    end: date,
    start: date = HISTORY_START,
    index_codes: tuple[str, ...] = INDEX_CODES,
) -> IndexBarsReport:
    """增量刷新基准指数日线到 ``end``（issue #67）。

    每只指数从缓存里的最新日期（含当日，用于覆盖修订）抓到 ``end``；缓存无该指数
    时从 ``start`` 全抓。合并按 ``(index_code, date)`` 去重 ``keep="last"``，同日重复
    跑幂等。单只失败 log+记账，不影响其余指数。
    """
    started = time.monotonic()
    data_dir = Path(data_dir)
    _ensure_data_dir(data_dir)
    report = IndexBarsReport()
    last_dates = _index_last_dates(data_dir)
    frames: list[pl.DataFrame] = []

    for index_code in index_codes:
        last = last_dates.get(index_code)
        if last is not None and last >= end:
            report.skipped += 1
            report.last_dates[index_code] = last.isoformat()
            continue
        fetch_start = last if last is not None else start
        try:
            frame = source.index_bars([index_code], fetch_start, end)
        except Exception as exc:  # noqa: BLE001 - 单只失败不影响其余
            report.failed += 1
            report.failures[index_code] = str(exc)[:500]
            logger.warning("抓取指数 %s 失败：%s", index_code, exc)
            continue
        frame = _normalize_index_bars(frame) if frame is not None else _empty(INDEX_BARS)
        if frame.height == 0:
            report.empty += 1
            if last is not None:
                report.last_dates[index_code] = last.isoformat()
            continue
        frames.append(frame)
        report.ok += 1
        report.last_dates[index_code] = frame["date"].max().isoformat()

    if frames:
        report.rows = _merge_index_bars(
            data_dir, pl.concat(frames, how="vertical_relaxed")
        )
    elif _index_bars_path(data_dir).exists():
        report.rows = pl.read_parquet(_index_bars_path(data_dir)).height

    report.elapsed_seconds = time.monotonic() - started
    return report


def load_index_bars(
    data_dir: Path,
    *,
    index_codes: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """读取指数日线，输出过 ``check_index_bars``；文件缺失返回空表。"""
    path = _index_bars_path(Path(data_dir))
    if not path.exists():
        return _empty(INDEX_BARS)
    df = pl.read_parquet(path)
    if index_codes is not None:
        df = df.filter(pl.col("index_code").is_in(list(index_codes)))
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    out = (
        df.select(list(INDEX_BARS.keys()))
        .cast(INDEX_BARS)
        .unique(subset=["index_code", "date"], keep="last")
        .sort(["index_code", "date"])
    )
    check_index_bars(out)
    return out


__all__ = [
    "DEFAULT_START",
    "FetchReport",
    "INDEX_BARS_FILE",
    "IndexBarsReport",
    "ProgressCallback",
    "fetch_full",
    "load_bars",
    "load_calendar",
    "load_corporate_actions",
    "load_index_bars",
    "load_instruments",
    "update_corporate_actions",
    "update_daily",
    "update_index_bars",
]
