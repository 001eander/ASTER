"""``scripts.daily_update`` 与 ``cache.update_corporate_actions`` 单元测试。

全部用例用内存 ``FakeSource`` + ``tmp_path``，不触网；ST 区间缓存预先写成空表，
避开 ``build_st_intervals`` 的曾用名接口探测。覆盖：

- 每日增量主流程（行情 + 公司行为 + 涨跌停一体化）；
- 同日重复跑幂等；
- 跨年 ``prev_close``：1 月 2 日的涨跌停用 12 月 31 日收盘价；
- 非交易日 ``end``：推进到最近开市日。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant.data.cache import (
    fetch_full,
    load_bars,
    load_corporate_actions,
    load_index_bars,
    load_industry,
)
from quant.data.limit import ST_INTERVALS
from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INDEX_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
)
from quant.data.source.base import DataSource
from scripts.daily_update import (
    affected_years,
    latest_open_date,
    manifest_max_bars_date,
    run_daily_update,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def _bar(day: date, instrument: str, close: float = 10.0) -> dict[str, object]:
    return {
        "date": day,
        "instrument": instrument,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "vwap": close,
        "volume": 1000.0,
        "amount": close * 1000.0,
        "adjfactor": 1.0,
        "limit_up": None,
        "limit_down": None,
    }


def _info(*instruments: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "instrument": instrument,
                "name": instrument,
                "board": "main",
                "list_date": date(2000, 1, 1),
                "delist_date": None,
            }
            for instrument in instruments
        ],
        schema=INSTRUMENT_INFO,
    )


def _ca(day: date, instrument: str) -> dict[str, object]:
    return {
        "date": day,
        "instrument": instrument,
        "cash_per_share": 0.5,
        "share_per_share": 0.1,
    }


def _index_row(day: date, code: str, close: float) -> dict[str, object]:
    return {
        "date": day,
        "index_code": code,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
    }


class FakeSource:
    """实现 ``DataSource`` 协议的内存数据源（参考 tests/test_cache.py）。"""

    def __init__(
        self,
        bars: dict[str, list[dict[str, object]]] | None = None,
        *,
        info: pl.DataFrame | None = None,
        actions: dict[str, list[dict[str, object]]] | None = None,
        index_rows: dict[str, list[dict[str, object]]] | None = None,
        industry: list[dict[str, object]] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self._bars = bars or {}
        self._info = info if info is not None else pl.DataFrame(schema=INSTRUMENT_INFO)
        self._actions = actions or {}
        self._index_rows = index_rows or {}
        self._industry = industry
        self._fail = set(fail or ())
        self.bar_calls: list[tuple[str, date, date]] = []
        self.ca_calls: list[tuple[str, date, date]] = []
        self.index_calls: list[tuple[str, date, date]] = []

    def daily_bars(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for instrument in instruments:
            if instrument in self._fail:
                raise RuntimeError(f"{instrument} 抓取失败")
            self.bar_calls.append((instrument, start, end))
            rows.extend(
                row
                for row in self._bars.get(instrument, [])
                if start <= row["date"] <= end  # type: ignore[operator]
            )
        if not rows:
            return pl.DataFrame(schema=DAILY_BARS)
        return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])

    def trade_calendar(self, start: date, end: date) -> pl.DataFrame:
        df = pl.DataFrame({"date": pl.date_range(start, end, interval="1d", eager=True)})
        return df.with_columns(
            (pl.col("date").dt.weekday() <= 5).alias("is_open")
        ).cast(TRADE_CALENDAR)

    def corporate_actions(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for instrument in instruments:
            self.ca_calls.append((instrument, start, end))
            rows.extend(
                row
                for row in self._actions.get(instrument, [])
                if start <= row["date"] <= end  # type: ignore[operator]
            )
        if not rows:
            return pl.DataFrame(schema=CORPORATE_ACTIONS)
        return pl.DataFrame(rows, schema=CORPORATE_ACTIONS).sort(["instrument", "date"])

    def instrument_info(self) -> pl.DataFrame:
        return self._info

    def index_bars(
        self, index_codes: list[str], start: date, end: date
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for code in index_codes:
            self.index_calls.append((code, start, end))
            rows.extend(
                row
                for row in self._index_rows.get(code, [])
                if start <= row["date"] <= end  # type: ignore[operator]
            )
        if not rows:
            return pl.DataFrame(schema=INDEX_BARS)
        return pl.DataFrame(rows, schema=INDEX_BARS).sort(["index_code", "date"])

    def industry_classification(self) -> pl.DataFrame:
        schema = {
            "instrument": pl.String,
            "industry_l1": pl.String,
            "industry_l2": pl.String,
        }
        rows = self._industry
        if rows is None:
            rows = [
                {"instrument": item, "industry_l1": "信息技术", "industry_l2": "半导体"}
                for item in self._info["instrument"].to_list()
            ]
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema).sort("instrument")


def _write_empty_st(data_dir: Path) -> None:
    """预置空 ST 区间缓存，让重算涨跌停时不触网。"""
    pl.DataFrame(schema=ST_INTERVALS).write_parquet(data_dir / "st_intervals.parquet")


def test_fake_source_satisfies_protocol() -> None:
    assert isinstance(FakeSource(), DataSource)


# ---------------------------------------------------------------------------
# 增量主流程
# ---------------------------------------------------------------------------


def test_daily_update_main_flow(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2025, 12, 31), "600000.SH", close=10.0),
            _bar(date(2026, 1, 2), "600000.SH", close=12.0),
        ],
        "000001.SZ": [
            _bar(date(2025, 12, 31), "000001.SZ", close=20.0),
            _bar(date(2026, 1, 2), "000001.SZ", close=21.0),
        ],
    }
    info = _info("600000.SH", "000001.SZ")
    actions = {"600000.SH": [_ca(date(2026, 1, 2), "600000.SH")]}

    fetch_full(
        FakeSource(bars, info=info, actions=actions),
        tmp_path,
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
    )
    _write_empty_st(tmp_path)

    source = FakeSource(bars, info=info, actions=actions)
    result = run_daily_update(source, tmp_path, end=date(2026, 1, 2))

    assert result.effective_end == date(2026, 1, 2)
    assert result.bars.ok == 2
    assert result.bars.failed == 0
    assert result.ca.ca_ok == 1
    assert result.ca.ca_empty == 1
    assert result.limit_years == {2026: 2}
    assert result.latest_data_date == date(2026, 1, 2)

    # 行业阶段（issue #65）：按 effective_end 落一份快照，覆盖全部未退市证券。
    assert result.industry.refreshed
    assert result.industry.effective_from == date(2026, 1, 2)
    assert result.industry.missing == 0
    assert result.industry.coverage == 1.0
    assert result.timings["industry"] >= 0.0
    assert load_industry(tmp_path).height == 2

    loaded = load_bars(tmp_path)
    assert loaded["date"].max() == date(2026, 1, 2)
    assert loaded.height == 4
    # 涨跌停已按前收填出：600000 前收 12/31=10.0，000001 前收 20.0。
    got = {
        (row["instrument"], row["date"]): (row["limit_up"], row["limit_down"])
        for row in loaded.iter_rows(named=True)
    }
    assert got[("600000.SH", date(2026, 1, 2))] == (11.0, 9.0)
    assert got[("000001.SZ", date(2026, 1, 2))] == (22.0, 18.0)
    # 12/31 是各票首行，前收为 null。
    assert got[("600000.SH", date(2025, 12, 31))] == (None, None)

    ca = load_corporate_actions(tmp_path)
    assert ca.height == 1
    assert ca["instrument"].to_list() == ["600000.SH"]


def test_daily_update_is_idempotent(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2025, 12, 31), "600000.SH", close=10.0),
            _bar(date(2026, 1, 2), "600000.SH", close=12.0),
        ]
    }
    info = _info("600000.SH")
    actions = {"600000.SH": [_ca(date(2026, 1, 2), "600000.SH")]}

    fetch_full(
        FakeSource(bars, info=info, actions=actions),
        tmp_path,
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
    )
    _write_empty_st(tmp_path)

    first = run_daily_update(
        FakeSource(bars, info=info, actions=actions), tmp_path, end=date(2026, 1, 2)
    )
    bars_snapshot = load_bars(tmp_path)
    ca_snapshot = load_corporate_actions(tmp_path)

    second_source = FakeSource(bars, info=info, actions=actions)
    second = run_daily_update(second_source, tmp_path, end=date(2026, 1, 2))

    # 第二次没有新增，全部跳过，不再重算涨跌停。
    assert second.bars.skipped == 1
    assert second.limit_years == {}
    assert second.latest_data_date == date(2026, 1, 2)
    # 缓存内容逐字节一致：无重复行、无重复公司行为。
    assert load_bars(tmp_path).equals(bars_snapshot)
    assert load_corporate_actions(tmp_path).equals(ca_snapshot)
    assert bars_snapshot.height == 2
    assert first.limit_years == {2026: 1}


def test_daily_update_lands_index_bars(tmp_path: Path) -> None:
    """指数行情阶段（issue #67）：落地 000300 并计入汇总，同日重跑跳过。"""
    bars = {
        "600000.SH": [
            _bar(date(2025, 12, 31), "600000.SH", close=10.0),
            _bar(date(2026, 1, 2), "600000.SH", close=12.0),
        ]
    }
    info = _info("600000.SH")
    index_rows = {
        "000300": [
            _index_row(date(2025, 12, 31), "000300", 4000.0),
            _index_row(date(2026, 1, 2), "000300", 4020.0),
        ]
    }

    fetch_full(FakeSource(bars, info=info), tmp_path, start=date(2025, 1, 1), end=date(2025, 12, 31), ca=False)
    _write_empty_st(tmp_path)

    result = run_daily_update(
        FakeSource(bars, info=info, index_rows=index_rows),
        tmp_path,
        end=date(2026, 1, 2),
    )

    assert result.index.ok == 1
    assert result.index.empty == 3
    assert result.index.rows == 2
    assert result.index.last_dates["000300"] == date(2026, 1, 2).isoformat()
    assert result.timings["index"] >= 0.0

    loaded = load_index_bars(tmp_path)
    assert loaded.height == 2
    assert loaded["close"].to_list() == [4000.0, 4020.0]

    second = run_daily_update(
        FakeSource(bars, info=info, index_rows=index_rows),
        tmp_path,
        end=date(2026, 1, 2),
    )
    # 000300 已到 end 短路跳过；其余 3 只无数据记 empty。
    assert second.index.skipped == 1
    assert second.index.empty == 3


# ---------------------------------------------------------------------------
# 跨年 prev_close
# ---------------------------------------------------------------------------


def test_daily_update_recomputes_limits_across_year_with_prev_close(
    tmp_path: Path,
) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2025, 12, 31), "600000.SH", close=8.0),
            _bar(date(2026, 1, 2), "600000.SH", close=9.0),
        ]
    }
    info = _info("600000.SH")
    fetch_full(
        FakeSource(bars, info=info),
        tmp_path,
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        ca=False,
    )
    _write_empty_st(tmp_path)

    result = run_daily_update(FakeSource(bars, info=info), tmp_path, end=date(2026, 1, 2))

    assert result.limit_years == {2026: 1}
    row = load_bars(tmp_path).filter(
        (pl.col("instrument") == "600000.SH") & (pl.col("date") == date(2026, 1, 2))
    )
    # 1 月 2 日的涨跌停必须用 12 月 31 日收盘 8.0，而非当日 9.0。
    assert row["limit_up"].to_list() == [8.8]
    assert row["limit_down"].to_list() == [7.2]


# ---------------------------------------------------------------------------
# 非交易日 end
# ---------------------------------------------------------------------------


def test_daily_update_non_trading_end_uses_latest_open(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2026, 1, 1), "600000.SH", close=10.0),  # 周四
            _bar(date(2026, 1, 2), "600000.SH", close=11.0),  # 周五
        ]
    }
    info = _info("600000.SH")
    fetch_full(
        FakeSource(bars, info=info),
        tmp_path,
        start=date(2025, 12, 31),
        end=date(2026, 1, 1),
        ca=False,
    )
    _write_empty_st(tmp_path)

    # 2026-01-03 是周六，非开市日；应推进到最近开市日 2026-01-02。
    result = run_daily_update(FakeSource(bars, info=info), tmp_path, end=date(2026, 1, 3))

    assert result.requested_end == date(2026, 1, 3)
    assert result.effective_end == date(2026, 1, 2)
    assert result.latest_data_date == date(2026, 1, 2)
    assert latest_open_date(tmp_path, date(2026, 1, 3)) == date(2026, 1, 2)
    assert load_bars(tmp_path)["date"].max() == date(2026, 1, 2)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def test_manifest_max_bars_date_and_affected_years(tmp_path: Path) -> None:
    assert manifest_max_bars_date(tmp_path) is None
    assert affected_years(tmp_path, None, date(2026, 1, 2)) == []

    bars = {"600000.SH": [_bar(date(2025, 12, 31), "600000.SH")]}
    fetch_full(
        FakeSource(bars, info=_info("600000.SH")),
        tmp_path,
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        ca=False,
    )
    assert manifest_max_bars_date(tmp_path) == date(2025, 12, 31)
    # 已有 2025 年文件，但没有 2026 年文件。
    assert affected_years(tmp_path, date(2025, 12, 31), date(2026, 1, 2)) == []
    assert affected_years(tmp_path, date(2024, 12, 31), date(2026, 1, 2)) == [2025]
