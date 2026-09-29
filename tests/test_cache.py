"""``quant.data.cache`` 单元测试。

全部用例用内存 ``FakeSource`` + ``tmp_path``，不触网。覆盖：

- 全量抓取落盘布局与账本；
- 中断后续传只补缺口（用 ``KeyboardInterrupt`` 模拟进程被打断）；
- 跨年合并到 ``bars/YYYY.parquet``；
- 重复抓取幂等（``unique`` 生效）；
- 失败票记账与重试、空数据记账；
- ``load_*`` 的过滤与 schema 校验；
- ``update_daily`` 增量补缺口。
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant.data.cache import (
    INSTRUMENTS_FILE,
    fetch_full,
    load_bars,
    load_calendar,
    load_corporate_actions,
    load_instruments,
    update_daily,
)
from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    check_daily_bars,
    check_schema,
)
from quant.data.source.base import DataSource


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


def _bar(
    day: date, instrument: str, close: float = 10.0, adjfactor: float = 1.0
) -> dict[str, object]:
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
        "adjfactor": adjfactor,
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


class FakeSource:
    """实现 ``DataSource`` 协议的内存数据源。"""

    def __init__(
        self,
        bars: dict[str, list[dict[str, object]]] | None = None,
        *,
        info: pl.DataFrame | None = None,
        actions: dict[str, list[dict[str, object]]] | None = None,
        fail: set[str] | None = None,
        crash: set[str] | None = None,
    ) -> None:
        self._bars = bars or {}
        self._info = info if info is not None else pl.DataFrame(schema=INSTRUMENT_INFO)
        self._actions = actions or {}
        self._fail = set(fail or ())
        self._crash = set(crash or ())
        #: 记录每次日线调用 (instrument, start, end)。
        self.bar_calls: list[tuple[str, date, date]] = []

    def daily_bars(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for instrument in instruments:
            if instrument in self._crash:
                raise KeyboardInterrupt(f"模拟进程在 {instrument} 处被打断")
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


def _manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(data_dir: Path, manifest: dict) -> None:
    (data_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def _ca(day: date, instrument: str) -> dict[str, object]:
    return {
        "date": day,
        "instrument": instrument,
        "cash_per_share": 0.5,
        "share_per_share": 0.1,
    }


def test_fake_source_satisfies_protocol() -> None:
    assert isinstance(FakeSource(), DataSource)


# ---------------------------------------------------------------------------
# 全量抓取
# ---------------------------------------------------------------------------


def test_fetch_full_writes_layout_and_manifest(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [_bar(date(2025, 12, 31), "600000.SH"), _bar(date(2026, 1, 2), "600000.SH")],
        "000001.SZ": [_bar(date(2026, 1, 2), "000001.SZ")],
    }
    source = FakeSource(
        bars,
        info=_info("600000.SH", "000001.SZ"),
        actions={"600000.SH": [_ca(date(2026, 1, 2), "600000.SH")]},
    )

    report = fetch_full(source, tmp_path, start=date(2025, 1, 1), end=date(2026, 1, 2))

    assert report.ok == 2
    assert report.failed == 0
    assert report.total == 2
    assert (tmp_path / "calendar.parquet").exists()
    assert (tmp_path / INSTRUMENTS_FILE).exists()
    assert (tmp_path / "corporate_actions.parquet").exists()
    assert (tmp_path / "bars" / "2025.parquet").exists()
    assert (tmp_path / "bars" / "2026.parquet").exists()

    loaded = load_bars(tmp_path)
    check_daily_bars(loaded)
    assert loaded.height == 3
    assert loaded["date"].min() == date(2025, 12, 31)
    assert loaded["date"].max() == date(2026, 1, 2)

    check_schema(load_calendar(tmp_path), TRADE_CALENDAR)
    check_schema(load_instruments(tmp_path), INSTRUMENT_INFO)
    actions = load_corporate_actions(tmp_path)
    check_schema(actions, CORPORATE_ACTIONS)
    assert actions.height == 1

    manifest = _manifest(tmp_path)
    assert manifest["bars"]["600000.SH"]["status"] == "ok"
    assert manifest["bars"]["600000.SH"]["last_date"] == "2026-01-02"
    assert manifest["bars"]["000001.SZ"]["rows"] == 1
    assert manifest["ca"]["600000.SH"]["status"] == "ok"
    assert manifest["updated_at"]


def test_fetch_full_with_explicit_instruments_uses_list(tmp_path: Path) -> None:
    bars = {"600519.SH": [_bar(date(2026, 9, 1), "600519.SH")]}
    source = FakeSource(bars, info=_info("600519.SH", "300750.SZ"))

    report = fetch_full(
        source,
        tmp_path,
        start=date(2026, 9, 1),
        end=date(2026, 9, 1),
        instruments=["600519.SH"],
    )

    assert report.ok == 1
    assert [call[0] for call in source.bar_calls] == ["600519.SH"]
    assert load_bars(tmp_path, instruments=["600519.SH"]).height == 1
    # 调试列表模式下也会补全 instruments.parquet。
    assert (tmp_path / INSTRUMENTS_FILE).exists()


# ---------------------------------------------------------------------------
# 断点续传
# ---------------------------------------------------------------------------


def test_crash_resume_only_fetches_missing(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [_bar(date(2026, 1, 2), "600000.SH")],
        "000001.SZ": [_bar(date(2026, 1, 2), "000001.SZ")],
        "300750.SZ": [_bar(date(2026, 1, 2), "300750.SZ")],
    }
    crash_source = FakeSource(
        bars, info=_info("600000.SH", "000001.SZ", "300750.SZ"), crash={"000001.SZ"}
    )

    with pytest.raises(KeyboardInterrupt):
        fetch_full(
            crash_source,
            tmp_path,
            start=date(2026, 1, 1),
            end=date(2026, 1, 2),
            instruments=["600000.SH", "000001.SZ", "300750.SZ"],
        )

    # 第一只票已落盘并记账；崩溃的那只没有记录。
    manifest = _manifest(tmp_path)
    assert manifest["bars"]["600000.SH"]["status"] == "ok"
    assert "000001.SZ" not in manifest["bars"]

    resume_source = FakeSource(bars, info=_info("600000.SH", "000001.SZ", "300750.SZ"))
    report = fetch_full(
        resume_source,
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
        instruments=["600000.SH", "000001.SZ", "300750.SZ"],
    )

    assert report.ok == 2
    assert report.skipped == 1
    assert [call[0] for call in resume_source.bar_calls] == ["000001.SZ", "300750.SZ"]
    assert load_bars(tmp_path).height == 3


def test_resume_fills_gap_across_years(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2025, 12, 31), "600000.SH"),
            _bar(date(2026, 1, 2), "600000.SH"),
            _bar(date(2026, 1, 5), "600000.SH"),
        ]
    }
    first = FakeSource(bars, info=_info("600000.SH"))
    report1 = fetch_full(
        first, tmp_path, start=date(2025, 12, 30), end=date(2025, 12, 31)
    )
    assert report1.ok == 1
    assert (tmp_path / "bars" / "2025.parquet").exists()

    second = FakeSource(bars, info=_info("600000.SH"))
    report2 = fetch_full(
        second, tmp_path, start=date(2025, 12, 30), end=date(2026, 1, 5)
    )

    assert second.bar_calls == [("600000.SH", date(2026, 1, 1), date(2026, 1, 5))]
    assert report2.ok == 1
    assert (tmp_path / "bars" / "2026.parquet").exists()
    loaded = load_bars(tmp_path)
    check_daily_bars(loaded)
    assert loaded["date"].to_list() == [
        date(2025, 12, 31),
        date(2026, 1, 2),
        date(2026, 1, 5),
    ]
    assert _manifest(tmp_path)["bars"]["600000.SH"]["last_date"] == "2026-01-05"


def test_refetch_same_range_is_idempotent(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2026, 1, 2), "600000.SH"),
            _bar(date(2026, 1, 5), "600000.SH", close=11.0),
        ]
    }
    fetch_full(
        FakeSource(bars, info=_info("600000.SH")),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
    )
    assert load_bars(tmp_path).height == 2

    # 抹掉账本进度强制重抓同一区间，unique(keep=last) 应保证不产生重复行。
    manifest = _manifest(tmp_path)
    manifest["bars"]["600000.SH"] = {
        "last_date": None,
        "rows": 0,
        "status": "failed",
        "error": "手动触发重抓",
    }
    _write_manifest(tmp_path, manifest)

    report = fetch_full(
        FakeSource(bars, info=_info("600000.SH")),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
    )
    assert report.ok == 1
    loaded = load_bars(tmp_path)
    check_daily_bars(loaded)
    assert loaded.height == 2
    assert loaded.filter(pl.col("close") == 11.0).height == 1


# ---------------------------------------------------------------------------
# 失败 / 空数据记账
# ---------------------------------------------------------------------------


def test_failed_ticket_recorded_and_retried(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [_bar(date(2026, 1, 2), "600000.SH")],
        "000001.SZ": [_bar(date(2026, 1, 2), "000001.SZ")],
    }
    info = _info("600000.SH", "000001.SZ")
    report = fetch_full(
        FakeSource(bars, info=info, fail={"000001.SZ"}),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
    )

    assert report.ok == 1
    assert report.failed == 1
    assert "000001.SZ" in report.failures
    manifest = _manifest(tmp_path)
    assert manifest["bars"]["000001.SZ"]["status"] == "failed"
    assert manifest["bars"]["000001.SZ"]["error"]

    retry_source = FakeSource(bars, info=info)
    retry = fetch_full(
        retry_source, tmp_path, start=date(2026, 1, 1), end=date(2026, 1, 2)
    )
    assert retry.ok == 1
    assert retry.skipped == 1
    assert retry_source.bar_calls == [("000001.SZ", date(2026, 1, 1), date(2026, 1, 2))]
    assert _manifest(tmp_path)["bars"]["000001.SZ"]["status"] == "ok"
    assert load_bars(tmp_path).height == 2


def test_empty_ticket_recorded(tmp_path: Path) -> None:
    report = fetch_full(
        FakeSource({}, info=_info("600000.SH")),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
        ca=False,
    )
    assert report.empty == 1
    assert report.ok == 0
    manifest = _manifest(tmp_path)
    assert manifest["bars"]["600000.SH"]["status"] == "empty"
    assert manifest["bars"]["600000.SH"]["last_date"] is None
    assert load_bars(tmp_path).height == 0


def test_fetch_full_creates_empty_corporate_actions_file(tmp_path: Path) -> None:
    report = fetch_full(
        FakeSource({"600000.SH": [_bar(date(2026, 1, 2), "600000.SH")]}, info=_info("600000.SH")),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
    )
    assert report.ca_empty == 1
    assert (tmp_path / "corporate_actions.parquet").exists()
    assert load_corporate_actions(tmp_path).height == 0


def test_corporate_actions_manifest_and_resume(tmp_path: Path) -> None:
    bars = {"600000.SH": [_bar(date(2026, 1, 2), "600000.SH")]}
    actions = {"600000.SH": [_ca(date(2026, 1, 2), "600000.SH")]}
    info = _info("600000.SH")
    fetch_full(
        FakeSource(bars, info=info, actions=actions),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
    )
    assert _manifest(tmp_path)["ca"]["600000.SH"]["status"] == "ok"

    # 再次全量抓取时已成功的公司行为不再重复请求（通过是否新增行间接判断）。
    fetch_full(
        FakeSource(bars, info=info, actions=actions),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
    )
    assert load_corporate_actions(tmp_path).height == 1


# ---------------------------------------------------------------------------
# 读取过滤
# ---------------------------------------------------------------------------


def test_load_bars_filters_by_instrument_and_range(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2025, 12, 31), "600000.SH"),
            _bar(date(2026, 1, 2), "600000.SH"),
        ],
        "000001.SZ": [
            _bar(date(2025, 12, 31), "000001.SZ"),
            _bar(date(2026, 1, 2), "000001.SZ"),
        ],
    }
    fetch_full(
        FakeSource(bars, info=_info("600000.SH", "000001.SZ")),
        tmp_path,
        start=date(2025, 1, 1),
        end=date(2026, 1, 2),
    )

    assert load_bars(tmp_path).height == 4
    assert load_bars(tmp_path, instruments=["600000.SH"]).height == 2
    assert load_bars(tmp_path, start=date(2026, 1, 1)).height == 2
    assert load_bars(tmp_path, end=date(2025, 12, 31)).height == 2
    filtered = load_bars(
        tmp_path,
        instruments=["600000.SH"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
    )
    check_daily_bars(filtered)
    assert filtered.rows() == [
        (
            date(2026, 1, 2),
            "600000.SH",
            10.0,
            11.0,
            9.0,
            10.0,
            10.0,
            1000.0,
            10000.0,
            1.0,
            None,
            None,
        )
    ]


def test_load_bars_missing_dir_returns_empty_schema(tmp_path: Path) -> None:
    out = load_bars(tmp_path / "不存在")
    assert out.height == 0
    check_schema(out, DAILY_BARS)


# ---------------------------------------------------------------------------
# 增量更新
# ---------------------------------------------------------------------------


def test_update_daily_appends_gap(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2026, 1, 2), "600000.SH"),
            _bar(date(2026, 1, 5), "600000.SH", close=11.0),
        ]
    }
    info = _info("600000.SH")
    fetch_full(
        FakeSource(bars, info=info),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
    )

    source = FakeSource(bars, info=info)
    report = update_daily(source, tmp_path, end=date(2026, 1, 5))

    # 增量多带锚点日 1/2 用于重定基（不落盘）。
    assert source.bar_calls == [("600000.SH", date(2026, 1, 2), date(2026, 1, 5))]
    assert report.ok == 1
    loaded = load_bars(tmp_path)
    check_daily_bars(loaded)
    assert loaded["date"].to_list() == [date(2026, 1, 2), date(2026, 1, 5)]
    assert _manifest(tmp_path)["bars"]["600000.SH"]["last_date"] == "2026-01-05"


def test_update_daily_skips_up_to_date(tmp_path: Path) -> None:
    bars = {"600000.SH": [_bar(date(2026, 1, 5), "600000.SH")]}
    info = _info("600000.SH")
    fetch_full(
        FakeSource(bars, info=info),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 5),
    )

    source = FakeSource(bars, info=info)
    report = update_daily(source, tmp_path, end=date(2026, 1, 5))

    assert report.skipped == 1
    assert source.bar_calls == []


# ---------------------------------------------------------------------------
# 增量复权因子重定基（issue #59）
# ---------------------------------------------------------------------------


def _load_adjfactors(data_dir: Path, instrument: str) -> dict[date, float]:
    frame = load_bars(data_dir, instruments=[instrument])
    return {row["date"]: row["adjfactor"] for row in frame.iter_rows(named=True)}


def test_update_daily_rebases_adjfactor_on_anchor(tmp_path: Path) -> None:
    info = _info("600000.SH")
    cached = {"600000.SH": [_bar(date(2026, 9, 28), "600000.SH", adjfactor=192.0)]}
    fetch_full(
        FakeSource(cached, info=info),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 9, 28),
    )
    assert _load_adjfactors(tmp_path, "600000.SH") == {date(2026, 9, 28): 192.0}

    incoming = {
        "600000.SH": [
            _bar(date(2026, 9, 28), "600000.SH", adjfactor=154.0),  # 锚点行
            _bar(date(2026, 9, 29), "600000.SH", adjfactor=155.0),
        ]
    }
    source = FakeSource(incoming, info=info)
    report = update_daily(source, tmp_path, end=date(2026, 9, 29))

    assert report.ok == 1
    assert source.bar_calls == [("600000.SH", date(2026, 9, 28), date(2026, 9, 29))]
    factors = _load_adjfactors(tmp_path, "600000.SH")
    # 锚点日保持缓存基期因子不被覆盖；新行整体重定基。
    assert factors[date(2026, 9, 28)] == 192.0
    assert factors[date(2026, 9, 29)] == pytest.approx(155.0 * 192.0 / 154.0)
    assert len(factors) == 2


def test_update_daily_anchor_missing_row_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="quant.data.cache")
    info = _info("600000.SH")
    fetch_full(
        FakeSource(
            {"600000.SH": [_bar(date(2026, 9, 28), "600000.SH", adjfactor=192.0)]},
            info=info,
        ),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 9, 28),
    )

    incoming = {"600000.SH": [_bar(date(2026, 9, 29), "600000.SH", adjfactor=155.0)]}
    report = update_daily(FakeSource(incoming, info=info), tmp_path, end=date(2026, 9, 29))

    assert report.ok == 1
    factors = _load_adjfactors(tmp_path, "600000.SH")
    # 无锚点行：原样落盘，缓存锚点日不变。
    assert factors[date(2026, 9, 28)] == 192.0
    assert factors[date(2026, 9, 29)] == 155.0
    assert any("缺锚点行" in record.message for record in caplog.records)


def test_update_daily_only_anchor_row_marks_ok(tmp_path: Path) -> None:
    info = _info("600000.SH")
    fetch_full(
        FakeSource(
            {"600000.SH": [_bar(date(2026, 9, 28), "600000.SH", adjfactor=192.0)]},
            info=info,
        ),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 9, 28),
    )

    # 数据源尚无新数据，只返回了锚点行。
    incoming = {"600000.SH": [_bar(date(2026, 9, 28), "600000.SH", adjfactor=154.0)]}
    report = update_daily(FakeSource(incoming, info=info), tmp_path, end=date(2026, 9, 29))

    assert report.ok == 1
    assert report.empty == 0
    assert _load_adjfactors(tmp_path, "600000.SH") == {date(2026, 9, 28): 192.0}
    manifest = _manifest(tmp_path)
    assert manifest["bars"]["600000.SH"]["status"] == "ok"
    assert manifest["bars"]["600000.SH"]["last_date"] == "2026-09-28"


def test_update_daily_anchor_scale_out_of_range_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="quant.data.cache")
    info = _info("600000.SH")
    fetch_full(
        FakeSource(
            {"600000.SH": [_bar(date(2026, 9, 28), "600000.SH", adjfactor=192.0)]},
            info=info,
        ),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 9, 28),
    )

    incoming = {
        "600000.SH": [
            _bar(date(2026, 9, 28), "600000.SH", adjfactor=19_200.0),  # scale=0.01
            _bar(date(2026, 9, 29), "600000.SH", adjfactor=155.0),
        ]
    }
    report = update_daily(FakeSource(incoming, info=info), tmp_path, end=date(2026, 9, 29))

    assert report.ok == 1
    factors = _load_adjfactors(tmp_path, "600000.SH")
    # scale 越界：不缩放，新行原样落盘。
    assert factors[date(2026, 9, 29)] == 155.0
    assert any("scale" in record.message for record in caplog.records)


def test_update_daily_anchor_idempotent(tmp_path: Path) -> None:
    info = _info("600000.SH")
    fetch_full(
        FakeSource(
            {"600000.SH": [_bar(date(2026, 9, 28), "600000.SH", adjfactor=192.0)]},
            info=info,
        ),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 9, 28),
    )
    incoming = {
        "600000.SH": [
            _bar(date(2026, 9, 28), "600000.SH", adjfactor=154.0),
            _bar(date(2026, 9, 29), "600000.SH", adjfactor=155.0),
        ]
    }
    update_daily(FakeSource(incoming, info=info), tmp_path, end=date(2026, 9, 29))
    snapshot = load_bars(tmp_path)

    second_source = FakeSource(incoming, info=info)
    second = update_daily(second_source, tmp_path, end=date(2026, 9, 29))

    assert second.skipped == 1
    assert second_source.bar_calls == []
    assert load_bars(tmp_path).equals(snapshot)


def test_fetch_full_keeps_source_adjfactor(tmp_path: Path) -> None:
    bars = {
        "600000.SH": [
            _bar(date(2026, 9, 28), "600000.SH", adjfactor=154.0),
            _bar(date(2026, 9, 29), "600000.SH", adjfactor=155.0),
        ]
    }
    fetch_full(
        FakeSource(bars, info=_info("600000.SH")),
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 9, 29),
    )
    # 全量首抓不锚定，因子原样落盘。
    assert load_bars(tmp_path)["adjfactor"].to_list() == [154.0, 155.0]
