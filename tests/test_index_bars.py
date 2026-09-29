"""指数日线落地（issue #67）单元测试：缓存读写 + akshare 源归一化。

缓存用例用内存 ``FakeIndexSource`` + ``tmp_path``；源用例 monkeypatch akshare 函数，
不触网（与 ``tests/test_akshare_source.py`` 同风格，测试夹具里用 pandas 还原输入）。
"""
from __future__ import annotations

import datetime as dt
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import quant.data.source.akshare as ak_source
from quant.data.cache import (
    load_index_bars,
    update_index_bars,
)
from quant.data.schema import INDEX_BARS, check_index_bars
from quant.data.source.akshare import AkshareSource

D1 = date(2021, 9, 29)
D2 = date(2021, 9, 30)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source, "REQUEST_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(ak_source, "RETRY_BASE_DELAY_SECONDS", 0.0)


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


class FakeIndexSource:
    """内存指数源，记录 ``index_bars`` 调用区间。"""

    def __init__(self, rows: dict[str, list[dict[str, object]]] | None = None) -> None:
        self._rows = rows or {}
        self.calls: list[tuple[str, date, date]] = []

    def index_bars(
        self, index_codes: list[str], start: date, end: date
    ) -> pl.DataFrame:
        out: list[dict[str, object]] = []
        for code in index_codes:
            self.calls.append((code, start, end))
            out.extend(
                row
                for row in self._rows.get(code, [])
                if start <= row["date"] <= end  # type: ignore[operator]
            )
        if not out:
            return pl.DataFrame(schema=INDEX_BARS)
        return pl.DataFrame(out, schema=INDEX_BARS).sort(["index_code", "date"])


# ---------------------------------------------------------------------------
# 缓存读写 / 幂等
# ---------------------------------------------------------------------------


def test_update_index_bars_writes_and_loads(tmp_path: Path) -> None:
    source = FakeIndexSource(
        {"000905": [_index_row(D1, "000905", 7000.0), _index_row(D2, "000905", 7010.0)]}
    )
    report = update_index_bars(source, tmp_path, end=D2, index_codes=("000905",))

    assert report.ok == 1
    assert report.failed == 0
    assert report.rows == 2
    assert report.last_dates["000905"] == D2.isoformat()

    loaded = load_index_bars(tmp_path)
    check_index_bars(loaded)
    assert loaded.columns == list(INDEX_BARS.keys())
    assert loaded["close"].to_list() == [7000.0, 7010.0]
    # 缓存无记录时从全系统历史起点起抓。
    assert source.calls == [("000905", date(2021, 9, 29), D2)]


def test_update_index_bars_is_incremental_and_idempotent(tmp_path: Path) -> None:
    rows = {
        "000905": [
            _index_row(D1, "000905", 7000.0),
            _index_row(D2, "000905", 7010.0),
            _index_row(date(2021, 10, 8), "000905", 7100.0),
        ]
    }
    first = FakeIndexSource(rows)
    update_index_bars(first, tmp_path, end=D2, index_codes=("000905",))
    assert first.calls == [("000905", D1, D2)]

    # 第二次推进到 10-08：从缓存最新日（含当日，覆盖修订）起抓。
    second = FakeIndexSource(rows)
    report = update_index_bars(second, tmp_path, end=date(2021, 10, 8), index_codes=("000905",))
    assert second.calls == [("000905", D2, date(2021, 10, 8))]
    assert report.ok == 1
    assert load_index_bars(tmp_path).height == 3

    # 第三次同日重跑：短路跳过，不再请求。
    third = FakeIndexSource(rows)
    skipped = update_index_bars(third, tmp_path, end=date(2021, 10, 8), index_codes=("000905",))
    assert third.calls == []
    assert skipped.skipped == 1
    assert skipped.rows == 3


def test_update_index_bars_dedup_keeps_last(tmp_path: Path) -> None:
    update_index_bars(
        FakeIndexSource({"000300": [_index_row(D1, "000300", 4000.0)]}),
        tmp_path,
        end=D1,
        index_codes=("000300",),
    )
    # 修订后的同日收盘应覆盖旧值。
    update_index_bars(
        FakeIndexSource({"000300": [_index_row(D1, "000300", 4001.0), _index_row(D2, "000300", 4002.0)]}),
        tmp_path,
        end=D2,
        index_codes=("000300",),
    )
    loaded = load_index_bars(tmp_path)
    assert loaded.height == 2
    assert loaded.filter(pl.col("date") == D1)["close"].to_list() == [4001.0]


def test_update_index_bars_failure_recorded(tmp_path: Path) -> None:
    class _Boom(FakeIndexSource):
        def index_bars(self, index_codes: list[str], start: date, end: date) -> pl.DataFrame:
            raise RuntimeError("网络不可用")

    report = update_index_bars(_Boom(), tmp_path, end=D2, index_codes=("000300", "000905"))
    assert report.failed == 2
    assert set(report.failures) == {"000300", "000905"}
    assert load_index_bars(tmp_path).height == 0


def test_load_index_bars_missing_file_returns_empty(tmp_path: Path) -> None:
    out = load_index_bars(tmp_path)
    assert out.height == 0
    check_index_bars(out)


def test_load_index_bars_filters(tmp_path: Path) -> None:
    rows = {
        "000300": [_index_row(D1, "000300", 4000.0), _index_row(D2, "000300", 4010.0)],
        "000905": [_index_row(D1, "000905", 7000.0)],
    }
    update_index_bars(
        FakeIndexSource(rows), tmp_path, end=D2, index_codes=("000300", "000905")
    )
    only = load_index_bars(tmp_path, index_codes=["000300"], start=D2)
    assert only.height == 1
    assert only["close"].to_list() == [4010.0]


# ---------------------------------------------------------------------------
# akshare 源归一化
# ---------------------------------------------------------------------------


def _pandas(df: pl.DataFrame) -> object:
    return pd.DataFrame(df.to_dict(as_series=False))


def _sina_raw() -> object:
    return _pandas(
        pl.DataFrame(
            {
                "date": [D1, D2],
                "open": [4000.0, 4010.0],
                "high": [4050.0, 4060.0],
                "low": [3950.0, 3960.0],
                "close": [4040.0, 4050.0],
                "volume": [1.0e10, 1.1e10],
            }
        )
    )


def _csindex_raw() -> object:
    return _pandas(
        pl.DataFrame(
            {
                "日期": [D1, D2],
                "指数代码": ["932000", "932000"],
                "开盘": [3000.0, 3010.0],
                "最高": [3050.0, 3060.0],
                "最低": [2950.0, 2960.0],
                "收盘": [3040.0, 3050.0],
                "成交量": [8.0e9, 8.1e9],
            }
        )
    )


def test_index_bars_sina_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ak_source.ak, "stock_zh_index_daily", lambda **kw: _sina_raw())
    out = AkshareSource().index_bars(["000300"], D1, D2)
    check_index_bars(out)
    assert out["index_code"].to_list() == ["000300", "000300"]
    assert out["close"].to_list() == [4040.0, 4050.0]
    assert out["volume"].to_list() == [1.0e10, 1.1e10]


def test_index_bars_932000_uses_csindex(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"sina": 0}

    def _sina(*_a: object, **_k: object) -> object:
        called["sina"] += 1
        raise AssertionError("932000 不应走新浪")

    monkeypatch.setattr(ak_source.ak, "stock_zh_index_daily", _sina)
    monkeypatch.setattr(ak_source.ak, "stock_zh_index_hist_csindex", lambda **kw: _csindex_raw())
    out = AkshareSource().index_bars(["932000"], D1, D2)
    check_index_bars(out)
    assert called["sina"] == 0
    assert out["close"].to_list() == [3040.0, 3050.0]


def test_index_bars_falls_back_to_csindex(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("新浪不可用")

    monkeypatch.setattr(ak_source.ak, "stock_zh_index_daily", _boom)
    monkeypatch.setattr(ak_source.ak, "stock_zh_index_hist_csindex", lambda **kw: _csindex_raw())
    out = AkshareSource().index_bars(["000300"], D1, D2)
    assert out.height == 2
    assert out["index_code"].to_list() == ["000300", "000300"]


def test_index_bars_all_sources_fail_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("全挂")

    monkeypatch.setattr(ak_source.ak, "stock_zh_index_daily", _boom)
    monkeypatch.setattr(ak_source.ak, "stock_zh_index_hist_csindex", _boom)
    out = AkshareSource().index_bars(["000300"], D1, D2)
    assert out.height == 0
    check_index_bars(out)
