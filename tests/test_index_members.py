"""``quant.data.index_members`` 与指数相关 validate 检查的单元测试。

全部用例用合成数据，不触网。覆盖：

- 定期调样生效日的日历推算；
- 成分快照按生效日展开到日频（含指数发布前无行）；
- 权重漂移（后复权收益复利、每日归一、锚日取官方、停牌收益为 0）；
- 漂移-官方对拍误差；
- ``build_daily_tables`` 端到端落盘；
- ``update_index_anchors`` 幂等与变更落盘；
- CSMAR 变更表解析（过滤 / 去重 / 归一化）；
- 逆放重建历史成分与正放不变式；
- 等效市值权重锚（含停牌票剔除）与官方锚优先合并；
- validate 的成分/权重/漂移检查触发与不触发。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant.data.index_history import (
    CHANGE_ADD,
    CHANGE_DROP,
    build_equivalent_anchors,
    check_reconstruction_invariant,
    compare_equivalent_to_official,
    market_cap_weights_on,
    read_csmar_changes,
    reconstruct_snapshots,
)
from quant.data.index_members import (
    build_daily_tables,
    drift_weights,
    expand_member_snapshots,
    merge_weight_anchors,
    read_anchor_weights,
    read_drift_check,
    read_index_members,
    read_index_weights,
    read_snapshots,
    rebalance_effective_dates,
    run_index_update,
    second_friday,
    update_index_anchors,
)
from quant.data.schema import (
    DAILY_BARS,
    INDEX_DRIFT_CHECK,
    INDEX_MEMBERS,
    INDEX_MEMBER_CHANGES,
    INDEX_MEMBER_SNAPSHOTS,
    INDEX_WEIGHTS,
    INDUSTRY,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
)
from quant.data.validate import validate

# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------

DAYS = [date(2024, 6, 1) + timedelta(days=i) for i in range(1, 8)]
# 2024-06-02 起：6/3 周一 ~ 6/7 周五
OPEN = [day for day in DAYS if day.weekday() < 5]


def _open_frame(days: list[date] | None = None) -> pl.DataFrame:
    days = days or OPEN
    return pl.DataFrame({"date": days}).cast({"date": pl.Date})


def _bar(
    day: date, instrument: str, close: float = 10.0, adjfactor: float = 1.0
) -> dict[str, object]:
    return {
        "date": day,
        "instrument": instrument,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "vwap": close,
        "volume": 1000.0,
        "amount": close * 1000.0,
        "adjfactor": adjfactor,
        "limit_up": None,
        "limit_down": None,
    }


def _snap(day: date, instrument: str, index_code: str = "000300") -> dict[str, object]:
    return {"snapshot_date": day, "instrument": instrument, "index_code": index_code}


def _member(day: date, instrument: str, index_code: str = "000300") -> dict[str, object]:
    return {"date": day, "instrument": instrument, "index_code": index_code}


def _anchor(
    day: date, instrument: str, weight: float, index_code: str = "000300"
) -> dict[str, object]:
    return {
        "date": day,
        "instrument": instrument,
        "index_code": index_code,
        "weight": weight,
    }


# ---------------------------------------------------------------------------
# 调样日历
# ---------------------------------------------------------------------------


def test_second_friday() -> None:
    assert second_friday(2024, 6) == date(2024, 6, 14)
    assert second_friday(2024, 12) == date(2024, 12, 13)
    assert second_friday(2021, 6) == date(2021, 6, 11)


def test_rebalance_effective_dates_skips_to_next_open() -> None:
    # 2024-06-14（周五）收市后生效 -> 下一交易日 2024-06-17（周一）。
    opens = [date(2024, 6, 13), date(2024, 6, 14), date(2024, 6, 17), date(2024, 6, 18)]
    assert rebalance_effective_dates(date(2024, 6, 1), date(2024, 6, 30), opens) == [
        date(2024, 6, 17)
    ]


def test_rebalance_effective_dates_both_half_years() -> None:
    opens = [
        date(2024, 6, 17),
        date(2024, 12, 16),
    ]
    assert rebalance_effective_dates(date(2024, 1, 1), date(2024, 12, 31), opens) == [
        date(2024, 6, 17),
        date(2024, 12, 16),
    ]


# ---------------------------------------------------------------------------
# 成分展开
# ---------------------------------------------------------------------------


def test_expand_member_snapshots_forward_fills() -> None:
    snapshots = pl.DataFrame(
        [
            _snap(date(2024, 6, 3), "600000.SH"),
            _snap(date(2024, 6, 3), "000001.SZ"),
            _snap(date(2024, 6, 5), "600000.SH"),
        ],
        schema=INDEX_MEMBER_SNAPSHOTS,
    )
    out = expand_member_snapshots(snapshots, _open_frame())
    got = {
        (row["date"], row["instrument"]) for row in out.iter_rows(named=True)
    }
    assert (date(2024, 6, 3), "000001.SZ") in got
    assert (date(2024, 6, 4), "000001.SZ") in got  # 6/3 名单延续
    assert (date(2024, 6, 5), "000001.SZ") not in got  # 6/5 起剔除
    assert (date(2024, 6, 7), "600000.SH") in got
    assert out.height == 2 + 2 + 1 + 1 + 1


def test_expand_member_snapshots_skips_before_first_snapshot() -> None:
    snapshots = pl.DataFrame(
        [_snap(date(2024, 6, 5), "600000.SH")], schema=INDEX_MEMBER_SNAPSHOTS
    )
    out = expand_member_snapshots(snapshots, _open_frame())
    assert out["date"].min() == date(2024, 6, 5)
    assert out.height == 3  # 6/5、6/6、6/7


# ---------------------------------------------------------------------------
# 权重漂移
# ---------------------------------------------------------------------------


def _drift_fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    anchors = pl.DataFrame(
        [
            _anchor(date(2024, 6, 3), "600000.SH", 0.5),
            _anchor(date(2024, 6, 3), "000001.SZ", 0.5),
            _anchor(date(2024, 6, 6), "600000.SH", 0.7),
            _anchor(date(2024, 6, 6), "000001.SZ", 0.3),
        ],
        schema=INDEX_WEIGHTS,
    )
    bars = pl.DataFrame(
        [
            _bar(date(2024, 6, 3), "600000.SH", close=10.0),
            _bar(date(2024, 6, 4), "600000.SH", close=11.0),
            _bar(date(2024, 6, 5), "600000.SH", close=11.0),
            _bar(date(2024, 6, 6), "600000.SH", close=12.0),
            _bar(date(2024, 6, 7), "600000.SH", close=12.0),
            _bar(date(2024, 6, 3), "000001.SZ", close=10.0),
            _bar(date(2024, 6, 4), "000001.SZ", close=10.0),
            _bar(date(2024, 6, 5), "000001.SZ", close=10.0),
            _bar(date(2024, 6, 6), "000001.SZ", close=10.0),
            _bar(date(2024, 6, 7), "000001.SZ", close=10.0),
        ],
        schema=DAILY_BARS,
    )
    return anchors, bars, _open_frame()


def test_drift_weights_compounds_and_normalizes() -> None:
    anchors, bars, opens = _drift_fixture()
    weights, checks = drift_weights(anchors, bars, opens)
    table = {
        (row["date"], row["instrument"]): row["weight"]
        for row in weights.iter_rows(named=True)
    }
    # 锚日：官方权重原样。
    assert table[(date(2024, 6, 3), "600000.SH")] == pytest.approx(0.5)
    assert table[(date(2024, 6, 6), "600000.SH")] == pytest.approx(0.7)
    # 6/4：A 涨 10%，B 不动；归一后 A=0.55/1.05。
    assert table[(date(2024, 6, 4), "600000.SH")] == pytest.approx(0.55 / 1.05)
    assert table[(date(2024, 6, 4), "000001.SZ")] == pytest.approx(0.50 / 1.05)
    # 每日权重和恒为 1。
    sums = weights.group_by("date").agg(pl.col("weight").sum().alias("s"))
    assert all(abs(value - 1.0) < 1e-12 for value in sums["s"].to_list())
    # 对拍：6/6 官方锚与漂移值比较。
    assert checks.height == 2
    check = {
        row["instrument"]: row for row in checks.iter_rows(named=True)
    }
    assert check["600000.SH"]["official_weight"] == pytest.approx(0.7)
    assert check["600000.SH"]["abs_deviation"] == pytest.approx(
        abs(check["600000.SH"]["drift_weight"] - 0.7)
    )


def test_drift_weights_suspension_is_zero_return() -> None:
    anchors = pl.DataFrame(
        [
            _anchor(date(2024, 6, 3), "600000.SH", 0.6),
            _anchor(date(2024, 6, 3), "000001.SZ", 0.4),
        ],
        schema=INDEX_WEIGHTS,
    )
    # B 在 6/4 停牌（无行），权重应保持不变，A 涨 10% 后归一。
    bars = pl.DataFrame(
        [
            _bar(date(2024, 6, 3), "600000.SH", close=10.0),
            _bar(date(2024, 6, 4), "600000.SH", close=11.0),
            _bar(date(2024, 6, 3), "000001.SZ", close=10.0),
        ],
        schema=DAILY_BARS,
    )
    weights, _ = drift_weights(anchors, bars, _open_frame())
    table = {
        (row["date"], row["instrument"]): row["weight"]
        for row in weights.iter_rows(named=True)
    }
    # B 停牌当日仍有权重，且等于前一日。
    assert table[(date(2024, 6, 4), "000001.SZ")] == pytest.approx(0.4 / 1.06)
    assert table[(date(2024, 6, 4), "600000.SH")] == pytest.approx(0.66 / 1.06)


def test_drift_weights_no_lookahead_from_future_anchor() -> None:
    """第一个锚区间的漂移不得使用后续锚的成分名单。"""
    anchors = pl.DataFrame(
        [
            _anchor(date(2024, 6, 3), "600000.SH", 0.5),
            _anchor(date(2024, 6, 3), "000001.SZ", 0.5),
            # 6/6 起 C 才进入指数。
            _anchor(date(2024, 6, 6), "600000.SH", 0.4),
            _anchor(date(2024, 6, 6), "000001.SZ", 0.3),
            _anchor(date(2024, 6, 6), "300750.SZ", 0.3),
        ],
        schema=INDEX_WEIGHTS,
    )
    bars = pl.DataFrame(
        [
            _bar(date(2024, 6, 3), "600000.SH", close=10.0),
            _bar(date(2024, 6, 4), "600000.SH", close=10.0),
            _bar(date(2024, 6, 5), "600000.SH", close=10.0),
            _bar(date(2024, 6, 3), "000001.SZ", close=10.0),
            _bar(date(2024, 6, 4), "000001.SZ", close=10.0),
            _bar(date(2024, 6, 5), "000001.SZ", close=10.0),
            _bar(date(2024, 6, 6), "300750.SZ", close=10.0),
        ],
        schema=DAILY_BARS,
    )
    weights, checks = drift_weights(anchors, bars, _open_frame())
    before = weights.filter(pl.col("date") < date(2024, 6, 6))
    assert "300750.SZ" not in set(before["instrument"].to_list())
    # 对拍只比较两段共有的成分，不含 6/6 新进的 C。
    assert "300750.SZ" not in set(checks["instrument"].to_list())


# ---------------------------------------------------------------------------
# 落盘与增量
# ---------------------------------------------------------------------------


def _write_base_cache(data_dir: Path, days: list[date]) -> None:
    bars_dir = data_dir / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        _bar(day, instrument, close=10.0 + offset)
        for instrument in ("600000.SH", "000001.SZ", "300750.SZ")
        for offset, day in enumerate(days)
    ]
    frame = pl.DataFrame(frames, schema=DAILY_BARS)
    for year in sorted(frame["date"].dt.year().unique().to_list()):
        frame.filter(pl.col("date").dt.year() == year).write_parquet(
            bars_dir / f"{year}.parquet"
        )
    pl.DataFrame({"date": days, "is_open": [True] * len(days)}).cast(
        TRADE_CALENDAR
    ).write_parquet(data_dir / "calendar.parquet")
    pl.DataFrame(
        [
            {
                "instrument": instrument,
                "name": instrument,
                "board": "main",
                "list_date": date(2000, 1, 1),
                "delist_date": None,
            }
            for instrument in ("600000.SH", "000001.SZ", "300750.SZ")
        ],
        schema=INSTRUMENT_INFO,
    ).write_parquet(data_dir / "instruments.parquet")
    # industry.parquet 是 validate 的必需表（issue #65），合成缓存补一份全归属快照。
    pl.DataFrame(
        [
            {
                "instrument": instrument,
                "industry_l1": "测试行业",
                "industry_l2": "测试行业",
                "effective_from": days[0],
            }
            for instrument in ("600000.SH", "000001.SZ", "300750.SZ")
        ],
        schema=INDUSTRY,
    ).write_parquet(data_dir / "industry.parquet")


def test_build_daily_tables_end_to_end(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    pl.DataFrame(
        [
            _snap(date(2024, 6, 3), "600000.SH"),
            _snap(date(2024, 6, 3), "000001.SZ"),
            _snap(date(2024, 6, 5), "600000.SH"),
        ],
        schema=INDEX_MEMBER_SNAPSHOTS,
    ).write_parquet(tmp_path / "index_member_snapshots.parquet")
    pl.DataFrame(
        [
            _anchor(date(2024, 6, 3), "600000.SH", 0.5),
            _anchor(date(2024, 6, 3), "000001.SZ", 0.5),
            _anchor(date(2024, 6, 6), "600000.SH", 0.6),
        ],
        schema=INDEX_WEIGHTS,
    ).write_parquet(tmp_path / "index_weight_anchors.parquet")

    counts = build_daily_tables(tmp_path)
    assert counts["members"] > 0
    assert counts["weights"] > 0

    members = read_index_members(tmp_path)
    assert set(members["instrument"].unique().to_list()) == {"600000.SH", "000001.SZ"}
    weights = read_anchor_weights(tmp_path)
    assert weights.height == 3
    derived = pl.read_parquet(tmp_path / "index_weights.parquet")
    assert set(derived["date"].unique().to_list()) == set(OPEN)
    derived_via_reader = read_index_weights(tmp_path)
    assert derived_via_reader.height == derived.height
    assert derived_via_reader.columns == list(INDEX_WEIGHTS.keys())
    assert read_drift_check(tmp_path).height == 1


def test_read_index_weights_missing_file_returns_empty(tmp_path: Path) -> None:
    out = read_index_weights(tmp_path)
    assert out.height == 0
    assert out.columns == list(INDEX_WEIGHTS.keys())


class _FakeAnchorSource:
    def __init__(self, snapshot: pl.DataFrame, anchor: pl.DataFrame) -> None:
        self._snapshot = snapshot
        self._anchor = anchor
        self.calls = 0

    def member_snapshot(self, index_code: str) -> pl.DataFrame:
        self.calls += 1
        return self._snapshot

    def weight_anchor(self, index_code: str) -> pl.DataFrame:
        return self._anchor


def test_update_index_anchors_idempotent(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    snapshot = pl.DataFrame(
        [_snap(date(2024, 6, 3), "600000.SH"), _snap(date(2024, 6, 3), "000001.SZ")],
        schema=INDEX_MEMBER_SNAPSHOTS,
    )
    anchor = pl.DataFrame(
        [_anchor(date(2024, 6, 3), "600000.SH", 0.5), _anchor(date(2024, 6, 3), "000001.SZ", 0.5)],
        schema=INDEX_WEIGHTS,
    )
    source = _FakeAnchorSource(snapshot, anchor)
    first = update_index_anchors(source, tmp_path, index_codes=("000300",))
    assert first["updated"] == 1
    assert read_snapshots(tmp_path).height == 2

    second = update_index_anchors(source, tmp_path, index_codes=("000300",))
    assert second["updated"] == 0  # 成员未变，不重复落盘
    assert read_snapshots(tmp_path).height == 2

    # 成员变化时追加新快照。
    changed = pl.DataFrame(
        [_snap(date(2024, 6, 5), "600000.SH")], schema=INDEX_MEMBER_SNAPSHOTS
    )
    third = update_index_anchors(
        _FakeAnchorSource(changed, anchor), tmp_path, index_codes=("000300",)
    )
    assert third["updated"] == 1
    assert read_snapshots(tmp_path).height == 3


def test_run_index_update_counts(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    snapshot = pl.DataFrame(
        [_snap(date(2024, 6, 3), "600000.SH"), _snap(date(2024, 6, 3), "000001.SZ")],
        schema=INDEX_MEMBER_SNAPSHOTS,
    )
    anchor = pl.DataFrame(
        [_anchor(date(2024, 6, 3), "600000.SH", 0.5), _anchor(date(2024, 6, 3), "000001.SZ", 0.5)],
        schema=INDEX_WEIGHTS,
    )
    counts = run_index_update(
        _FakeAnchorSource(snapshot, anchor), tmp_path, index_codes=("000300",)
    )
    assert counts["updated"] == 1
    assert counts["weights"] > 0


# ---------------------------------------------------------------------------
# validate 扩展
# ---------------------------------------------------------------------------


def _weights_frame(day: date, *pairs: tuple[str, float]) -> list[dict[str, object]]:
    return [_anchor(day, instrument, weight) for instrument, weight in pairs]


def test_validate_index_tables_all_green(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    pl.DataFrame(
        [
            _snap(date(2024, 6, 3), "600000.SH"),
            _snap(date(2024, 6, 3), "000001.SZ"),
        ],
        schema=INDEX_MEMBER_SNAPSHOTS,
    ).write_parquet(tmp_path / "index_member_snapshots.parquet")
    pl.DataFrame(
        _weights_frame(date(2024, 6, 3), ("600000.SH", 0.5), ("000001.SZ", 0.5)), schema=INDEX_WEIGHTS
    ).write_parquet(tmp_path / "index_weight_anchors.parquet")
    build_daily_tables(tmp_path)

    report = validate(tmp_path)
    assert report.ok
    assert "index_weight_sum_out_of_range" not in {issue.check for issue in report.issues}
    assert "index_member_change_off_schedule" not in {
        issue.check for issue in report.issues
    }


def test_validate_index_weight_sum_out_of_range(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    pl.DataFrame(
        _weights_frame(date(2024, 6, 3), ("600000.SH", 0.5), ("000001.SZ", 0.2)), schema=INDEX_WEIGHTS
    ).write_parquet(tmp_path / "index_weights.parquet")
    report = validate(tmp_path)
    assert not report.ok
    assert "index_weight_sum_out_of_range" in {issue.check for issue in report.issues}


def test_validate_index_weight_duplicate_key(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    rows = _weights_frame(date(2024, 6, 3), ("600000.SH", 0.5), ("600000.SH", 0.5))
    pl.DataFrame(rows, schema=INDEX_WEIGHTS).write_parquet(
        tmp_path / "index_weights.parquet"
    )
    report = validate(tmp_path)
    assert not report.ok
    assert "index_weights_duplicate_key" in {issue.check for issue in report.issues}


def test_validate_index_member_hole_is_warning(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    # 6/4 漏掉 B 的行，构成区间空档。
    rows = [
        _member(date(2024, 6, 3), "600000.SH"),
        _member(date(2024, 6, 3), "000001.SZ"),
        _member(date(2024, 6, 4), "600000.SH"),
        _member(date(2024, 6, 5), "600000.SH"),
        _member(date(2024, 6, 5), "000001.SZ"),
        _member(date(2024, 6, 6), "600000.SH"),
        _member(date(2024, 6, 7), "600000.SH"),
    ]
    pl.DataFrame(rows, schema=INDEX_MEMBERS).write_parquet(
        tmp_path / "index_members.parquet"
    )
    report = validate(tmp_path)
    assert report.ok
    assert "index_member_hole" in {issue.check for issue in report.issues}


def test_validate_off_schedule_member_change_is_warning(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    rows = [
        _member(date(2024, 6, 3), "600000.SH"),
        _member(date(2024, 6, 4), "600000.SH"),  # 6/4 非调样日也变更
        _member(date(2024, 6, 4), "000001.SZ"),
    ]
    pl.DataFrame(rows, schema=INDEX_MEMBERS).write_parquet(
        tmp_path / "index_members.parquet"
    )
    report = validate(tmp_path)
    assert report.ok
    assert "index_member_change_off_schedule" in {
        issue.check for issue in report.issues
    }


def test_validate_drift_deviation_warns(tmp_path: Path) -> None:
    _write_base_cache(tmp_path, OPEN)
    drift = pl.DataFrame(
        [
            {
                "anchor_date": date(2024, 6, 6),
                "index_code": "000300",
                "instrument": "600000.SH",
                "drift_weight": 0.2,
                "official_weight": 0.7,
                "abs_deviation": 0.5,
            }
        ],
        schema=INDEX_DRIFT_CHECK,
    )
    drift.write_parquet(tmp_path / "index_weight_drift.parquet")
    report = validate(tmp_path)
    assert report.ok
    assert "index_weight_drift_deviation" in {issue.check for issue in report.issues}


# ---------------------------------------------------------------------------
# CSMAR 变更表解析
# ---------------------------------------------------------------------------

_CSMAR_HEADER = (
    "Indexcd,Chgsmp01,Chgsmp02,Chgsmp03,Chgsmp04,Chgsmp05,Chgsmp06,Chgsmp07"
)


def _write_csmar(tmp_path: Path, name: str, rows: list[tuple[str, ...]]) -> Path:
    path = tmp_path / name
    body = "\n".join(",".join(row) for row in rows)
    path.write_text(f"{_CSMAR_HEADER}\n{body}\n", encoding="utf-8")
    return path


def test_read_csmar_changes_filters_and_dedups(tmp_path: Path) -> None:
    path = _write_csmar(
        tmp_path,
        "IDX_Chgsmp.csv",
        [
            # 目标指数 + 股票类 + 前导零代码。
            ("000300", "2024-06-17", "000001", "平安银行", "1", "1", "2024-06-07", "SZSE"),
            ("000300", "2024-06-17", "600000", "浦发银行", "2", "1", "2024-06-07", "SSE"),
            # 完全重复的一行，应被去重。
            ("000300", "2024-06-17", "000001", "平安银行", "1", "1", "2024-06-07", "SZSE"),
            # 基金类（Chgsmp05=2）应过滤。
            ("000300", "2024-06-17", "510300", "沪深300ETF", "1", "2", "2024-06-07", "SSE"),
            # 非目标指数应过滤。
            ("000905", "2024-06-17", "600519", "贵州茅台", "1", "1", "2024-06-07", "SSE"),
        ],
    )
    out = read_csmar_changes([path], index_codes=("000300",))
    assert out.height == 2
    assert set(out["instrument"].to_list()) == {"000001.SZ", "600000.SH"}
    assert set(out["change_type"].to_list()) == {CHANGE_ADD, CHANGE_DROP}
    assert out["effective_date"].to_list() == [date(2024, 6, 17)] * 2
    assert out.schema == pl.Schema(INDEX_MEMBER_CHANGES)


def test_read_csmar_changes_empty_when_no_files() -> None:
    out = read_csmar_changes([])
    assert out.height == 0
    assert out.schema == pl.Schema(INDEX_MEMBER_CHANGES)


# ---------------------------------------------------------------------------
# 逆放重建与正放不变式
# ---------------------------------------------------------------------------


def _changes_frame(rows: list[tuple[str, date, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "index_code": code,
                "effective_date": day,
                "instrument": instrument,
                "change_type": mode,
            }
            for code, day, instrument, mode in rows
        ],
        schema=INDEX_MEMBER_CHANGES,
    )


def _base_frame(code: str, day: date, instruments: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"snapshot_date": day, "instrument": instrument, "index_code": code}
            for instrument in instruments
        ],
        schema=INDEX_MEMBER_SNAPSHOTS,
    )


def test_reconstruct_snapshots_reverse_apply() -> None:
    # S0={000001.SZ,000002.SZ,000003.SZ} -> 6/13 换入 000004.SZ 换出 000001.SZ
    # -> S1={000002.SZ,000003.SZ,000004.SZ} -> 6/17 换入 000005.SZ 换出 000002.SZ
    # -> S2={000003.SZ,000004.SZ,000005.SZ}
    base = _base_frame(
        "000300",
        date(2024, 6, 20),
        ["000003.SZ", "000004.SZ", "000005.SZ"],
    )
    changes = _changes_frame(
        [
            ("000300", date(2024, 6, 13), "000004.SZ", CHANGE_ADD),
            ("000300", date(2024, 6, 13), "000001.SZ", CHANGE_DROP),
            ("000300", date(2024, 6, 17), "000005.SZ", CHANGE_ADD),
            ("000300", date(2024, 6, 17), "000002.SZ", CHANGE_DROP),
        ]
    )
    out = reconstruct_snapshots(changes, base, coverage_start=None)
    by_date = {
        day: set(
            out.filter(pl.col("snapshot_date") == day)["instrument"].to_list()
        )
        for day in (date(2024, 6, 13), date(2024, 6, 17))
    }
    assert by_date[date(2024, 6, 17)] == {"000003.SZ", "000004.SZ", "000005.SZ"}
    assert by_date[date(2024, 6, 13)] == {"000002.SZ", "000003.SZ", "000004.SZ"}
    # 正放回推与逆放路径一致。
    check_reconstruction_invariant(changes, out)


def test_reconstruct_snapshots_coverage_start_keeps_latest_pre_window() -> None:
    base = _base_frame(
        "000300",
        date(2024, 6, 20),
        ["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    changes = _changes_frame(
        [
            ("000300", date(2024, 6, 3), "000001.SZ", CHANGE_ADD),
            ("000300", date(2024, 6, 13), "000002.SZ", CHANGE_ADD),
            ("000300", date(2024, 6, 17), "000003.SZ", CHANGE_ADD),
        ]
    )
    out = reconstruct_snapshots(changes, base, coverage_start=date(2024, 6, 14))
    # 只保留「最近一次 <= 6/14 的变更日」（6/13）及其之后。
    assert set(out["snapshot_date"].unique().to_list()) == {
        date(2024, 6, 13),
        date(2024, 6, 17),
    }


def test_check_reconstruction_invariant_raises_on_tamper() -> None:
    # 与逆放用例同构：S0={1,2,3} -> 6/13 换入 4 换出 1 -> 6/17 换入 5 换出 2。
    base = _base_frame(
        "000300",
        date(2024, 6, 20),
        ["000003.SZ", "000004.SZ", "000005.SZ"],
    )
    changes = _changes_frame(
        [
            ("000300", date(2024, 6, 13), "000004.SZ", CHANGE_ADD),
            ("000300", date(2024, 6, 13), "000001.SZ", CHANGE_DROP),
            ("000300", date(2024, 6, 17), "000005.SZ", CHANGE_ADD),
            ("000300", date(2024, 6, 17), "000002.SZ", CHANGE_DROP),
        ]
    )
    out = reconstruct_snapshots(changes, base, coverage_start=None)
    assert out.filter(pl.col("snapshot_date") == date(2024, 6, 13)).height == 3
    check_reconstruction_invariant(changes, out)  # 原样应通过
    # 篡改 6/13 快照（漏掉 6/13 换入、6/17 仍在的 000004.SZ）后不变式应失败。
    tampered = out.filter(
        ~(
            (pl.col("snapshot_date") == date(2024, 6, 13))
            & (pl.col("instrument") == "000004.SZ")
        )
    )
    with pytest.raises(ValueError, match="不变式失败"):
        check_reconstruction_invariant(changes, tampered)


# ---------------------------------------------------------------------------
# 等效市值权重锚
# ---------------------------------------------------------------------------


def test_market_cap_weights_drops_suspended() -> None:
    members = pl.DataFrame({"instrument": ["600000.SH", "000001.SZ", "300750.SZ"]})
    bars = pl.DataFrame(
        [
            _bar(date(2024, 6, 17), "600000.SH", close=10.0, adjfactor=1.0),
            _bar(date(2024, 6, 17), "000001.SZ", close=20.0, adjfactor=2.0),
            # 300750.SZ 当日报价缺失（停牌），应剔除。
        ],
        schema=DAILY_BARS,
    )
    weights, dropped = market_cap_weights_on(members, bars, date(2024, 6, 17))
    table = {row["instrument"]: row["weight"] for row in weights.iter_rows(named=True)}
    assert table["600000.SH"] == pytest.approx(10.0 / 50.0)
    assert table["000001.SZ"] == pytest.approx(40.0 / 50.0)
    assert dropped["instrument"].to_list() == ["300750.SZ"]


def test_build_equivalent_anchors_clamps_to_open_day() -> None:
    # 快照日 6/15 非开市日，锚日顺延到 6/17。
    snapshots = _base_frame("000300", date(2024, 6, 15), ["600000.SH", "000001.SZ"])
    bars = pl.DataFrame(
        [
            _bar(date(2024, 6, 17), "600000.SH", close=10.0, adjfactor=1.0),
            _bar(date(2024, 6, 17), "000001.SZ", close=30.0, adjfactor=1.0),
        ],
        schema=DAILY_BARS,
    )
    anchors, dropped = build_equivalent_anchors(
        snapshots, bars, [date(2024, 6, 14), date(2024, 6, 17)], index_codes=("000300",)
    )
    assert anchors["date"].unique().to_list() == [date(2024, 6, 17)]
    assert anchors["weight"].sum() == pytest.approx(1.0)
    assert dropped.height == 0


def test_compare_equivalent_to_official_reports_deviation() -> None:
    official = pl.DataFrame(
        [
            _anchor(date(2024, 6, 17), "600000.SH", 0.5),
            _anchor(date(2024, 6, 17), "000001.SZ", 0.5),
        ],
        schema=INDEX_WEIGHTS,
    )
    bars = pl.DataFrame(
        [
            _bar(date(2024, 6, 17), "600000.SH", close=10.0, adjfactor=1.0),
            _bar(date(2024, 6, 17), "000001.SZ", close=30.0, adjfactor=1.0),
        ],
        schema=DAILY_BARS,
    )
    out = compare_equivalent_to_official(official, bars)
    assert out.height == 2
    row = out.filter(pl.col("instrument") == "000001.SZ").row(0, named=True)
    assert row["equivalent_weight"] == pytest.approx(0.75)
    assert row["official_weight"] == pytest.approx(0.5)
    assert row["abs_deviation"] == pytest.approx(0.25)


def test_merge_weight_anchors_official_priority() -> None:
    # 锚按 (index_code, date) 整份覆盖：同一日的官方锚整体替换近似锚。
    approx = pl.DataFrame(
        [
            _anchor(date(2024, 6, 13), "600000.SH", 0.5),
            _anchor(date(2024, 6, 17), "600000.SH", 0.5),
            _anchor(date(2024, 6, 17), "000001.SZ", 0.5),
        ],
        schema=INDEX_WEIGHTS,
    )
    official = pl.DataFrame(
        [
            _anchor(date(2024, 6, 17), "600000.SH", 0.7),
            _anchor(date(2024, 6, 17), "000001.SZ", 0.3),
        ],
        schema=INDEX_WEIGHTS,
    )
    merged = merge_weight_anchors(approx, official)
    day17 = {
        row["instrument"]: row["weight"]
        for row in merged.filter(pl.col("date") == date(2024, 6, 17)).iter_rows(named=True)
    }
    assert day17 == {"600000.SH": pytest.approx(0.7), "000001.SZ": pytest.approx(0.3)}
    # 无官方锚的日期保留近似值。
    assert merged.filter(pl.col("date") == date(2024, 6, 13)).height == 1
