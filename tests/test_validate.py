"""``quant.data.validate`` 单元测试。

全部用例用合成数据写到 ``tmp_path``，不触网；覆盖每条检查的触发与不触发，
并验证 ``ok`` 属性与 CLI 退出码。
"""
from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
)
from quant.data.validate import validate

# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------

_MAIN = "600001.SH"
_CYB = "300001.SZ"

_DEFAULT_BAR: dict[str, object] = {
    "open": 10.0,
    "high": 10.0,
    "low": 10.0,
    "close": 10.0,
    "vwap": 10.0,
    "volume": 1000.0,
    "amount": 10000.0,
    "adjfactor": 1.0,
    "limit_up": None,
    "limit_down": None,
}


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


DAYS: list[date] = _weekdays(date(2024, 1, 2), 20)


def _row(day: date, instrument: str, **overrides: object) -> dict[str, object]:
    row = {"date": day, "instrument": instrument, **_DEFAULT_BAR}
    row.update(overrides)
    return row


def _base_bars(days: list[date] | None = None) -> list[dict[str, object]]:
    days = days or DAYS
    return [_row(day, instrument) for instrument in (_MAIN, _CYB) for day in days]


def _instrument_rows(
    list_dates: dict[str, date] | None = None,
) -> list[dict[str, object]]:
    list_dates = list_dates or {}
    return [
        {
            "instrument": _MAIN,
            "name": "测试A",
            "board": "main",
            "list_date": list_dates.get(_MAIN, date(2023, 12, 1)),
            "delist_date": None,
        },
        {
            "instrument": _CYB,
            "name": "测试B",
            "board": "cyb",
            "list_date": list_dates.get(_CYB, date(2023, 12, 1)),
            "delist_date": None,
        },
    ]


def _write_cache(
    tmp_path: Path,
    bars: list[dict[str, object]],
    *,
    days: list[date] | None = None,
    instruments: list[dict[str, object]] | None = None,
    actions: list[dict[str, object]] | None = None,
) -> Path:
    days = days or DAYS
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    if bars:
        frame = pl.DataFrame(bars, schema=DAILY_BARS)
        for year in sorted(frame["date"].dt.year().unique().to_list()):
            frame.filter(pl.col("date").dt.year() == year).write_parquet(
                bars_dir / f"{year}.parquet"
            )
    else:
        pl.DataFrame(schema=DAILY_BARS).write_parquet(bars_dir / "2024.parquet")

    openset = set(days)
    span = [
        min(days) + timedelta(days=offset)
        for offset in range((max(days) - min(days)).days + 1)
    ]
    pl.DataFrame(
        {"date": span, "is_open": [item in openset for item in span]}
    ).cast(TRADE_CALENDAR).write_parquet(tmp_path / "calendar.parquet")

    pl.DataFrame(
        instruments if instruments is not None else _instrument_rows(),
        schema=INSTRUMENT_INFO,
    ).write_parquet(tmp_path / "instruments.parquet")

    pl.DataFrame(actions or [], schema=CORPORATE_ACTIONS).write_parquet(
        tmp_path / "corporate_actions.parquet"
    )
    return tmp_path


def _apply(
    bars: list[dict[str, object]],
    instrument: str,
    predicate,
    **overrides: object,
) -> list[dict[str, object]]:
    """把某只证券满足条件的行替换为带 overrides 的新行。"""
    return [
        _row(row["date"], row["instrument"], **overrides)
        if row["instrument"] == instrument and predicate(row["date"])
        else row
        for row in bars
    ]


def _checks(report) -> set[str]:
    return {issue.check for issue in report.issues}


def _severity(report, check: str) -> str | None:
    for issue in report.issues:
        if issue.check == check:
            return issue.severity
    return None


# ---------------------------------------------------------------------------
# 正常数据
# ---------------------------------------------------------------------------


def test_valid_data_all_green(tmp_path: Path) -> None:
    data_dir = _write_cache(tmp_path, _base_bars())
    report = validate(data_dir)
    assert report.ok
    assert report.issues == []
    assert report.checked_rows == 40
    assert report.checked_instruments == 2


# ---------------------------------------------------------------------------
# 异常值
# ---------------------------------------------------------------------------


def test_price_nonpositive(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, open=-1.0)
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "price_nonpositive") == "error"


def test_high_low_inverted(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, high=9.0, low=11.0)
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "high_low_inverted") == "error"


def test_open_close_out_of_range(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, open=11.0, high=10.5, low=9.5, close=10.0)
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "open_close_out_of_range") == "error"


def test_vwap_out_of_range_is_warning(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, vwap=12.0)
    report = validate(_write_cache(tmp_path, bars))
    assert report.ok
    assert _severity(report, "vwap_out_of_range") == "warning"


def test_vwap_within_tolerance_not_flagged(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, high=10.0, low=9.0, close=9.5, vwap=10.05)
    report = validate(_write_cache(tmp_path, bars))
    assert "vwap_out_of_range" not in _checks(report)


def test_limit_move_exceeded(tmp_path: Path) -> None:
    bars = _apply(
        _base_bars(),
        _MAIN,
        lambda day: day == DAYS[-1],
        close=12.0,
        open=12.0,
        high=12.0,
        low=11.9,
        vwap=11.9,
    )
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "limit_move_exceeded") == "error"


def test_limit_move_within_band_not_flagged(tmp_path: Path) -> None:
    bars = _apply(
        _base_bars(),
        _MAIN,
        lambda day: day == DAYS[-1],
        close=10.9,
        open=10.5,
        high=10.9,
        low=10.4,
        vwap=10.7,
    )
    report = validate(_write_cache(tmp_path, bars))
    assert "limit_move_exceeded" not in _checks(report)


def test_new_listing_grace_skips_limit_check(tmp_path: Path) -> None:
    bars = [_row(day, _MAIN) for day in DAYS]
    bars += [_row(DAYS[5], _CYB)]
    bars += [_row(day, _CYB, close=20.0, open=20.0, high=20.0, low=19.0, vwap=19.5)
             for day in DAYS[6:]]
    data_dir = _write_cache(
        tmp_path,
        bars,
        instruments=_instrument_rows({_CYB: DAYS[5]}),
    )
    report = validate(data_dir)
    assert "limit_move_exceeded" not in _checks(report)


def test_volume_negative(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, volume=-1.0)
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "negative_value") == "error"


def test_volume_amount_mismatch(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, volume=1000.0, amount=0.0)
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "volume_amount_mismatch") == "error"


# ---------------------------------------------------------------------------
# 复权一致性
# ---------------------------------------------------------------------------


def test_adjfactor_nonpositive(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, adjfactor=0.0)
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "adjfactor_nonpositive") == "error"


def test_adjfactor_jump_without_ca_is_warning(tmp_path: Path) -> None:
    bars = _apply(_base_bars(), _MAIN, lambda day: day >= DAYS[5], adjfactor=2.0)
    report = validate(_write_cache(tmp_path, bars))
    assert report.ok
    assert _severity(report, "adjfactor_jump") == "warning"


def test_adjfactor_jump_explained_by_corporate_action(tmp_path: Path) -> None:
    bars = _apply(_base_bars(), _MAIN, lambda day: day >= DAYS[5], adjfactor=2.0)
    actions = [
        {
            "date": DAYS[5],
            "instrument": _MAIN,
            "cash_per_share": 0.5,
            "share_per_share": 0.0,
        }
    ]
    report = validate(_write_cache(tmp_path, bars, actions=actions))
    assert report.ok
    assert "adjfactor_jump" not in _checks(report)


def test_adjfactor_decrease_is_warning(tmp_path: Path) -> None:
    bars = _apply(_base_bars(), _MAIN, lambda day: day >= DAYS[5], adjfactor=0.5)
    report = validate(_write_cache(tmp_path, bars))
    assert report.ok
    assert _severity(report, "adjfactor_decrease") == "warning"
    assert _severity(report, "adjfactor_jump") == "warning"


# ---------------------------------------------------------------------------
# 完整性
# ---------------------------------------------------------------------------


def test_coverage_low(tmp_path: Path) -> None:
    bars = [
        row
        for row in _base_bars()
        if not (row["instrument"] == _CYB and row["date"] == DAYS[3])
    ]
    report = validate(_write_cache(tmp_path, bars))
    assert report.ok
    assert _severity(report, "coverage_low") == "warning"


def test_instrument_no_data(tmp_path: Path) -> None:
    bars = [row for row in _base_bars() if row["instrument"] == _MAIN]
    report = validate(_write_cache(tmp_path, bars))
    assert report.ok
    assert _severity(report, "instrument_no_data") == "warning"


def test_date_gap(tmp_path: Path) -> None:
    dropped = set(DAYS[2:14])
    bars = [
        row
        for row in _base_bars()
        if not (row["instrument"] == _MAIN and row["date"] in dropped)
    ]
    report = validate(_write_cache(tmp_path, bars))
    assert report.ok
    assert _severity(report, "date_gap") == "warning"


def test_short_gap_not_flagged(tmp_path: Path) -> None:
    bars = [row for row in _base_bars() if row["date"] != DAYS[3] or row["instrument"] == _CYB]
    report = validate(_write_cache(tmp_path, bars))
    assert "date_gap" not in _checks(report)


# ---------------------------------------------------------------------------
# schema / 重复键
# ---------------------------------------------------------------------------


def test_duplicate_key_is_error(tmp_path: Path) -> None:
    bars = _base_bars()
    bars.append(dict(bars[0]))
    report = validate(_write_cache(tmp_path, bars))
    assert not report.ok
    assert _severity(report, "duplicate_key") == "error"


def test_schema_mismatch_is_error(tmp_path: Path) -> None:
    data_dir = _write_cache(tmp_path, _base_bars())
    path = data_dir / "bars" / "2024.parquet"
    raw = pl.read_parquet(path).drop("vwap")
    raw.write_parquet(path)
    report = validate(data_dir)
    assert not report.ok
    assert _severity(report, "schema_mismatch") == "error"


def test_missing_calendar_is_error(tmp_path: Path) -> None:
    data_dir = _write_cache(tmp_path, _base_bars())
    (data_dir / "calendar.parquet").unlink()
    report = validate(data_dir)
    assert not report.ok
    assert _severity(report, "trade_calendar_missing") == "error"


# ---------------------------------------------------------------------------
# 窗口与合法性
# ---------------------------------------------------------------------------


def test_lookback_narrows_window(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, open=-1.0)
    data_dir = _write_cache(tmp_path, bars)

    full = validate(data_dir)
    assert not full.ok

    recent = validate(data_dir, lookback_days=3)
    assert recent.ok
    assert "price_nonpositive" not in _checks(recent)
    assert recent.checked_rows == 6


def test_invalid_lookback_raises(tmp_path: Path) -> None:
    data_dir = _write_cache(tmp_path, _base_bars())
    with pytest.raises(ValueError):
        validate(data_dir, lookback_days=0)


def test_end_truncates_window(tmp_path: Path) -> None:
    bars = _apply(_base_bars(), _MAIN, lambda day: day == DAYS[-1], open=-1.0)
    data_dir = _write_cache(tmp_path, bars)
    assert not validate(data_dir).ok
    truncated = validate(data_dir, end=DAYS[5])
    assert truncated.ok
    assert truncated.checked_rows == 12  # 6 个开市日 × 2 只


def test_ok_property_ignores_warnings(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, vwap=12.0)
    report = validate(_write_cache(tmp_path, bars))
    assert report.warnings
    assert not report.errors
    assert report.ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cli():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_data.py"
    spec = importlib.util.spec_from_file_location("validate_data_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_exit_zero_on_valid(tmp_path: Path) -> None:
    data_dir = _write_cache(tmp_path, _base_bars())
    cli = _load_cli()
    assert cli.main(["--data-dir", str(data_dir), "--full"]) == 0


def test_cli_exit_one_on_error(tmp_path: Path) -> None:
    bars = _base_bars()
    bars[0] = _row(DAYS[0], _MAIN, open=-1.0)
    data_dir = _write_cache(tmp_path, bars)
    cli = _load_cli()
    assert cli.main(["--data-dir", str(data_dir), "--full"]) == 1


def test_cli_rejects_full_with_lookback(tmp_path: Path) -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit):
        cli.main(["--data-dir", str(tmp_path), "--full", "--lookback-days", "5"])
