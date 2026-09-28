import datetime as dt

import polars as pl
import pytest

from quant.data.schema import (
    DAILY_BARS,
    SchemaError,
    board_of,
    check_daily_bars,
    check_schema,
    exchange_of_digits,
    normalize_instrument,
)


def _bars(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=DAILY_BARS,
        orient="row",
    )


class TestInstrumentCode:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("600000", "600000.SH"),
            ("688981", "688981.SH"),
            ("000001", "000001.SZ"),
            ("300750", "300750.SZ"),
            ("430047", "430047.BJ"),
            ("830799", "830799.BJ"),
            ("920001", "920001.BJ"),
            ("600000.SH", "600000.SH"),
            ("600000.sh", "600000.SH"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_instrument(raw) == expected

    def test_normalize_rejects_garbage(self) -> None:
        with pytest.raises(ValueError):
            normalize_instrument("ABC123")

    def test_exchange_of_digits_unknown_segment(self) -> None:
        with pytest.raises(ValueError):
            exchange_of_digits("700000")

    @pytest.mark.parametrize(
        "instrument, board",
        [
            ("600000.SH", "main"),
            ("000001.SZ", "main"),
            ("002594.SZ", "main"),
            ("300750.SZ", "cyb"),
            ("301308.SZ", "cyb"),
            ("688981.SH", "kcb"),
            ("689009.SH", "kcb"),
            ("430047.BJ", "bj"),
            ("920001.BJ", "bj"),
        ],
    )
    def test_board_of(self, instrument: str, board: str) -> None:
        assert board_of(instrument) == board


class TestCheckSchema:
    def test_ok(self) -> None:
        df = _bars(
            [
                (dt.date(2026, 9, 25), "600000.SH", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None)
            ]
        )
        check_schema(df, DAILY_BARS, name="daily_bars")

    def test_missing_column(self) -> None:
        df = _bars(
            [
                (dt.date(2026, 9, 25), "600000.SH", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None)
            ]
        ).drop("vwap")
        with pytest.raises(SchemaError, match="列不符"):
            check_schema(df, DAILY_BARS, name="daily_bars")

    def test_wrong_dtype(self) -> None:
        df = _bars(
            [
                (dt.date(2026, 9, 25), "600000.SH", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None)
            ]
        ).with_columns(pl.col("close").cast(pl.Int64))
        with pytest.raises(SchemaError, match="dtype 不符"):
            check_schema(df, DAILY_BARS, name="daily_bars")


class TestCheckDailyBars:
    def _row(self, d: dt.date, ins: str) -> tuple:
        return (d, ins, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None)

    def test_unsorted_rejected(self) -> None:
        df = _bars(
            [
                self._row(dt.date(2026, 9, 25), "600000.SH"),
                self._row(dt.date(2026, 9, 24), "600000.SH"),
            ]
        )
        with pytest.raises(SchemaError, match="排序"):
            check_daily_bars(df)

    def test_duplicate_key_rejected(self) -> None:
        row = self._row(dt.date(2026, 9, 24), "600000.SH")
        df = _bars([row, row])
        with pytest.raises(SchemaError, match="重复"):
            check_daily_bars(df)

    def test_sorted_multi_instrument_ok(self) -> None:
        df = _bars(
            [
                self._row(dt.date(2026, 9, 24), "000001.SZ"),
                self._row(dt.date(2026, 9, 25), "000001.SZ"),
                self._row(dt.date(2026, 9, 24), "600000.SH"),
            ]
        )
        check_daily_bars(df)
