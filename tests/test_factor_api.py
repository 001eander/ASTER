"""``quant.factor_api`` 的单元测试：动态加载与输入输出 schema 校验。

全部使用合成的 3 只证券 × 若干交易日的面板数据，不触网、不读 ``data/`` 真实缓存。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import SchemaError
from quant.factor_api import (
    FACTOR_INPUT_SCHEMA,
    FACTOR_OUTPUT_SCHEMA,
    FactorLoadError,
    load_factor,
    validate_input,
    validate_output,
)

INSTRUMENTS: tuple[str, ...] = ("000001.SZ", "300750.SZ", "600000.SH")
DAYS: int = 6

#: 合法因子源码：对 close 取 2 日差分，输出 ``(date, instrument, value)``。
VALID_FACTOR_SRC: str = '''
"""测试因子：close 的 2 日差分。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    ordered = data.sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("close") - pl.col("close").shift(1).over("instrument")).alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
'''


def _input_frame() -> pl.DataFrame:
    """3 只证券 × 6 个交易日的合法输入，按 (instrument, date) 排序。"""
    rows: list[tuple] = []
    for instrument in INSTRUMENTS:
        for step in range(DAYS):
            day = dt.date(2026, 1, 5) + dt.timedelta(days=step)
            base = 10.0 + step
            rows.append(
                (
                    day,
                    instrument,
                    base,
                    base + 1.0,
                    base - 1.0,
                    base + 0.5,
                    base,
                    1000.0,
                    base * 1000.0,
                    1.0,
                )
            )
    return pl.DataFrame(rows, schema=FACTOR_INPUT_SCHEMA, orient="row")


def _output_frame() -> pl.DataFrame:
    """合法输出：每个 (date, instrument) 一行，value 可为 null。"""
    rows: list[tuple] = []
    for instrument in INSTRUMENTS:
        for step in range(DAYS):
            day = dt.date(2026, 1, 5) + dt.timedelta(days=step)
            rows.append((day, instrument, None if step == 0 else float(step)))
    return pl.DataFrame(rows, schema=FACTOR_OUTPUT_SCHEMA, orient="row")


def _write_factor(tmp_path: Path, source: str, name: str = "demo_factor") -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


class TestLoadFactor:
    def test_loads_valid_factor_and_runs(self, tmp_path: Path) -> None:
        path = _write_factor(tmp_path, VALID_FACTOR_SRC)
        compute = load_factor(path)
        assert callable(compute)

        data = _input_frame()
        validate_input(data)
        result = compute(data)
        validate_output(result)
        assert result.columns == ["date", "instrument", "value"]
        assert result.height == len(INSTRUMENTS) * DAYS
        first = result.filter(pl.col("instrument") == INSTRUMENTS[0]).sort("date")
        assert first["value"][0] is None
        assert first["value"][1] == pytest.approx(1.0)

    def test_same_path_loads_repeatedly(self, tmp_path: Path) -> None:
        path = _write_factor(tmp_path, VALID_FACTOR_SRC)
        first = load_factor(path)
        second = load_factor(path)
        data = _input_frame()
        assert first(data).equals(second(data))

    def test_distinct_paths_do_not_collide(self, tmp_path: Path) -> None:
        first_path = _write_factor(tmp_path, VALID_FACTOR_SRC, name="factor_a")
        second_path = _write_factor(tmp_path, VALID_FACTOR_SRC, name="factor_a_copy")
        first = load_factor(first_path)
        second = load_factor(second_path)
        assert first is not second

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FactorLoadError, match="不存在"):
            load_factor(tmp_path / "nope.py")

    def test_non_py_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "factor.txt"
        path.write_text(VALID_FACTOR_SRC, encoding="utf-8")
        with pytest.raises(FactorLoadError, match=r"\.py"):
            load_factor(path)

    def test_missing_compute(self, tmp_path: Path) -> None:
        path = _write_factor(tmp_path, "def other(data):\n    return data\n")
        with pytest.raises(FactorLoadError, match="未定义 compute"):
            load_factor(path)

    def test_compute_not_callable(self, tmp_path: Path) -> None:
        path = _write_factor(tmp_path, "compute = 123\n")
        with pytest.raises(FactorLoadError, match="不可调用"):
            load_factor(path)

    def test_compute_wrong_signature(self, tmp_path: Path) -> None:
        source = "def compute(data, window):\n    return data\n"
        path = _write_factor(tmp_path, source)
        with pytest.raises(FactorLoadError, match="恰好 1 个参数"):
            load_factor(path)

    def test_import_error_wrapped(self, tmp_path: Path) -> None:
        path = _write_factor(tmp_path, "raise RuntimeError('boom')\n")
        with pytest.raises(FactorLoadError, match="导入因子文件失败"):
            load_factor(path)


class TestValidateInput:
    def test_ok(self) -> None:
        validate_input(_input_frame())

    def test_missing_column(self) -> None:
        df = _input_frame().drop("vwap")
        with pytest.raises(SchemaError, match="列不符"):
            validate_input(df)

    def test_wrong_order(self) -> None:
        base = _input_frame()
        reordered = base.select("instrument", *[c for c in base.columns if c != "instrument"])
        with pytest.raises(SchemaError, match="列不符"):
            validate_input(reordered)

    def test_wrong_dtype(self) -> None:
        df = _input_frame().with_columns(pl.col("close").cast(pl.Int64))
        with pytest.raises(SchemaError, match="dtype 不符"):
            validate_input(df)


class TestValidateOutput:
    def test_ok(self) -> None:
        validate_output(_output_frame())

    def test_missing_column(self) -> None:
        df = _output_frame().drop("value")
        with pytest.raises(SchemaError, match="列不符"):
            validate_output(df)

    def test_extra_column(self) -> None:
        df = _output_frame().with_columns(pl.lit(1.0).alias("extra"))
        with pytest.raises(SchemaError, match="列不符"):
            validate_output(df)

    def test_wrong_dtype(self) -> None:
        df = _output_frame().with_columns(pl.col("value").cast(pl.Int64))
        with pytest.raises(SchemaError, match="dtype 不符"):
            validate_output(df)

    def test_duplicate_key(self) -> None:
        row = (dt.date(2026, 1, 5), INSTRUMENTS[0], 1.0)
        df = pl.DataFrame(
            [row, row], schema=FACTOR_OUTPUT_SCHEMA, orient="row"
        )
        with pytest.raises(SchemaError, match="重复"):
            validate_output(df)
