"""``quant.factor_lib.correlation`` 的单元测试：行为相关性口径。

全部用确定性合成值长表，不触网、不读真实数据。覆盖：完全正 / 负相关、
逐日截面样本数下限、绝对值取最大、空库。
"""
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable

import polars as pl
import pytest

from quant.factor_lib.correlation import (
    cross_section_corr,
    iter_library_values,
    load_library_values,
    max_library_corr,
)

START: dt.date = dt.date(2024, 1, 1)


def _panel(
    n_instruments: int,
    n_days: int,
    value_fn: Callable[[int, int], float],
) -> pl.DataFrame:
    """构造 ``(date, instrument, value)`` 长表，``value_fn(day, instrument)`` 决定取值。"""
    rows: list[dict[str, object]] = []
    for day in range(n_days):
        for index in range(n_instruments):
            rows.append(
                {
                    "date": START + dt.timedelta(days=day),
                    "instrument": f"{600000 + index:06d}.SH",
                    "value": value_fn(day, index),
                }
            )
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "instrument": pl.String, "value": pl.Float64},
    )


def _signal(day: int, index: int) -> float:
    """确定性的截面变化序列（非随机），逐日独立。"""
    return math.sin(1.7 * index + 0.3 * day) + 0.05 * (index % 7)


class TestCrossSectionCorr:
    def test_perfect_positive_is_one(self) -> None:
        library = _panel(40, 6, _signal)
        new = library.with_columns((2.0 * pl.col("value") + 3.0).alias("value"))

        assert cross_section_corr(new, library) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_is_minus_one(self) -> None:
        library = _panel(40, 6, _signal)
        new = library.with_columns((-3.0 * pl.col("value")).alias("value"))

        assert cross_section_corr(new, library) == pytest.approx(-1.0, abs=1e-9)

    def test_days_below_min_count_are_skipped(self) -> None:
        library = _panel(10, 6, _signal)
        new = library.with_columns((2.0 * pl.col("value")).alias("value"))

        # 逐日只有 10 只证券，低于默认下限 30，全部日子跳过 → None。
        assert cross_section_corr(new, library) is None
        # 放宽下限后可得有限相关。
        relaxed = cross_section_corr(new, library, min_count=5)
        assert relaxed is not None and math.isfinite(relaxed)

    def test_no_overlap_returns_none(self) -> None:
        library = _panel(40, 6, _signal)
        shifted = library.with_columns(
            (pl.col("date") + pl.duration(days=100)).alias("date")
        )

        assert cross_section_corr(shifted, library) is None


class TestMaxLibraryCorr:
    def test_uses_absolute_value_of_negative_corr(self) -> None:
        library = _panel(40, 6, _signal)
        new = library.with_columns((2.0 * pl.col("value")).alias("value"))

        report = max_library_corr(new, {"neg": library.with_columns((-pl.col("value")).alias("value"))})

        assert report.per_factor["neg"] == pytest.approx(-1.0, abs=1e-9)
        assert report.max_corr == pytest.approx(1.0, abs=1e-9)

    def test_empty_library_gives_none(self) -> None:
        new = _panel(40, 6, _signal)

        report = max_library_corr(new, {})

        assert report.per_factor == {}
        assert report.max_corr is None

    def test_all_unavailable_gives_none(self) -> None:
        library = _panel(10, 6, _signal)

        report = max_library_corr(library, {"tiny": library})

        assert report.per_factor == {"tiny": None}
        assert report.max_corr is None

    def test_iterable_input_matches_mapping(self) -> None:
        """可迭代输入（查重阶段逐条取库的用法）与映射输入结果一致。"""
        library = _panel(40, 6, _signal)
        other = library.with_columns((pl.col("value") * 0.5 + 1.0).alias("value"))
        new = library.with_columns((2.0 * pl.col("value") + 3.0).alias("value"))
        mapping = {"a": library, "b": other}

        from_mapping = max_library_corr(new, mapping)
        from_iterable = max_library_corr(new, iter(mapping.items()))

        assert from_iterable.per_factor == from_mapping.per_factor
        assert from_iterable.max_corr == from_mapping.max_corr


# ---------------------------------------------------------------------------
# 逐条取库（OOM 口径）
# ---------------------------------------------------------------------------


_FACTOR_SOURCE = '''\
"""测试用最小因子：value = close。"""

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date", "instrument", pl.col("close").cast(pl.Float64).alias("value")
    )
'''

_BROKEN_FACTOR_SOURCE = """def compute(data):\n    raise RuntimeError("boom")\n"""


def _registry_payload(code_path: str) -> dict[str, object]:
    return {
        "factor_id": code_path.rsplit("/", 1)[-1].removesuffix(".py"),
        "hypothesis": "测试因子",
        "code_path": code_path,
        "metrics": {},
        "direction": {
            "signal_source": "price",
            "time_scale": "short",
            "mechanism": "test",
        },
        "lineage": {"op": "seed", "parents": [], "run_id": "test", "generation": 0},
        "status": "pool",
    }


def _write_library(root: Path, entries: list[tuple[str, str]]) -> Path:
    """在 ``root`` 下搭一个最小因子库：``factor_library/`` 含 registry 与因子文件。"""
    import json

    library_dir = root / "factor_library"
    library_dir.mkdir(parents=True)
    for filename, source in entries:
        (library_dir / filename).write_text(source, encoding="utf-8")
    payload = {
        "version": 1,
        "factors": [
            _registry_payload(f"factor_library/{filename}") for filename, _ in entries
        ],
    }
    (library_dir / "registry.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return library_dir


def _factor_input() -> pl.DataFrame:
    rows = []
    for day in range(30):
        for index in range(3):
            rows.append(
                {
                    "date": START + dt.timedelta(days=day),
                    "instrument": f"{600000 + index:06d}.SH",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0 + index + day * 0.01,
                    "vwap": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows)


class TestIterLibraryValues:
    def test_missing_registry_yields_nothing(self, tmp_path: Path) -> None:
        assert list(iter_library_values(tmp_path, _factor_input())) == []
        assert load_library_values(tmp_path, _factor_input()) == {}

    def test_yields_pool_factor_values(self, tmp_path: Path) -> None:
        library_dir = _write_library(tmp_path, [("close_factor.py", _FACTOR_SOURCE)])
        data = _factor_input()

        pairs = list(iter_library_values(library_dir, data))

        assert [factor_id for factor_id, _ in pairs] == ["close_factor"]
        values = pairs[0][1]
        assert values.columns == ["date", "instrument", "value"]
        assert values.height == data.height

    def test_broken_factor_is_skipped(self, tmp_path: Path) -> None:
        library_dir = _write_library(
            tmp_path,
            [("broken.py", _BROKEN_FACTOR_SOURCE), ("close_factor.py", _FACTOR_SOURCE)],
        )

        pairs = list(iter_library_values(library_dir, _factor_input()))

        assert [factor_id for factor_id, _ in pairs] == ["close_factor"]

    def test_load_library_values_matches_iter(self, tmp_path: Path) -> None:
        """批量取库与逐条取库取值口径一致，只差生命周期。"""
        library_dir = _write_library(tmp_path, [("close_factor.py", _FACTOR_SOURCE)])
        data = _factor_input()

        loaded = load_library_values(library_dir, data)
        iterated = dict(iter_library_values(library_dir, data))

        assert loaded.keys() == iterated.keys()
        for factor_id in loaded:
            assert loaded[factor_id].equals(iterated[factor_id])
