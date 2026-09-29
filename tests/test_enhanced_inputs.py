"""``quant.portfolio.enhanced_inputs`` 单测（issue #71）。

覆盖 PIT 截取（行业 / 流通市值 / 六风格 / 基准权重 / 收益矩阵）与两处退化分支
（真实流通市值缺失 → 等效市值；行业历史快照缺失 → 最新一份）。
全部用 ``tmp_path`` 下的合成缓存，不碰真实 ``data/``。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import (
    DAILY_BARS,
    FLOAT_MV,
    INDEX_WEIGHTS,
    INDUSTRY,
)
from quant.portfolio import enhanced_inputs as ei

DAY1 = date(2026, 1, 5)
DAY2 = date(2026, 1, 6)
DAY3 = date(2026, 1, 7)
INSTRUMENTS = ["600000.SH", "600001.SH"]


def _write_industry(data_dir: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows, schema=INDUSTRY).write_parquet(data_dir / "industry.parquet")


def _write_float_mv(data_dir: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows, schema=FLOAT_MV).write_parquet(data_dir / "float_mv.parquet")


def _write_index_weights(data_dir: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows, schema=INDEX_WEIGHTS).write_parquet(
        data_dir / "index_weights.parquet"
    )


def _bars() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, instrument in enumerate(INSTRUMENTS):
        for step, day in enumerate((DAY1, DAY2, DAY3)):
            close = 10.0 + index + step
            rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "vwap": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows, schema=DAILY_BARS)


# ---------------------------------------------------------------------------
# 基准权重
# ---------------------------------------------------------------------------


def test_benchmark_weights_takes_latest_not_after(tmp_path: Path) -> None:
    _write_index_weights(
        tmp_path,
        [
            {"date": DAY1, "instrument": "600000.SH", "index_code": "000852", "weight": 0.6},
            {"date": DAY1, "instrument": "600001.SH", "index_code": "000852", "weight": 0.4},
            {"date": DAY3, "instrument": "600000.SH", "index_code": "000852", "weight": 0.9},
            {"date": DAY3, "instrument": "600001.SH", "index_code": "000852", "weight": 0.1},
            {"date": DAY3, "instrument": "600000.SH", "index_code": "000300", "weight": 1.0},
        ],
    )
    assert ei.benchmark_weights(tmp_path, "000852", DAY2) == {
        "600000.SH": pytest.approx(0.6),
        "600001.SH": pytest.approx(0.4),
    }
    # 早于所有记录 / 指数不存在 → 空字典
    assert ei.benchmark_weights(tmp_path, "000852", date(2025, 1, 1)) == {}
    assert ei.benchmark_weights(tmp_path, "000905", DAY3) == {}


# ---------------------------------------------------------------------------
# 行业：PIT 与退化
# ---------------------------------------------------------------------------


def test_industry_table_pit_then_fallback(tmp_path: Path) -> None:
    _write_industry(
        tmp_path,
        [
            {
                "instrument": "600000.SH",
                "industry_l1": "银行",
                "industry_l2": "银行",
                "effective_from": DAY1,
            },
            {
                "instrument": "600000.SH",
                "industry_l1": "非银金融",
                "industry_l2": "证券",
                "effective_from": DAY3,
            },
        ],
    )
    frame, fallback = ei.industry_table(tmp_path, DAY2, INSTRUMENTS)
    assert not fallback
    assert ei.industry_map(frame) == {"600000.SH": "银行"}

    # 早于首份快照：退化到最新一份并置 fallback。
    frame, fallback = ei.industry_table(tmp_path, date(2025, 12, 31), INSTRUMENTS)
    assert fallback
    assert ei.industry_map(frame) == {"600000.SH": "非银金融"}


def test_industry_table_empty_when_no_snapshot(tmp_path: Path) -> None:
    frame, fallback = ei.industry_table(tmp_path, DAY1, INSTRUMENTS)
    assert fallback
    assert frame.height == 0
    assert ei.industry_map(frame) == {}


# ---------------------------------------------------------------------------
# 流通市值：真实表优先，缺失退化到等效市值
# ---------------------------------------------------------------------------


def test_float_mv_frame_prefers_real_table(tmp_path: Path) -> None:
    _write_float_mv(
        tmp_path,
        [
            {"date": DAY1, "instrument": "600000.SH", "float_mv": 1.0e6},
            {"date": DAY3, "instrument": "600000.SH", "float_mv": 2.0e6},
            {"date": DAY2, "instrument": "600001.SH", "float_mv": 3.0e6},
        ],
    )
    frame = ei.float_mv_frame(tmp_path, _bars())
    assert ei.float_mv_map(frame, DAY2) == {
        "600000.SH": pytest.approx(1.0e6),
        "600001.SH": pytest.approx(3.0e6),
    }
    # cutoff 之后的行不可见
    assert ei.float_mv_map(frame, DAY1) == {"600000.SH": pytest.approx(1.0e6)}


def test_float_mv_frame_falls_back_to_equivalent_mv(tmp_path: Path) -> None:
    frame = ei.float_mv_frame(tmp_path, _bars())
    assert ei.FLOAT_MV_COL in frame.columns
    # 等效市值 = close × adjfactor；DAY1 两票分别为 10 / 11。
    assert ei.float_mv_map(frame, DAY1) == {
        "600000.SH": pytest.approx(10.0),
        "600001.SH": pytest.approx(11.0),
    }
    assert ei.float_mv_map(frame, DAY3) == {
        "600000.SH": pytest.approx(12.0),
        "600001.SH": pytest.approx(13.0),
    }


# ---------------------------------------------------------------------------
# 六风格与收益矩阵
# ---------------------------------------------------------------------------


def test_style_map_takes_latest_row_not_after(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "date": [DAY1, DAY2, DAY3],
            "instrument": ["600000.SH"] * 3,
            "beta": [0.1, 0.2, 0.3],
            "momentum": [1.0, 2.0, 3.0],
            "nlsize": [0.0, 0.0, 0.0],
            "reverse": [0.0, 0.0, 0.0],
            "sigma": [0.0, 0.0, 0.0],
            "turnover": [0.0, 0.0, 0.0],
        },
        schema={
            "date": pl.Date,
            "instrument": pl.String,
            "beta": pl.Float64,
            "momentum": pl.Float64,
            "nlsize": pl.Float64,
            "reverse": pl.Float64,
            "sigma": pl.Float64,
            "turnover": pl.Float64,
        },
    )
    mapped = ei.style_map(frame, DAY2)
    assert mapped["600000.SH"]["beta"] == pytest.approx(0.2)
    assert mapped["600000.SH"]["momentum"] == pytest.approx(2.0)
    assert ei.style_map(frame, date(2025, 12, 31)) == {}


def test_returns_matrix_aligns_columns_and_fills_missing(tmp_path: Path) -> None:
    bars = _bars()
    instruments = ["600001.SH", "600000.SH", "999999.SH"]
    matrix = ei.returns_matrix(bars, instruments, DAY3, window=5)
    assert list(matrix.columns) == instruments
    # 窗口内首日没有前一交易日，收益缺失填 0（与流水线口径一致）。
    assert matrix.height == 3
    assert matrix["999999.SH"].to_list() == [0.0, 0.0, 0.0]
    # 后复权收益：10 → 11 → 12 对应 0.1 / 1/11
    assert matrix["600000.SH"].to_list()[1] == pytest.approx(0.1)


def test_latest_rows_empty_input(tmp_path: Path) -> None:
    out = ei.latest_rows(pl.DataFrame(), DAY1, ["float_mv"])
    assert out.height == 0
    assert out.columns == [ei.INSTRUMENT_COL, "float_mv"]


def test_subset_weights_drops_outside_and_non_positive() -> None:
    weights = {"A": 0.5, "B": 0.0, "C": -0.1, "D": 0.2}
    assert ei.subset_weights(weights, ["A", "B"]) == {"A": 0.5}


def test_returns_window_default_constant() -> None:
    # 与 pipeline 的 RETURNS_WINDOW_DEFAULT 保持一致（协方差窗口口径）。
    from quant.daily import pipeline

    assert ei.DEFAULT_RETURNS_WINDOW == pipeline.RETURNS_WINDOW_DEFAULT


def test_module_date_import_sanity() -> None:
    assert isinstance(DAY1 + timedelta(days=1), date)
