"""``scripts.report_factor_library`` 的单元测试：相关矩阵组装与热力图落盘。

合成值长表与临时因子库，不触网、不读真实行情。覆盖：矩阵形状 / 对称性 /
对角为 1、不可得对为 NaN、PNG 落盘、CLI 退出码与只读口径。
"""
from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant.data.schema import DAILY_BARS
from quant.factor_lib.prune import build_corr_matrix, cluster_corr_threshold
from scripts.report_factor_library import (
    EXIT_DATA_ERROR,
    EXIT_OK,
    EXIT_REGISTRY_ERROR,
    assemble_corr_grid,
    main,
    plot_corr_heatmap,
)

START: dt.date = dt.date(2024, 1, 2)


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


# ---------------------------------------------------------------------------
# assemble_corr_grid：画图前的数据组装
# ---------------------------------------------------------------------------


class TestAssembleCorrGrid:
    def test_shape_symmetry_diagonal(self) -> None:
        base = _panel(40, 6, _signal)
        values = {
            "alpha": base,
            "beta": base.with_columns((2.0 * pl.col("value") + 3.0).alias("value")),
            "gamma": _panel(40, 6, lambda day, index: math.cos(0.9 * index - 0.2 * day)),
        }
        matrix = build_corr_matrix(values)

        grid = assemble_corr_grid(["alpha", "beta", "gamma"], matrix)

        assert grid.shape == (3, 3)
        np.testing.assert_allclose(grid, grid.T, equal_nan=True)
        np.testing.assert_allclose(np.diag(grid), np.ones(3))
        # beta = 2 * alpha + 3，逐日截面相关恒为 1
        assert grid[0, 1] == pytest.approx(1.0, abs=1e-9)
        assert grid[0, 1] == pytest.approx(grid[1, 0], abs=1e-12)

    def test_unavailable_pair_is_nan(self) -> None:
        base = _panel(40, 6, _signal)
        # 逐日只有 10 只证券，低于默认截面下限 30，相关不可得 → 该对不进矩阵。
        tiny = _panel(10, 6, _signal)
        matrix = build_corr_matrix({"alpha": base, "tiny": tiny})

        grid = assemble_corr_grid(["alpha", "tiny"], matrix)

        assert math.isnan(grid[0, 1])
        assert math.isnan(grid[1, 0])
        assert grid[0, 0] == 1.0
        assert grid[1, 1] == 1.0

    def test_factor_without_values_gets_nan_off_diagonal(self) -> None:
        # pool 因子取值缺失（计算失败被跳过）时仍在轴上占位，非对角为 NaN。
        base = _panel(40, 6, _signal)
        matrix = build_corr_matrix({"alpha": base})

        grid = assemble_corr_grid(["alpha", "ghost"], matrix)

        assert math.isnan(grid[0, 1])
        assert math.isnan(grid[1, 0])
        np.testing.assert_allclose(np.diag(grid), np.ones(2))


# ---------------------------------------------------------------------------
# plot_corr_heatmap：无显示环境落盘
# ---------------------------------------------------------------------------


class TestPlotCorrHeatmap:
    def test_writes_png(self, tmp_path: Path) -> None:
        grid = np.array([[1.0, 0.8, np.nan], [0.8, 1.0, -0.5], [np.nan, -0.5, 1.0]])
        out = tmp_path / "nested" / "heat.png"

        path = plot_corr_heatmap(
            ["a", "b", "c"],
            grid,
            threshold=cluster_corr_threshold(),
            out_path=out,
        )

        assert path == out
        assert out.is_file()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# main：一条命令产出 PNG，只读 factor_library/
# ---------------------------------------------------------------------------

_FACTOR_CLOSE = '''\
"""测试用最小因子：value = close。"""

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date", "instrument", pl.col("close").cast(pl.Float64).alias("value")
    )
'''

_FACTOR_VWAP = '''\
"""测试用最小因子：value = vwap。"""

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date", "instrument", pl.col("vwap").cast(pl.Float64).alias("value")
    )
'''

_BROKEN_FACTOR_SOURCE = """def compute(data):\n    raise RuntimeError("boom")\n"""


def _registry_payload(factor_id: str, code_path: str) -> dict[str, object]:
    return {
        "factor_id": factor_id,
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


def _write_library(root: Path, entries: list[tuple[str, str, str]]) -> Path:
    """在 ``root`` 下搭最小因子库；entries 为 ``(factor_id, 文件名, 源码)``。"""
    library_dir = root / "factor_library"
    library_dir.mkdir(parents=True)
    for _, filename, source in entries:
        (library_dir / filename).write_text(source, encoding="utf-8")
    payload = {
        "version": 1,
        "factors": [
            _registry_payload(factor_id, f"factor_library/{filename}")
            for factor_id, filename, _ in entries
        ],
    }
    (library_dir / "registry.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return library_dir


def _write_bars(root: Path) -> Path:
    """写一份 40 只证券 × 6 个交易日的日线缓存（data/bars/2024.parquet）。"""
    rows: list[dict[str, object]] = []
    for day in range(6):
        for index in range(40):
            close = 10.0 + index + 0.5 * _signal(day, index)
            rows.append(
                {
                    "date": START + dt.timedelta(days=day),
                    "instrument": f"{600000 + index:06d}.SH",
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "vwap": close + 0.1 * math.cos(index + day),
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    data_dir = root / "data"
    (data_dir / "bars").mkdir(parents=True)
    frame = pl.DataFrame(rows).cast(DAILY_BARS)
    frame.write_parquet(data_dir / "bars" / "2024.parquet")
    return data_dir


class TestMain:
    def test_produces_heatmap_and_keeps_library_readonly(self, tmp_path: Path) -> None:
        library_dir = _write_library(
            tmp_path,
            [
                ("close_factor", "close_factor.py", _FACTOR_CLOSE),
                ("vwap_factor", "vwap_factor.py", _FACTOR_VWAP),
            ],
        )
        data_dir = _write_bars(tmp_path)
        registry_before = (library_dir / "registry.json").read_bytes()
        out = tmp_path / "runs" / "reports" / "factor_corr_heatmap.png"

        code = main(
            [
                "--data-dir",
                str(data_dir),
                "--factor-library-dir",
                str(library_dir),
                "--out",
                str(out),
            ]
        )

        assert code == EXIT_OK
        assert out.is_file()
        assert out.stat().st_size > 0
        assert (library_dir / "registry.json").read_bytes() == registry_before

    def test_missing_registry_exits_2(self, tmp_path: Path) -> None:
        data_dir = _write_bars(tmp_path)

        code = main(
            [
                "--data-dir",
                str(data_dir),
                "--factor-library-dir",
                str(tmp_path / "nope"),
                "--out",
                str(tmp_path / "out.png"),
            ]
        )

        assert code == EXIT_REGISTRY_ERROR

    def test_empty_bars_exits_1(self, tmp_path: Path) -> None:
        library_dir = _write_library(
            tmp_path, [("close_factor", "close_factor.py", _FACTOR_CLOSE)]
        )

        code = main(
            [
                "--data-dir",
                str(tmp_path / "empty-data"),
                "--factor-library-dir",
                str(library_dir),
                "--out",
                str(tmp_path / "out.png"),
            ]
        )

        assert code == EXIT_DATA_ERROR

    def test_empty_pool_exits_0_without_png(self, tmp_path: Path) -> None:
        library_dir = _write_library(tmp_path, [])
        data_dir = _write_bars(tmp_path)
        out = tmp_path / "out.png"

        code = main(
            [
                "--data-dir",
                str(data_dir),
                "--factor-library-dir",
                str(library_dir),
                "--out",
                str(out),
            ]
        )

        assert code == EXIT_OK
        assert not out.exists()

    def test_broken_factor_stays_on_axis_with_nan(self, tmp_path: Path) -> None:
        # 单个 pool 因子计算失败：不阻断出图，轴上占位、格子留白。
        library_dir = _write_library(
            tmp_path,
            [
                ("close_factor", "close_factor.py", _FACTOR_CLOSE),
                ("broken", "broken.py", _BROKEN_FACTOR_SOURCE),
            ],
        )
        data_dir = _write_bars(tmp_path)
        out = tmp_path / "out.png"

        code = main(
            [
                "--data-dir",
                str(data_dir),
                "--factor-library-dir",
                str(library_dir),
                "--out",
                str(out),
            ]
        )

        assert code == EXIT_OK
        assert out.is_file()

    def test_invalid_date_exits_via_argparse(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["--start", "2024/01/01"])
