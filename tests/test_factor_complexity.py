"""``quant.factor_api.complexity`` 的单元测试：radon 圈复杂度 + AST 节点 + 字段统计。

全部使用合成源码字符串写入 ``tmp_path``，不触网、不读 ``data/`` 真实缓存。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quant.factor_api.complexity import (
    MAX_AST_NODES,
    MAX_CYCLOMATIC,
    MAX_FIELDS,
    ComplexityReport,
    measure_complexity,
)

#: 仓库根目录，用于定位真实因子库文件。
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

#: mom_5 风格简单因子源码：字段少、结构平坦，应通过。
SIMPLE_FACTOR_SRC: str = '''
"""测试因子：close × adjfactor 的 5 日动量。"""
from __future__ import annotations

import polars as pl

WINDOW: int = 5
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
        )
        .with_columns(
            (
                pl.col("_adj_close")
                / pl.col("_adj_close").shift(WINDOW).over("instrument")
                - 1.0
            ).alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
'''


def _write(tmp_path: Path, source: str, name: str = "factor") -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def _nested_if_source(levels: int) -> str:
    """生成 ``levels`` 层嵌套 if 的函数，圈复杂度随层数线性增长。"""
    lines = ["def compute(data):", "    value = 0"]
    for level in range(levels):
        indent = "    " * (level + 1)
        lines.append(f"{indent}if data is not None:")
        lines.append(f"{indent}    value += {level}")
    lines.append("    return value")
    return "\n".join(lines) + "\n"


def _long_body_source(statements: int) -> str:
    """生成含 ``statements`` 条赋值语句的函数，用于撑大 AST 节点数。"""
    lines = ["def compute(data):", "    value = 0"]
    for index in range(statements):
        lines.append(f"    value = {index}")
    lines.append("    return value")
    return "\n".join(lines) + "\n"


class TestMeasureComplexity:
    def test_simple_factor_passes(self, tmp_path: Path) -> None:
        report = measure_complexity(_write(tmp_path, SIMPLE_FACTOR_SRC))
        assert isinstance(report, ComplexityReport)
        assert report.ok is True
        assert report.violations == ()
        assert report.cyclomatic_max >= 1
        assert report.cyclomatic_total >= report.cyclomatic_max
        assert report.ast_nodes > 0

    def test_real_mom_5_factor_passes(self) -> None:
        report = measure_complexity(REPO_ROOT / "factor_library" / "mom_5.py")
        assert report.ok is True
        assert report.violations == ()
        assert report.cyclomatic_max <= MAX_CYCLOMATIC
        assert report.ast_nodes <= MAX_AST_NODES
        assert report.fields_used == ("adjfactor", "close", "date", "instrument")

    def test_high_cyclomatic_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _nested_if_source(MAX_CYCLOMATIC + 5))
        report = measure_complexity(path)
        assert report.ok is False
        assert report.cyclomatic_max > MAX_CYCLOMATIC
        assert any("圈复杂度" in item for item in report.violations)

    def test_excessive_ast_nodes_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _long_body_source(300))
        report = measure_complexity(path)
        assert report.ok is False
        assert report.ast_nodes > MAX_AST_NODES
        assert any("AST 节点数" in item for item in report.violations)

    def test_injectable_thresholds(self, tmp_path: Path) -> None:
        path = _write(tmp_path, SIMPLE_FACTOR_SRC)
        assert measure_complexity(path).ok is True

        too_tight_nodes = measure_complexity(path, max_ast_nodes=1)
        assert too_tight_nodes.ok is False
        assert any("AST 节点数" in item for item in too_tight_nodes.violations)

        too_tight_cyclomatic = measure_complexity(path, max_cyclomatic=0)
        assert too_tight_cyclomatic.ok is False
        assert any("圈复杂度" in item for item in too_tight_cyclomatic.violations)

        too_tight_fields = measure_complexity(path, max_fields=1)
        assert too_tight_fields.ok is False
        assert any("使用字段数" in item for item in too_tight_fields.violations)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            measure_complexity(tmp_path / "nope.py")


class TestFieldDetection:
    def test_all_writing_styles_captured(self, tmp_path: Path) -> None:
        source = '''
from __future__ import annotations

import polars as pl

REQUIRED_COLUMNS: tuple[str, ...] = ("adjfactor",)


def compute(data: pl.DataFrame) -> pl.DataFrame:
    selected = data.select("amount")
    direct = data["volume"]
    expr = pl.col("close")
    bogus = "not_a_col"
    return selected
'''
        report = measure_complexity(_write(tmp_path, source))
        assert report.fields_used == ("adjfactor", "amount", "close", "volume")
        assert "not_a_col" not in report.fields_used

    def test_nonexistent_column_not_counted(self, tmp_path: Path) -> None:
        source = 'value = "no_such_column"\nother = "close"\n'
        report = measure_complexity(_write(tmp_path, source))
        assert report.fields_used == ("close",)

    def test_fields_are_deduplicated_and_sorted(self, tmp_path: Path) -> None:
        source = 'a = "volume"\nb = "close"\nc = "volume"\n'
        report = measure_complexity(_write(tmp_path, source))
        assert report.fields_used == ("close", "volume")
        assert len(report.fields_used) == len(set(report.fields_used))
        assert report.fields_used == tuple(sorted(report.fields_used))

    def test_field_threshold_constant_matches_input_columns(self) -> None:
        from quant.factor_api.spec import FACTOR_INPUT_COLUMNS

        assert MAX_FIELDS == len(FACTOR_INPUT_COLUMNS)
