"""factor_api 复杂度度量：圈复杂度 + AST 节点数 + 使用字段数。

Agent 生成的因子若过于复杂，审计成本高，也更容易藏前视或随机性。
本模块对单个因子 ``.py`` 文件做静态度量，三个维度任一超过阈值即判 ``ok=False``：

- **圈复杂度**：用 ``radon.complexity.cc_visit`` 计算，取所有函数/方法的最大值与总和。
- **AST 节点数**：``ast.parse`` 后 ``ast.walk`` 计数，粗略反映代码体量。
- **使用字段数**：收集源码里全部字符串字面量，与因子输入列集合
  :data:`quant.factor_api.spec.FACTOR_INPUT_COLUMNS` 求交集。

字段口径说明
    按字符串字面量取交集，能稳定捕获 ``pl.col("close")``、``data["close"]``、
    ``select("close")`` 以及模块级 ``REQUIRED_COLUMNS`` 常量等各种写法，
    不依赖对 polars 调用的语义分析。代价是会多算：源码里恰好与输入列同名的
    字符串（例如注释外的日志文本）也会被计入。误报方向偏严，对 Agent 安全。

阈值取舍
    因子输入一共 10 列，``MAX_FIELDS`` 取 10 等于全部放行，卡控主力落在
    圈复杂度与 AST 节点数两项。这样既不误伤正常的多字段因子，又能拦住
    结构上明显失控的实现。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from radon.complexity import cc_visit
from radon.visitors import Class, Function

from quant.factor_api.spec import FACTOR_INPUT_COLUMNS

#: 单函数最高圈复杂度阈值。
MAX_CYCLOMATIC: int = 15

#: AST 节点总数阈值。
MAX_AST_NODES: int = 500

#: 使用字段数阈值（输入共 10 列，等于全放行）。
MAX_FIELDS: int = 10

#: 因子输入列集合，用于字符串字面量求交集。
_INPUT_COLUMNS: frozenset[str] = frozenset(FACTOR_INPUT_COLUMNS)


@dataclass(frozen=True)
class ComplexityReport:
    """单个因子的复杂度度量结果。

    :param cyclomatic_max: 所有函数/方法中最高的圈复杂度。
    :param cyclomatic_total: 所有函数/方法的圈复杂度之和。
    :param ast_nodes: AST 节点总数。
    :param fields_used: 命中的输入字段名，去重后升序。
    :param ok: 三项度量是否全部在阈值内。
    :param violations: 超阈值项的说明，中文描述，无违规时为空元组。
    """

    cyclomatic_max: int
    cyclomatic_total: int
    ast_nodes: int
    fields_used: tuple[str, ...]
    ok: bool
    violations: tuple[str, ...]


def measure_complexity(
    path: Path,
    *,
    max_cyclomatic: int = MAX_CYCLOMATIC,
    max_ast_nodes: int = MAX_AST_NODES,
    max_fields: int = MAX_FIELDS,
) -> ComplexityReport:
    """度量 ``path`` 指向的因子源码，返回 :class:`ComplexityReport`。

    三个阈值均可注入，便于测试或后续按策略收紧。

    :param path: 因子 ``.py`` 文件路径。
    :param max_cyclomatic: 单函数最高圈复杂度阈值。
    :param max_ast_nodes: AST 节点总数阈值。
    :param max_fields: 使用字段数阈值。
    :raises FileNotFoundError: 路径不存在。
    :raises SyntaxError: 源码无法解析。
    """
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = list(_iter_functions(cc_visit(source)))
    complexities = [block.complexity for block in functions]
    cyclomatic_max = max(complexities, default=0)
    cyclomatic_total = sum(complexities)
    ast_nodes = sum(1 for _ in ast.walk(tree))
    fields_used = _collect_fields(tree)

    violations: list[str] = []
    if cyclomatic_max > max_cyclomatic:
        violations.append(
            f"圈复杂度最高 {cyclomatic_max}，超过阈值 {max_cyclomatic}"
        )
    if ast_nodes > max_ast_nodes:
        violations.append(f"AST 节点数 {ast_nodes}，超过阈值 {max_ast_nodes}")
    if len(fields_used) > max_fields:
        violations.append(
            f"使用字段数 {len(fields_used)}，超过阈值 {max_fields}："
            f"{', '.join(fields_used)}"
        )

    return ComplexityReport(
        cyclomatic_max=cyclomatic_max,
        cyclomatic_total=cyclomatic_total,
        ast_nodes=ast_nodes,
        fields_used=fields_used,
        ok=not violations,
        violations=tuple(violations),
    )


def _iter_functions(blocks: list) -> Iterator[Function]:
    """展开 radon 的代码块，产出其中的函数与方法（含类内方法）。"""
    for block in blocks:
        if isinstance(block, Function):
            yield block
        elif isinstance(block, Class):
            yield from _iter_functions(block.methods)


def _collect_fields(tree: ast.AST) -> tuple[str, ...]:
    """收集源码字符串字面量与输入列名的交集，去重升序。"""
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _INPUT_COLUMNS:
                used.add(node.value)
    return tuple(sorted(used))
