"""标签定义：全项目唯一的标签口径所在。

下游的 IC / RankIC / ICIR / 分层只在 ``quant.eval.metrics`` 实现，标签只在
``quant.labels`` 定义。
"""
from quant.labels.open_to_open import (
    DEFAULT_HORIZON,
    REQUIRED_COLUMNS,
    attach_label,
    open_to_open_label,
)

__all__ = [
    "DEFAULT_HORIZON",
    "REQUIRED_COLUMNS",
    "attach_label",
    "open_to_open_label",
]
