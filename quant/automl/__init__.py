"""``quant.automl``：AutoGluon 基线训练管线公开接口。

- :func:`build_dataset`：因子面板 → 截面标准化后的单窗口训练集。
- :func:`missing_rate`：特征缺测率报告。
- :class:`BaselineTrainer`：包一层 AutoGluon ``TabularPredictor`` 的回归基线。
- :func:`gpu_available` / :func:`resolve_num_gpus`：GPU 探测与开关解析。

walk-forward 切分归 ``quant/eval/model.py``（issue #33），本包只管单窗口训练与预测。
"""
from __future__ import annotations

from quant.automl.dataset import (
    DATE_COL,
    DELAY_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    FactorCompute,
    build_dataset,
    missing_rate,
)
from quant.automl.trainer import (
    DEFAULT_EVAL_METRIC,
    DEFAULT_MODEL_DIR,
    DEFAULT_PRESETS,
    DEFAULT_TIME_LIMIT,
    BaselineTrainer,
    gpu_available,
    resolve_num_gpus,
)

__all__ = [
    "DATE_COL",
    "DEFAULT_EVAL_METRIC",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_PRESETS",
    "DEFAULT_TIME_LIMIT",
    "DELAY_COL",
    "INSTRUMENT_COL",
    "LABEL_COL",
    "BaselineTrainer",
    "FactorCompute",
    "build_dataset",
    "gpu_available",
    "missing_rate",
    "resolve_num_gpus",
]
