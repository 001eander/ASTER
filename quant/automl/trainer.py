"""AutoGluon 基线训练器：单窗口训练 + 预测，供 walk-forward 循环复用。

职责范围
--------
本模块只包一层 :class:`autogluon.tabular.TabularPredictor`：给定一个已经构造好的
训练宽表（见 :mod:`quant.automl.dataset`），跑一次回归训练并输出 ``(date,
instrument, score)``。walk-forward 的窗口切分、样本外拼接归 ``quant/eval/model.py``
（issue #33），本模块不介入。

AutoGluon API 依据
------------------
AutoGluon 1.6.3（``autogluon.tabular``）：

- 回归任务：``TabularPredictor(label=..., problem_type="regression", ...)``。
- GPU：``fit(..., num_gpus=N)`` 为官方入口，N 为分配给整个 predictor 的 GPU 数；
  单模型级别可用 ``hyperparameters={"GBM": {"ag_args_fit": {"num_gpus": n}}}`` 细化。
  本基线只给顶层 ``num_gpus``（0 = 纯 CPU，1 = 单卡），模型自动继承。
- 保存/加载：``predictor.save()`` 写入它自己的 ``path``；``TabularPredictor.load(path)``。
- 预测：``predictor.predict(X)`` 返回 ``pandas.Series``。

pandas 边界
-----------
AutoGluon 只接受 ``pandas.DataFrame``，因此本模块**仅在** ``to_pandas()`` 这一处
接触 pandas，其余数据流保持 polars。pyarrow 为 ``to_pandas`` 的依赖。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from autogluon.tabular import TabularPredictor

from quant.automl.dataset import (
    DATE_COL,
    DELAY_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    missing_rate,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: AutoGluon 任务类型：回归。
PROBLEM_TYPE: str = "regression"

#: 默认预设（起步档，后续 issue #35 调优）。
DEFAULT_PRESETS: str = "medium_quality"

#: 默认模型集（issue #63）：精简到 GBM / XGB / CAT / NN_TORCH 四个族。
#:
#: issue #16 实跑暴露 16GB 内存瓶颈：RandomForest / ExtraTrees 估算需求
#: ~20GB 被 AutoGluon 直接跳过，NeuralNetFastAI 因内存余量不足被跳过，
#: LightGBMXT 吃掉 71% 预算且验证分更差。精简模型集让预算集中到有效模型上，
#: WeightedEnsemble 不再因 OOM 跳过而缺失模型族。传 ``hyperparameters=None``
#: 可回到 AutoGluon 全集（见 :class:`BaselineTrainer`）。
DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "GBM": {},
    "XGB": {},
    "CAT": {},
    "NN_TORCH": {},
}

#: 默认训练时限（秒）。
DEFAULT_TIME_LIMIT: float = 600.0

#: 默认评估指标。
DEFAULT_EVAL_METRIC: str = "root_mean_squared_error"

#: 默认 verbosity。
DEFAULT_VERBOSITY: int = 2

#: 默认模型落盘目录（``runs/`` 不入 git）。
DEFAULT_MODEL_DIR: str = "runs/automl/baseline"

#: 预测输出列名。
SCORE_COL: str = "score"

#: 不属于特征的列。
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    DATE_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    DELAY_COL,
)


# ---------------------------------------------------------------------------
# GPU 探测
# ---------------------------------------------------------------------------


def gpu_available() -> bool:
    """探测当前环境是否有可用 CUDA GPU。

    ``torch`` 为 AutoGluon 的依赖，正常装好；装不上 CUDA 版 torch 时返回 False，
    训练自动退回 CPU：``torch.cuda.is_available()`` 即官方判定入口。
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - 依赖缺失的兜底
        return False
    return bool(torch.cuda.is_available())


def resolve_num_gpus(use_gpu: bool | None) -> int:
    """把 GPU 开关解析成 ``fit(num_gpus=...)`` 的参数。

    ``use_gpu=None`` 时自动探测；显式 ``True`` / ``False`` 时以调用方为准。
    """
    enabled = gpu_available() if use_gpu is None else use_gpu
    return 1 if enabled else 0


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------


class BaselineTrainer:
    """包一层 AutoGluon ``TabularPredictor`` 的单窗口回归基线。

    典型用法::

        trainer = BaselineTrainer(path="runs/automl/window_00")
        trainer.train(train_df, valid_df=valid_df, time_limit=300)
        scores = trainer.predict(test_df)  # (date, instrument, score)
        trainer.save()
        ...
        same = BaselineTrainer.load("runs/automl/window_00")
        scores = same.predict(test_df)

    ``train_df`` / ``valid_df`` / ``test_df`` 均为 :func:`quant.automl.dataset.build_dataset`
    的输出形状：``date, instrument, <特征列...>, label, delay_days``。
    """

    def __init__(
        self,
        *,
        label: str = LABEL_COL,
        feature_columns: Sequence[str] | None = None,
        presets: str = DEFAULT_PRESETS,
        time_limit: float = DEFAULT_TIME_LIMIT,
        eval_metric: str = DEFAULT_EVAL_METRIC,
        path: str | Path = DEFAULT_MODEL_DIR,
        use_gpu: bool | None = None,
        verbosity: int = DEFAULT_VERBOSITY,
        hyperparameters: dict[str, Any] | str | None = DEFAULT_HYPERPARAMETERS,
        num_bag_folds: int | None = None,
        num_stack_levels: int | None = None,
        hyperparameter_tune_kwargs: dict[str, Any] | None = None,
        recipe: str | None = None,
    ) -> None:
        self.label = label
        self.feature_columns = (
            list(feature_columns) if feature_columns is not None else None
        )
        self.presets = presets
        self.time_limit = float(time_limit)
        self.eval_metric = eval_metric
        self.path = str(path)
        self.use_gpu = gpu_available() if use_gpu is None else use_gpu
        self.num_gpus = resolve_num_gpus(self.use_gpu)
        self.verbosity = verbosity
        self.hyperparameters = hyperparameters
        self.num_bag_folds = num_bag_folds
        self.num_stack_levels = num_stack_levels
        self.hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
        if recipe is not None:
            self._apply_recipe(recipe)

        self.predictor: TabularPredictor | None = None
        #: 训练/加载后解析出的特征列。
        self.feature_columns_: list[str] | None = (
            list(self.feature_columns) if self.feature_columns is not None else None
        )
        #: 训练集各特征缺测率报告（``train`` 后可用）。
        self.feature_missing_rate_: pl.DataFrame | None = None

    def _apply_recipe(self, name: str) -> None:
        """按命名配方覆盖训练旋钮（recipe 的字段优先于构造参数）。"""
        from quant.automl.recipes import get_recipe

        found = get_recipe(name)
        self.presets = found.presets
        self.hyperparameters = found.hyperparameters
        self.num_bag_folds = found.num_bag_folds
        self.num_stack_levels = found.num_stack_levels
        self.hyperparameter_tune_kwargs = found.hyperparameter_tune_kwargs

    # -- 状态 ---------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """是否已有可用的已训练 predictor。"""
        return self.predictor is not None

    def _require_fitted(self) -> TabularPredictor:
        if self.predictor is None:
            raise RuntimeError("trainer 尚未训练或加载，无法预测")
        return self.predictor

    # -- 列解析 -------------------------------------------------------------

    def _feature_names_for(self, df: pl.DataFrame) -> list[str]:
        if self.feature_columns is not None:
            names = list(self.feature_columns)
        elif self.feature_columns_ is not None:
            names = list(self.feature_columns_)
        else:
            names = [col for col in df.columns if col not in NON_FEATURE_COLUMNS]
        missing = [col for col in names if col not in df.columns]
        if missing:
            raise ValueError(f"输入缺少特征列：{missing}")
        return names

    def _to_pandas(
        self, df: pl.DataFrame, feature_columns: Sequence[str], *, require_label: bool
    ) -> Any:
        """``df`` → AutoGluon 需要的 pandas DataFrame（唯一 pandas 边界）。"""
        if require_label and self.label not in df.columns:
            raise ValueError(f"训练/验证数据缺少标签列：{self.label!r}")
        columns = list(feature_columns)
        if require_label:
            columns.append(self.label)
        selected = df.select(*columns)
        if require_label:
            selected = selected.drop_nulls(self.label)
        return selected.to_pandas()

    # -- 训练 / 预测 --------------------------------------------------------

    def train(
        self,
        train_df: pl.DataFrame,
        *,
        valid_df: pl.DataFrame | None = None,
        time_limit: float | None = None,
    ) -> BaselineTrainer:
        """训练单个窗口，返回自身。

        ``valid_df`` 非空时作为 AutoGluon 的 ``tuning_data`` 传给 ``fit``，用于
        stacking 层与早停的模型选择；``time_limit`` 覆盖构造时的默认上限。
        """
        feature_columns = self._feature_names_for(train_df)
        self.feature_columns_ = feature_columns
        self.feature_missing_rate_ = missing_rate(train_df, tuple(feature_columns))

        train_pandas = self._to_pandas(
            train_df, feature_columns, require_label=True
        )
        valid_pandas = (
            self._to_pandas(valid_df, feature_columns, require_label=True)
            if valid_df is not None
            else None
        )
        limit = self.time_limit if time_limit is None else float(time_limit)

        predictor = TabularPredictor(
            label=self.label,
            problem_type=PROBLEM_TYPE,
            eval_metric=self.eval_metric,
            path=self.path,
            verbosity=self.verbosity,
        )
        fit_kwargs: dict[str, Any] = {
            "train_data": train_pandas,
            "tuning_data": valid_pandas,
            "time_limit": limit,
            "presets": self.presets,
            "num_gpus": self.num_gpus,
        }
        if self.hyperparameters is not None:
            fit_kwargs["hyperparameters"] = self.hyperparameters
        if self.num_bag_folds is not None:
            fit_kwargs["num_bag_folds"] = self.num_bag_folds
        if self.num_stack_levels is not None:
            fit_kwargs["num_stack_levels"] = self.num_stack_levels
        if self.hyperparameter_tune_kwargs is not None:
            fit_kwargs["hyperparameter_tune_kwargs"] = self.hyperparameter_tune_kwargs
        predictor.fit(**fit_kwargs)
        self.predictor = predictor
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """对 ``df`` 打分，返回 ``(date, instrument, score)``，行序与输入一致。"""
        predictor = self._require_fitted()
        feature_columns = self._feature_names_for(df)
        features_pandas = df.select(*feature_columns).to_pandas()
        scores = list(predictor.predict(features_pandas))
        return pl.DataFrame(
            {
                DATE_COL: df.get_column(DATE_COL),
                INSTRUMENT_COL: df.get_column(INSTRUMENT_COL),
                SCORE_COL: scores,
            }
        )

    # -- 持久化 -------------------------------------------------------------

    def save(self) -> str:
        """把 predictor 落盘到 ``self.path``，返回该路径。"""
        predictor = self._require_fitted()
        predictor.save()
        return self.path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        label: str = LABEL_COL,
        feature_columns: Sequence[str] | None = None,
        use_gpu: bool | None = None,
        verbosity: int = DEFAULT_VERBOSITY,
    ) -> BaselineTrainer:
        """从 ``path`` 加载已保存的 predictor，构造可直接预测的 trainer。

        ``feature_columns`` 省略时从 predictor 的特征元数据读取。
        """
        predictor = TabularPredictor.load(str(path))
        trainer = cls(
            label=label,
            feature_columns=feature_columns,
            path=str(path),
            use_gpu=use_gpu,
            verbosity=verbosity,
        )
        trainer.predictor = predictor
        if trainer.feature_columns_ is None:
            trainer.feature_columns_ = _feature_names_from(predictor)
        return trainer


def _feature_names_from(predictor: TabularPredictor) -> list[str] | None:
    """从已加载的 predictor 读取输入特征名，取不到时返回 None。"""
    metadata = getattr(predictor, "feature_metadata_in", None)
    names = getattr(metadata, "names", None)
    if names is None:
        return None
    return [str(name) for name in names]


__all__ = [
    "DEFAULT_EVAL_METRIC",
    "DEFAULT_HYPERPARAMETERS",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_PRESETS",
    "DEFAULT_TIME_LIMIT",
    "PROBLEM_TYPE",
    "SCORE_COL",
    "BaselineTrainer",
    "gpu_available",
    "resolve_num_gpus",
]
