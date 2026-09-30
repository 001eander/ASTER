"""AutoGluon 训练配方库（issue #35）。

配方结论
--------
配方调优的结论是**沉淀默认值**，而非堆叠选项。issue #63 的实跑证据（全量
训练 2021-09 ~ 2024-12，4M 行 × 12 特征，time_limit=1800s，16GB 内存）：

- RandomForest / ExtraTrees 内存估算 ~20GB，被 AutoGluon 整族跳过；
- NeuralNetFastAI 差 11% 内存余量被跳过；
- LightGBMXT 吃掉 71% 预算，验证分反而低于 131s 跑完的 LightGBM；
- 最终 WeightedEnsemble 只剩 LightGBM + XGBoost，多样性受损。

因此默认配方 ``memory_safe`` 把模型集精简到 GBM / XGB / CAT / NN_TORCH
四个族（``medium_quality`` 预设下它们都能在 16GB 内跑完），把预算让给
有效模型。其余配方保留为对照与升级路径：

- ``full``：AutoGluon 全集（对照组，复现 #63 的 OOM 跳过行为）；
- ``bagged``：``good_quality`` 预设（bagging 5 折 + 1 层 stacking）+
  精简模型集，预算充足时换质量；
- ``hpo``：精简模型集 + AutoGluon 内置 HPO（10 次试验），验证超参收益。

用法::

    trainer = BaselineTrainer(recipe="bagged", path=...)
    # 或显式展开
    BaselineTrainer(**recipe_kwargs("hpo"), path=...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quant.automl.trainer import DEFAULT_HYPERPARAMETERS

# ---------------------------------------------------------------------------
# 配方
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """一组训练旋钮的组合。

    ``hyperparameters=None`` 表示 AutoGluon 全集（不做模型集裁剪）。
    其余 ``None`` 字段表示不覆盖 AutoGluon 预设自身的默认。
    """

    presets: str
    hyperparameters: dict[str, Any] | str | None
    num_bag_folds: int | None = None
    num_stack_levels: int | None = None
    hyperparameter_tune_kwargs: dict[str, Any] | None = None
    description: str = ""


#: 命名配方注册表。
RECIPES: dict[str, Recipe] = {
    "memory_safe": Recipe(
        presets="medium_quality",
        hyperparameters=dict(DEFAULT_HYPERPARAMETERS),
        description=(
            "默认。精简模型集（GBM/XGB/CAT/NN_TORCH），16GB 内存下不再整族 OOM"
        ),
    ),
    "full": Recipe(
        presets="medium_quality",
        hyperparameters=None,
        description="对照组：AutoGluon 全集，复现 issue #63 的 OOM 跳过",
    ),
    "bagged": Recipe(
        presets="good_quality",
        hyperparameters=dict(DEFAULT_HYPERPARAMETERS),
        description="bagging 5 折 + 1 层 stacking + 精简模型集，预算充足时换质量",
    ),
    "hpo": Recipe(
        presets="medium_quality",
        hyperparameters=dict(DEFAULT_HYPERPARAMETERS),
        hyperparameter_tune_kwargs={
            "num_trials": 10,
            "scheduler": "local",
            "searcher": "auto",
        },
        description="精简模型集 + 内置 HPO（10 次试验），验证超参收益",
    ),
}

#: 默认配方名。
DEFAULT_RECIPE: str = "memory_safe"


class RecipeError(ValueError):
    """配方名不存在。"""


def get_recipe(name: str) -> Recipe:
    """按名取配方；不存在时抛 :class:`RecipeError` 并列出可选名。"""
    try:
        return RECIPES[name]
    except KeyError as exc:
        raise RecipeError(
            f"未知配方 {name!r}；可选：{sorted(RECIPES)}"
        ) from exc


def recipe_kwargs(name: str) -> dict[str, Any]:
    """把命名配方展开为 :class:`BaselineTrainer` 的构造参数。"""
    found = get_recipe(name)
    return {
        "presets": found.presets,
        "hyperparameters": found.hyperparameters,
        "num_bag_folds": found.num_bag_folds,
        "num_stack_levels": found.num_stack_levels,
        "hyperparameter_tune_kwargs": found.hyperparameter_tune_kwargs,
    }


__all__ = [
    "DEFAULT_RECIPE",
    "RECIPES",
    "Recipe",
    "RecipeError",
    "get_recipe",
    "recipe_kwargs",
]
