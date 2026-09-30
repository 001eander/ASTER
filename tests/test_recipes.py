"""``quant.automl.recipes`` 与 trainer 配方旋钮的单元测试（issue #35 / #63）。"""
from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import quant.automl.trainer as trainer_module
from quant.automl.dataset import build_dataset
from quant.automl.recipes import (
    DEFAULT_RECIPE,
    RECIPES,
    RecipeError,
    get_recipe,
    recipe_kwargs,
)
from quant.automl.trainer import (
    DEFAULT_HYPERPARAMETERS,
    BaselineTrainer,
)
from quant.labels.open_to_open import attach_label
from quant.data.schema import DAILY_BARS
import datetime as dt


# ---------------------------------------------------------------------------
# 配方注册表
# ---------------------------------------------------------------------------


class TestRecipes:
    def test_default_recipe_is_memory_safe(self) -> None:
        assert DEFAULT_RECIPE == "memory_safe"
        assert DEFAULT_RECIPE in RECIPES

    def test_memory_safe_prunes_model_set(self) -> None:
        recipe = get_recipe("memory_safe")
        assert recipe.presets == "medium_quality"
        assert recipe.hyperparameters is not None
        assert set(recipe.hyperparameters) == {"GBM", "XGB", "CAT", "NN_TORCH"}
        # 16GB 下放不下的族被裁掉（issue #63）
        assert "RF" not in recipe.hyperparameters
        assert "XT" not in recipe.hyperparameters
        assert "FASTAI" not in recipe.hyperparameters

    def test_full_recipe_keeps_autogluon_default_set(self) -> None:
        assert get_recipe("full").hyperparameters is None

    def test_bagged_recipe_enables_bagging(self) -> None:
        assert get_recipe("bagged").presets == "good_quality"

    def test_hpo_recipe_has_tune_kwargs(self) -> None:
        tune = get_recipe("hpo").hyperparameter_tune_kwargs
        assert tune is not None and tune["num_trials"] > 0

    def test_unknown_recipe_raises(self) -> None:
        with pytest.raises(RecipeError, match="未知配方"):
            get_recipe("不存在")

    def test_recipe_kwargs_shape(self) -> None:
        kwargs = recipe_kwargs("memory_safe")
        assert kwargs["presets"] == "medium_quality"
        assert kwargs["hyperparameters"] == DEFAULT_HYPERPARAMETERS


# ---------------------------------------------------------------------------
# trainer 旋钮传递
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakePredictor:
    """记录 fit 参数的假 predictor。"""

    instances: list[_FakePredictor] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.fit_kwargs: dict[str, Any] | None = None
        self.feature_metadata_in = _FakeMeta(["f1"])
        _FakePredictor.instances.append(self)

    def fit(self, **kwargs: Any) -> _FakePredictor:
        self.fit_kwargs = kwargs
        return self

    def predict(self, data: Any) -> list[float]:
        return [0.0] * len(data)

    def save(self) -> None:
        return None


@pytest.fixture()
def fake_predictor(monkeypatch: pytest.MonkeyPatch) -> type[_FakePredictor]:
    _FakePredictor.instances = []
    monkeypatch.setattr(trainer_module, "TabularPredictor", _FakePredictor)
    return _FakePredictor


def _toy_dataset() -> pl.DataFrame:
    rows = []
    start = dt.date(2026, 1, 5)
    for i, instrument in enumerate(["600000.SH", "600001.SH", "600002.SH"]):
        for d in range(12):
            day = start + dt.timedelta(days=d)
            price = 10.0 + i + d
            rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "vwap": price,
                    "volume": 1000.0,
                    "amount": price * 1000.0,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    bars = pl.DataFrame(rows, schema=DAILY_BARS)

    def f1(data: pl.DataFrame) -> pl.DataFrame:
        return data.select("date", "instrument", pl.col("close").alias("value"))

    return build_dataset(bars, {"f1": f1})


class TestTrainerKnobs:
    def test_default_fit_uses_pruned_model_set(
        self, fake_predictor: type[_FakePredictor]
    ) -> None:
        trainer = BaselineTrainer(use_gpu=False)
        trainer.train(_toy_dataset())
        fit_kwargs = fake_predictor.instances[0].fit_kwargs
        assert fit_kwargs is not None
        assert fit_kwargs["hyperparameters"] == DEFAULT_HYPERPARAMETERS
        assert "num_bag_folds" not in fit_kwargs
        assert "num_stack_levels" not in fit_kwargs
        assert "hyperparameter_tune_kwargs" not in fit_kwargs

    def test_explicit_none_restores_full_set(
        self, fake_predictor: type[_FakePredictor]
    ) -> None:
        trainer = BaselineTrainer(use_gpu=False, hyperparameters=None)
        trainer.train(_toy_dataset())
        fit_kwargs = fake_predictor.instances[0].fit_kwargs
        assert fit_kwargs is not None
        assert "hyperparameters" not in fit_kwargs

    def test_bag_stack_hpo_passed_through(
        self, fake_predictor: type[_FakePredictor]
    ) -> None:
        trainer = BaselineTrainer(
            use_gpu=False,
            num_bag_folds=5,
            num_stack_levels=1,
            hyperparameter_tune_kwargs={"num_trials": 3},
        )
        trainer.train(_toy_dataset())
        fit_kwargs = fake_predictor.instances[0].fit_kwargs
        assert fit_kwargs is not None
        assert fit_kwargs["num_bag_folds"] == 5
        assert fit_kwargs["num_stack_levels"] == 1
        assert fit_kwargs["hyperparameter_tune_kwargs"] == {"num_trials": 3}

    def test_recipe_overrides_knobs(
        self, fake_predictor: type[_FakePredictor]
    ) -> None:
        trainer = BaselineTrainer(
            use_gpu=False, presets="medium_quality", recipe="bagged"
        )
        assert trainer.presets == "good_quality"
        trainer.train(_toy_dataset())
        fit_kwargs = fake_predictor.instances[0].fit_kwargs
        assert fit_kwargs is not None
        assert fit_kwargs["presets"] == "good_quality"
        assert fit_kwargs["hyperparameters"] == DEFAULT_HYPERPARAMETERS

    def test_unknown_recipe_raises(self) -> None:
        with pytest.raises(RecipeError, match="未知配方"):
            BaselineTrainer(use_gpu=False, recipe="不存在")
