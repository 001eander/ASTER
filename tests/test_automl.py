"""``quant.automl`` 单元测试。

覆盖范围：

- ``build_dataset`` 拼接正确性、列名、标签行日期对齐、尾部 null 不丢行
- 逐日截面 z-score（均值 ≈ 0、样本 std ≈ 1）
- ``inf → null`` 与缺测率统计
- ``BaselineTrainer`` 的数据流转、GPU 参数、save/load（monkeypatch 掉 TabularPredictor）
- 真实训练的冒烟用例标记 ``slow``，默认由 pyproject 的 ``-m "not slow"`` 跳过
"""
from __future__ import annotations

import datetime as dt
import random
from typing import Any

import polars as pl
import pytest

from quant.automl import trainer as trainer_module
from quant.automl.dataset import build_dataset, missing_rate
from quant.automl.trainer import BaselineTrainer, gpu_available, resolve_num_gpus
from quant.data.schema import DAILY_BARS

DAY0 = dt.date(2026, 1, 5)  # 周一
A = "600000.SH"
B = "000001.SZ"
C = "300750.SZ"


# ---------------------------------------------------------------------------
# 合成数据与玩具因子
# ---------------------------------------------------------------------------


def _bars(
    instruments: list[str],
    n_days: int,
    price_fn: Any,
) -> pl.DataFrame:
    """由 ``price_fn(i, d)`` 生成完整 DAILY_BARS，open=close=price、adjfactor=1。"""
    rows = [
        {
            "date": DAY0 + dt.timedelta(days=d),
            "instrument": ins,
            "open": float(price_fn(i, d)),
            "high": float(price_fn(i, d)),
            "low": float(price_fn(i, d)),
            "close": float(price_fn(i, d)),
            "vwap": float(price_fn(i, d)),
            "volume": 100.0 * (i + 1) + d,
            "amount": float(price_fn(i, d)),
            "adjfactor": 1.0,
            "limit_up": None,
            "limit_down": None,
        }
        for i, ins in enumerate(instruments)
        for d in range(n_days)
    ]
    return pl.DataFrame(rows, schema=DAILY_BARS)


def _factor_open(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date", "instrument", pl.col("open").cast(pl.Float64).alias("value")
    )


def _factor_volume(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date", "instrument", pl.col("volume").cast(pl.Float64).alias("value")
    )


def _factor_inf(data: pl.DataFrame) -> pl.DataFrame:
    """当日开盘价最高的那只记为 +inf，验证 inf → null。"""
    return data.select(
        "date",
        "instrument",
        pl.when(pl.col("open") == pl.col("open").max().over("date"))
        .then(pl.lit(float("inf")))
        .otherwise(pl.col("open"))
        .alias("value"),
    )


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_shape_columns_and_label_alignment() -> None:
    """3 票 × 30 天 → 90 行；标签对齐 T+1 开盘 → T+2 开盘。"""
    bars = _bars([A, B, C], 30, lambda i, d: 10.0 + i + d)
    ds = build_dataset(bars, {"price": _factor_open, "volume": _factor_volume})

    assert ds.height == 90
    assert ds.columns == ["date", "instrument", "price", "volume", "label", "delay_days"]

    # 信号日 T = 2026-01-05（d=0）：open 为 [10, 11, 12]，截面 z-score 后 A 为 -1
    row = ds.filter((pl.col("instrument") == A) & (pl.col("date") == DAY0)).row(0, named=True)
    assert row["price"] == pytest.approx(-1.0)
    assert row["label"] == pytest.approx(12.0 / 11.0 - 1.0)

    # 尾部 horizon+1 天没有未来行情，label 为 null，但行保留
    tail = ds.filter(
        (pl.col("instrument") == A) & (pl.col("date") >= DAY0 + dt.timedelta(days=29))
    )
    assert tail.height == 1
    assert tail["label"][0] is None


def test_build_dataset_does_not_drop_tail_rows() -> None:
    """标签为 null 的尾部行不被丢弃。"""
    bars = _bars([A], 5, lambda i, d: 10.0 + d)
    ds = build_dataset(bars, {"price": _factor_open})
    assert ds.height == 5
    assert ds.filter(pl.col("label").is_null()).height == 2  # D4, D5


def test_build_dataset_rejects_reserved_factor_name() -> None:
    bars = _bars([A], 3, lambda i, d: 10.0 + d)
    with pytest.raises(ValueError, match="保留列"):
        build_dataset(bars, {"label": _factor_open})


def test_build_dataset_rejects_bad_factor_output() -> None:
    bars = _bars([A], 3, lambda i, d: 10.0 + d)
    with pytest.raises(ValueError, match="输出缺少列"):
        build_dataset(bars, {"bad": lambda data: data.select("date", "instrument")})


# ---------------------------------------------------------------------------
# 截面标准化
# ---------------------------------------------------------------------------


def test_cross_section_zscore_mean_zero_std_one() -> None:
    """单日截面标准化后均值 ≈ 0、样本 std ≈ 1。"""
    bars = _bars([A, B, C, "600519.SH", "601318.SH"], 1, lambda i, d: 10.0 + 2 * i)
    ds = build_dataset(bars, {"price": _factor_open})

    values = ds["price"]
    assert values.null_count() == 0
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(values.std()) == pytest.approx(1.0, rel=1e-12)


def test_cross_section_zscore_is_per_day() -> None:
    """不同交易日的截面分别标准化，互不影响。"""
    bars = _bars([A, B, C], 2, lambda i, d: 10.0 + 2 * i + d)
    ds = build_dataset(bars, {"price": _factor_open})
    for day in ds["date"].unique().to_list():
        values = ds.filter(pl.col("date") == day)["price"]
        assert float(values.mean()) == pytest.approx(0.0, abs=1e-12)
        assert float(values.std()) == pytest.approx(1.0, rel=1e-9)


def test_cross_section_zscore_all_null_when_no_variance() -> None:
    """截面无方差（常数）→ std=0 → 标准化值全部为 null。"""
    bars = _bars([A, B, C], 1, lambda i, d: 10.0)
    ds = build_dataset(bars, {"price": _factor_open})
    assert ds["price"].null_count() == 3


def test_inf_becomes_null() -> None:
    """因子输出 ±inf 先转 null，且不污染截面统计。"""
    bars = _bars([A, B, C, "600519.SH", "601318.SH"], 1, lambda i, d: 10.0 + 2 * i)
    ds = build_dataset(bars, {"price": _factor_inf})

    top = ds.filter(pl.col("instrument") == "601318.SH")
    assert top["price"][0] is None
    rest = ds.filter(pl.col("price").is_not_null())["price"]
    assert rest.len() == 4
    assert float(rest.mean()) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# missing_rate
# ---------------------------------------------------------------------------


def test_missing_rate_counts_inf_null() -> None:
    bars = _bars([A, B, C, "600519.SH", "601318.SH"], 1, lambda i, d: 10.0 + 2 * i)
    ds = build_dataset(bars, {"price": _factor_inf, "volume": _factor_volume})
    report = missing_rate(ds, ("price", "volume"))

    assert report.columns == ["feature", "n", "n_missing", "missing_rate"]
    by_feature = {row["feature"]: row for row in report.to_dicts()}
    assert by_feature["price"]["n_missing"] == 1
    assert by_feature["price"]["missing_rate"] == pytest.approx(0.2)
    assert by_feature["volume"]["n_missing"] == 0
    assert by_feature["volume"]["missing_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 训练器（mock AutoGluon）
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakePredictor:
    """记录 AutoGluon 调用参数的假 predictor。"""

    instances: list[_FakePredictor] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.fit_kwargs: dict[str, Any] | None = None
        self.saved = False
        self.path = kwargs.get("path")
        self.feature_metadata_in = _FakeMeta(["price", "volume"])
        _FakePredictor.instances.append(self)

    def fit(self, **kwargs: Any) -> _FakePredictor:
        self.fit_kwargs = kwargs
        return self

    def predict(self, data: Any) -> list[float]:
        return [0.25] * len(data)

    def save(self) -> None:
        self.saved = True

    @classmethod
    def load(cls, path: str) -> _FakePredictor:
        obj = cls(path=path)
        obj.loaded_from = path
        return obj


@pytest.fixture()
def fake_predictor(monkeypatch: pytest.MonkeyPatch) -> type[_FakePredictor]:
    _FakePredictor.instances = []
    monkeypatch.setattr(trainer_module, "TabularPredictor", _FakePredictor)
    return _FakePredictor


def _toy_dataset(n_days: int = 10) -> pl.DataFrame:
    bars = _bars([A, B, C], n_days, lambda i, d: 10.0 + i + d)
    return build_dataset(bars, {"price": _factor_open, "volume": _factor_volume})


def test_trainer_train_predict_data_flow(fake_predictor: type[_FakePredictor]) -> None:
    ds = _toy_dataset()
    trainer = BaselineTrainer(
        feature_columns=["price", "volume"],
        path="runs/test-window",
        use_gpu=False,
    )
    assert trainer.num_gpus == 0

    trainer.train(ds, time_limit=7.0)

    fake = fake_predictor.instances[-1]
    assert fake.init_kwargs["label"] == "label"
    assert fake.init_kwargs["problem_type"] == "regression"
    assert fake.init_kwargs["path"] == "runs/test-window"

    fit_kwargs = fake.fit_kwargs
    assert fit_kwargs is not None
    assert fit_kwargs["num_gpus"] == 0
    assert fit_kwargs["time_limit"] == 7.0
    assert fit_kwargs["presets"] == "medium_quality"
    assert fit_kwargs["tuning_data"] is None
    # 训练集只保留 label 非 null 的行，特征 + label 列
    assert len(fit_kwargs["train_data"]) == ds.filter(pl.col("label").is_not_null()).height
    assert list(fit_kwargs["train_data"].columns) == ["price", "volume", "label"]

    scores = trainer.predict(ds)
    assert scores.columns == ["date", "instrument", "score"]
    assert scores.height == ds.height
    assert scores["score"].to_list() == [0.25] * ds.height


def test_trainer_passes_valid_df_as_tuning_data(
    fake_predictor: type[_FakePredictor],
) -> None:
    ds = _toy_dataset()
    trainer = BaselineTrainer(feature_columns=["price", "volume"], use_gpu=False)
    trainer.train(ds, valid_df=ds, time_limit=3.0)
    fake = fake_predictor.instances[-1]
    assert fake.fit_kwargs is not None
    assert fake.fit_kwargs["tuning_data"] is not None
    assert list(fake.fit_kwargs["tuning_data"].columns) == ["price", "volume", "label"]


def test_trainer_gpu_flag_and_auto_detection(
    fake_predictor: type[_FakePredictor],
) -> None:
    ds = _toy_dataset()
    forced = BaselineTrainer(feature_columns=["price", "volume"], use_gpu=True)
    assert forced.num_gpus == 1
    forced.train(ds, time_limit=1.0)
    assert fake_predictor.instances[-1].fit_kwargs is not None
    assert fake_predictor.instances[-1].fit_kwargs["num_gpus"] == 1

    auto = BaselineTrainer(feature_columns=["price", "volume"], use_gpu=None)
    assert auto.num_gpus == (1 if gpu_available() else 0)


def test_trainer_infers_feature_columns(fake_predictor: type[_FakePredictor]) -> None:
    ds = _toy_dataset()
    trainer = BaselineTrainer(use_gpu=False)
    trainer.train(ds, time_limit=1.0)
    assert trainer.feature_columns_ == ["price", "volume"]
    assert trainer.feature_missing_rate_ is not None


def test_trainer_save_and_load(fake_predictor: type[_FakePredictor]) -> None:
    ds = _toy_dataset()
    trainer = BaselineTrainer(feature_columns=["price", "volume"], path="runs/save-me", use_gpu=False)
    trainer.train(ds, time_limit=1.0)

    assert trainer.save() == "runs/save-me"
    assert fake_predictor.instances[-1].saved is True

    loaded = BaselineTrainer.load("runs/save-me", feature_columns=["price", "volume"])
    assert loaded.is_fitted
    assert loaded.feature_columns_ == ["price", "volume"]
    assert loaded.predict(ds).height == ds.height

    # 不传 feature_columns 时从 predictor 元数据读取
    loaded_auto = BaselineTrainer.load("runs/save-me")
    assert loaded_auto.feature_columns_ == ["price", "volume"]


def test_trainer_predict_before_fit_raises() -> None:
    trainer = BaselineTrainer(feature_columns=["price"], use_gpu=False)
    with pytest.raises(RuntimeError, match="尚未训练"):
        trainer.predict(_toy_dataset())


def test_trainer_missing_feature_column_raises(
    fake_predictor: type[_FakePredictor],
) -> None:
    ds = _toy_dataset()
    trainer = BaselineTrainer(feature_columns=["nope"], use_gpu=False)
    with pytest.raises(ValueError, match="缺少特征列"):
        trainer.train(ds, time_limit=1.0)


def test_resolve_num_gpus_matches_detection() -> None:
    assert resolve_num_gpus(True) == 1
    assert resolve_num_gpus(False) == 0
    assert resolve_num_gpus(None) == (1 if gpu_available() else 0)


# ---------------------------------------------------------------------------
# 真实训练冒烟（默认跳过）
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_autogluon_smoke(tmp_path: Any) -> None:
    """100+ 行玩具数据 + 10s 限时跑一次真实 AutoGluon 回归训练。"""
    rng = random.Random(20260105)
    n = 120
    rows = []
    for i in range(n):
        f1 = rng.gauss(0.0, 1.0)
        f2 = rng.gauss(0.0, 1.0)
        rows.append(
            {
                "date": DAY0 + dt.timedelta(days=i // 3),
                "instrument": [A, B, C][i % 3],
                "f1": f1,
                "f2": f2,
                "label": 0.5 * f1 - 0.3 * f2 + rng.gauss(0.0, 0.1),
                "delay_days": 1,
            }
        )
    ds = pl.DataFrame(rows)
    train_df = ds.head(100)
    test_df = ds.tail(20)

    trainer = BaselineTrainer(
        feature_columns=["f1", "f2"],
        path=str(tmp_path / "automl-smoke"),
        use_gpu=None,
        verbosity=1,
    )
    trainer.train(train_df, time_limit=10.0)
    scores = trainer.predict(test_df)

    assert scores.columns == ["date", "instrument", "score"]
    assert scores.height == test_df.height
    assert scores["score"].null_count() == 0
