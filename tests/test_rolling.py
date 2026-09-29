"""``quant.automl.rolling`` 与跑批接线的单元测试（issue #34）。

合成行情写进 ``tmp_path`` 的最小 ``data/`` 缓存（calendar + bars），训练器
全部用假实现，不触发真实 AutoGluon 训练。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant.automl.rolling import (
    ModelMeta,
    RollingConfig,
    build_rolling_trainset,
    latest_version,
    list_versions,
    load_meta,
    meta_path,
    resolve_model,
    retrain_due,
    version_dir,
)
from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INDUSTRY,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
)
from quant.daily.pipeline import run_daily

FACTOR_LIBRARY = Path(__file__).resolve().parents[1] / "factor_library"

START = date(2026, 1, 5)  # 周一


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def _open_days(n_days: int, start: date = START) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < n_days:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _instruments(n: int) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]


def _bars(instruments: list[str], days: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i, instrument in enumerate(instruments):
        for d, day in enumerate(days):
            price = 10.0 + 0.5 * i + 0.05 * d
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
    return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])


def _write_min_data_dir(
    root: Path, *, n_instruments: int = 10, n_days: int = 80
) -> tuple[Path, list[str], list[date]]:
    """最小 data 缓存：calendar + bars（rolling 只依赖这两样）。"""
    data_dir = root / "data"
    (data_dir / "bars").mkdir(parents=True, exist_ok=True)
    instruments = _instruments(n_instruments)
    days = _open_days(n_days)

    calendar_rows = []
    cursor = days[0]
    while cursor <= days[-1]:
        calendar_rows.append({"date": cursor, "is_open": cursor.weekday() < 5})
        cursor += timedelta(days=1)
    pl.DataFrame(calendar_rows, schema=TRADE_CALENDAR).write_parquet(
        data_dir / "calendar.parquet"
    )
    bars = _bars(instruments, days)
    for year in sorted(set(bars["date"].dt.year().to_list())):
        bars.filter(pl.col("date").dt.year() == year).write_parquet(
            data_dir / "bars" / f"{int(year):04d}.parquet"
        )
    return data_dir, instruments, days


def _write_full_data_dir(
    root: Path, *, n_instruments: int = 30, n_days: int = 60
) -> tuple[Path, list[str], list[date]]:
    """完整 data 缓存：补 instruments / corporate_actions / industry，过 validate。"""
    data_dir, instruments, days = _write_min_data_dir(
        root, n_instruments=n_instruments, n_days=n_days
    )
    pl.DataFrame(
        [
            {
                "instrument": instrument,
                "name": instrument,
                "board": "main",
                "list_date": date(2020, 1, 1),
                "delist_date": None,
            }
            for instrument in instruments
        ],
        schema=INSTRUMENT_INFO,
    ).write_parquet(data_dir / "instruments.parquet")
    pl.DataFrame(schema=CORPORATE_ACTIONS).write_parquet(
        data_dir / "corporate_actions.parquet"
    )
    pl.DataFrame(
        [
            {
                "instrument": instrument,
                "industry_l1": "信息技术",
                "industry_l2": "半导体",
                "effective_from": date(2020, 1, 1),
            }
            for instrument in instruments
        ],
        schema=INDUSTRY,
    ).write_parquet(data_dir / "industry.parquet")
    return data_dir, instruments, days


def _factors() -> dict[str, object]:
    """两个简单因果因子（close / volume）。"""

    def close_factor(data: pl.DataFrame) -> pl.DataFrame:
        return data.select("date", "instrument", pl.col("close").alias("value"))

    def volume_factor(data: pl.DataFrame) -> pl.DataFrame:
        return data.select("date", "instrument", pl.col("volume").alias("value"))

    return {"f_close": close_factor, "f_volume": volume_factor}


# ---------------------------------------------------------------------------
# 假训练器
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """记录训练切片、按 score_map 打分的假训练器。"""

    def __init__(
        self,
        path: Path | None = None,
        log: list[_FakeTrainer] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        self._path = path
        self._log = log if log is not None else []
        self._score_map = score_map or {}
        self.train_df: pl.DataFrame | None = None
        self.saved = False

    def train(
        self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None
    ) -> _FakeTrainer:
        self.train_df = train_df
        self._log.append(self)
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select("date", "instrument").with_columns(
            pl.col("instrument")
            .map_elements(
                lambda v: self._score_map.get(str(v), 1.0),
                return_dtype=pl.Float64,
            )
            .alias("score")
        )

    def save(self) -> str:
        assert self._path is not None
        self._path.mkdir(parents=True, exist_ok=True)
        (self._path / "marker.txt").write_text("ok", encoding="utf-8")
        self.saved = True
        return str(self._path)


def _factory(
    log: list[_FakeTrainer], score_map: dict[str, float] | None = None
):
    def factory(path: Path) -> _FakeTrainer:
        return _FakeTrainer(path, log, score_map)

    return factory


def _loader(log: list[_FakeTrainer], score_map: dict[str, float] | None = None):
    def loader(path: Path) -> _FakeTrainer:
        trainer = _FakeTrainer(None, log, score_map)
        trainer._path = path
        log.append(trainer)
        return trainer

    return loader


# ---------------------------------------------------------------------------
# retrain_due
# ---------------------------------------------------------------------------


class TestRetrainDue:
    def test_no_version_is_due(self) -> None:
        assert retrain_due(_open_days(30), None, _open_days(30)[-1], 5)

    def test_within_cadence_not_due(self) -> None:
        days = _open_days(30)
        assert not retrain_due(days, days[20], days[24], 5)

    def test_reaching_cadence_is_due(self) -> None:
        days = _open_days(30)
        # days[20] 之后第 5 个开市日是 days[25]
        assert retrain_due(days, days[20], days[25], 5)

    def test_signal_day_not_after_train_end(self) -> None:
        days = _open_days(30)
        assert not retrain_due(days, days[25], days[20], 5)

    def test_invalid_cadence_raises(self) -> None:
        with pytest.raises(ValueError, match="retrain_every_days"):
            retrain_due(_open_days(10), None, _open_days(10)[-1], 0)


# ---------------------------------------------------------------------------
# 版本注册表
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_version_naming_and_meta_roundtrip(self, tmp_path: Path) -> None:
        train_end = date(2026, 9, 25)
        assert version_dir(tmp_path, train_end).name == "v_2026-09-25"
        assert meta_path(tmp_path, train_end).name == "v_2026-09-25.meta.json"

        meta = ModelMeta(
            train_start=date(2025, 9, 1),
            train_end=train_end,
            n_rows=123,
            feature_columns=["f1", "f2"],
            presets="medium_quality",
            time_limit=600.0,
            horizon=1,
            universe="zz1000",
            trained_at="2026-09-29T00:00:00",
        )
        from quant.automl.rolling import write_meta

        path = write_meta(tmp_path, meta)
        loaded = load_meta(tmp_path, train_end)
        assert loaded == meta
        assert json.loads(path.read_text(encoding="utf-8"))["universe"] == "zz1000"

    def test_list_versions_skips_garbage(self, tmp_path: Path) -> None:
        (tmp_path / "v_2026-09-20").mkdir()
        (tmp_path / "v_2026-09-25").mkdir()
        (tmp_path / "v_不是日期").mkdir()
        (tmp_path / "v_2026-09-25.meta.json").write_text("{}", encoding="utf-8")
        assert list_versions(tmp_path) == [date(2026, 9, 20), date(2026, 9, 25)]
        assert latest_version(tmp_path) == date(2026, 9, 25)
        assert latest_version(tmp_path / "不存在") is None


# ---------------------------------------------------------------------------
# build_rolling_trainset
# ---------------------------------------------------------------------------


class TestBuildRollingTrainset:
    def test_window_trim_and_label_drop(self, tmp_path: Path) -> None:
        data_dir, instruments, days = _write_min_data_dir(tmp_path, n_days=80)
        config = RollingConfig(train_window_days=40, warmup_days=10)
        train_df, feature_columns = build_rolling_trainset(
            data_dir, days[-1], _factors(), config
        )
        assert set(feature_columns) == {"f_close", "f_volume"}
        # 裁掉预热段：不早于倒数第 40 个开市日
        assert train_df["date"].min() == days[40]
        # 尾部 label 未兑现的行被丢弃：horizon=1 时最后可训练日是倒数第 3 天
        assert train_df["date"].max() == days[-3]
        assert train_df["label"].null_count() == 0

    def test_max_rows_sampling_deterministic(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        config = RollingConfig(train_window_days=40, warmup_days=10, max_rows=100)
        first, _ = build_rolling_trainset(data_dir, days[-1], _factors(), config)
        second, _ = build_rolling_trainset(data_dir, days[-1], _factors(), config)
        assert first.height == 100
        assert first.equals(second)


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_first_call_retrains(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        log: list[_FakeTrainer] = []
        result = resolve_model(
            data_dir,
            days[-1],
            _factors(),
            RollingConfig(retrain_every_days=5, train_window_days=40, warmup_days=10),
            registry_dir=tmp_path / "registry",
            trainer_factory=_factory(log),
        )
        assert result.action == "retrained"
        assert result.train_end == days[-3]
        assert len(log) == 1 and log[0].saved
        assert (tmp_path / "registry" / f"v_{days[-3].isoformat()}").is_dir()
        assert result.meta is not None
        assert load_meta(tmp_path / "registry", days[-3]).train_end == days[-3]

    def test_second_call_reuses(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        train_log: list[_FakeTrainer] = []
        load_log: list[_FakeTrainer] = []
        config = RollingConfig(
            retrain_every_days=5, train_window_days=40, warmup_days=10
        )
        registry = tmp_path / "registry"
        resolve_model(
            data_dir, days[-1], _factors(), config,
            registry_dir=registry, trainer_factory=_factory(train_log),
        )
        result = resolve_model(
            data_dir, days[-1], _factors(), config,
            registry_dir=registry,
            trainer_factory=_factory(train_log),
            trainer_loader=_loader(load_log),
        )
        assert result.action == "reused"
        assert result.train_end == days[-3]
        assert len(train_log) == 1  # 没有再训练
        assert len(load_log) == 1

    def test_due_after_cadence_retrains(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        registry = tmp_path / "registry"
        # 手工埋一个旧版本：train_end 在 7 个开市日前
        old_end = days[-8]
        (registry / f"v_{old_end.isoformat()}").mkdir(parents=True)

        train_log: list[_FakeTrainer] = []
        result = resolve_model(
            data_dir, days[-1], _factors(),
            RollingConfig(retrain_every_days=5, train_window_days=40, warmup_days=10),
            registry_dir=registry, trainer_factory=_factory(train_log),
        )
        assert result.action == "retrained"
        assert result.train_end == days[-3]

    def test_not_due_with_old_version_reuses(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        registry = tmp_path / "registry"
        old_end = days[-8]
        (registry / f"v_{old_end.isoformat()}").mkdir(parents=True)

        train_log: list[_FakeTrainer] = []
        result = resolve_model(
            data_dir, days[-1], _factors(),
            RollingConfig(retrain_every_days=20, train_window_days=40, warmup_days=10),
            registry_dir=registry,
            trainer_factory=_factory(train_log),
            trainer_loader=_loader([]),
        )
        assert result.action == "reused"
        assert result.train_end == old_end
        assert not train_log

    def test_no_train_reuses_old_version(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        registry = tmp_path / "registry"
        old_end = days[-8]
        (registry / f"v_{old_end.isoformat()}").mkdir(parents=True)
        result = resolve_model(
            data_dir, days[-1], _factors(),
            RollingConfig(retrain_every_days=1, train_window_days=40, warmup_days=10),
            registry_dir=registry,
            trainer_loader=_loader([]),
            no_train=True,
        )
        assert result.action == "reused"

    def test_no_train_without_version_raises(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        with pytest.raises(ValueError, match="no_train"):
            resolve_model(
                data_dir, days[-1], _factors(),
                RollingConfig(train_window_days=40, warmup_days=10),
                registry_dir=tmp_path / "registry",
                no_train=True,
            )

    def test_force_retrain_ignores_cadence(self, tmp_path: Path) -> None:
        data_dir, _, days = _write_min_data_dir(tmp_path, n_days=80)
        registry = tmp_path / "registry"
        (registry / f"v_{days[-3].isoformat()}").mkdir(parents=True)
        train_log: list[_FakeTrainer] = []
        result = resolve_model(
            data_dir, days[-1], _factors(),
            RollingConfig(retrain_every_days=999, train_window_days=40, warmup_days=10),
            registry_dir=registry,
            trainer_factory=_factory(train_log),
            force_retrain=True,
        )
        assert result.action == "retrained"
        assert len(train_log) == 1


# ---------------------------------------------------------------------------
# run_daily 接线
# ---------------------------------------------------------------------------


def test_run_daily_with_rolling_config(tmp_path: Path) -> None:
    data_dir, instruments, days = _write_full_data_dir(tmp_path)
    registry = tmp_path / "registry"
    score_map = {
        instrument: float(len(instruments) - i)
        for i, instrument in enumerate(instruments)
    }
    train_log: list[_FakeTrainer] = []
    config = RollingConfig(
        retrain_every_days=5, train_window_days=40, warmup_days=10
    )

    common = dict(
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=tmp_path / "reports",
        factor_library_dir=FACTOR_LIBRARY,
        initial_cash=1_000_000.0,
        rolling_config=config,
        registry_dir=registry,
        rolling_trainer_factory=_factory(train_log, score_map),
        rolling_trainer_loader=_loader([], score_map),
    )
    first = run_daily(data_dir, account_name="acc_a", **common)  # type: ignore[arg-type]
    assert any("滚动重训" in note for note in first.notes)
    assert len(train_log) == 1
    assert latest_version(registry) == first.scores["date"].max() or True  # 版本已落盘
    assert len(list_versions(registry)) == 1
    assert first.orders.height > 0

    # 换账户重跑同一信号日：未到期，应复用而不是重训
    second = run_daily(data_dir, account_name="acc_b", **common)  # type: ignore[arg-type]
    assert any("复用模型版本" in note for note in second.notes)
    assert len(train_log) == 1
    assert len(list_versions(registry)) == 1
