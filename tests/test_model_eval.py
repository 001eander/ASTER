"""``quant.eval.model`` 的单元测试：walk-forward 窗口切分与模型级评估。

全部使用合成数据与假训练器，不触发真实 AutoGluon 训练。合成面板的
``label`` 与特征 ``f1`` 强相关，假训练器直接以 ``f1`` 打分，样本外 RankIC
应显著为正。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random

import polars as pl
import pytest

from quant.eval.model import (
    ModelEvaluation,
    WalkForwardConfig,
    evaluate_walk_forward,
    split_windows,
)

# ---------------------------------------------------------------------------
# 合成数据与假训练器
# ---------------------------------------------------------------------------

N_INSTRUMENTS: int = 40
N_DAYS: int = 80
START: dt.date = dt.date(2021, 1, 4)
#: 标签对 f1 的暴露，保证 score=f1 时 RankIC 显著为正。
SIGNAL_LOADING: float = 0.8


def _dates(n: int = N_DAYS) -> list[dt.date]:
    return [START + dt.timedelta(days=i) for i in range(n)]


def _dataset(n_instruments: int = N_INSTRUMENTS, n_days: int = N_DAYS) -> pl.DataFrame:
    rng = random.Random(20260929)
    rows: list[dict[str, object]] = []
    for day in _dates(n_days):
        for i in range(n_instruments):
            f1 = rng.gauss(0.0, 1.0)
            f2 = rng.gauss(0.0, 1.0)
            label = SIGNAL_LOADING * f1 + math.sqrt(1 - SIGNAL_LOADING**2) * rng.gauss(
                0.0, 1.0
            )
            rows.append(
                {
                    "date": day,
                    "instrument": f"sz{i:06d}",
                    "f1": f1,
                    "f2": f2,
                    "label": label,
                    "delay_days": 1,
                }
            )
    return pl.DataFrame(rows).sort("instrument", "date")


class _FakeTrainer:
    """记录训练切片信息、以 f1 打分的假训练器。"""

    def __init__(self, log: list[_FakeTrainer]) -> None:
        self._log = log
        self.train_max_date: dt.date | None = None
        self.train_min_date: dt.date | None = None
        self.n_train_rows = 0

    def train(
        self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None
    ) -> _FakeTrainer:
        self.train_max_date = train_df["date"].max()
        self.train_min_date = train_df["date"].min()
        self.n_train_rows = train_df.height
        self._log.append(self)
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select("date", "instrument", pl.col("f1").alias("score"))


class _OutOfWindowTrainer(_FakeTrainer):
    """打分越出测试区间的坏训练器。"""

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        out = super().predict(df)
        return out.with_columns(pl.col("date") + dt.timedelta(days=365))


def _fake_importance(trainer: _FakeTrainer, train_df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["f1", "f2"],
            "importance": [1.0 + 0.01 * len(trainer._log), 0.5],
        }
    )


def _fake_leaderboard(trainer: _FakeTrainer) -> str:
    return f"rows={trainer.n_train_rows}"


def _run_eval(
    dataset: pl.DataFrame | None = None,
    config: WalkForwardConfig | None = None,
    **kwargs: object,
) -> tuple[ModelEvaluation, list[_FakeTrainer]]:
    log: list[_FakeTrainer] = []

    def factory() -> _FakeTrainer:
        return _FakeTrainer(log)

    result = evaluate_walk_forward(
        dataset if dataset is not None else _dataset(),
        factory,
        config
        if config is not None
        else WalkForwardConfig(train_window_days=20, test_days=10, embargo_days=1),
        top_n=10,
        **kwargs,  # type: ignore[arg-type]
    )
    return result, log


# ---------------------------------------------------------------------------
# split_windows
# ---------------------------------------------------------------------------


class TestSplitWindows:
    def test_rolling_boundaries(self) -> None:
        dates = _dates(100)
        windows = split_windows(
            dates, WalkForwardConfig(train_window_days=20, test_days=10, embargo_days=1)
        )
        # first_test = 20 + 1 = 21，步长 10 → 测试起点 21/31/.../91，共 8 个窗口
        assert len(windows) == 8
        first = windows[0]
        assert first.train_start == dates[0]
        assert first.train_end == dates[19]
        assert first.test_start == dates[21]
        assert first.test_end == dates[30]
        last = windows[-1]
        assert last.test_start == dates[91]
        assert last.test_end == dates[99]  # 尾巴不足一块也保留

    def test_rolling_windows_do_not_overlap_in_test(self) -> None:
        dates = _dates(100)
        windows = split_windows(
            dates, WalkForwardConfig(train_window_days=20, test_days=10, embargo_days=1)
        )
        for prev, cur in zip(windows, windows[1:]):
            assert prev.test_end < cur.test_start
            assert cur.index == prev.index + 1

    def test_embargo_gap(self) -> None:
        dates = _dates(100)
        embargo = 3
        windows = split_windows(
            dates,
            WalkForwardConfig(train_window_days=20, test_days=10, embargo_days=embargo),
        )
        index = {day: i for i, day in enumerate(dates)}
        for window in windows:
            assert index[window.test_start] - index[window.train_end] == embargo + 1

    def test_expanding_keeps_train_start(self) -> None:
        dates = _dates(100)
        windows = split_windows(
            dates,
            WalkForwardConfig(
                train_window_days=20,
                test_days=10,
                embargo_days=1,
                min_train_days=20,
                expanding=True,
            ),
        )
        assert windows[0].train_start == dates[0]
        assert windows[-1].train_start == dates[0]
        assert windows[-1].train_end > windows[0].train_end

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="不足以开出任何窗口"):
            split_windows(
                _dates(21),
                WalkForwardConfig(train_window_days=20, test_days=10, embargo_days=1),
            )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="为空"):
            split_windows([], WalkForwardConfig())

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError, match="train_window_days"):
            WalkForwardConfig(train_window_days=0)
        with pytest.raises(ValueError, match="test_days"):
            WalkForwardConfig(test_days=0)
        with pytest.raises(ValueError, match="embargo_days"):
            WalkForwardConfig(embargo_days=-1)


# ---------------------------------------------------------------------------
# evaluate_walk_forward
# ---------------------------------------------------------------------------


class TestEvaluateWalkForward:
    def test_window_count_and_coverage(self) -> None:
        result, log = _run_eval()
        assert result.windows.height == 6
        assert len(log) == 6
        # 样本外拼接恰好覆盖所有窗口的测试区间，且不重叠
        oos_dates = result.oos["date"].unique().sort().to_list()
        expected: list[dt.date] = []
        for row in result.windows.iter_rows(named=True):
            lo, hi = row["test_start"], row["test_end"]
            expected.extend(day for day in _dates() if lo <= day <= hi)
        assert oos_dates == expected

    def test_no_lookahead_train_before_test(self) -> None:
        result, log = _run_eval()
        boundaries = {
            row["window"]: (row["train_end"], row["test_start"])
            for row in result.windows.iter_rows(named=True)
        }
        for index, trainer in enumerate(log):
            train_end, test_start = boundaries[index]
            assert trainer.train_max_date == train_end
            assert trainer.train_max_date < test_start
            assert trainer.n_train_rows > 0

    def test_oos_ic_positive_on_signal(self) -> None:
        result, _ = _run_eval()
        assert result.ic.mean is not None and result.ic.mean > 0.5
        assert result.ic.n_days > 0
        assert result.monotonicity is not None and result.monotonicity > 0.9
        assert result.turnover_mean is not None

    def test_window_metrics_populated(self) -> None:
        result, _ = _run_eval()
        for row in result.windows.iter_rows(named=True):
            assert row["n_train_rows"] > 0
            assert row["n_test_rows"] > 0
            assert row["ic_mean"] is not None
        assert result.stability["n_windows"] == 6
        assert result.stability["window_positive_rate"] == 1.0

    def test_importance_and_leaderboard_hooks(self) -> None:
        result, _ = _run_eval(
            importance_fn=_fake_importance, leaderboard_fn=_fake_leaderboard
        )
        assert result.feature_importance.height == 2
        top = result.feature_importance.row(0, named=True)
        assert top["feature"] == "f1"
        assert top["n_windows"] == 6
        assert top["mean_rank"] < result.feature_importance.row(1, named=True)[
            "mean_rank"
        ]
        assert len(result.leaderboards) == 6
        assert "rows=" in result.leaderboards[0]

    def test_without_hooks_importance_empty(self) -> None:
        result, _ = _run_eval()
        assert result.feature_importance.height == 0
        assert result.leaderboards == {}

    def test_report_json_and_markdown(self, tmp_path: object) -> None:
        result, _ = _run_eval(
            importance_fn=_fake_importance, leaderboard_fn=_fake_leaderboard
        )
        json_path = result.write_json(f"{tmp_path}/report.json")
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["config"]["train_window_days"] == 20
        assert loaded["ic"]["n_days"] == result.ic.n_days
        assert len(loaded["windows"]) == 6
        assert loaded["feature_importance"][0]["feature"] == "f1"

        md_path = result.write_markdown(f"{tmp_path}/report.md")
        text = md_path.read_text(encoding="utf-8")
        assert "walk-forward" in text
        assert "逐窗口指标" in text
        assert "因子重要性" in text

    def test_train_slice_drops_null_labels(self) -> None:
        dataset = _dataset().with_columns(
            pl.when(pl.col("date") == _dates()[5])
            .then(None)
            .otherwise(pl.col("label"))
            .alias("label")
        )
        result, log = _run_eval(dataset)
        for trainer in log:
            assert trainer.n_train_rows > 0
        # 第一天窗口的训练区间覆盖 date[5]，其 40 行 label 为 null 应被丢弃
        first = log[0]
        assert first.n_train_rows == 20 * N_INSTRUMENTS - N_INSTRUMENTS

    def test_missing_label_column_raises(self) -> None:
        with pytest.raises(ValueError, match="缺少必需列"):
            _run_eval(_dataset().drop("label"))

    def test_no_feature_columns_raises(self) -> None:
        bare = _dataset().select("date", "instrument", "label", "delay_days")
        with pytest.raises(ValueError, match="没有任何特征列"):
            _run_eval(bare)

    def test_scores_outside_window_raise(self) -> None:
        with pytest.raises(ValueError, match="越出测试区间"):
            evaluate_walk_forward(
                _dataset(),
                lambda: _OutOfWindowTrainer([]),
                WalkForwardConfig(train_window_days=20, test_days=10, embargo_days=1),
                top_n=10,
            )
