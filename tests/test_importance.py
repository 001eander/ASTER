"""``quant.automl.importance`` 与因子贡献度 CLI 的单元测试（issue #36）。

全部使用合成数据与假 predictor，不触发真实 AutoGluon 训练。假 predictor 的
``feature_importance`` 返回一个最小的类 DataFrame 对象（提供 ``index`` 与
``__getitem__``），因此测试连 pandas 都不需要接触。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant.automl.importance import (
    CONTRIBUTION_COLUMNS,
    aggregate_importance,
    factor_contribution,
    make_importance_fn,
    write_contribution_report,
)
from scripts import factor_contribution as fc

# ---------------------------------------------------------------------------
# 假 predictor / 假训练器
# ---------------------------------------------------------------------------


class _FakeImportanceFrame:
    """模仿 AutoGluon ``feature_importance`` 返回对象：``index`` + 列取值。"""

    def __init__(self, names: list[str], values: list[float]) -> None:
        self.index = names
        self._values = values

    def __getitem__(self, key: str) -> list[float]:
        if key == "importance":
            return self._values
        raise KeyError(key)


class _FakePredictor:
    """记录传入行数、按需抛错或返回固定重要性表的假 predictor。"""

    def __init__(
        self,
        frame: _FakeImportanceFrame | None = None,
        error: Exception | None = None,
    ) -> None:
        self.frame = frame
        self.error = error
        self.calls: list[int] = []

    def feature_importance(self, data: Any, silent: bool = True) -> _FakeImportanceFrame:
        self.calls.append(len(data))
        if self.error is not None:
            raise self.error
        assert self.frame is not None
        return self.frame


class _FakeTrainer:
    def __init__(self, predictor: Any, feature_columns_: list[str] | None = None) -> None:
        self.predictor = predictor
        if feature_columns_ is not None:
            self.feature_columns_ = feature_columns_


def _train_frame(n: int = 10, *, start: dt.date = dt.date(2026, 1, 5)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [start + dt.timedelta(days=i) for i in range(n)],
            "instrument": [f"{600000 + i % 3:06d}.SH" for i in range(n)],
            "f1": [float(i) for i in range(n)],
            "f2": [float(i % 2) for i in range(n)],
            "label": [float(i % 3) for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# make_importance_fn
# ---------------------------------------------------------------------------


class TestMakeImportanceFn:
    def test_normal_returns_frame_and_caps_rows(self) -> None:
        predictor = _FakePredictor(_FakeImportanceFrame(["f1", "f2"], [0.9, 0.1]))
        fn = make_importance_fn(["f1", "f2"], rows=3)
        result = fn(_FakeTrainer(predictor), _train_frame(10))

        assert result is not None
        assert result.columns == ["feature", "importance"]
        assert result["feature"].to_list() == ["f1", "f2"]
        assert result["importance"].to_list() == pytest.approx([0.9, 0.1])
        assert predictor.calls == [3]  # 采样上限生效

    def test_predictor_absent_returns_none(self) -> None:
        fn = make_importance_fn(["f1"])
        assert fn(object(), _train_frame()) is None

    def test_error_logs_warning_and_returns_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        predictor = _FakePredictor(error=RuntimeError("boom"))
        fn = make_importance_fn(["f1", "f2"])
        with caplog.at_level(logging.WARNING):
            assert fn(_FakeTrainer(predictor), _train_frame()) is None
        assert "不可用" in caplog.text
        assert len(predictor.calls) == 1

    def test_too_few_rows_returns_none(self) -> None:
        predictor = _FakePredictor(_FakeImportanceFrame(["f1"], [0.5]))
        fn = make_importance_fn(["f1"])
        assert fn(_FakeTrainer(predictor), _train_frame(1)) is None
        assert predictor.calls == []

    def test_all_null_labels_returns_none(self) -> None:
        frame = _train_frame(5).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("label")
        )
        predictor = _FakePredictor(_FakeImportanceFrame(["f1"], [0.5]))
        result = make_importance_fn(["f1"])(_FakeTrainer(predictor), frame)
        assert result is None
        assert predictor.calls == []

    def test_missing_feature_column_returns_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        predictor = _FakePredictor(_FakeImportanceFrame(["nope"], [0.5]))
        with caplog.at_level(logging.WARNING):
            assert make_importance_fn(["nope"])(
                _FakeTrainer(predictor), _train_frame()
            ) is None
        assert "样本构造失败" in caplog.text

    def test_rows_validation(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            make_importance_fn(["f1"], rows=0)


# ---------------------------------------------------------------------------
# 跨窗口聚合与归因表
# ---------------------------------------------------------------------------


def _aggregated() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["alpha", "gamma", "beta"],
            "n_windows": [2, 2, 2],
            "mean": [1.2, 0.5, -0.4],
            "std": [0.2, 0.25, 0.2],
            "mean_rank": [1.0, 2.0, 3.0],
        }
    )


class TestAggregateImportance:
    def test_empty_parts_returns_empty_schema(self) -> None:
        result = aggregate_importance([])
        assert result.height == 0
        assert set(result.columns) == {"feature", "n_windows", "mean", "std", "mean_rank"}

    def test_combines_window_parts(self) -> None:
        part0 = pl.DataFrame(
            {"feature": ["f1", "f2"], "importance": [1.0, 0.5], "_window": [0, 0]}
        )
        part1 = pl.DataFrame(
            {"feature": ["f1", "f2"], "importance": [3.0, 0.0], "_window": [1, 1]}
        )
        result = aggregate_importance([part0, part1])

        by_feature = {row["feature"]: row for row in result.to_dicts()}
        assert by_feature["f1"]["n_windows"] == 2
        assert by_feature["f1"]["mean"] == pytest.approx(2.0)
        assert by_feature["f2"]["mean"] == pytest.approx(0.25)


class TestFactorContribution:
    def test_sorted_by_mean_rank_and_keeps_sign(self) -> None:
        table = factor_contribution(_aggregated())

        assert table.columns == list(CONTRIBUTION_COLUMNS)
        assert table["factor"].to_list() == ["alpha", "gamma", "beta"]
        by_factor = {row["factor"]: row for row in table.to_dicts()}
        assert by_factor["beta"]["mean_importance"] == pytest.approx(-0.4)
        assert by_factor["gamma"]["stability"] == pytest.approx(0.25 / 0.5)
        # 负重要性参与变异系数，符号保留
        assert by_factor["beta"]["stability"] == pytest.approx(0.2 / -0.4)

    def test_tie_broken_by_mean_importance_descending(self) -> None:
        frame = pl.DataFrame(
            {
                "feature": ["low", "high"],
                "n_windows": [2, 2],
                "mean": [0.1, 0.9],
                "std": [0.05, 0.1],
                "mean_rank": [1.0, 1.0],
            }
        )
        assert factor_contribution(frame)["factor"].to_list() == ["high", "low"]

    def test_zero_or_missing_mean_gives_null_stability(self) -> None:
        frame = pl.DataFrame(
            {
                "feature": ["zero", "missing"],
                "n_windows": [2, 2],
                "mean": [0.0, None],
                "std": [0.2, 0.2],
                "mean_rank": [1.0, 2.0],
            }
        )
        by_factor = {row["factor"]: row for row in factor_contribution(frame).to_dicts()}
        assert by_factor["zero"]["stability"] is None
        assert by_factor["missing"]["stability"] is None

    def test_empty_returns_schema(self) -> None:
        empty = pl.DataFrame(
            schema={
                "feature": pl.String,
                "n_windows": pl.Int64,
                "mean": pl.Float64,
                "std": pl.Float64,
                "mean_rank": pl.Float64,
            }
        )
        table = factor_contribution(empty)
        assert table.height == 0
        assert table.columns == list(CONTRIBUTION_COLUMNS)

    def test_missing_required_column_raises(self) -> None:
        frame = pl.DataFrame({"feature": ["f1"], "mean_rank": [1.0]})
        with pytest.raises(ValueError, match="缺少必需列"):
            factor_contribution(frame)


# ---------------------------------------------------------------------------
# 报告落盘
# ---------------------------------------------------------------------------


class TestWriteContributionReport:
    def test_writes_json_and_markdown(self, tmp_path: Path) -> None:
        report = write_contribution_report(_aggregated(), tmp_path)

        assert report.json_path.exists()
        assert report.markdown_path.exists()
        payload = json.loads(report.json_path.read_text(encoding="utf-8"))
        assert payload["n_factors"] == 3
        assert [row["factor"] for row in payload["factor_contribution"]] == [
            "alpha",
            "gamma",
            "beta",
        ]
        assert payload["factor_contribution"][2]["mean_importance"] == pytest.approx(-0.4)

        text = report.markdown_path.read_text(encoding="utf-8")
        assert "因子贡献度分析" in text
        assert "因子归因表" in text
        assert "| factor |" in text

    def test_accepts_evaluation_like_object(self, tmp_path: Path) -> None:
        class _EvalLike:
            feature_importance = _aggregated()

        report = write_contribution_report(_EvalLike(), tmp_path)
        assert report.table.height == 3

    def test_invalid_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="ModelEvaluation"):
            write_contribution_report(object(), tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_walkforward_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": {"train_window_days": 5, "test_days": 2, "embargo_days": 1},
        "windows": [
            {
                "window": 0,
                "train_start": "2021-01-04",
                "train_end": "2021-01-08",
                "test_start": "2021-01-11",
                "test_end": "2021-01-12",
            },
            {
                "window": 1,
                "train_start": "2021-01-11",
                "train_end": "2021-01-15",
                "test_start": "2021-01-18",
                "test_end": "2021-01-19",
            },
        ],
        "feature_importance": [
            {
                "feature": "f1",
                "n_windows": 2,
                "mean": 0.8,
                "std": 0.1,
                "mean_rank": 1.0,
            },
            {
                "feature": "f2",
                "n_windows": 2,
                "mean": -0.2,
                "std": 0.05,
                "mean_rank": 2.0,
            },
        ],
    }
    payload.update(overrides)
    return payload


class TestFactorContributionCli:
    def test_json_mode_reads_aggregated_importance(self, tmp_path: Path) -> None:
        payload_path = _write_walkforward_json(tmp_path / "walkforward.json", _payload())
        out_dir = tmp_path / "out"

        code = fc.main(
            [
                "--walkforward-json",
                str(payload_path),
                "--output-dir",
                str(out_dir),
            ]
        )

        assert code == 0
        report_path = out_dir / "factor_contribution.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert [row["factor"] for row in data["factor_contribution"]] == ["f1", "f2"]
        assert data["factor_contribution"][1]["mean_importance"] == pytest.approx(-0.2)

    def test_json_mode_without_importance_requires_models_dir(
        self, tmp_path: Path
    ) -> None:
        payload_path = _write_walkforward_json(
            tmp_path / "walkforward.json", _payload(feature_importance=[])
        )
        with pytest.raises(SystemExit, match="models-dir"):
            fc.main(
                [
                    "--walkforward-json",
                    str(payload_path),
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )

    def test_recompute_mode_loads_window_models(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload_path = _write_walkforward_json(tmp_path / "walkforward.json", _payload())
        out_dir = tmp_path / "out"
        dataset = _train_frame(10, start=dt.date(2021, 1, 4))

        monkeypatch.setattr(fc, "build_dataset_for_args", lambda args: dataset)

        class _Loader:
            def __init__(self) -> None:
                self.paths: list[Path] = []
                self.values = [(1.0, 0.5), (3.0, 0.0)]

            def __call__(self, path: Path) -> _FakeTrainer:
                self.paths.append(Path(path))
                f1, f2 = self.values[len(self.paths) - 1]
                predictor = _FakePredictor(
                    _FakeImportanceFrame(["f1", "f2"], [f1, f2])
                )
                return _FakeTrainer(predictor, feature_columns_=["f1", "f2"])

        loader = _Loader()
        monkeypatch.setattr(fc, "load_window_trainer", loader)

        code = fc.main(
            [
                "--walkforward-json",
                str(payload_path),
                "--models-dir",
                str(tmp_path / "models"),
                "--output-dir",
                str(out_dir),
            ]
        )

        assert code == 0
        assert [p.name for p in loader.paths] == ["window_00", "window_01"]
        data = json.loads(
            (out_dir / "factor_contribution.json").read_text(encoding="utf-8")
        )
        by_factor = {row["factor"]: row for row in data["factor_contribution"]}
        assert by_factor["f1"]["mean_importance"] == pytest.approx(2.0)
        assert by_factor["f2"]["mean_importance"] == pytest.approx(0.25)
        assert by_factor["f1"]["n_windows"] == 2

    def test_payload_helpers(self) -> None:
        payload = _payload()
        aggregated = fc.importance_from_payload(payload)
        assert aggregated.height == 2
        windows = fc.windows_from_payload(payload)
        assert [w.index for w in windows] == [0, 1]
        assert windows[0].train_start == dt.date(2021, 1, 4)

        empty = fc.importance_from_payload({"feature_importance": []})
        assert empty.height == 0
