"""``scripts/reproduce_flagship.py`` 的单元测试（issue #113）。

不真跑子进程：plan 构造是纯函数，执行路径用 monkeypatch 拦 ``subprocess.run``。
覆盖：链路步骤与关键参数、模型已存在时跳过训练、强制重训、失败中止、dry-run。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import reproduce_flagship as rf


class TestBuildPlan:
    def test_steps_and_key_args(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)  # 模型目录不存在 → 训练步在链路里
        steps = rf.build_plan("data")

        assert [step.name for step in steps] == [
            "数据校验",
            "旗舰股票池重建",
            "旗舰模型训练",
            "端到端回测",
        ]
        backtest = steps[-1].argv
        assert "backtest_e2e.py" in backtest[1]
        for expected in (
            "--initial-cash",
            str(rf.INITIAL_CASH),
            "--top-k",
            str(rf.TOP_K),
            "--rebalance-freq",
            rf.REBALANCE_FREQ,
            "--benchmark",
            rf.BENCHMARK,
            "--strategy-config",
            rf.STRATEGY_CONFIG,
            "--universe",
            rf.UNIVERSE,
            "--risk-report",
        ):
            assert expected in backtest

    def test_train_window_and_backtest_window_do_not_overlap(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        steps = rf.build_plan("data")
        train = steps[2].argv
        backtest = steps[3].argv
        assert rf.TRAIN_END < rf.BACKTEST_START
        assert rf.TRAIN_START in train and rf.TRAIN_END in train
        assert rf.BACKTEST_START in backtest and rf.BACKTEST_END in backtest

    def test_skips_train_when_model_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / rf.MODEL_DIR
        model_dir.mkdir(parents=True)
        (model_dir / rf.PREDICTOR_FILE).write_bytes(b"")

        steps = rf.build_plan("data")

        assert [step.name for step in steps] == ["数据校验", "旗舰股票池重建", "端到端回测"]

    def test_force_retrain_keeps_train_step(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / rf.MODEL_DIR
        model_dir.mkdir(parents=True)
        (model_dir / rf.PREDICTOR_FILE).write_bytes(b"")

        steps = rf.build_plan("data", force_retrain=True)

        assert "旗舰模型训练" in [step.name for step in steps]


class TestMain:
    def test_runs_steps_in_order(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, check: calls.append(tuple(argv)) or subprocess.CompletedProcess(argv, 0),
        )

        assert rf.main([]) == 0
        assert len(calls) == 4
        assert "validate_data.py" in calls[0][1]
        assert "backtest_e2e.py" in calls[-1][1]

    def test_aborts_on_step_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, check: subprocess.CompletedProcess(argv, 3),
        )

        assert rf.main([]) == 3
        assert "中止" in capsys.readouterr().err

    def test_dry_run_executes_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, check: calls.append(tuple(argv)) or subprocess.CompletedProcess(argv, 0),
        )

        assert rf.main(["--dry-run"]) == 0
        assert calls == []
