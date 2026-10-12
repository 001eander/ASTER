"""``scripts/reproduce_flagship.py`` 的单元测试（issue #113）。

不真跑子进程：plan 构造是纯函数，执行路径用 monkeypatch 拦 ``subprocess.run``。
覆盖：链路步骤与关键参数、策略参数来自配置 JSON（唯一真源）、模型已存在时
跳过训练、强制重训、配置不可用时退出码、失败中止、dry-run。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import reproduce_flagship as rf

#: 与仓库 config/strategy-flagship.json 同形的测试配置。
_CONFIG = {
    "strategy": "stock_selection",
    "universe": "config/universe/flagship_pool.parquet",
    "benchmark": "000300",
    "top_k": 15,
    "rebalance_freq": "W",
    "optimize": {"w_max": 0.1, "max_turnover": 0.6, "kappa": 0.002},
}


def _write_config(root: Path, payload: dict[str, object] | None = None) -> Path:
    path = root / "config" / "strategy-flagship.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _CONFIG), encoding="utf-8")
    return path


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """切到临时工作目录并写入策略配置，返回配置路径。"""
    monkeypatch.chdir(tmp_path)
    return _write_config(tmp_path)


class TestBuildPlan:
    def test_steps_and_key_args(self, workspace: Path) -> None:
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
            str(_CONFIG["top_k"]),
            "--rebalance-freq",
            _CONFIG["rebalance_freq"],
            "--benchmark",
            _CONFIG["benchmark"],
            "--universe",
            _CONFIG["universe"],
            "--risk-report",
        ):
            assert expected in backtest
        # 配置路径指向临时工作目录里写入的那份 JSON（按名匹配，不绑死相对/绝对写法）。
        config_arg = backtest[backtest.index("--strategy-config") + 1]
        assert config_arg.endswith("strategy-flagship.json")

    def test_strategy_args_come_from_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """策略参数唯一真源是配置 JSON：改配置即改命令，脚本不另存一份。"""
        monkeypatch.chdir(tmp_path)
        config_path = _write_config(tmp_path, {**_CONFIG, "top_k": 20, "benchmark": "000905"})

        steps = rf.build_plan("data", strategy_config_path=config_path)
        train = steps[2].argv
        backtest = steps[3].argv

        assert "20" in backtest and "000905" in backtest
        assert _CONFIG["universe"] in train  # 训练池与回测池同一份配置

    def test_train_window_and_backtest_window_do_not_overlap(self, workspace: Path) -> None:
        steps = rf.build_plan("data")
        train = steps[2].argv
        backtest = steps[3].argv
        assert rf.TRAIN_END < rf.BACKTEST_START
        assert rf.TRAIN_START in train and rf.TRAIN_END in train
        assert rf.BACKTEST_START in backtest and rf.BACKTEST_END in backtest

    def test_skips_train_when_model_exists(self, workspace: Path, tmp_path: Path) -> None:
        model_dir = tmp_path / rf.MODEL_DIR
        model_dir.mkdir(parents=True)
        (model_dir / rf.PREDICTOR_FILE).write_bytes(b"")

        steps = rf.build_plan("data")

        assert [step.name for step in steps] == ["数据校验", "旗舰股票池重建", "端到端回测"]

    def test_force_retrain_keeps_train_step(self, workspace: Path, tmp_path: Path) -> None:
        model_dir = tmp_path / rf.MODEL_DIR
        model_dir.mkdir(parents=True)
        (model_dir / rf.PREDICTOR_FILE).write_bytes(b"")

        steps = rf.build_plan("data", force_retrain=True)

        assert "旗舰模型训练" in [step.name for step in steps]

    def test_config_missing_field_raises(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, {"strategy": "stock_selection"})

        with pytest.raises(ValueError, match="缺少字段"):
            rf.build_plan("data", strategy_config_path=config_path)


class TestMain:
    def test_runs_steps_in_order(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_aborts_on_step_failure(self, workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, check: subprocess.CompletedProcess(argv, 3),
        )

        assert rf.main([]) == 3
        assert "中止" in capsys.readouterr().err

    def test_dry_run_executes_nothing(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, check: calls.append(tuple(argv)) or subprocess.CompletedProcess(argv, 0),
        )

        assert rf.main(["--dry-run"]) == 0
        assert calls == []

    def test_missing_config_exits_2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)  # 不写配置

        assert rf.main([]) == 2
        assert "策略配置不可用" in capsys.readouterr().err
