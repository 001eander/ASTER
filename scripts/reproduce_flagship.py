"""旗舰策略一条命令复现（issue #113）：数据校验 → 建池 → 训练 → 端到端回测。

链路
----
1. ``scripts/validate_data.py``：跑批前强制数据体检，不过即中止（量化纪律第 6 条）；
2. ``scripts/build_flagship_universe.py``：重建季度刷新的旗舰股票池
   （``config/universe/flagship_pool.parquet``，300 只低价高流动性主板 / 创业板）；
3. ``scripts/train_baseline.py``：在旗舰池上训练 AutoGluon 模型到
   ``runs/automl/flagship``（模型目录已存在且未给 ``--force-retrain`` 时跳过，
   训练一次约半小时，复现回测不必每次重训）；
4. ``scripts/backtest_e2e.py``：10 万本金、周频调仓、top_k=15、基准 000300，
   产物（净值图 / 报告 / 持仓 / 风险四表）落 ``runs/flagship/``。

训练窗口（2021-09-29 ~ 2024-12-31）与回测窗口（2025-01-02 起）不重叠，
回测段是纯样本外；因子挖掘的评估用了全样本行情，因子入选本身含全样本信息，
这一泄漏边界在 wiki 策略细节页如实标注。

用法::

    # 全链路（数据已抓取的前提下）
    uv run python scripts/reproduce_flagship.py

    # 只看不跑
    uv run python scripts/reproduce_flagship.py --dry-run

    # 强制重训模型
    uv run python scripts/reproduce_flagship.py --force-retrain

数据初始化（首次全量抓取）不在本脚本内：依赖网络与数据源配额，见 README「快速开始」。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# 配置（旗舰样板口径，与 config/strategy-flagship.json 逐项一致）
# ---------------------------------------------------------------------------

#: 训练窗口（样本内）。
TRAIN_START: str = "2021-09-29"
TRAIN_END: str = "2024-12-31"

#: 回测窗口（纯样本外，接训练窗口之后）。
BACKTEST_START: str = "2025-01-02"
BACKTEST_END: str = "2026-09-28"

#: 本金口径：个人交易者样板 10 万。
INITIAL_CASH: int = 100_000

#: 策略配置 JSON：strategy / universe / benchmark / top_k / rebalance_freq 的唯一真源，
#: 本脚本不另存一份参数，直接读它拼回测命令（backtest_e2e 会逐项校验一致性）。
STRATEGY_CONFIG: str = "config/strategy-flagship.json"

#: 模型与产物目录。
MODEL_DIR: str = "runs/automl/flagship"
OUT_DIR: str = "runs/flagship"

#: 训练算力预算。
TIME_LIMIT: int = 1800
PRESETS: str = "medium_quality"

#: AutoGluon 落盘的预测器文件，存在即视为模型已训练。
PREDICTOR_FILE: str = "predictor.pkl"

DEFAULT_DATA_DIR: str = "data"

SCRIPTS_DIR: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    """复现链路中的一步：名称 + 子进程 argv（含解释器与脚本路径）。"""

    name: str
    argv: tuple[str, ...]


def _script_argv(script: str, *args: object) -> tuple[str, ...]:
    """拼一步的子进程命令：``python scripts/<script> <args...>``。"""
    return (
        sys.executable,
        str(SCRIPTS_DIR / script),
        *(str(arg) for arg in args),
    )


def _load_strategy_config(path: Path) -> dict[str, object]:
    """读策略配置 JSON，缺文件或缺字段时直接报错（口径唯一真源，不默认兜底）。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = ("strategy", "universe", "benchmark", "top_k", "rebalance_freq")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"策略配置 {path} 缺少字段：{missing}")
    return raw


def build_plan(
    data_dir: str,
    *,
    force_retrain: bool = False,
    strategy_config_path: Path = Path(STRATEGY_CONFIG),
) -> list[Step]:
    """生成复现链路各步命令；模型已存在且未强制重训时训练步跳过。

    回测步的策略参数逐项从 ``strategy_config_path`` 读取，与生产跑批共用同一份 JSON。
    """
    config = _load_strategy_config(strategy_config_path)
    universe = str(config["universe"])
    steps = [
        Step(
            "数据校验",
            _script_argv("validate_data.py", "--data-dir", data_dir),
        ),
        Step(
            "旗舰股票池重建",
            _script_argv("build_flagship_universe.py", "--data-dir", data_dir),
        ),
    ]
    model_ready = (Path(MODEL_DIR) / PREDICTOR_FILE).is_file()
    if force_retrain or not model_ready:
        steps.append(
            Step(
                "旗舰模型训练",
                _script_argv(
                    "train_baseline.py",
                    "--data-dir", data_dir,
                    "--start", TRAIN_START,
                    "--end", TRAIN_END,
                    "--model-dir", MODEL_DIR,
                    "--universe", universe,
                    "--time-limit", TIME_LIMIT,
                    "--presets", PRESETS,
                ),
            )
        )
    steps.append(
        Step(
            "端到端回测",
            _script_argv(
                "backtest_e2e.py",
                "--data-dir", data_dir,
                "--strategy-config", str(strategy_config_path),
                "--strategy", config["strategy"],
                "--universe", universe,
                "--benchmark", config["benchmark"],
                "--top-k", config["top_k"],
                "--rebalance-freq", config["rebalance_freq"],
                "--initial-cash", INITIAL_CASH,
                "--model-dir", MODEL_DIR,
                "--start", BACKTEST_START,
                "--end", BACKTEST_END,
                "--out-dir", OUT_DIR,
                "--risk-report",
            ),
        )
    )
    return steps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="旗舰策略一条命令复现（issue #113）")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--strategy-config",
        default=STRATEGY_CONFIG,
        help=f"策略配置 JSON 路径，默认 {STRATEGY_CONFIG}",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="模型目录已存在也强制重训（默认跳过训练步）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印各步命令，不执行")
    return parser


def main(argv: list[str] | None = None) -> int:
    """逐步执行复现链路，任一步失败即中止并透出退出码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    try:
        steps = build_plan(
            args.data_dir,
            force_retrain=args.force_retrain,
            strategy_config_path=Path(args.strategy_config),
        )
    except (OSError, ValueError) as exc:
        print(f"策略配置不可用：{exc}", file=sys.stderr)
        return 2
    for index, step in enumerate(steps, 1):
        print(f"[{index}/{len(steps)}] {step.name}")
        print("      " + " ".join(step.argv))
        if args.dry_run:
            continue
        completed = subprocess.run(step.argv, check=False)
        if completed.returncode != 0:
            print(f"步骤「{step.name}」失败（退出码 {completed.returncode}），中止", file=sys.stderr)
            return completed.returncode
    if args.dry_run:
        print("dry-run：未执行任何步骤")
    else:
        print(f"复现完成，产物见 {OUT_DIR}/（nav.png / report.md / holdings.parquet / risk_report.md）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
