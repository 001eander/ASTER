"""walk-forward 模型级评估入口（issue #33）。

链路：``load_bars`` → 池内裁决（可选）→ ``discover_factors`` →
``build_dataset`` → :func:`quant.eval.model.evaluate_walk_forward` 逐窗口训练
AutoGluon → 样本外拼接 → 指标汇总 → JSON / markdown / 样本外打分明细落盘。

用法::

    uv run python scripts/eval_model_walkforward.py \
        --data-dir data --start 2021-09-29 --end 2024-12-31 \
        --train-window-days 500 --test-days 20 \
        --presets medium_quality --time-limit 600

小样本试跑加 ``--universe zz1000`` 收窄股票池，或缩短 ``--start/--end``。
逐窗口模型落盘到 ``--models-dir``（默认 ``runs/automl/walkforward``）。
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.automl.dataset import (  # noqa: E402
    build_dataset,
)
from quant.automl.importance import make_importance_fn  # noqa: E402
from quant.automl.trainer import (  # noqa: E402
    DEFAULT_PRESETS,
    DEFAULT_TIME_LIMIT,
    BaselineTrainer,
)
from quant.daily.pipeline import (  # noqa: E402
    DEFAULT_FACTOR_LIBRARY_DIR,
    discover_factors,
)
from quant.data.cache import load_bars  # noqa: E402
from quant.eval.model import (  # noqa: E402
    DEFAULT_TEST_DAYS,
    DEFAULT_TOP_N,
    DEFAULT_TRAIN_WINDOW_DAYS,
    NON_FEATURE_COLUMNS,
    ModelTrainer,
    WalkForwardConfig,
    evaluate_walk_forward,
)
from quant.eval.metrics import DEFAULT_LAYERS  # noqa: E402
from quant.labels.open_to_open import DEFAULT_HORIZON  # noqa: E402
from quant.universe.members import filter_bars_to_universe  # noqa: E402

logger = logging.getLogger("eval_model_walkforward")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR: str = "data"
DEFAULT_OUTPUT_DIR: str = "runs/model_eval"
DEFAULT_MODELS_DIR: str = "runs/automl/walkforward"

#: leaderboard 打印行数。
LEADERBOARD_ROWS: int = 10


# ---------------------------------------------------------------------------
# AutoGluon 钩子（importance_fn / leaderboard_fn）
# ---------------------------------------------------------------------------


def leaderboard_fn(trainer: ModelTrainer) -> str | None:
    """逐窗口 leaderboard 文本钩子。"""
    predictor = getattr(trainer, "predictor", None)
    if predictor is None:
        return None
    try:
        board = predictor.leaderboard(silent=True)
        return str(board.head(LEADERBOARD_ROWS).to_string())
    except Exception as exc:  # noqa: BLE001
        logger.warning("window leaderboard 不可用：%s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="walk-forward 模型级评估（issue #33）")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument("--start", type=_parse_date, default=None, help="评估起始日")
    parser.add_argument("--end", type=_parse_date, default=None, help="评估截止日")
    parser.add_argument(
        "--factor-library",
        default=str(DEFAULT_FACTOR_LIBRARY_DIR),
        help=f"因子库目录，默认 {DEFAULT_FACTOR_LIBRARY_DIR}",
    )
    parser.add_argument("--universe", default=None, help="股票池（命名池或自定义路径）")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="标签持有期")
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_DAYS,
        help=f"滚动训练窗口交易日数，默认 {DEFAULT_TRAIN_WINDOW_DAYS}",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        help=f"重训周期（每窗口样本外交易日数），默认 {DEFAULT_TEST_DAYS}",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        help="训练截止与测试起始的隔离交易日数，默认 = horizon",
    )
    parser.add_argument("--expanding", action="store_true", help="扩张窗口（默认滚动）")
    parser.add_argument("--presets", default=DEFAULT_PRESETS, help="AutoGluon 预设")
    parser.add_argument(
        "--recipe",
        default=None,
        help="命名训练配方（memory_safe/full/bagged/hpo）；给出后覆盖 --presets",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=f"单窗口训练时限（秒），默认 {DEFAULT_TIME_LIMIT:.0f}",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="换手臂数")
    parser.add_argument(
        "--n-layers", type=int, default=DEFAULT_LAYERS, help="分层数"
    )
    parser.add_argument(
        "--no-importance", action="store_true", help="跳过逐窗口因子重要性（省时）"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="报告落盘目录"
    )
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR, help="逐窗口模型落盘目录"
    )
    parser.add_argument("--use-gpu", action="store_true", default=None, help="强制 GPU")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="强制 CPU")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    data_dir = Path(args.data_dir)
    bars = load_bars(data_dir, start=args.start, end=args.end)
    if bars.height == 0:
        raise SystemExit(f"行情窗口 [{args.start}, {args.end}] 内没有数据：{data_dir}")
    if args.universe is not None:
        bars = filter_bars_to_universe(bars, args.universe, data_dir=data_dir)
        if bars.height == 0:
            raise SystemExit(f"股票池 {args.universe!r} 在窗口内没有行情")

    factors = discover_factors(args.factor_library)
    logger.info("构造数据集：%d 个因子，%d 行行情", len(factors), bars.height)
    dataset = build_dataset(bars, factors, horizon=args.horizon)
    feature_columns = [
        col for col in dataset.columns if col not in NON_FEATURE_COLUMNS
    ]

    models_dir = Path(args.models_dir)
    counter = itertools.count()

    def trainer_factory() -> BaselineTrainer:
        index = next(counter)
        return BaselineTrainer(
            feature_columns=feature_columns,
            presets=args.presets,
            time_limit=args.time_limit,
            path=models_dir / f"window_{index:02d}",
            use_gpu=args.use_gpu,
            recipe=args.recipe,
        )

    config = WalkForwardConfig(
        train_window_days=args.train_window_days,
        test_days=args.test_days,
        embargo_days=(
            args.embargo_days if args.embargo_days is not None else args.horizon
        ),
        expanding=args.expanding,
    )
    result = evaluate_walk_forward(
        dataset,
        trainer_factory,
        config,
        n_layers=args.n_layers,
        top_n=args.top_n,
        importance_fn=(
            None if args.no_importance else make_importance_fn(feature_columns)
        ),
        leaderboard_fn=leaderboard_fn,
    )

    output_dir = Path(args.output_dir)
    json_path = result.write_json(output_dir / "walkforward.json")
    md_path = result.write_markdown(output_dir / "walkforward.md")
    oos_path = output_dir / "oos_scores.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.oos.write_parquet(oos_path)

    ic = result.ic
    print("\n# walk-forward 模型级评估完成")
    print(f"窗口数：{result.windows.height}")
    print(f"样本外：{json_path} / {md_path} / {oos_path}")
    print(
        f"RankIC 均值 {ic.mean}，ICIR {ic.icir}，"
        f"胜率 {ic.ic_win_rate}，有效 {ic.n_days} 日"
    )
    print(f"分层单调性 {result.monotonicity}，Top-{args.top_n} 日均换手 "
          f"{result.turnover_mean}")
    print(f"滚动稳定性：{result.stability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
