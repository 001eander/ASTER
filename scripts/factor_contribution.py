"""因子贡献度报告入口（issue #36）。

从已有的 walk-forward 产物生成因子维度归因报告，支持两种输入：

1. 直接读 ``walkforward.json`` 里已聚合的 ``feature_importance``（默认路径
   ``runs/model_eval/walkforward.json``）；
2. 给出 ``--models-dir``，加载 ``runs/automl/walkforward`` 下逐窗口 predictor，
   按 walkforward.json 记录的窗口区间重建训练切片、重算置换重要性，再聚合。

数据集构造（``load_bars`` → 可选股票池裁决 → ``discover_factors`` →
``build_dataset``）与 :mod:`scripts.eval_model_walkforward` 一致。

用法::

    # 模式 1：直接读已聚合的重要性
    uv run python scripts/factor_contribution.py \
        --walkforward-json runs/model_eval/walkforward.json \
        --output-dir runs/model_eval

    # 模式 2：从逐窗口模型重算（walkforward.json 没跑重要性时）
    uv run python scripts/factor_contribution.py \
        --models-dir runs/automl/walkforward \
        --data-dir data --start 2021-09-29 --end 2024-12-31 \
        --universe zz1000 --output-dir runs/model_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from quant.automl.dataset import (  # noqa: E402
    DATE_COL,
    build_dataset,
)
from quant.automl.importance import (  # noqa: E402
    DEFAULT_IMPORTANCE_ROWS,
    aggregate_importance,
    make_importance_fn,
    write_contribution_report,
)
from quant.automl.trainer import BaselineTrainer  # noqa: E402
from quant.daily.pipeline import (  # noqa: E402
    DEFAULT_FACTOR_LIBRARY_DIR,
    discover_factors,
)
from quant.data.cache import load_bars  # noqa: E402
from quant.eval.model import NON_FEATURE_COLUMNS, ModelTrainer  # noqa: E402
from quant.labels.open_to_open import DEFAULT_HORIZON  # noqa: E402
from quant.universe.members import filter_bars_to_universe  # noqa: E402

logger = logging.getLogger("factor_contribution")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR: str = "data"
DEFAULT_WALKFORWARD_JSON: str = "runs/model_eval/walkforward.json"
DEFAULT_MODELS_DIR: str = "runs/automl/walkforward"
DEFAULT_OUTPUT_DIR: str = "runs/model_eval"

#: 逐窗口模型目录命名，与 :mod:`scripts.eval_model_walkforward` 对齐。
WINDOW_DIR_FMT: str = "window_{index:02d}"

_IMPORTANCE_SCHEMA = pl.Schema(
    {
        "feature": pl.String,
        "n_windows": pl.Int64,
        "mean": pl.Float64,
        "std": pl.Float64,
        "mean_rank": pl.Float64,
    }
)


# ---------------------------------------------------------------------------
# walk-forward 产物读取
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSlice:
    """从 walkforward.json 还原的单个训练窗口区间。"""

    index: int
    train_start: date
    train_end: date


def load_walkforward_payload(path: str | Path) -> dict[str, Any]:
    """读取 walkforward.json，返回解析后的字典。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"walkforward.json 顶层应为对象：{path}")
    return payload


def importance_from_payload(payload: dict[str, Any]) -> pl.DataFrame:
    """从 walkforward.json 的 ``feature_importance`` 还原跨窗口聚合表。"""
    records = payload.get("feature_importance") or []
    if not records:
        return pl.DataFrame(schema=_IMPORTANCE_SCHEMA)
    rows = [
        {
            "feature": None if rec.get("feature") is None else str(rec["feature"]),
            "n_windows": rec.get("n_windows"),
            "mean": rec.get("mean"),
            "std": rec.get("std"),
            "mean_rank": rec.get("mean_rank"),
        }
        for rec in records
    ]
    return pl.DataFrame(rows, schema=_IMPORTANCE_SCHEMA)


def windows_from_payload(payload: dict[str, Any]) -> list[WindowSlice]:
    """从 walkforward.json 的 ``windows`` 记录还原窗口训练区间。"""
    records = payload.get("windows") or []
    windows: list[WindowSlice] = []
    for rec in records:
        windows.append(
            WindowSlice(
                index=int(rec["window"]),
                train_start=_parse_iso_date(rec["train_start"]),
                train_end=_parse_iso_date(rec["train_end"]),
            )
        )
    return windows


def _parse_iso_date(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


# ---------------------------------------------------------------------------
# 从落盘模型重算重要性
# ---------------------------------------------------------------------------


def load_window_trainer(path: str | Path) -> BaselineTrainer:
    """加载单个窗口的已落盘 predictor，供 CLI 与测试复用。"""
    return BaselineTrainer.load(path)


def recompute_importance(
    dataset: pl.DataFrame,
    windows: Sequence[WindowSlice],
    models_dir: str | Path,
    *,
    rows: int = DEFAULT_IMPORTANCE_ROWS,
    trainer_loader: Callable[[Path], ModelTrainer] | None = None,
    feature_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """按窗口区间重算置换重要性，返回跨窗口聚合表。

    每个窗口从 ``models_dir/window_XX`` 加载 predictor，切出训练切片后调用
    :func:`quant.automl.importance.make_importance_fn`。单个窗口失败只记 warning，
    不影响其余窗口。
    """
    loader = load_window_trainer if trainer_loader is None else trainer_loader
    parts: list[pl.DataFrame] = []
    for window in windows:
        model_path = Path(models_dir) / WINDOW_DIR_FMT.format(index=window.index)
        trainer = loader(model_path)
        columns = (
            list(feature_columns)
            if feature_columns is not None
            else _feature_columns_for(trainer, dataset)
        )
        if not columns:
            logger.warning("窗口 %d 无法确定特征列，跳过", window.index)
            continue
        train_df = dataset.filter(
            (pl.col(DATE_COL) >= window.train_start)
            & (pl.col(DATE_COL) <= window.train_end)
        )
        if train_df.height < 2:
            logger.warning("窗口 %d 训练切片不足（%d 行），跳过", window.index, train_df.height)
            continue
        importance = make_importance_fn(columns, rows=rows)(trainer, train_df)
        if importance is not None and importance.height:
            parts.append(
                importance.with_columns(pl.lit(window.index).alias("_window"))
            )
        else:
            logger.warning("窗口 %d 因子重要性不可用", window.index)
    return aggregate_importance(parts)


def _feature_columns_for(
    trainer: ModelTrainer, dataset: pl.DataFrame
) -> list[str]:
    """优先用 predictor 的特征元数据，缺失时回落到数据集的特征列。"""
    names = getattr(trainer, "feature_columns_", None)
    if names:
        return [str(name) for name in names]
    return [col for col in dataset.columns if col not in NON_FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# 数据集构造
# ---------------------------------------------------------------------------


def build_dataset_for_args(args: argparse.Namespace) -> pl.DataFrame:
    """按 CLI 参数构造 walk-forward 数据集（与 eval_model_walkforward 同口径）。"""
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
    return build_dataset(bars, factors, horizon=args.horizon)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="因子贡献度报告（issue #36）")
    parser.add_argument(
        "--walkforward-json",
        default=DEFAULT_WALKFORWARD_JSON,
        help=f"walk-forward 产物，默认 {DEFAULT_WALKFORWARD_JSON}",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help=(
            "逐窗口 predictor 目录；给出则从落盘模型重算重要性"
            f"（约定路径 {DEFAULT_MODELS_DIR}），缺省只读 JSON 已聚合结果"
        ),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument("--start", type=_parse_date, default=None, help="数据集起始日")
    parser.add_argument("--end", type=_parse_date, default=None, help="数据集截止日")
    parser.add_argument(
        "--factor-library",
        default=str(DEFAULT_FACTOR_LIBRARY_DIR),
        help=f"因子库目录，默认 {DEFAULT_FACTOR_LIBRARY_DIR}",
    )
    parser.add_argument("--universe", default=None, help="股票池（命名池或自定义路径）")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="标签持有期")
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_IMPORTANCE_ROWS,
        help=f"置换重要性采样行数，默认 {DEFAULT_IMPORTANCE_ROWS}",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="报告落盘目录"
    )
    parser.add_argument("--title", default="因子贡献度分析", help="报告标题")
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

    payload = load_walkforward_payload(args.walkforward_json)
    if args.models_dir is not None:
        windows = windows_from_payload(payload)
        if not windows:
            raise SystemExit("walkforward.json 缺少 windows 明细，无法定位逐窗口模型")
        dataset = build_dataset_for_args(args)
        aggregated = recompute_importance(
            dataset, windows, args.models_dir, rows=args.rows
        )
        mode = "recompute"
    else:
        aggregated = importance_from_payload(payload)
        mode = "json"
        if aggregated.height == 0:
            raise SystemExit(
                "walkforward.json 不含 feature_importance；"
                "请用 --models-dir 从落盘模型重算"
            )

    report = write_contribution_report(
        aggregated, args.output_dir, title=args.title
    )

    print("\n# 因子贡献度分析完成")
    print(f"输入模式：{mode}，因子数：{report.table.height}")
    print(f"报告：{report.json_path} / {report.markdown_path}")
    for row in report.table.head(10).iter_rows(named=True):
        print(
            f"  {row['factor']}: mean={row['mean_importance']}, "
            f"stability={row['stability']}, mean_rank={row['mean_rank']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
