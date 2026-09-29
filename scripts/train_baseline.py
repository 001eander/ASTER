"""端到端 M1（issue #16）：手工因子 → 数据集 → AutoGluon 基线训练 → 模型落盘。

链路位置
--------
本脚本是「训练侧」入口，与 ``scripts/backtest_e2e.py``（回测侧）配对：

1. ``load_bars`` 读窗口内全市场行情；
2. ``discover_factors`` 加载 ``factor_library/`` 下全部手工因子；
3. ``build_dataset`` 拼出「每票每日一行」的宽表并做逐日截面 z-score；
4. 丢弃 ``label`` 为空或全部特征为空的样本（打印行数与缺失统计）；
5. ``BaselineTrainer.train`` 训练并 ``save`` 到 ``--model-dir``；
6. 打印 leaderboard 与特征重要性，并把训练配置写到 model-dir 旁边的
   ``<model-dir 名>_train_config.json``，供回测脚本核对特征列一致性。

训练配置的落盘位置
------------------
AutoGluon 的 ``TabularPredictor.save`` 会独占 ``--model-dir``，因此配置写在**同级**
的 ``<名字>_train_config.json``（见 :func:`default_train_config_path`），回测脚本按同一
约定读取。

用法::

    uv run python scripts/train_baseline.py \
        --data-dir data --start 2021-09-29 --end 2024-12-31 \
        --model-dir runs/automl/baseline --time-limit 1800 --presets medium_quality

小样本试跑可用 ``--max-rows 200000`` 截断训练集；``--max-rows 0``（默认）不截断。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from quant.automl.dataset import (  # noqa: E402
    DATE_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    build_dataset,
    missing_rate,
)
from quant.automl.trainer import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_PRESETS,
    DEFAULT_TIME_LIMIT,
    NON_FEATURE_COLUMNS,
    BaselineTrainer,
)
from quant.daily.pipeline import (  # noqa: E402
    DEFAULT_FACTOR_LIBRARY_DIR,
    discover_factors,
)
from quant.data.cache import load_bars  # noqa: E402
from quant.labels.open_to_open import DEFAULT_HORIZON  # noqa: E402
from quant.universe.members import (  # noqa: E402
    filter_bars_to_universe,
    mean_daily_instruments,
)

logger = logging.getLogger("train_baseline")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 默认数据目录。
DEFAULT_DATA_DIR: str = "data"

#: 训练配置 JSON 的文件名后缀（放到 model-dir 同级目录）。
TRAIN_CONFIG_SUFFIX: str = "_train_config.json"

#: ``--max-rows`` 截断时的抽样随机种子，固定以保证可复现。
SAMPLING_SEED: int = 20260929

#: leaderboard 打印行数。
LEADERBOARD_ROWS: int = 5

#: 特征重要性计算使用的样本行数（喂给 AutoGluon 的置换重要性）。
FEATURE_IMPORTANCE_ROWS: int = 2000


# ---------------------------------------------------------------------------
# 训练配置的读写（回测脚本共用）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """一次训练落盘的元信息，供回测脚本核对特征列与窗口一致性。"""

    data_dir: str
    start: str | None
    end: str | None
    n_rows: int
    feature_columns: list[str]
    presets: str
    time_limit: float
    horizon: int
    max_rows: int
    #: 训练所用的股票池（命名池名或自定义池路径）；``None`` 表示全市场。
    universe: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "start": self.start,
            "end": self.end,
            "n_rows": self.n_rows,
            "feature_columns": list(self.feature_columns),
            "presets": self.presets,
            "time_limit": self.time_limit,
            "horizon": self.horizon,
            "max_rows": self.max_rows,
            "universe": self.universe,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        universe = data.get("universe")
        return cls(
            data_dir=str(data.get("data_dir", "")),
            start=data.get("start"),
            end=data.get("end"),
            n_rows=int(data.get("n_rows", 0)),
            feature_columns=[str(c) for c in data.get("feature_columns", [])],
            presets=str(data.get("presets", DEFAULT_PRESETS)),
            time_limit=float(data.get("time_limit", DEFAULT_TIME_LIMIT)),
            horizon=int(data.get("horizon", DEFAULT_HORIZON)),
            max_rows=int(data.get("max_rows", 0)),
            universe=None if universe is None else str(universe),
        )


def default_train_config_path(model_dir: str | Path) -> Path:
    """返回 ``--model-dir`` 对应的训练配置路径（同级、带后缀）。"""
    model_dir = Path(model_dir)
    return model_dir.parent / f"{model_dir.name}{TRAIN_CONFIG_SUFFIX}"


def write_train_config(path: str | Path, config: TrainConfig) -> Path:
    """把训练配置写成 UTF-8 JSON，返回落盘路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_train_config(path: str | Path) -> TrainConfig:
    """读取 :func:`write_train_config` 写出的 JSON；文件不存在时抛 FileNotFoundError。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"训练配置 JSON 顶层应为对象：{path}")
    return TrainConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# 数据集准备
# ---------------------------------------------------------------------------


@dataclass
class DatasetSummary:
    """训练集构造的统计摘要。"""

    total_rows: int
    kept_rows: int
    dropped_null_label: int
    dropped_all_null_feature: int
    n_dates: int
    n_instruments: int
    feature_columns: tuple[str, ...]
    missing: pl.DataFrame = field(default_factory=pl.DataFrame)
    #: 训练股票池（``None`` 全市场）。
    universe: str | None = None
    #: 池内日均票数（按交易日统计的证券数均值）。
    mean_daily_instruments: float = 0.0


def prepare_training_dataset(
    bars: pl.DataFrame,
    factors: dict[str, Any],
    *,
    horizon: int = DEFAULT_HORIZON,
    max_rows: int = 0,
    seed: int = SAMPLING_SEED,
) -> tuple[pl.DataFrame, DatasetSummary]:
    """``build_dataset`` + 丢弃不可训练行 + 可选抽样，返回 ``(数据集, 摘要)``。

    丢弃规则（与纪律一致，训练集不喂入无效标签）：

    - ``label`` 为空：序列尾部没有未来行情；
    - 全部特征为空：该样本没有任何可用信息。

    ``max_rows > 0`` 时对保留行做固定种子的随机抽样，便于小样本人工试跑。
    """
    if bars.height == 0:
        raise ValueError("行情为空，无法构造训练集")
    dataset = build_dataset(bars, factors, horizon=horizon)
    feature_columns = tuple(
        col for col in dataset.columns if col not in NON_FEATURE_COLUMNS
    )
    if not feature_columns:
        raise ValueError("数据集没有任何特征列")

    null_exprs = [pl.col(col).is_null() for col in feature_columns]
    enriched = dataset.with_columns(
        pl.all_horizontal(null_exprs).alias("_all_null")
    )
    total = enriched.height
    dropped_null_label = enriched.filter(pl.col(LABEL_COL).is_null()).height
    dropped_all_null = enriched.filter(pl.col("_all_null")).height

    keep = pl.col(LABEL_COL).is_not_null() & ~pl.col("_all_null")
    kept = enriched.filter(keep).drop("_all_null")
    if max_rows > 0 and kept.height > max_rows:
        kept = kept.sample(n=max_rows, seed=seed)
    kept = kept.sort([INSTRUMENT_COL, DATE_COL])

    missing = missing_rate(kept, feature_columns)
    summary = DatasetSummary(
        total_rows=total,
        kept_rows=kept.height,
        dropped_null_label=dropped_null_label,
        dropped_all_null_feature=dropped_all_null,
        n_dates=kept[DATE_COL].n_unique(),
        n_instruments=kept[INSTRUMENT_COL].n_unique(),
        feature_columns=feature_columns,
        missing=missing,
    )
    return kept, summary


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


@dataclass
class TrainReport:
    """一次训练的产出摘要。"""

    model_dir: Path
    config_path: Path
    summary: DatasetSummary
    time_limit: float
    presets: str
    leaderboard: str | None = None
    feature_importance: str | None = None
    elapsed_seconds: float = 0.0


def train_baseline(
    data_dir: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    factor_library_dir: str | Path = DEFAULT_FACTOR_LIBRARY_DIR,
    start: date | None = None,
    end: date | None = None,
    presets: str = DEFAULT_PRESETS,
    time_limit: float = DEFAULT_TIME_LIMIT,
    max_rows: int = 0,
    horizon: int = DEFAULT_HORIZON,
    use_gpu: bool | None = None,
    seed: int = SAMPLING_SEED,
    trainer: BaselineTrainer | None = None,
    config_path: str | Path | None = None,
    universe: str | None = None,
) -> TrainReport:
    """跑完整训练链路并落盘模型与配置，返回 :class:`TrainReport`。

    ``trainer`` 用于依赖注入（测试传假 trainer，跳过真实 AutoGluon 训练）；省略时按
    参数构造 :class:`BaselineTrainer`。

    ``universe`` 非空时，行情先按池内 PIT 成员裁剪再进 :func:`build_dataset`，因此
    截面 z-score 只在池内计算（与全市场口径不同）；池名会写进训练配置，供回测校验。
    """
    started = time.monotonic()
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    bars = load_bars(data_dir, start=start, end=end)
    if bars.height == 0:
        raise ValueError(f"行情窗口 [{start}, {end}] 内没有数据：{data_dir}")

    if universe is not None:
        bars = filter_bars_to_universe(bars, universe, data_dir=data_dir)
        if bars.height == 0:
            raise ValueError(
                f"股票池 {universe!r} 在窗口 [{start}, {end}] 内没有行情"
            )

    factors = discover_factors(factor_library_dir)
    dataset, summary = prepare_training_dataset(
        bars, factors, horizon=horizon, max_rows=max_rows, seed=seed
    )
    summary.universe = universe
    summary.mean_daily_instruments = mean_daily_instruments(bars)
    logger.info(
        "训练集：%d 行 / %d 只 / %d 个交易日（原始 %d 行，label 空 %d，全特征空 %d）",
        summary.kept_rows,
        summary.n_instruments,
        summary.n_dates,
        summary.total_rows,
        summary.dropped_null_label,
        summary.dropped_all_null_feature,
    )

    active = trainer
    if active is None:
        active = BaselineTrainer(
            label=LABEL_COL,
            feature_columns=list(summary.feature_columns),
            presets=presets,
            time_limit=time_limit,
            path=model_dir,
            use_gpu=use_gpu,
        )
    active.train(dataset)
    active.save()

    resolved_config = (
        Path(config_path) if config_path is not None
        else default_train_config_path(model_dir)
    )
    write_train_config(
        resolved_config,
        TrainConfig(
            data_dir=str(data_dir),
            start=start.isoformat() if start is not None else None,
            end=end.isoformat() if end is not None else None,
            n_rows=summary.kept_rows,
            feature_columns=list(summary.feature_columns),
            presets=presets,
            time_limit=float(time_limit),
            horizon=horizon,
            max_rows=max_rows,
            universe=universe,
        ),
    )

    return TrainReport(
        model_dir=model_dir,
        config_path=resolved_config,
        summary=summary,
        time_limit=float(time_limit),
        presets=presets,
        leaderboard=_leaderboard_text(active),
        feature_importance=_feature_importance_text(
            active, dataset, summary.feature_columns
        ),
        elapsed_seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# 训练报告（leaderboard / 特征重要性）
# ---------------------------------------------------------------------------


def _leaderboard_text(trainer: Any) -> str | None:
    """取 ``predictor.leaderboard`` 前几行，取不到时返回 None。"""
    predictor = getattr(trainer, "predictor", None)
    if predictor is None:
        return None
    try:
        board = predictor.leaderboard(silent=True)
    except Exception as exc:  # noqa: BLE001 - 报告失败不影响训练产物
        logger.warning("leaderboard 不可用：%s", exc)
        return None
    try:
        return str(board.head(LEADERBOARD_ROWS).to_string())
    except Exception:  # noqa: BLE001
        return str(board)


def _feature_importance_text(
    trainer: Any,
    dataset: pl.DataFrame,
    feature_columns: tuple[str, ...],
    rows: int = FEATURE_IMPORTANCE_ROWS,
) -> str | None:
    """取 ``predictor.feature_importance``，取不到时返回 None。

    置换重要性需要一份带标签的数据；这里只取前 ``rows`` 行控制成本。``to_pandas``
    是喂给 AutoGluon 的唯一 pandas 边界（与本项目其它模块一致）。
    """
    predictor = getattr(trainer, "predictor", None)
    if predictor is None:
        return None
    sample = (
        dataset.select(*feature_columns, LABEL_COL)
        .drop_nulls(LABEL_COL)
        .head(rows)
    )
    if sample.height < 2:
        return None
    try:
        importance = predictor.feature_importance(sample.to_pandas(), silent=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("feature_importance 不可用：%s", exc)
        return None
    try:
        return str(importance.to_string())
    except Exception:  # noqa: BLE001
        return str(importance)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 AutoGluon 基线模型（M1 端到端）")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR, help=f"模型落盘目录，默认 {DEFAULT_MODEL_DIR}"
    )
    parser.add_argument(
        "--factor-library",
        default=str(DEFAULT_FACTOR_LIBRARY_DIR),
        help=f"因子库目录，默认 {DEFAULT_FACTOR_LIBRARY_DIR}",
    )
    parser.add_argument("--start", type=_parse_date, default=None, help="训练窗口起始日 YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, default=None, help="训练窗口结束日 YYYY-MM-DD")
    parser.add_argument(
        "--presets", default=DEFAULT_PRESETS, help=f"AutoGluon 预设，默认 {DEFAULT_PRESETS}"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=f"训练时限（秒），默认 {DEFAULT_TIME_LIMIT:.0f}",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="训练集行数上限，0 表示不截断（默认）",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON, help=f"标签持有期，默认 {DEFAULT_HORIZON}"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=None,
        help="强制使用 GPU；缺省自动探测",
    )
    parser.add_argument(
        "--no-gpu", dest="use_gpu", action="store_false", help="强制只用 CPU"
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="训练配置 JSON 落盘路径，默认放 model-dir 同级",
    )
    parser.add_argument(
        "--universe",
        default=None,
        help="股票池：命名池名（hs300/zz500/zz1000/zz2000）或自定义池文件路径；缺省全市场",
    )
    return parser


def _print_summary(report: TrainReport) -> None:
    summary = report.summary
    print("\n# 基线训练完成")
    print(f"模型目录：{report.model_dir}")
    print(f"训练配置：{report.config_path}")
    print(
        f"训练集：{summary.kept_rows} 行 / {summary.n_instruments} 只 / "
        f"{summary.n_dates} 个交易日"
    )
    if summary.universe is not None:
        print(
            f"股票池：{summary.universe}    池内日均票数 {summary.mean_daily_instruments:.2f}"
        )
    print(
        f"丢弃：label 空 {summary.dropped_null_label} 行，"
        f"全特征空 {summary.dropped_all_null_feature} 行（原始 {summary.total_rows} 行）"
    )
    print(f"特征列（{len(summary.feature_columns)}）：{', '.join(summary.feature_columns)}")
    if summary.missing.height:
        print("特征缺测率：")
        for row in summary.missing.sort("missing_rate", descending=True).iter_rows(
            named=True
        ):
            rate = row["missing_rate"]
            shown = "n/a" if rate is None else f"{rate:.2%}"
            print(f"  {row['feature']:<20} {shown}")
    print(f"presets={report.presets}    time_limit={report.time_limit:.0f}s")
    if report.leaderboard:
        print("\nleaderboard（前几行）：")
        print(report.leaderboard)
    if report.feature_importance:
        print("\nfeature_importance：")
        print(report.feature_importance)
    print(f"\n耗时 {report.elapsed_seconds:.1f}s")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    report = train_baseline(
        Path(args.data_dir),
        Path(args.model_dir),
        factor_library_dir=Path(args.factor_library),
        start=args.start,
        end=args.end,
        presets=args.presets,
        time_limit=args.time_limit,
        max_rows=args.max_rows,
        horizon=args.horizon,
        use_gpu=args.use_gpu,
        config_path=args.train_config,
        universe=args.universe,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
