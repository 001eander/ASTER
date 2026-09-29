"""模型级评估：walk-forward 滚动重训 → 纯样本外拼接 → 指标汇总（issue #33）。

与因子级评估的区别
------------------
因子级评估（:mod:`quant.eval.factor`）考察单个因子的截面预测力；本模块考察
AutoML 合成信号在**真实重训节奏**下的样本外表现：每个滚动窗口各自训练、各自
预测，把所有窗口的样本外打分按日期拼接成一条序列，再交给
:mod:`quant.eval.metrics` 的指标内核汇总。AutoGluon 只负责单窗口训练，窗口
切分归本模块（架构 3.4 / M5 决策注）。

无前视
------
- 训练切片 ``date <= train_end``，测试切片 ``date >= test_start``，两者之间
  留 ``embargo_days`` 个交易日（默认 = 标签持有期 ``horizon``）。训练集最后一行
  的标签用到 ``train_end + 1 + horizon`` 的价格，embargo 保证它不晚于测试第一行
  的建仓日，训练标签与测试区间不重叠。
- 特征列由上游 ``build_dataset`` 逐日截面 z-score 产生，截面内标准化不携带
  跨期信息，整表一次构造后按日期切片是安全的。

产出
----
:class:`ModelEvaluation` 汇总三层内容：整条样本外序列的 IC / ICIR / 分层 /
换手（复用 :mod:`quant.eval.metrics`），逐窗口指标表（滚动稳定性），以及可选的
逐窗口因子重要性与 leaderboard（通过 ``importance_fn`` / ``leaderboard_fn``
钩子注入，#36 提供重要性实现）。报告可落盘 JSON 与 markdown。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl

from quant.automl.dataset import (
    DATE_COL,
    DELAY_COL,
    INSTRUMENT_COL,
    LABEL_COL,
)
from quant.eval.metrics import (
    DEFAULT_LAYERS,
    ICSummary,
    ic_series,
    layer_monotonicity,
    layered_returns,
    summarize_ic,
    turnover,
)
from quant.labels.open_to_open import DEFAULT_HORIZON

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

#: 滚动训练窗口的交易日数。
DEFAULT_TRAIN_WINDOW_DAYS: int = 500

#: 每个窗口的样本外测试交易日数（= 重训周期）。
DEFAULT_TEST_DAYS: int = 20

#: 扩张窗口模式下的最小训练交易日数。
DEFAULT_MIN_TRAIN_DAYS: int = 250

#: 打分 Top-N 组合的换手臂数。
DEFAULT_TOP_N: int = 50

#: 特征列之外的元信息列（特征列 = 数据集其余列）。
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    DATE_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    DELAY_COL,
)

#: 打分列名（与 :data:`quant.automl.trainer.SCORE_COL` 一致，本模块不引 autogluon）。
SCORE_COL: str = "score"

#: 指标内核的因子列名约定（:mod:`quant.eval.metrics` 的 FACTOR_COL）。
_FACTOR_COL: str = "factor"

_IMPORTANCE_SCHEMA = pl.Schema({"feature": pl.String, "importance": pl.Float64})

_WINDOW_SCHEMA = pl.Schema(
    {
        "window": pl.Int64,
        "train_start": pl.Date,
        "train_end": pl.Date,
        "test_start": pl.Date,
        "test_end": pl.Date,
        "n_train_rows": pl.Int64,
        "n_test_rows": pl.Int64,
        "ic_mean": pl.Float64,
        "ic_std": pl.Float64,
        "icir": pl.Float64,
        "ic_win_rate": pl.Float64,
        "ic_days": pl.Int64,
    }
)

_OOS_SCHEMA = pl.Schema(
    {
        DATE_COL: pl.Date,
        INSTRUMENT_COL: pl.String,
        _FACTOR_COL: pl.Float64,
        LABEL_COL: pl.Float64,
    }
)


# ---------------------------------------------------------------------------
# 训练器协议
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelTrainer(Protocol):
    """walk-forward 单窗口训练器的最小协议。

    :class:`quant.automl.trainer.BaselineTrainer` 天然满足该协议；测试与
    其它模型实现只需提供 ``train`` / ``predict`` 两个方法。
    """

    def train(
        self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None
    ) -> ModelTrainer:
        """在单个窗口的训练切片上拟合。"""
        ...

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """对 ``df`` 打分，返回 ``(date, instrument, score)``。"""
        ...


#: 每个窗口产出 ``(feature, importance)`` 长表的钩子；返回 None 表示该窗口不可用。
ImportanceFn = Callable[[ModelTrainer, pl.DataFrame], pl.DataFrame | None]

#: 每个窗口产出 leaderboard 文本的钩子；返回 None 表示该窗口不可用。
LeaderboardFn = Callable[[ModelTrainer], str | None]


# ---------------------------------------------------------------------------
# 窗口切分
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """walk-forward 的窗口配置。

    Attributes
    ----------
    train_window_days:
        滚动模式下每个窗口的训练交易日数。``expanding=True`` 时该值只作为
        首个窗口的下界（训练起点固定为数据起点）。
    test_days:
        每个窗口的样本外测试交易日数，即重训周期。
    embargo_days:
        训练截止日到测试起始日之间的隔离交易日数。必须 ``>= horizon``，
        保证训练标签（T+1 开盘 → T+1+horizon 开盘）不伸进测试区间。
    min_train_days:
        扩张窗口模式下允许开窗的最小训练交易日数。滚动模式每个窗口恒为
        ``train_window_days`` 天，该参数不生效。
    expanding:
        ``True`` 时训练起点固定（扩张窗口），``False`` 时定长滚动。
    """

    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS
    test_days: int = DEFAULT_TEST_DAYS
    embargo_days: int = DEFAULT_HORIZON
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS
    expanding: bool = False

    def __post_init__(self) -> None:
        for name in ("train_window_days", "test_days", "min_train_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是 >= 1 的整数，收到 {value!r}")
        if (
            isinstance(self.embargo_days, bool)
            or not isinstance(self.embargo_days, int)
            or self.embargo_days < 0
        ):
            raise ValueError(
                f"embargo_days 必须是 >= 0 的整数，收到 {self.embargo_days!r}"
            )


@dataclass(frozen=True)
class WalkWindow:
    """一个 walk-forward 窗口的日期边界（均为交易日，闭区间）。"""

    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def split_windows(
    dates: Sequence[date], config: WalkForwardConfig | None = None
) -> list[WalkWindow]:
    """把升序交易日序列切成 walk-forward 窗口。

    测试块定长 ``test_days`` 顺序推进，最后不足一个完整块的尾巴也保留为一个
    窗口（样本外数据不应被丢弃）。每个窗口的训练区间为测试起点往前隔
    ``embargo_days`` 个交易日、再往前取 ``train_window_days`` 天（滚动）或
    取到序列起点（扩张）。有效窗口数为零时抛 :class:`ValueError`。
    """
    cfg = config if config is not None else WalkForwardConfig()
    ordered = sorted(set(dates))
    n = len(ordered)
    if n == 0:
        raise ValueError("交易日序列为空，无法切分窗口")

    first_test = (
        cfg.min_train_days if cfg.expanding else cfg.train_window_days
    ) + cfg.embargo_days
    windows: list[WalkWindow] = []
    test_lo = first_test
    while test_lo < n:
        train_hi = test_lo - cfg.embargo_days - 1
        train_lo = 0 if cfg.expanding else train_hi - cfg.train_window_days + 1
        train_days = train_hi - train_lo + 1
        enough = train_days >= cfg.min_train_days if cfg.expanding else True
        if train_lo >= 0 and enough:
            test_hi = min(test_lo + cfg.test_days - 1, n - 1)
            windows.append(
                WalkWindow(
                    index=len(windows),
                    train_start=ordered[train_lo],
                    train_end=ordered[train_hi],
                    test_start=ordered[test_lo],
                    test_end=ordered[test_hi],
                )
            )
        test_lo += cfg.test_days
    if not windows:
        raise ValueError(
            f"交易日数 {n} 不足以开出任何窗口（需要 >= "
            f"{first_test + 1} 个交易日）"
        )
    return windows


# ---------------------------------------------------------------------------
# 评估主流程
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowEvaluation:
    """单个窗口的样本外指标。"""

    window: WalkWindow
    n_train_rows: int
    n_test_rows: int
    ic: ICSummary


@dataclass
class ModelEvaluation:
    """walk-forward 模型级评估的完整产出。

    Attributes
    ----------
    config:
        本次评估的窗口配置。
    windows:
        逐窗口指标表（滚动稳定性的原始素材）。
    oos:
        全部窗口拼接的纯样本外长表 ``(date, instrument, factor, label)``。
    ic:
        样本外序列的 IC 汇总。
    monotonicity:
        分层单调性（层号 vs 层均收益的 Spearman 相关）。
    layered:
        分层收益表 ``(date, layer, ret, count)``。
    turnover_mean:
        Top-N 组合日均换手率（首日无前值，剔除）。
    feature_importance:
        逐特征跨窗口重要性汇总 ``(feature, n_windows, mean, std, mean_rank)``；
        未提供 ``importance_fn`` 时为空表。
    leaderboards:
        ``{window_index: leaderboard 文本}``，未提供 ``leaderboard_fn`` 时为空。
    stability:
        滚动稳定性摘要：窗口 IC 均值的均值 / 标准差、IC 均值为正的窗口占比。
    """

    config: WalkForwardConfig
    windows: pl.DataFrame
    oos: pl.DataFrame
    ic: ICSummary
    monotonicity: float | None
    layered: pl.DataFrame
    turnover_mean: float | None
    feature_importance: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "feature": pl.String,
                "n_windows": pl.Int64,
                "mean": pl.Float64,
                "std": pl.Float64,
                "mean_rank": pl.Float64,
            }
        )
    )
    leaderboards: dict[int, str] = field(default_factory=dict)
    stability: dict[str, float | int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（不含 ``oos`` 原始明细）。"""
        return {
            "config": {
                "train_window_days": self.config.train_window_days,
                "test_days": self.config.test_days,
                "embargo_days": self.config.embargo_days,
                "min_train_days": self.config.min_train_days,
                "expanding": self.config.expanding,
            },
            "oos_range": {
                "start": _iso(self.oos[DATE_COL].min()),
                "end": _iso(self.oos[DATE_COL].max()),
                "n_days": int(self.oos[DATE_COL].n_unique()),
                "n_rows": int(self.oos.height),
            },
            "ic": {
                "mean": self.ic.mean,
                "std": self.ic.std,
                "icir": self.ic.icir,
                "ic_win_rate": self.ic.ic_win_rate,
                "n_days": self.ic.n_days,
            },
            "monotonicity": self.monotonicity,
            "turnover_mean": self.turnover_mean,
            "stability": dict(self.stability),
            "windows": _records(self.windows),
            "layered_daily_mean": _records(
                self.layered.group_by("layer")
                .agg(pl.col("ret").mean().alias("ret_mean"))
                .sort("layer")
            )
            if self.layered.height
            else [],
            "feature_importance": _records(self.feature_importance),
            "leaderboards": {str(k): v for k, v in self.leaderboards.items()},
        }

    def write_json(self, path: str | Path) -> Path:
        """落盘 JSON 报告。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def render_markdown(self, *, title: str = "walk-forward 模型级评估") -> str:
        """渲染 markdown 报告。"""
        lines = [f"# {title}", ""]
        cfg = self.config
        mode = "扩张窗口" if cfg.expanding else "滚动窗口"
        lines.append(
            f"- 窗口：{mode}，训练 {cfg.train_window_days} 交易日，"
            f"测试 {cfg.test_days} 交易日，embargo {cfg.embargo_days} 个交易日"
        )
        lines.append(
            f"- 样本外区间：{self.to_dict()['oos_range']['start']} ~ "
            f"{self.to_dict()['oos_range']['end']}，共 {self.ic.n_days} 个有效 IC 日"
        )
        lines.append("")
        lines.append("## 样本外 IC 汇总")
        lines.append("")
        lines.append(
            f"- RankIC 均值 {_fmt(self.ic.mean)}，ICIR {_fmt(self.ic.icir)}，"
            f"胜率 {_fmt(self.ic.ic_win_rate)}"
        )
        lines.append(f"- 分层单调性 {_fmt(self.monotonicity)}")
        lines.append(f"- Top-N 日均换手 {_fmt(self.turnover_mean)}")
        if self.stability:
            lines.append(
                f"- 滚动稳定性：窗口 IC 均值 {_fmt(self.stability.get('window_ic_mean'))}"
                f" ± {_fmt(self.stability.get('window_ic_std'))}，"
                f"正窗口占比 {_fmt(self.stability.get('window_positive_rate'))}"
            )
        lines.append("")
        lines.append("## 逐窗口指标")
        lines.append("")
        lines.append(_frame_to_markdown(self.windows))
        lines.append("")
        if self.layered.height:
            lines.append("## 分层收益（跨日均值）")
            lines.append("")
            lines.append(
                _frame_to_markdown(
                    self.layered.group_by("layer")
                    .agg(pl.col("ret").mean().alias("ret_mean"))
                    .sort("layer")
                )
            )
            lines.append("")
        if self.feature_importance.height:
            lines.append("## 因子重要性（跨窗口汇总）")
            lines.append("")
            lines.append(_frame_to_markdown(self.feature_importance))
            lines.append("")
        if self.leaderboards:
            lines.append("## 各窗口 leaderboard")
            lines.append("")
            for index in sorted(self.leaderboards):
                lines.append(f"### window {index}")
                lines.append("")
                lines.append("```")
                lines.append(self.leaderboards[index])
                lines.append("```")
                lines.append("")
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path, *, title: str = "walk-forward 模型级评估") -> Path:
        """落盘 markdown 报告。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_markdown(title=title), encoding="utf-8")
        return path


def evaluate_walk_forward(
    dataset: pl.DataFrame,
    trainer_factory: Callable[[], ModelTrainer],
    config: WalkForwardConfig | None = None,
    *,
    n_layers: int = DEFAULT_LAYERS,
    top_n: int = DEFAULT_TOP_N,
    min_ic_count: int | None = None,
    importance_fn: ImportanceFn | None = None,
    leaderboard_fn: LeaderboardFn | None = None,
) -> ModelEvaluation:
    """执行 walk-forward 模型级评估。

    参数
    ----
    dataset:
        ``build_dataset`` 形态的宽表 ``(date, instrument, <特征...>, label,
        delay_days)``，覆盖全部训练与测试区间。函数内部只做按日期切片，
        不改动行的时间语义。
    trainer_factory:
        无参工厂，每个窗口调用一次得到全新的训练器实例。
    config:
        窗口配置；缺省为 :class:`WalkForwardConfig` 默认值。
    n_layers / top_n / min_ic_count:
        透传给 :mod:`quant.eval.metrics` 的分层数、Top-N 换手臂数与
        IC 最小截面证券数（缺省用 metrics 模块默认值）。
    importance_fn / leaderboard_fn:
        可选钩子，分别在每个窗口训练完成后调用，产出因子重要性长表与
        leaderboard 文本（#36 提供重要性实现）。

    返回
    ----
    :class:`ModelEvaluation`，``oos`` 为拼接后的纯样本外 ``(date, instrument,
    factor, label)`` 长表。
    """
    cfg = config if config is not None else WalkForwardConfig()
    _require_columns(dataset, (DATE_COL, INSTRUMENT_COL, LABEL_COL))
    feature_columns = [
        col for col in dataset.columns if col not in NON_FEATURE_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("数据集没有任何特征列")

    dates = dataset.get_column(DATE_COL).unique().sort().to_list()
    windows = split_windows(dates, cfg)
    logger.info("walk-forward：%d 个窗口，样本外 %s ~ %s", len(windows),
                windows[0].test_start, windows[-1].test_end)

    window_rows: list[dict[str, Any]] = []
    oos_parts: list[pl.DataFrame] = []
    importance_parts: list[pl.DataFrame] = []
    leaderboards: dict[int, str] = {}

    labels = dataset.select(DATE_COL, INSTRUMENT_COL, LABEL_COL)
    for window in windows:
        train_df = _train_slice(dataset, window, feature_columns)
        test_df = dataset.filter(
            (pl.col(DATE_COL) >= window.test_start)
            & (pl.col(DATE_COL) <= window.test_end)
        )
        trainer = trainer_factory()
        trainer.train(train_df)
        scores = trainer.predict(test_df)
        window_oos = _attach_labels(scores, labels, window)
        oos_parts.append(window_oos)

        ic_kwargs: dict[str, Any] = {}
        if min_ic_count is not None:
            ic_kwargs["min_count"] = min_ic_count
        window_ic = summarize_ic(ic_series(window_oos, **ic_kwargs))
        window_rows.append(
            {
                "window": window.index,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "n_train_rows": train_df.height,
                "n_test_rows": test_df.height,
                "ic_mean": window_ic.mean,
                "ic_std": window_ic.std,
                "icir": window_ic.icir,
                "ic_win_rate": window_ic.ic_win_rate,
                "ic_days": window_ic.n_days,
            }
        )
        logger.info(
            "window %d：train %s~%s（%d 行），test %s~%s（%d 行），"
            "RankIC 均值 %s",
            window.index,
            window.train_start,
            window.train_end,
            train_df.height,
            window.test_start,
            window.test_end,
            test_df.height,
            _fmt(window_ic.mean),
        )

        if importance_fn is not None:
            importance = importance_fn(trainer, train_df)
            if importance is not None and importance.height:
                importance_parts.append(
                    importance.select(
                        pl.col("feature").cast(pl.String),
                        pl.col("importance").cast(pl.Float64),
                    ).with_columns(pl.lit(window.index).alias("_window"))
                )
        if leaderboard_fn is not None:
            board = leaderboard_fn(trainer)
            if board:
                leaderboards[window.index] = board

    oos = (
        pl.concat(oos_parts).sort(DATE_COL, INSTRUMENT_COL)
        if oos_parts
        else pl.DataFrame(schema=_OOS_SCHEMA)
    )

    ic_kwargs = {}
    if min_ic_count is not None:
        ic_kwargs["min_count"] = min_ic_count
    ic = summarize_ic(ic_series(oos, **ic_kwargs))

    if oos.height:
        layered = layered_returns(oos, n_layers=n_layers)
        monotonicity = layer_monotonicity(layered)
        turnover_series = turnover(oos, top_n=top_n)
        valid_turnover = turnover_series.get_column("turnover").drop_nulls()
        turnover_mean = (
            float(valid_turnover.mean()) if valid_turnover.len() else None
        )
    else:
        layered = pl.DataFrame(
            schema={"date": pl.Date, "layer": pl.Int64, "ret": pl.Float64, "count": pl.Int64}
        )
        monotonicity = None
        turnover_mean = None

    window_frame = pl.DataFrame(window_rows, schema=_WINDOW_SCHEMA)
    return ModelEvaluation(
        config=cfg,
        windows=window_frame,
        oos=oos,
        ic=ic,
        monotonicity=monotonicity,
        layered=layered,
        turnover_mean=turnover_mean,
        feature_importance=_aggregate_importance(importance_parts),
        leaderboards=leaderboards,
        stability=_stability(window_frame),
    )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _require_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"输入缺少必需列：{missing}；实际列：{df.columns}")


def _train_slice(
    dataset: pl.DataFrame, window: WalkWindow, feature_columns: Sequence[str]
) -> pl.DataFrame:
    """切出窗口训练集，丢弃不可训练行（label 空 / 全部特征空）。

    与 ``scripts/train_baseline.prepare_training_dataset`` 的丢弃规则一致：
    训练集不喂入无效标签与空信息样本。
    """
    sliced = dataset.filter(
        (pl.col(DATE_COL) >= window.train_start)
        & (pl.col(DATE_COL) <= window.train_end)
    )
    any_feature = pl.any_horizontal(*[pl.col(c).is_not_null() for c in feature_columns])
    kept = sliced.filter(pl.col(LABEL_COL).is_not_null() & any_feature)
    if kept.height == 0:
        raise ValueError(
            f"窗口 {window.index}（{window.train_start} ~ {window.train_end}）"
            "没有任何可训练行"
        )
    return kept


def _attach_labels(
    scores: pl.DataFrame, labels: pl.DataFrame, window: WalkWindow
) -> pl.DataFrame:
    """把窗口打分拼上标签，规整为指标内核需要的 ``(date, instrument, factor, label)``。"""
    _require_columns(scores, (DATE_COL, INSTRUMENT_COL, SCORE_COL))
    merged = scores.select(DATE_COL, INSTRUMENT_COL, SCORE_COL).join(
        labels, on=[DATE_COL, INSTRUMENT_COL], how="left"
    )
    outside = merged.filter(
        (pl.col(DATE_COL) < window.test_start) | (pl.col(DATE_COL) > window.test_end)
    )
    if outside.height:
        raise ValueError(
            f"窗口 {window.index} 的打分越出测试区间 "
            f"[{window.test_start}, {window.test_end}]"
        )
    return merged.select(
        DATE_COL,
        INSTRUMENT_COL,
        pl.col(SCORE_COL).cast(pl.Float64).alias(_FACTOR_COL),
        LABEL_COL,
    )


def _aggregate_importance(parts: list[pl.DataFrame]) -> pl.DataFrame:
    """把逐窗口 ``(feature, importance, _window)`` 汇总为跨窗口统计。

    ``mean`` / ``std`` 为重要性取值的跨窗口均值 / 样本标准差；``mean_rank``
    为窗口内重要性降序名次（1 = 最重要）的跨窗口均值。
    """
    schema = {
        "feature": pl.String,
        "n_windows": pl.Int64,
        "mean": pl.Float64,
        "std": pl.Float64,
        "mean_rank": pl.Float64,
    }
    if not parts:
        return pl.DataFrame(schema=schema)
    stacked = pl.concat(parts).with_columns(
        pl.col("importance")
        .rank(method="average", descending=True)
        .over("_window")
        .alias("_rank")
    )
    return (
        stacked.group_by("feature")
        .agg(
            pl.col("_window").n_unique().alias("n_windows"),
            pl.col("importance").mean().alias("mean"),
            pl.col("importance").std().alias("std"),
            pl.col("_rank").mean().alias("mean_rank"),
        )
        .sort("mean", descending=True)
        .cast(schema)
    )


def _stability(windows: pl.DataFrame) -> dict[str, float | int | None]:
    """滚动稳定性摘要：逐窗口 RankIC 均值的离散程度。"""
    means = windows.get_column("ic_mean").drop_nulls()
    n = means.len()
    if n == 0:
        return {
            "n_windows": int(windows.height),
            "window_ic_mean": None,
            "window_ic_std": None,
            "window_positive_rate": None,
        }
    return {
        "n_windows": int(windows.height),
        "window_ic_mean": float(means.mean()),
        "window_ic_std": float(means.std()) if n >= 2 else None,
        "window_positive_rate": float((means > 0).sum()) / n,
    }


def _records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        out.append(
            {
                key: (value.isoformat() if isinstance(value, date) else value)
                for key, value in row.items()
            }
        )
    return out


def _iso(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    return None if value is None else str(value)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _frame_to_markdown(df: pl.DataFrame, *, digits: int = 4) -> str:
    """把宽表渲染为 markdown 表（日期列输出 ISO 字符串）。

    与 :func:`quant.eval.risk_report.frame_to_markdown` 同风格；单独实现一份
    以免本模块拖入 cvxpy 依赖链。
    """
    if df.height == 0:
        return "（空）"
    headers = ["日期" if col == DATE_COL else str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in df.iter_rows():
        cells: list[str] = []
        for col, value in zip(df.columns, row):
            if col == DATE_COL or isinstance(value, date):
                cells.append(_iso(value) or "-")
            elif value is None:
                cells.append("-")
            elif isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MIN_TRAIN_DAYS",
    "DEFAULT_TEST_DAYS",
    "DEFAULT_TOP_N",
    "DEFAULT_TRAIN_WINDOW_DAYS",
    "NON_FEATURE_COLUMNS",
    "SCORE_COL",
    "ImportanceFn",
    "LeaderboardFn",
    "ModelEvaluation",
    "ModelTrainer",
    "WalkForwardConfig",
    "WalkWindow",
    "WindowEvaluation",
    "evaluate_walk_forward",
    "split_windows",
]
