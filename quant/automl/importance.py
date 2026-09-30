"""因子贡献度分析：逐窗口置换重要性钩子 + 跨窗口因子归因（issue #36）。

职责范围
--------
本模块把「AutoGluon 置换重要性」包装成 :mod:`quant.eval.model` 消费的
``importance_fn`` 钩子，并在跨窗口聚合之上产出因子维度的归因表：

- :func:`make_importance_fn`：逐窗口 ``(feature, importance)`` 长表钩子，
  采样训练集前 ``rows`` 行控制成本，异常时记 warning 返回 ``None``，
  不让重要性计算拖垮整轮 walk-forward 评估。
- :func:`aggregate_importance`：把逐窗口长表汇总为
  ``(feature, n_windows, mean, std, mean_rank)``，与
  :attr:`quant.eval.model.ModelEvaluation.feature_importance` 同形。
- :func:`factor_contribution`：在聚合表之上产出因子归因表，给出带符号的
  重要性均值、跨窗口稳定性（变异系数 ``std / mean``）与平均名次。
- :func:`write_contribution_report`：归因结果落盘 JSON + markdown。

口径
----
AutoGluon 的置换重要性可以为负：负值表示打乱该特征后验证指标反而变好，
即该特征对预测有害。归因表保留这个符号，不取绝对值。特征名即因子库里的
因子名，不需要额外的名称映射。

pandas 边界
-----------
``predictor.feature_importance`` 需要 ``pandas.DataFrame``；本模块只在
``sample.to_pandas()`` 这一处接触 pandas，其余数据流保持 polars。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from quant.automl.dataset import LABEL_COL
from quant.eval.model import (
    ImportanceFn,
    ModelEvaluation,
    ModelTrainer,
    _aggregate_importance,
    _frame_to_markdown,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

#: 置换重要性使用的训练样本行数上限（控制单窗口成本）。
DEFAULT_IMPORTANCE_ROWS: int = 2000

#: 归因报告的默认标题。
DEFAULT_CONTRIBUTION_TITLE: str = "因子贡献度分析"

#: 归因报告文件名前缀（``<stem>.json`` / ``<stem>.md``）。
DEFAULT_REPORT_STEM: str = "factor_contribution"

#: 归因表列顺序。
CONTRIBUTION_COLUMNS: tuple[str, ...] = (
    "factor",
    "n_windows",
    "mean_importance",
    "std",
    "stability",
    "mean_rank",
)

_CONTRIBUTION_SCHEMA = pl.Schema(
    {
        "factor": pl.String,
        "n_windows": pl.Int64,
        "mean_importance": pl.Float64,
        "std": pl.Float64,
        "stability": pl.Float64,
        "mean_rank": pl.Float64,
    }
)


# ---------------------------------------------------------------------------
# 逐窗口重要性钩子
# ---------------------------------------------------------------------------


def make_importance_fn(
    feature_columns: Sequence[str], rows: int = DEFAULT_IMPORTANCE_ROWS
) -> ImportanceFn:
    """返回逐窗口因子重要性钩子：AutoGluon 置换重要性，取训练集前 ``rows`` 行。

    钩子在任何异常（predictor 缺失、样本不足、AutoGluon 报错、结果解析失败）
    下都记 warning 并返回 ``None``，walk-forward 评估继续跑下一个窗口。
    """
    if rows < 1:
        raise ValueError(f"rows 必须是 >= 1 的整数，收到 {rows!r}")
    columns = [str(name) for name in feature_columns]

    def importance_fn(
        trainer: ModelTrainer, train_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        predictor = getattr(trainer, "predictor", None)
        if predictor is None:
            return None
        try:
            sample = (
                train_df.select(*columns, LABEL_COL)
                .drop_nulls(LABEL_COL)
                .head(rows)
            )
        except Exception as exc:  # noqa: BLE001 - 样本构造失败不拖垮评估
            logger.warning("窗口因子重要性样本构造失败：%s", exc)
            return None
        if sample.height < 2:
            logger.warning("窗口因子重要性样本不足（%d 行），跳过", sample.height)
            return None
        try:
            importance = predictor.feature_importance(sample.to_pandas(), silent=True)
        except Exception as exc:  # noqa: BLE001 - 重要性失败不拖垮评估
            logger.warning("窗口因子重要性不可用：%s", exc)
            return None
        try:
            return pl.DataFrame(
                {
                    "feature": [str(name) for name in importance.index],
                    "importance": [float(value) for value in importance["importance"]],
                }
            )
        except Exception as exc:  # noqa: BLE001 - 解析失败不拖垮评估
            logger.warning("窗口因子重要性解析失败：%s", exc)
            return None

    return importance_fn


# ---------------------------------------------------------------------------
# 跨窗口聚合与因子归因
# ---------------------------------------------------------------------------


def aggregate_importance(parts: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """把逐窗口 ``(feature, importance, _window)`` 汇总为跨窗口统计表。

    委托 :func:`quant.eval.model._aggregate_importance`，产出
    ``(feature, n_windows, mean, std, mean_rank)``，与 walk-forward 评估的
    ``feature_importance`` 完全同形，便于直接复用。
    """
    return _aggregate_importance(list(parts))


def factor_contribution(feature_importance: pl.DataFrame) -> pl.DataFrame:
    """在跨窗口聚合表之上产出因子维度归因表。

    参数
    ----
    feature_importance
        ``(feature, n_windows, mean, std, mean_rank)`` 的聚合表。``n_windows``
        与 ``std`` 允许缺列（缺列时归因表对应列记 null）。``mean`` / ``mean_rank``
        必须存在。

    返回
    ----
    ``(factor, n_windows, mean_importance, std, stability, mean_rank)``，按
    ``mean_rank`` 升序（1 = 最重要）排列，同名词次按重要性均值降序破平。
    ``stability = std / mean``（变异系数）；``mean`` 为 0 或缺失时记 null。
    重要性均值保留符号，负值表示该因子对预测有害。

    异常
    ----
    输入非空但缺少必需列时抛 :class:`ValueError`；空表按归因表 schema 返回空表。
    """
    if feature_importance.height == 0:
        return pl.DataFrame(schema=_CONTRIBUTION_SCHEMA)

    required = ("feature", "mean", "mean_rank")
    missing = [col for col in required if col not in feature_importance.columns]
    if missing:
        raise ValueError(
            f"重要性聚合表缺少必需列：{missing}；实际列：{feature_importance.columns}"
        )

    mean = pl.col("mean").cast(pl.Float64)
    std = (
        pl.col("std").cast(pl.Float64)
        if "std" in feature_importance.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    n_windows = (
        pl.col("n_windows").cast(pl.Int64)
        if "n_windows" in feature_importance.columns
        else pl.lit(None, dtype=pl.Int64)
    )
    stability = (
        pl.when(mean.is_not_null() & std.is_not_null() & (mean != 0.0))
        .then(std / mean)
        .otherwise(None)
    )

    return (
        feature_importance.select(
            pl.col("feature").cast(pl.String).alias("factor"),
            n_windows.alias("n_windows"),
            mean.alias("mean_importance"),
            std.alias("std"),
            stability.alias("stability"),
            pl.col("mean_rank").cast(pl.Float64).alias("mean_rank"),
        )
        .select(*CONTRIBUTION_COLUMNS)
        .sort(
            ["mean_rank", "mean_importance"],
            descending=[False, True],
            nulls_last=True,
        )
    )


# ---------------------------------------------------------------------------
# 报告落盘
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContributionReport:
    """归因报告的落盘结果。

    Attributes
    ----------
    table:
        因子归因表 ``(factor, n_windows, mean_importance, std, stability,
        mean_rank)``。
    aggregated:
        归因表的原始素材，跨窗口聚合表 ``(feature, n_windows, mean, std,
        mean_rank)``。
    json_path / markdown_path:
        两份报告的落盘路径。
    """

    table: pl.DataFrame
    aggregated: pl.DataFrame
    json_path: Path
    markdown_path: Path


def write_contribution_report(
    source: ModelEvaluation | pl.DataFrame,
    out_dir: str | Path,
    *,
    title: str = DEFAULT_CONTRIBUTION_TITLE,
    stem: str = DEFAULT_REPORT_STEM,
) -> ContributionReport:
    """把归因结果落盘为 ``<stem>.json`` 与 ``<stem>.md``。

    ``source`` 可以是 :class:`quant.eval.model.ModelEvaluation`（取其
    ``feature_importance``），也可以直接是跨窗口聚合表。
    """
    aggregated = _as_importance_frame(source)
    table = factor_contribution(aggregated)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    markdown_path = out / f"{stem}.md"

    payload: dict[str, Any] = {
        "title": title,
        "n_factors": table.height,
        "sort": "mean_rank ascending",
        "factor_contribution": table.to_dicts(),
        "feature_importance": aggregated.to_dicts(),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(
        render_contribution_markdown(table, title=title), encoding="utf-8"
    )
    return ContributionReport(
        table=table,
        aggregated=aggregated,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def render_contribution_markdown(
    table: pl.DataFrame, *, title: str = DEFAULT_CONTRIBUTION_TITLE
) -> str:
    """把因子归因表渲染为 markdown。"""
    lines = [f"# {title}", ""]
    lines.append(f"- 因子数：{table.height}")
    lines.append("- 排序：平均名次 mean_rank 升序（1 = 最重要）")
    lines.append(
        "- 稳定性 stability = std / mean（变异系数）；mean_importance 可为负，"
        "负值表示该因子对预测有害"
    )
    lines.append("")
    lines.append("## 因子归因表")
    lines.append("")
    lines.append(_frame_to_markdown(table))
    lines.append("")
    return "\n".join(lines) + "\n"


def _as_importance_frame(source: ModelEvaluation | pl.DataFrame) -> pl.DataFrame:
    """从 ``ModelEvaluation`` 或裸聚合表取出跨窗口重要性表。"""
    if isinstance(source, pl.DataFrame):
        return source
    importance = getattr(source, "feature_importance", None)
    if isinstance(importance, pl.DataFrame):
        return importance
    raise TypeError(
        "source 必须是 ModelEvaluation 或重要性聚合表，"
        f"收到 {type(source).__name__}"
    )


__all__ = [
    "CONTRIBUTION_COLUMNS",
    "DEFAULT_CONTRIBUTION_TITLE",
    "DEFAULT_IMPORTANCE_ROWS",
    "DEFAULT_REPORT_STEM",
    "ContributionReport",
    "aggregate_importance",
    "factor_contribution",
    "make_importance_fn",
    "render_contribution_markdown",
    "write_contribution_report",
]
