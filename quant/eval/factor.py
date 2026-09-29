"""因子级评估管线：schema 校验 → 截断重算 → 复杂度 → IC 门控 → ``score.json``。

本模块把 issue #17（``factor_api`` 接口契约与 schema 校验）、#18（截断重算检测前视）、
#19（复杂度度量）与既有的指标内核 ``quant.eval.metrics`` 串成一条流水线，输入一个
因子 ``.py`` 文件与一份常驻内存的行情面板，输出 :class:`FactorEvaluation`。CLI
``python -m quant.eval.factor`` 在其上落盘 ``score.json``，供 hyra-pi 内循环沙盒
（``eval.sh``）读取。

管线顺序与短路
--------------
每一步失败即返回，``stage`` 指明失败发生处，``error`` 为中文明细：

1. ``complexity``：``measure_complexity`` 超阈值（#19）。
2. ``load``：``load_factor`` 无法加载为可用的 ``compute``（#17）。
3. ``schema``：输入面板 ``select`` 前 10 列后过 ``validate_input`` 失败。这是管线
   自检，防止缓存 schema 漂移；因子输出过 ``validate_output`` 失败同样归此类。
4. ``compute``：因子 ``compute`` 抛异常，``error`` 取 traceback 末行。
5. ``truncation``：``check_truncation`` 检出前视，或数据太短不可检（#18）。
6. ``metrics``：IC / 分层 / 换手等指标计算失败，或行为相关性查重失败（registry 非法）。
7. ``done``：全流程跑完，``gate_passed`` 表示是否过门控。

门控除 IC / ICIR / 分层单调性外，还含行为相关性：新因子与库内 pool 因子的
``max(|逐日截面相关均值|)`` 超过 :data:`MAX_CORR_REJECT` 时判冗余并拒绝
（issue #26）。本阶段只拒绝与落盘指标，不折减 ``score``。

奖励信号一致性
--------------
跑完评估的因子，``score = rank_ic_mean``，这是连续的质量信号，取值可负，门控只决定
入库与否（``gate_passed``），不截断分数。Agent 因此能从分数梯度学习，不存在「分数高
但被拒」的隐藏规则（AGENTS.md 量化纪律第 4 条）。硬失败（``stage != "done"``）不写
``score.json``，由退出码表达。

性能预算
--------
数据单次加载后常驻内存，截断重算共享同一份输入裁剪（polars ``filter`` 廉价），全流程
向量化。单因子全市场评估目标秒级；含截断重算的数次重算在内，宽限 60s。

n_days 为 0（没有任何有效 IC 日）时 ``score`` 写 0.0，并在 ``notes`` 说明。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from quant.data.cache import load_bars
from quant.eval.metrics import (
    DEFAULT_LAYERS,
    ic_series,
    layer_monotonicity,
    layered_returns,
    summarize_ic,
    turnover,
)
from quant.factor_api.complexity import (
    MAX_AST_NODES,
    MAX_CYCLOMATIC,
    MAX_FIELDS,
    ComplexityReport,
    measure_complexity,
)
from quant.factor_api.loader import FactorLoadError, load_factor
from quant.factor_api.schema import validate_input, validate_output
from quant.factor_api.spec import FACTOR_INPUT_COLUMNS
from quant.factor_api.truncation import (
    DEFAULT_WARMUP_DAYS,
    TruncationResult,
    check_truncation,
)
from quant.factor_lib.correlation import load_library_values, max_library_corr
from quant.labels import attach_label

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: RankIC 均值下限，低于此值不过门控。
RANK_IC_MIN: float = 0.02

#: ICIR 下限，低于此值不过门控；ICIR 为 None 时同样不过。
ICIR_MIN: float = 0.2

#: 分层单调性下限（Spearman），严格要求大于此值。
MONO_MIN: float = 0.0

#: 与库内因子行为相关性的拒绝阈值：``|max_corr| > 此值`` 即判冗余，不过门控。
MAX_CORR_REJECT: float = 0.7

#: 换手率口径：Top-50 组合。
TURNOVER_TOP_N: int = 50

#: 分层评估的层数，复用指标内核的默认值。
EVAL_LAYERS: int = DEFAULT_LAYERS

#: CLI 默认数据目录。
DEFAULT_DATA_DIR: str = "data"

#: CLI 默认 score.json 输出路径。
DEFAULT_OUT: str = "score.json"

#: CLI 默认因子库目录（含 registry.json）。
DEFAULT_FACTOR_LIBRARY_DIR: str = "factor_library"

#: 默认标签 horizon，与 ``quant.labels`` 一致。
DEFAULT_HORIZON: int = 1

#: ``FactorEvaluation.metrics`` 的字段类型。
Metrics = dict[str, float | int | None]


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorEvaluation:
    """单个因子的评估结果。

    :param ok: 管线是否跑完（``stage == "done"``）。
    :param stage: 失败发生的阶段：``load`` / ``complexity`` / ``schema`` /
        ``truncation`` / ``compute`` / ``metrics``，成功为 ``"done"``。
    :param error: 失败明细，成功为 None。
    :param gate_passed: 是否过 IC 门控；硬失败时为 False。
    :param metrics: 指标明细，键为 ``rank_ic_mean`` / ``rank_ic_std`` / ``icir`` /
        ``ic_win_rate`` / ``n_days`` / ``mono`` / ``turnover_mean`` / ``max_corr``，
        不可得为 None。
    :param complexity: 复杂度度量明细，度量未执行时为 None。
    :param truncation: 截断重算明细，检测未执行时为 None。
    """

    ok: bool
    stage: str
    error: str | None
    gate_passed: bool
    metrics: Metrics
    complexity: dict[str, object] | None
    truncation: dict[str, object] | None


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _complexity_dict(report: ComplexityReport) -> dict[str, object]:
    """把 :class:`ComplexityReport` 展开成可 JSON 序列化的字典。"""
    return {
        "ok": report.ok,
        "cyclomatic_max": report.cyclomatic_max,
        "cyclomatic_total": report.cyclomatic_total,
        "ast_nodes": report.ast_nodes,
        "fields_used": list(report.fields_used),
        "violations": list(report.violations),
    }


def _truncation_dict(result: TruncationResult) -> dict[str, object]:
    """把 :class:`TruncationResult` 展开成可 JSON 序列化的字典。"""
    return {
        "ok": result.ok,
        "n_checks": result.n_checks,
        "skipped_reason": result.skipped_reason,
        "failures": [
            {
                "date": failure.date.isoformat(),
                "n_mismatch": failure.n_mismatch,
                "max_abs_diff": failure.max_abs_diff,
            }
            for failure in result.failures
        ],
    }


def _failure(
    stage: str,
    error: str,
    *,
    complexity: dict[str, object] | None = None,
    truncation: dict[str, object] | None = None,
) -> FactorEvaluation:
    """构造一个硬失败的评估结果。"""
    return FactorEvaluation(
        ok=False,
        stage=stage,
        error=error,
        gate_passed=False,
        metrics={},
        complexity=complexity,
        truncation=truncation,
    )


def _factor_warmup(compute: object) -> int:
    """读取因子模块级 ``WARMUP`` 常量，缺省或非法时回退到 ``DEFAULT_WARMUP_DAYS``。"""
    module = sys.modules.get(getattr(compute, "__module__", ""))
    value = getattr(module, "WARMUP", DEFAULT_WARMUP_DAYS) if module is not None else DEFAULT_WARMUP_DAYS
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return DEFAULT_WARMUP_DAYS
    return value


def _last_traceback_line() -> str:
    """取当前异常的 traceback 末行，须在 ``except`` 块内调用。"""
    lines = traceback.format_exc().rstrip().splitlines()
    return lines[-1] if lines else "未知异常"


def _compute_metrics(
    data: pl.DataFrame,
    factor_output: pl.DataFrame,
    *,
    horizon: int,
) -> Metrics:
    """跑 IC / ICIR / 分层 / 换手指标，全部复用 ``quant.eval.metrics``。"""
    labelled = attach_label(data, horizon=horizon)
    factor = factor_output.rename({"value": "factor"})
    sample = labelled.select("date", "instrument", "label").join(
        factor, on=["date", "instrument"], how="inner"
    )

    summary = summarize_ic(ic_series(sample))
    layered = layered_returns(sample, n_layers=EVAL_LAYERS)
    mono = layer_monotonicity(layered)

    turnover_series = turnover(
        sample.select("date", "instrument", "factor"), top_n=TURNOVER_TOP_N
    )["turnover"]
    finite = turnover_series.filter(turnover_series.is_finite())
    turnover_mean: float | None = float(finite.mean()) if finite.len() else None

    return {
        "rank_ic_mean": summary.mean,
        "rank_ic_std": summary.std,
        "icir": summary.icir,
        "ic_win_rate": summary.ic_win_rate,
        "n_days": summary.n_days,
        "mono": mono,
        "turnover_mean": turnover_mean,
    }


# ---------------------------------------------------------------------------
# 评估管线
# ---------------------------------------------------------------------------


def evaluate_factor(
    factor_path: Path,
    data: pl.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    factor_library_dir: str | Path | None = None,
) -> FactorEvaluation:
    """对单个因子跑完整评估管线。

    ``data`` 为已加载的全样本行情（12 列 ``DAILY_BARS`` 或前 10 列均可，函数内
    自行 ``select``）。任一步失败即短路返回，``ok=False``，``stage`` / ``error``
    指明失败处。全流程跑完返回 ``ok=True``、``stage="done"``，``gate_passed``
    由 :data:`RANK_IC_MIN` / :data:`ICIR_MIN` / :data:`MONO_MIN` 与行为相关性
    阈值 :data:`MAX_CORR_REJECT` 共同决定。

    ``factor_library_dir`` 给出因子库目录时，指标阶段后计算新因子与库内 pool 因子的
    行为相关性，写入 ``metrics["max_corr"]``（``max(|逐日截面相关均值|)``），
    ``max_corr > MAX_CORR_REJECT`` 判冗余并拒绝。目录下无 ``registry.json`` 或库为空时
    跳过该阶段，``max_corr`` 为 None，行为与不传该参数一致。
    """
    factor_path = Path(factor_path)

    # 1. 复杂度度量：超标直接判不合格。
    try:
        complexity = measure_complexity(
            factor_path,
            max_cyclomatic=MAX_CYCLOMATIC,
            max_ast_nodes=MAX_AST_NODES,
            max_fields=MAX_FIELDS,
        )
    except Exception:  # noqa: BLE001 - 文件缺失 / 语法错误交由 load 阶段报错
        complexity = None
    complexity_dict = _complexity_dict(complexity) if complexity is not None else None
    if complexity is not None and not complexity.ok:
        detail = "；".join(complexity.violations) or "复杂度超标"
        return _failure("complexity", f"复杂度超标：{detail}", complexity=complexity_dict)

    # 2. 动态加载因子。
    try:
        compute = load_factor(factor_path)
    except FactorLoadError as exc:
        return _failure("load", str(exc), complexity=complexity_dict)

    # 3. 输入 schema 自检，防缓存漂移。
    try:
        factor_input = data.select(list(FACTOR_INPUT_COLUMNS))
        validate_input(factor_input)
    except Exception as exc:  # noqa: BLE001 - schema / 缺列统一归 schema 阶段
        return _failure("schema", f"因子输入 schema 不符：{exc}", complexity=complexity_dict)

    # 4. 计算因子并校验输出。
    try:
        raw_output = compute(factor_input)
    except Exception:  # noqa: BLE001 - 任何因子内部异常都记 compute 阶段
        return _failure(
            "compute",
            f"因子计算抛异常：{_last_traceback_line()}",
            complexity=complexity_dict,
        )
    try:
        validate_output(raw_output)
    except Exception as exc:  # noqa: BLE001
        return _failure("schema", f"因子输出 schema 不符：{exc}", complexity=complexity_dict)

    # 5. 截断重算检测前视。
    warmup = _factor_warmup(compute)
    try:
        truncation = check_truncation(compute, factor_input, warmup=warmup)
    except Exception as exc:  # noqa: BLE001 - 因子在截断输入上抛错视同检测失败
        return _failure(
            "truncation", f"截断重算执行失败：{exc}", complexity=complexity_dict
        )
    truncation_dict = _truncation_dict(truncation)
    if not truncation.ok:
        if truncation.skipped_reason is not None:
            error = f"截断重算不可执行：{truncation.skipped_reason}"
        else:
            first = truncation.failures[0]
            error = (
                f"截断重算检出前视：首个不一致日 {first.date}，"
                f"不一致 {first.n_mismatch} 只（共 {truncation.n_checks} 个检测日）"
            )
        return _failure(
            "truncation", error, complexity=complexity_dict, truncation=truncation_dict
        )

    # 6. 指标。
    try:
        metrics = _compute_metrics(data, raw_output, horizon=horizon)
    except Exception as exc:  # noqa: BLE001
        return _failure(
            "metrics",
            f"指标计算失败：{exc}",
            complexity=complexity_dict,
            truncation=truncation_dict,
        )

    # 7. 行为相关性查重：与库内 pool 因子逐日截面 Pearson 相关的时间序列均值，
    #    取绝对值最大者。factor_library_dir 为空 / registry 缺失 / 库为空时 max_corr
    #    保持 None，跳过该阶段。
    max_corr: float | None = None
    if factor_library_dir is not None:
        try:
            library_values = load_library_values(factor_library_dir, factor_input)
            if library_values:
                max_corr = max_library_corr(raw_output, library_values).max_corr
        except Exception as exc:  # noqa: BLE001 - registry 非法等归 metrics 阶段
            return _failure(
                "metrics",
                f"相关性查重失败：{exc}",
                complexity=complexity_dict,
                truncation=truncation_dict,
            )
    metrics["max_corr"] = max_corr

    # 8. 门控。行为相关性超过阈值即判冗余，与 IC 门控一并决定入库。
    rank_ic_mean = metrics["rank_ic_mean"]
    icir = metrics["icir"]
    mono = metrics["mono"]
    redundant = isinstance(max_corr, float) and max_corr > MAX_CORR_REJECT
    gate_passed = (
        isinstance(rank_ic_mean, float)
        and rank_ic_mean >= RANK_IC_MIN
        and isinstance(icir, float)
        and icir >= ICIR_MIN
        and isinstance(mono, float)
        and mono > MONO_MIN
        and not redundant
    )
    return FactorEvaluation(
        ok=True,
        stage="done",
        error=None,
        gate_passed=gate_passed,
        metrics=metrics,
        complexity=complexity_dict,
        truncation=truncation_dict,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(value: str) -> date:
    """把 ``YYYY-MM-DD`` 解析为 :class:`datetime.date`。"""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{value!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="对单个因子跑完整评估管线并输出 score.json"
    )
    parser.add_argument("factor", help="因子 .py 文件路径")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument("--out", default=DEFAULT_OUT, help="score.json 输出路径")
    parser.add_argument("--start", type=_parse_date, default=None, help="起始日 YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, default=None, help="结束日 YYYY-MM-DD")
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON, help=f"标签 horizon，默认 {DEFAULT_HORIZON}"
    )
    parser.add_argument(
        "--factor-library-dir",
        default=DEFAULT_FACTOR_LIBRARY_DIR,
        help=f"因子库目录（含 registry.json），默认 {DEFAULT_FACTOR_LIBRARY_DIR}/；目录不存在则跳过查重",
    )
    return parser


def _fmt(value: object, digits: int) -> str:
    """格式化为定宽小数，非 float 记 ``n/a``。"""
    return "n/a" if not isinstance(value, float) else f"{value:.{digits}f}"


def _notes(evaluation: FactorEvaluation) -> str:
    """生成 ``score.json`` 的 ``notes`` 文本。"""
    metrics = evaluation.metrics
    n_days = metrics.get("n_days")
    max_corr = _fmt(metrics.get("max_corr"), 2)
    if not isinstance(n_days, int) or n_days == 0:
        return f"无有效 IC 日（n_days=0），max_corr={max_corr}，score 记 0"
    body = (
        f"rank_ic={_fmt(metrics.get('rank_ic_mean'), 4)} "
        f"icir={_fmt(metrics.get('icir'), 2)} "
        f"mono={_fmt(metrics.get('mono'), 2)} "
        f"max_corr={max_corr} "
        f"n_days={n_days}"
    )
    return f"过门控：{body}" if evaluation.gate_passed else f"未过门控：{body}"


def _score_payload(evaluation: FactorEvaluation) -> dict[str, object]:
    """由评估结果构造 ``score.json`` 内容。

    ``score = rank_ic_mean``（连续质量信号，可负）；n_days 为 0 时记 0.0。
    """
    rank_ic_mean = evaluation.metrics.get("rank_ic_mean")
    score = float(rank_ic_mean) if isinstance(rank_ic_mean, float) else 0.0
    return {
        "score": score,
        "higher_is_better": True,
        "notes": _notes(evaluation),
        "details": {
            "gate_passed": evaluation.gate_passed,
            "metrics": evaluation.metrics,
            "complexity": evaluation.complexity,
            "truncation": evaluation.truncation,
        },
    }


def _reconfigure_stdout() -> None:
    """Windows 控制台下把 stdout 切到 UTF-8，失败则忽略。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：成功跑完写 ``score.json`` 返回 0；硬失败不写文件返回 1。"""
    _reconfigure_stdout()
    args = _build_parser().parse_args(argv)

    data = load_bars(Path(args.data_dir), start=args.start, end=args.end)
    if data.height == 0:
        print("未读到任何行情，检查 --data-dir / --start / --end", file=sys.stderr)
        return 1

    evaluation = evaluate_factor(
        Path(args.factor),
        data,
        horizon=args.horizon,
        factor_library_dir=Path(args.factor_library_dir),
    )
    if not evaluation.ok:
        print(f"stage={evaluation.stage} error={evaluation.error}", file=sys.stderr)
        return 1

    payload = _score_payload(evaluation)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"score={payload['score']} gate_passed={evaluation.gate_passed} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
