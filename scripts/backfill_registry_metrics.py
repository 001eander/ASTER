"""因子库 metrics 回填 CLI：批量评估库内因子并把 rank_ic / icir / max_corr 写回 registry。

用法::

    # 只看结果，不写盘（推荐先跑）
    uv run python scripts/backfill_registry_metrics.py --data-dir data --dry-run

    # 真实回填（data/ 在主仓库，worktree 里用绝对路径）
    uv run python scripts/backfill_registry_metrics.py \
        --data-dir C:\\Users\\陶唐\\Workspace\\ASTER\\data

背景（issue #108）：#25 建 registry 时 13 个种子因子的 metrics 各键均为 null，
整库脚本 :mod:`quant.factor_lib.prune` 的质量排序把「未评估」当最弱，对真实库跑
会把 pool 从 13 降到 6。本脚本用既有评估管线 :func:`quant.eval.factor.evaluate_factor`
对库内因子批量补跑一次，把 :data:`RANK_IC_KEY` / ``icir`` / ``max_corr`` 写回。

实现要点
--------
- 行情只经 :func:`quant.data.cache.load_bars` 加载一次，循环因子复用同一份 ``data``。
- 库因子取值只经 :func:`quant.factor_lib.correlation.load_library_values` 取一次；
  逐个被评因子时从该映射里剔除自身再传给 ``evaluate_factor(library_values=...)``。
  库里含被评因子自身，不剔除会让自相关恒为 1.0 污染 ``max_corr``。
- 评估硬失败（``ok=False``）的因子保留原 metrics 不动，记入失败清单；
  只要有一次硬失败脚本以退出码 :data:`EXIT_EVAL_ERROR` 结束。

退出码：0 全部评估成功（含 dry-run）；1 读不到行情；2 registry 缺失或非法；
3 有因子评估硬失败（已写入成功部分）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.cache import load_bars  # noqa: E402
from quant.eval.factor import (  # noqa: E402
    DEFAULT_HORIZON,
    FactorEvaluation,
    evaluate_factor,
)
from quant.factor_api.spec import FACTOR_INPUT_COLUMNS  # noqa: E402
from quant.factor_lib.correlation import load_library_values  # noqa: E402
from quant.factor_lib.registry import load_registry, save_registry  # noqa: E402
from quant.factor_lib.schema import (  # noqa: E402
    FactorEntry,
    FactorLibError,
    Registry,
)

#: 默认行情缓存目录。
DEFAULT_DATA_DIR: str = "data"

#: 默认因子库目录（含 registry.json）。
DEFAULT_FACTOR_LIBRARY_DIR: str = "factor_library"

#: registry ``metrics`` 里写回的质量键，取评估指标 ``rank_ic_mean``。
RANK_IC_KEY: str = "rank_ic"

#: registry ``metrics`` 里写回的 ICIR 键。
ICIR_KEY: str = "icir"

#: registry ``metrics`` 里写回的相关性键。
MAX_CORR_KEY: str = "max_corr"

#: 退出码：全部评估成功（含 dry-run）。
EXIT_OK: int = 0

#: 退出码：读不到行情。
EXIT_DATA_ERROR: int = 1

#: 退出码：registry 缺失或非法。
EXIT_REGISTRY_ERROR: int = 2

#: 退出码：有因子评估硬失败（成功部分已写盘）。
EXIT_EVAL_ERROR: int = 3


def _parse_date(value: str) -> date:
    """把 ``YYYY-MM-DD`` 解析为 :class:`datetime.date`。"""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{value!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量评估库内因子并把 rank_ic / icir / max_corr 写回 registry"
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--factor-library-dir",
        default=DEFAULT_FACTOR_LIBRARY_DIR,
        help=f"因子库目录（含 registry.json），默认 {DEFAULT_FACTOR_LIBRARY_DIR}/",
    )
    parser.add_argument(
        "--start", type=_parse_date, default=None, help="评估窗口起始日 YYYY-MM-DD，默认全样本"
    )
    parser.add_argument(
        "--end", type=_parse_date, default=None, help="评估窗口结束日 YYYY-MM-DD，默认全样本"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"标签 horizon，默认 {DEFAULT_HORIZON}",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写回 registry.json")
    return parser


def _resolve_code_path(entry: FactorEntry, repo_root: Path) -> Path:
    """把条目 ``code_path`` 解析为绝对路径；相对路径以仓库根为基准。"""
    code_path = Path(entry.code_path)
    if not code_path.is_absolute():
        code_path = repo_root / code_path
    return code_path


def _metric(evaluation: FactorEvaluation, key: str) -> float | None:
    """取评估 ``metrics[key]``，非 float 一律视为不可得（None）。"""
    value = evaluation.metrics.get(key)
    return value if isinstance(value, float) else None


def _backfilled_metrics(entry: FactorEntry, evaluation: FactorEvaluation) -> dict[str, float | None]:
    """保留原 metrics 的其余键，覆写 rank_ic / icir / max_corr。"""
    metrics = dict(entry.metrics)
    metrics[RANK_IC_KEY] = _metric(evaluation, "rank_ic_mean")
    metrics[ICIR_KEY] = _metric(evaluation, "icir")
    metrics[MAX_CORR_KEY] = _metric(evaluation, "max_corr")
    return metrics


def _fmt(value: object, digits: int = 4) -> str:
    """格式化为定宽小数，非 float 记 ``n/a``。"""
    return f"{value:.{digits}f}" if isinstance(value, float) else "n/a"


def _print_table(rows: list[tuple[str, str, str, str, str]]) -> None:
    """打印 ``factor_id / rank_ic / icir / max_corr / note`` 汇总表。"""
    header = ("factor_id", "rank_ic", "icir", "max_corr", "note")
    widths = [
        max(len(header[index]), max((len(row[index]) for row in rows), default=0))
        for index in range(len(header))
    ]
    line = "  ".join(cell.ljust(width) for cell, width in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def _evaluate_entries(
    registry: Registry,
    library_dir: Path,
    data: pl.DataFrame,
    *,
    horizon: int,
) -> tuple[Registry, list[tuple[str, str, str, str, str]], list[tuple[str, str]]]:
    """逐个评估库内条目，返回 ``(新 registry, 表格行, 失败清单)``。

    ``data`` 为完整行情面板；库因子取值只在评估开始前取一次，逐个条目剔除自身复用。
    """
    factor_input = data.select(list(FACTOR_INPUT_COLUMNS))
    library_values = load_library_values(library_dir, factor_input)
    repo_root = library_dir.resolve().parent

    rows: list[tuple[str, str, str, str, str]] = []
    failures: list[tuple[str, str]] = []
    updated: list[FactorEntry] = []

    for entry in registry.factors:
        others = {
            factor_id: values
            for factor_id, values in library_values.items()
            if factor_id != entry.factor_id
        }
        evaluation = evaluate_factor(
            _resolve_code_path(entry, repo_root),
            data,
            horizon=horizon,
            library_values=others,
        )
        if not evaluation.ok:
            failures.append((entry.factor_id, f"{evaluation.stage}: {evaluation.error}"))
            rows.append((entry.factor_id, "ERR", "-", "-", f"{evaluation.stage} 失败"))
            updated.append(entry)
            continue

        metrics = _backfilled_metrics(entry, evaluation)
        updated.append(replace(entry, metrics=metrics))
        rows.append(
            (
                entry.factor_id,
                _fmt(metrics[RANK_IC_KEY]),
                _fmt(metrics[ICIR_KEY], 2),
                _fmt(metrics[MAX_CORR_KEY], 2),
                "已回填",
            )
        )

    return Registry(version=registry.version, factors=tuple(updated)), rows, failures


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：一次加载行情，批量评估并（非 dry-run 时）原子写回 registry。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    library_dir = Path(args.factor_library_dir)

    try:
        registry = load_registry(library_dir)
    except FactorLibError as exc:
        print(f"registry 不可用：{exc}", file=sys.stderr)
        return EXIT_REGISTRY_ERROR

    data = load_bars(Path(args.data_dir), start=args.start, end=args.end)
    if data.height == 0:
        print("未读到任何行情，检查 --data-dir / --start / --end", file=sys.stderr)
        return EXIT_DATA_ERROR

    try:
        updated, rows, failures = _evaluate_entries(
            registry, library_dir, data, horizon=args.horizon
        )
    except FactorLibError as exc:
        print(f"registry 不可用：{exc}", file=sys.stderr)
        return EXIT_REGISTRY_ERROR

    _print_table(rows)
    nulls_replaced = sum(
        1
        for entry in updated.factors
        if any(value is not None for value in entry.metrics.values())
    )
    print(f"\n共 {len(updated.factors)} 个条目，{nulls_replaced} 个已有数值指标。")

    if failures:
        print(f"以下 {len(failures)} 个因子评估失败，metrics 保持原值：", file=sys.stderr)
        for factor_id, detail in failures:
            print(f"  {factor_id}：{detail}", file=sys.stderr)

    if args.dry_run:
        print("dry-run：未写回 registry.json")
        return EXIT_EVAL_ERROR if failures else EXIT_OK

    path = save_registry(updated, library_dir)
    print(f"已写回 {path}")
    return EXIT_EVAL_ERROR if failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
