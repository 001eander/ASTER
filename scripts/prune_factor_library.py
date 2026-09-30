"""因子库定期整库 CLI：冗余聚簇降级 + pool 容量上限。

用法::

    # 只看计划，不写盘（推荐先跑）
    uv run python scripts/prune_factor_library.py --dry-run

    # 只用一个窗口的相关性（默认全样本）
    uv run python scripts/prune_factor_library.py --start 2023-01-01 --end 2024-12-31

规则见 :mod:`quant.factor_lib.prune`：``|corr| > 0.7`` 连边取连通分量，每簇留
``rank_ic`` 最高者；随后 pool 数不超过库内总条目数的一半（向下取整），超出部分按
质量从弱到强降级。被降级的条目留在 lib 里、保留血统，只是不再参与建模。

这是按需手动运行的脚本，不挂在 daily pipeline 上；跑批前先 ``--dry-run`` 看计划。

退出码：0 表示计划算出（含无需整库与 dry-run）；1 表示读不到行情；2 表示 registry
缺失或非法。写盘用 :func:`quant.factor_lib.registry.save_registry` 的原子写。
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.cache import load_bars  # noqa: E402
from quant.factor_lib.correlation import load_library_values  # noqa: E402
from quant.factor_lib.prune import (  # noqa: E402
    REASON_CLUSTER,
    PrunePlan,
    apply_prune,
    plan_prune,
)
from quant.factor_lib.registry import (  # noqa: E402
    load_registry,
    pool_factors,
    save_registry,
)
from quant.factor_lib.schema import FactorLibError  # noqa: E402

#: 默认行情缓存目录。
DEFAULT_DATA_DIR: str = "data"

#: 默认因子库目录（含 registry.json）。
DEFAULT_FACTOR_LIBRARY_DIR: str = "factor_library"

#: 退出码：计划跑完（含干跑 / 无需整库）。
EXIT_OK: int = 0

#: 退出码：读不到行情。
EXIT_DATA_ERROR: int = 1

#: 退出码：registry 缺失或非法。
EXIT_REGISTRY_ERROR: int = 2


def _parse_date(value: str) -> date:
    """把 ``YYYY-MM-DD`` 解析为 :class:`datetime.date`。"""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{value!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="因子库定期整库：冗余聚簇降级与 pool 容量上限"
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--factor-library-dir",
        default=DEFAULT_FACTOR_LIBRARY_DIR,
        help=f"因子库目录（含 registry.json），默认 {DEFAULT_FACTOR_LIBRARY_DIR}/",
    )
    parser.add_argument(
        "--start", type=_parse_date, default=None, help="相关性窗口起始日 YYYY-MM-DD，默认全样本"
    )
    parser.add_argument(
        "--end", type=_parse_date, default=None, help="相关性窗口结束日 YYYY-MM-DD，默认全样本"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印计划，不写回 registry.json"
    )
    return parser


def _describe(demotion_reason: str, kept: str | None) -> str:
    """把降级原因写成中文说明。"""
    if demotion_reason == REASON_CLUSTER:
        return f"簇内冗余，同簇保留 {kept}"
    return "超出容量上限"


def _print_report(
    plan: PrunePlan,
    *,
    unvaluable: list[str],
) -> None:
    """打印中文计划报告。"""
    if unvaluable:
        print(f"未取到值的 pool 因子（不参与聚簇）：{'、'.join(unvaluable)}")
    if plan.is_empty:
        print(
            f"无需整库：pool {plan.pool_before} 个，"
            f"容量上限 {plan.capacity_limit} 个（库内总条目数的一半）"
        )
        return
    print(
        f"pool {plan.pool_before} → {plan.pool_after} 个，"
        f"容量上限 {plan.capacity_limit} 个；降级 {len(plan.demotions)} 个："
    )
    for demotion in plan.demotions:
        print(
            f"  降级 {demotion.factor_id}："
            f"{_describe(demotion.reason, demotion.kept)}"
        )
    if plan.capacity_limit == 0 and plan.pool_after == 0:
        print("注意：库内总条目数不足 2，容量上限为 0，写盘会清空 pool；确认后再去掉 --dry-run")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：算出计划并（非 dry-run 时）原子写回 registry。"""
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
        values = load_library_values(library_dir, data)
    except FactorLibError as exc:
        print(f"registry 不可用：{exc}", file=sys.stderr)
        return EXIT_REGISTRY_ERROR

    plan = plan_prune(registry, values)
    unvaluable = sorted(
        entry.factor_id
        for entry in pool_factors(registry)
        if entry.factor_id not in values
    )
    _print_report(plan, unvaluable=unvaluable)

    if plan.is_empty or args.dry_run:
        if not plan.is_empty:
            print("dry-run：计划如上，未写盘")
        return EXIT_OK

    path = save_registry(apply_prune(registry, plan), library_dir)
    print(f"已写回 {path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
