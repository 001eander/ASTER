"""基线因子集评估 CLI：对 12 个手工因子逐个跑 IC / 分层。

用法::

    uv run python scripts/eval_baseline_factors.py --data-dir data
    uv run python scripts/eval_baseline_factors.py --start 2020-01-01 --end 2023-12-31 \
        --instruments 600519,300750

流程对每个因子：``compute`` → ``attach_label``（T+1 开盘 → T+2 开盘）→
``ic_series`` / ``summarize_ic`` + ``layered_returns``，最后打印一张汇总表。
不触网；数据来自 ``data/`` 缓存（``quant.data.cache.load_bars``）。
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.cache import load_bars  # noqa: E402
from quant.eval.metrics import (  # noqa: E402
    DEFAULT_LAYERS,
    ic_series,
    layer_monotonicity,
    layered_returns,
    summarize_ic,
)
from quant.labels import attach_label  # noqa: E402

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 默认数据目录。
DEFAULT_DATA_DIR: str = "data"

#: 因子库目录（相对仓库根）。
FACTOR_LIBRARY_DIR: str = "factor_library"

#: 分层评估的层数。
EVAL_LAYERS: int = DEFAULT_LAYERS

logger = logging.getLogger("eval_baseline_factors")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{value!r}") from exc


def _discover_factors() -> list[str]:
    """列出 factor_library 下的因子模块名（忽略下划线开头文件）。"""
    root = Path(__file__).resolve().parents[1] / FACTOR_LIBRARY_DIR
    return sorted(p.stem for p in root.glob("*.py") if not p.stem.startswith("_"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对基线因子集跑 IC / 分层评估")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument("--start", type=_parse_date, default=None, help="起始日 YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, default=None, help="结束日 YYYY-MM-DD")
    parser.add_argument(
        "--instruments",
        default=None,
        help="逗号分隔的证券代码，默认全市场；示例 600519,300750",
    )
    parser.add_argument(
        "--layers", type=int, default=EVAL_LAYERS, help=f"分层数，默认 {EVAL_LAYERS}"
    )
    return parser


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    instruments = (
        [item.strip() for item in args.instruments.split(",") if item.strip()]
        if args.instruments
        else None
    )
    bars = load_bars(
        Path(args.data_dir), instruments=instruments, start=args.start, end=args.end
    )
    if bars.height == 0:
        logger.error("未读到任何行情，检查 --data-dir / --start / --end / --instruments")
        return 1
    logger.info(
        "载入 %d 行 / %d 只证券 / %d 个交易日",
        bars.height,
        bars["instrument"].n_unique(),
        bars["date"].n_unique(),
    )

    labelled = attach_label(bars)

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for name in _discover_factors():
        module = importlib.import_module(f"{FACTOR_LIBRARY_DIR}.{name}")
        factor = module.compute(bars).rename({"value": "factor"})
        sample = labelled.select("date", "instrument", "label").join(
            factor, on=["date", "instrument"], how="inner"
        )
        try:
            summary = summarize_ic(ic_series(sample))
            layered = layered_returns(sample, n_layers=args.layers)
            mono = layer_monotonicity(layered)
        except ValueError as exc:  # 有效证券数不足等，跳过该因子不中断整批
            logger.warning("因子 %s 评估失败：%s", name, exc)
            rows.append((name, "ERR", "-", "-", "-", "-", "-"))
            continue
        rows.append(
            (
                name,
                str(summary.n_days),
                _fmt(summary.mean),
                _fmt(summary.std),
                _fmt(summary.icir),
                _fmt(summary.ic_win_rate),
                _fmt(mono),
            )
        )

    header = ("factor", "n_days", "mean_IC", "std_IC", "ICIR", "win_rate", "mono")
    widths = [
        max(len(header[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(header))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    print(
        f"\nIC 方向：因子值越大越看多；mono 为分层收益对层号的 Spearman 相关，"
        f"1.0 表示层号越高收益越高。层数 {args.layers}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
