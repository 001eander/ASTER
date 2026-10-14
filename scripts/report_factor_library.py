"""因子库报告 CLI（issue #116 第一阶段）：pool 因子两两相关热力图。

从 registry 读 pool 因子集，用 :func:`quant.factor_lib.correlation.load_library_values`
在同一份行情面板上现算取值，:func:`quant.factor_lib.prune.build_corr_matrix` 出两两
「逐日截面相关均值」——与入库查重、定期整库完全同一口径。图上注明冗余边界
``|corr| > threshold``，阈值来自 :func:`quant.factor_lib.prune.cluster_corr_threshold`
（单一来源 :data:`quant.eval.factor.MAX_CORR_REJECT`），源码不硬编码数值。

用法::

    # 默认：全样本窗口，输出 runs/reports/factor_corr_heatmap.png
    uv run python scripts/report_factor_library.py

    # 指定相关性窗口与输出路径
    uv run python scripts/report_factor_library.py \
        --start 2023-01-01 --end 2024-12-31 --out runs/reports/corr_2024.png

图面文本（轴标签、标题、色标）只用 ASCII：factor_id 本就是 ASCII，无显示环境
（``Agg``）下不依赖中文字体，不出豆腐块。取值缺失或相关不可得的格子留白（灰底）。

只读口径：脚本只读 registry 与行情缓存，不修改 ``factor_library/``；产物落 ``runs/``。

退出码：0 表示报告产出（含 pool 为空的空跑）；1 表示读不到行情；2 表示 registry
缺失或非法。
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 无显示环境，必须在 pyplot 之前
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quant.data.cache import load_bars  # noqa: E402
from quant.factor_lib.correlation import load_library_values  # noqa: E402
from quant.factor_lib.prune import (  # noqa: E402
    build_corr_matrix,
    cluster_corr_threshold,
)
from quant.factor_lib.registry import load_registry, pool_factors  # noqa: E402
from quant.factor_lib.schema import FactorLibError  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 默认行情缓存目录。
DEFAULT_DATA_DIR: str = "data"

#: 默认因子库目录（含 registry.json）。
DEFAULT_FACTOR_LIBRARY_DIR: str = "factor_library"

#: 默认热力图输出路径（``runs/`` 不入 git）。
DEFAULT_OUT: str = "runs/reports/factor_corr_heatmap.png"

#: 热力图 DPI。
PNG_DPI: int = 120

#: 色标范围：相关系数的理论区间。
CORR_VMIN: float = -1.0
CORR_VMAX: float = 1.0

#: 超过该因子数时不再逐格标注数值（避免格子过密看不清）。
ANNOTATE_MAX_FACTORS: int = 40

#: 退出码：报告产出（含 pool 为空的空跑）。
EXIT_OK: int = 0

#: 退出码：读不到行情。
EXIT_DATA_ERROR: int = 1

#: 退出码：registry 缺失或非法。
EXIT_REGISTRY_ERROR: int = 2


# ---------------------------------------------------------------------------
# 数据组装（画图前的纯函数，单测锚点）
# ---------------------------------------------------------------------------


def assemble_corr_grid(
    factor_ids: Sequence[str],
    matrix: Mapping[tuple[str, str], float],
) -> np.ndarray:
    """把上三角相关字典展开成 N×N 网格：对角为 1，对称，缺失对为 NaN。

    :param factor_ids: 轴上的因子顺序（调用方给排序后的 pool 因子集）。
    :param matrix: :func:`build_corr_matrix` 的输出，``{(较小 id, 较大 id): 相关}``。
    :return: ``float64`` 网格，``grid[i, j]`` 为 ``factor_ids[i]`` × ``factor_ids[j]``
        的逐日截面相关均值；任一侧取值缺失或相关不可得时为 NaN（图上留白）。
    """
    n = len(factor_ids)
    grid = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(grid, 1.0)
    position = {factor_id: index for index, factor_id in enumerate(factor_ids)}
    for (left, right), corr in matrix.items():
        if left not in position or right not in position:
            continue
        i, j = position[left], position[right]
        grid[i, j] = corr
        grid[j, i] = corr
    return grid


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------


def plot_corr_heatmap(
    factor_ids: Sequence[str],
    grid: np.ndarray,
    *,
    threshold: float,
    out_path: Path,
) -> Path:
    """画 N×N 相关热力图并落盘，返回输出路径。

    色标居中于 0、范围 ``[-1, 1]``；NaN 格子（取值缺失 / 相关不可得）留白。
    冗余边界 ``|corr| > threshold`` 写进标题与色标说明，阈值由调用方从
    :func:`cluster_corr_threshold` 取，本函数不硬编码。
    """
    n = len(factor_ids)
    side = max(6.0, 0.5 * n + 2.5)
    fig, ax = plt.subplots(figsize=(side, side))

    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#d9d9d9")
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(masked, cmap=cmap, vmin=CORR_VMIN, vmax=CORR_VMAX)

    tick_fontsize = max(5.0, min(9.0, 180.0 / max(n, 1)))
    ax.set_xticks(range(n), factor_ids, rotation=90, fontsize=tick_fontsize)
    ax.set_yticks(range(n), factor_ids, fontsize=tick_fontsize)
    ax.set_title(
        "Factor library pairwise correlation\n"
        f"(mean of daily cross-section Pearson; redundant boundary |corr| > {threshold:.2f})"
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("corr in [-1, 1]; grey = unavailable; dashed = redundant boundary")
    for sign in (1.0, -1.0):
        colorbar.ax.axhline(sign * threshold, color="black", linewidth=1.0, linestyle="--")

    if n <= ANNOTATE_MAX_FACTORS:
        for i in range(n):
            for j in range(n):
                value = grid[i, j]
                if not np.isfinite(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=max(4.0, tick_fontsize - 2.0),
                    color="black" if abs(value) < 0.6 else "white",
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI)
    plt.close(fig)
    return out_path


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
    parser = argparse.ArgumentParser(
        description="因子库报告：pool 因子两两相关热力图（issue #116 第一阶段）"
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
        "--out", default=DEFAULT_OUT, help=f"热力图输出路径，默认 {DEFAULT_OUT}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：现算 pool 因子取值 → 相关矩阵 → 热力图 PNG。"""
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

    pool_ids = sorted(entry.factor_id for entry in pool_factors(registry))
    if not pool_ids:
        print("registry 中没有 pool 因子，无图可出")
        return EXIT_OK

    data = load_bars(Path(args.data_dir), start=args.start, end=args.end)
    if data.height == 0:
        print("未读到任何行情，检查 --data-dir / --start / --end", file=sys.stderr)
        return EXIT_DATA_ERROR

    # registry 已在本函数前段加载成功，这里只可能因并发改动再抛 FactorLibError。
    values = load_library_values(library_dir, data)

    unvaluable = [factor_id for factor_id in pool_ids if factor_id not in values]
    if unvaluable:
        print(f"未取到值的 pool 因子（图上留白）：{'、'.join(unvaluable)}")

    matrix = build_corr_matrix(
        {factor_id: values[factor_id] for factor_id in pool_ids if factor_id in values}
    )
    grid = assemble_corr_grid(pool_ids, matrix)

    threshold = cluster_corr_threshold()
    redundant = [
        (left, right) for (left, right), corr in matrix.items() if abs(corr) > threshold
    ]

    out_path = plot_corr_heatmap(
        pool_ids, grid, threshold=threshold, out_path=Path(args.out)
    )
    print(
        f"pool 因子 {len(pool_ids)} 个，相关对 {len(matrix)} 对，"
        f"越过冗余边界（|corr| > {threshold:.2f}）{len(redundant)} 对"
    )
    for left, right in redundant:
        print(f"  冗余对：{left} × {right} = {matrix[(left, right)]:.4f}")
    print(f"热力图已写入 {out_path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
